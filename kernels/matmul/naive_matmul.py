"""
Kernel:   naive_matmul
Category: matmul
Complexity: O(M × N × K)
Memory bound: No — compute bound
PyTorch equivalent: torch.matmul(A, B)
References:
  - Triton matmul tutorial: https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html

How it works:
Matrix multiply C = A @ B where A is (M, K) and B is (K, N).

One Triton program per BLOCK_M × BLOCK_N output tile of C. Each program:
  1. Iterates over the K dimension in steps of BLOCK_K
  2. Loads a (BLOCK_M, BLOCK_K) tile of A and a (BLOCK_K, BLOCK_N) tile of B from HBM
  3. Accumulates the partial dot product into a fp32 accumulator via tl.dot (tensor cores)
  4. Writes the completed (BLOCK_M, BLOCK_N) tile of C to HBM

What makes this "naive":
  - Fixed block sizes (BLOCK_M=BLOCK_N=BLOCK_K=32) — no autotuning
  - No group ordering — programs are assigned output tiles in row-major order with
    no regard for L2 cache reuse. Adjacent programs in the same SM group share
    almost no input data, so each K-loop iteration is a cold HBM read.
  - No software prefetching or pipeline staging

This is the baseline. It shows that even without optimisation, the tiled structure
and tl.dot (tensor core) usage produce respectable TFLOPS. tiled_matmul will show
what happens when group ordering and autotuning are added on top.

TFLOPS metric: (2 × M × N × K × 1e-12) / (ms × 1e-3)
  — factor 2: each multiply-add is 2 floating-point operations
"""

import torch
import triton
import triton.language as tl


# ── 1. Triton kernel ──────────────────────────────────────────────────────────

@triton.jit
def naive_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Row and column offsets for this tile
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # fp32 accumulator for this output tile
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        rk = k * BLOCK_K + tl.arange(0, BLOCK_K)

        # Load A tile (BLOCK_M, BLOCK_K) — cold HBM read every iteration
        A = tl.load(
            A_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak,
            mask=(rm[:, None] < M) & (rk[None, :] < K),
            other=0.0,
        )
        # Load B tile (BLOCK_K, BLOCK_N) — cold HBM read every iteration
        B = tl.load(
            B_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn,
            mask=(rk[:, None] < K) & (rn[None, :] < N),
            other=0.0,
        )

        acc += tl.dot(A, B)

    # Write output tile — cast accumulator back to input dtype
    C = acc.to(A_ptr.dtype.element_ty)
    tl.store(
        C_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn,
        C,
        mask=(rm[:, None] < M) & (rn[None, :] < N),
    )


# ── 2. Python wrapper ─────────────────────────────────────────────────────────

BLOCK_M = 32
BLOCK_N = 32
BLOCK_K = 32

def naive_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda, "Inputs must be on CUDA"
    assert A.is_contiguous() and B.is_contiguous(), "Inputs must be contiguous"
    assert A.ndim == 2 and B.ndim == 2, "Inputs must be 2D"
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, f"Inner dimensions must match: A is ({M},{K}), B is ({K2},{N})"

    C = torch.empty((M, N), device=A.device, dtype=A.dtype)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    naive_matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return C


# ── 3. Correctness tests ──────────────────────────────────────────────────────

def test_naive_matmul():
    configs = [
        (64, 64, 64),
        (128, 256, 64),
        (256, 128, 512),
        (512, 512, 512),
        (1024, 1024, 1024),
        (127, 255, 63),   # non-powers-of-2
    ]
    for M, N, K in configs:
        A = torch.randn(M, K, device="cuda", dtype=torch.float32)
        B = torch.randn(K, N, device="cuda", dtype=torch.float32)
        ref = torch.matmul(A, B)
        got = naive_matmul(A, B)
        torch.testing.assert_close(got, ref, rtol=1e-2, atol=1e-2)

    print("test_naive_matmul: PASSED")


# ── 4. Benchmarks ─────────────────────────────────────────────────────────────
#
# Square matrices M=N=K, varying size — the standard matmul benchmark.

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[256, 512, 1024, 2048, 4096],
        x_log=True,
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton (naive)", "PyTorch (cuBLAS)"],
        styles=[("blue", "-"), ("green", "--")],
        ylabel="TFLOPS",
        plot_name="naive_matmul",
        args={},
    )
)
def benchmark_naive_matmul(N, provider):
    A = torch.randn(N, N, device="cuda", dtype=torch.float32)
    B = torch.randn(N, N, device="cuda", dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]

    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: naive_matmul(A, B), warmup=25, rep=100, quantiles=quantiles
        )
    else:
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch.matmul(A, B), warmup=25, rep=100, quantiles=quantiles
        )

    tflops = 2 * N * N * N * 1e-12
    return tflops / (ms * 1e-3), tflops / (max_ms * 1e-3), tflops / (min_ms * 1e-3)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_naive_matmul()
    benchmark_naive_matmul.run(print_data=True, show_plots=True)
