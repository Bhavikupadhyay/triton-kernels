"""
Kernel:   batched_matmul
Category: matmul
Complexity: O(B × M × N × K)
Memory bound: No — compute bound
PyTorch equivalent: torch.bmm(A, B)
References:
  - Extends tiled_matmul.py — see that file for the tiling and group ordering explanation.

How it works:
Batched matrix multiply: C[b] = A[b] @ B[b] for b in 0..B-1.
A is (B, M, K), B is (B, K, N), C is (B, M, N).

The kernel is tiled_matmul with one extra program dimension:
  pid   = tl.program_id(0) — tile index within one (M, N) output matrix (same as tiled)
  pid_b = tl.program_id(1) — batch index

Grid: (cdiv(M, BLOCK_M) * cdiv(N, BLOCK_N), B_sz)

Each program adds pid_b * stride_batch to the A, B, and C base pointers before
running the identical group-ordered K-loop. Everything else — group ordering,
autotuning configs, accumulator, masking — is identical to tiled_matmul.

TFLOPS metric: (2 × B × M × N × K × 1e-12) / (ms × 1e-3)
"""

import torch
import triton
import triton.language as tl


# ── 1. Autotune configs (same set as tiled_matmul) ────────────────────────────

def get_autotune_configs():
    return [
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64,  "GROUP_SIZE": 8}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 256, "BLOCK_K": 32,  "GROUP_SIZE": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32,  "GROUP_SIZE": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 32,  "GROUP_SIZE": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 32,  "GROUP_SIZE": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32,  "BLOCK_K": 32,  "GROUP_SIZE": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 32,  "BLOCK_K": 32,  "GROUP_SIZE": 8}, num_stages=5, num_warps=2),
        triton.Config({"BLOCK_M": 32,  "BLOCK_N": 64,  "BLOCK_K": 32,  "GROUP_SIZE": 8}, num_stages=5, num_warps=2),
    ]


# ── 2. Triton kernel ──────────────────────────────────────────────────────────

@triton.autotune(configs=get_autotune_configs(), key=["M", "N", "K"])
@triton.jit
def batched_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_ab, stride_am, stride_ak,
    stride_bb, stride_bk, stride_bn,
    stride_cb, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    pid   = tl.program_id(0)   # tile index within the (M, N) plane
    pid_b = tl.program_id(1)   # batch index

    # Offset base pointers to this batch element
    A_ptr = A_ptr + pid_b * stride_ab
    B_ptr = B_ptr + pid_b * stride_bb
    C_ptr = C_ptr + pid_b * stride_cb

    # Group ordering — identical to tiled_matmul
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE * num_pid_n
    group_id     = pid // num_pid_in_group
    first_pid_m  = group_id * GROUP_SIZE
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        rk = k * BLOCK_K + tl.arange(0, BLOCK_K)

        A = tl.load(
            A_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak,
            mask=(rm[:, None] < M) & (rk[None, :] < K),
            other=0.0,
        )
        B = tl.load(
            B_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn,
            mask=(rk[:, None] < K) & (rn[None, :] < N),
            other=0.0,
        )

        acc += tl.dot(A, B)

    C = acc.to(A_ptr.dtype.element_ty)
    tl.store(
        C_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn,
        C,
        mask=(rm[:, None] < M) & (rn[None, :] < N),
    )


# ── 3. Python wrapper ─────────────────────────────────────────────────────────

def batched_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda, "Inputs must be on CUDA"
    assert A.is_contiguous() and B.is_contiguous(), "Inputs must be contiguous"
    assert A.ndim == 3 and B.ndim == 3, "Inputs must be 3D (B, M, K) and (B, K, N)"
    B_sz, M, K  = A.shape
    B_sz2, K2, N = B.shape
    assert B_sz == B_sz2, f"Batch sizes must match: {B_sz} vs {B_sz2}"
    assert K == K2,       f"Inner dimensions must match: {K} vs {K2}"

    C = torch.empty((B_sz, M, N), device=A.device, dtype=A.dtype)
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),
        B_sz,
    )

    batched_matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1), A.stride(2),
        B.stride(0), B.stride(1), B.stride(2),
        C.stride(0), C.stride(1), C.stride(2),
    )
    return C


# ── 4. Correctness tests ──────────────────────────────────────────────────────

def test_batched_matmul():
    configs = [
        (1,  64,  64,  64),
        (4,  128, 256, 64),
        (8,  256, 128, 512),
        (16, 512, 512, 512),
        (2,  127, 255, 63),   # non-powers-of-2
    ]
    for B_sz, M, N, K in configs:
        A = torch.randn(B_sz, M, K, device="cuda", dtype=torch.float32)
        B = torch.randn(B_sz, K, N, device="cuda", dtype=torch.float32)
        ref = torch.bmm(A, B)
        got = batched_matmul(A, B)
        torch.testing.assert_close(got, ref, rtol=1e-2, atol=1e-2)

    print("test_batched_matmul: PASSED")


# ── 5. Benchmarks ─────────────────────────────────────────────────────────────
#
# Fix B=16, square M=N=K, vary matrix size — mirrors tiled_matmul benchmark
# so the per-matrix overhead of the batch dimension is visible.

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[256, 512, 1024, 2048, 4096],
        x_log=True,
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton (batched)", "PyTorch (bmm)"],
        styles=[("blue", "-"), ("green", "--")],
        ylabel="TFLOPS",
        plot_name="batched_matmul",
        args={"B_sz": 16},
    )
)
def benchmark_batched_matmul(B_sz, N, provider):
    A = torch.randn(B_sz, N, N, device="cuda", dtype=torch.float32)
    B = torch.randn(B_sz, N, N, device="cuda", dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]

    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: batched_matmul(A, B), warmup=25, rep=100, quantiles=quantiles
        )
    else:
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch.bmm(A, B), warmup=25, rep=100, quantiles=quantiles
        )

    tflops = 2 * B_sz * N * N * N * 1e-12
    return tflops / (ms * 1e-3), tflops / (max_ms * 1e-3), tflops / (min_ms * 1e-3)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[256, 512, 1024, 2048, 4096],
        x_log=True,
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton (batched)", "PyTorch (bmm)"],
        styles=[("blue", "-"), ("green", "--")],
        ylabel="TFLOPS",
        plot_name="batched_matmul_steady",
        args={"B_sz": 16},
    )
)
def benchmark_batched_matmul_steady(B_sz, N, provider):
    """Same as benchmark_batched_matmul — run immediately after to capture
    steady-state numbers once the autotuner cache is populated."""
    A = torch.randn(B_sz, N, N, device="cuda", dtype=torch.float32)
    B = torch.randn(B_sz, N, N, device="cuda", dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]

    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: batched_matmul(A, B), warmup=25, rep=100, quantiles=quantiles
        )
    else:
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch.bmm(A, B), warmup=25, rep=100, quantiles=quantiles
        )

    tflops = 2 * B_sz * N * N * N * 1e-12
    return tflops / (ms * 1e-3), tflops / (max_ms * 1e-3), tflops / (min_ms * 1e-3)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_batched_matmul()
    benchmark_batched_matmul.run(print_data=True, show_plots=True)
    benchmark_batched_matmul_steady.run(print_data=True, show_plots=True)
