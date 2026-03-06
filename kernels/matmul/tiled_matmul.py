"""
Kernel:   tiled_matmul
Category: matmul
Complexity: O(M × N × K)
Memory bound: No — compute bound
PyTorch equivalent: torch.matmul(A, B)
References:
  - Triton matmul tutorial: https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
  - "Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking":
      https://arxiv.org/abs/1804.06826

What this adds over naive_matmul:

  1. Group ordering (L2 cache swizzle)
     The launch grid maps program ID → (pid_m, pid_n) so that GROUP_SIZE
     consecutive programs all share the same row of A tiles. They load different
     columns of B but the same A stripe, so if they run on nearby SMs the A tile
     stays in L2 across all GROUP_SIZE programs rather than being evicted after
     one use. This reduces HBM reads of A by up to GROUP_SIZE×.

     Mapping:
       pid_m = (pid % programs_in_group) + group_id * GROUP_SIZE
       pid_n =  pid // programs_in_group
     where programs_in_group = min(GROUP_SIZE, cdiv(M, BLOCK_M) - group_id * GROUP_SIZE)

  2. @triton.autotune
     Compiles and measures several (BLOCK_M, BLOCK_N, BLOCK_K, num_stages,
     num_warps) configurations on a warm-up call and caches the winner for the
     given (M, N, K) shape. This is hardware measurement, not heuristics.

TFLOPS metric: (2 × M × N × K × 1e-12) / (ms × 1e-3)
"""

import torch
import triton
import triton.language as tl


# ── 1. Autotune configs ───────────────────────────────────────────────────────

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
def tiled_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    # Group ordering: map pid → (pid_m, pid_n) so GROUP_SIZE programs share
    # the same row of A tiles, improving L2 reuse of A across those programs.
    num_pid_in_group = GROUP_SIZE * num_pid_n
    group_id        = pid // num_pid_in_group
    first_pid_m     = group_id * GROUP_SIZE
    group_size_m    = min(num_pid_m - first_pid_m, GROUP_SIZE)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Tile offsets
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

def tiled_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda, "Inputs must be on CUDA"
    assert A.is_contiguous() and B.is_contiguous(), "Inputs must be contiguous"
    assert A.ndim == 2 and B.ndim == 2, "Inputs must be 2D"
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, f"Inner dimensions must match: A is ({M},{K}), B is ({K2},{N})"

    C = torch.empty((M, N), device=A.device, dtype=A.dtype)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)

    tiled_matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
    )
    return C


# ── 4. Correctness tests ──────────────────────────────────────────────────────

def test_tiled_matmul():
    configs = [
        (64, 64, 64),
        (128, 256, 64),
        (256, 128, 512),
        (512, 512, 512),
        (1024, 1024, 1024),
        (127, 255, 63),
    ]
    for M, N, K in configs:
        A = torch.randn(M, K, device="cuda", dtype=torch.float32)
        B = torch.randn(K, N, device="cuda", dtype=torch.float32)
        ref = torch.matmul(A, B)
        got = tiled_matmul(A, B)
        torch.testing.assert_close(got, ref, rtol=1e-2, atol=1e-2)

    print("test_tiled_matmul: PASSED")


# ── 5. Benchmarks ─────────────────────────────────────────────────────────────

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[256, 512, 1024, 2048, 4096],
        x_log=True,
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton (tiled)", "PyTorch (cuBLAS)"],
        styles=[("blue", "-"), ("green", "--")],
        ylabel="TFLOPS",
        plot_name="tiled_matmul",
        args={},
    )
)
def benchmark_tiled_matmul(N, provider):
    A = torch.randn(N, N, device="cuda", dtype=torch.float32)
    B = torch.randn(N, N, device="cuda", dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]

    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: tiled_matmul(A, B), warmup=25, rep=100, quantiles=quantiles
        )
    else:
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch.matmul(A, B), warmup=25, rep=100, quantiles=quantiles
        )

    tflops = 2 * N * N * N * 1e-12
    return tflops / (ms * 1e-3), tflops / (max_ms * 1e-3), tflops / (min_ms * 1e-3)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_tiled_matmul()
    benchmark_tiled_matmul.run(print_data=True, show_plots=True)
