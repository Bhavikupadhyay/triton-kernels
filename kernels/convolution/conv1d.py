"""
Kernel:   conv1d
Category: convolution
Complexity: O(B × C_out × N_out × C_in × K)
Memory bound: Yes — arithmetic intensity ≈ 2·K·C_in / (K·C_in·dtype_bytes + dtype_bytes)
              is sub-1 FLOPs/byte for typical (K, C_in); HBM bandwidth is the bottleneck.
PyTorch equivalent: torch.nn.functional.conv1d(x, weight, padding=0)
References:
  - Direct convolution: https://en.wikipedia.org/wiki/Convolution#Discrete_convolution

Layout:
  x      : (B, C_in, N)       — batch, input channels, signal length
  weight : (C_out, C_in, K)   — output channels, input channels, kernel size
  y      : (B, C_out, N_out)  — N_out = N - K + 1  (valid convolution, no padding)

Parallelism — 2D output tile:
  One program per (b, c_out_block, n_out_block).
  Grid: (B × cdiv(C_out, BLOCK_C_OUT), cdiv(N_out, BLOCK_N_OUT))

Accumulation (per program):
  Outer loop: for c_in in [0, C_in)                 ← runtime, pipelined via num_stages
  Inner loop: for k in tl.static_range(K)           ← constexpr → fully unrolled by LLVM
    x_val  ← x[b, c_in, offs_n + k]                  (BLOCK_N_OUT,) — shared across c_out
    w_val  ← w[offs_cout, c_in, k]                    (BLOCK_C_OUT,) — one weight per c_out
    acc   += w_val[:, None] * x_val[None, :]           outer product → (BLOCK_C_OUT, BLOCK_N_OUT)

  Why K must be constexpr: with K as a runtime int, `for k in range(K)` compiles to a
  GPU loop with a branch on every iteration. The 7 independent x-loads (for k=0..6 at
  a fixed c_in) are serialised behind the loop counter — the compiler cannot pipeline
  them. With K as constexpr, tl.static_range(K) unrolls all K iterations into independent
  instructions; LLVM can issue all K loads simultaneously, hiding global memory latency
  and recovering the throughput that the loop overhead was stealing.

  Mask on offs_n < N_out is sufficient to keep offs_n + k in-bounds on x:
    offs_n + k ≤ (N_out − 1) + (K − 1) = N − 1.

TFLOPS metric: (2 × B × C_out × N_out × C_in × K × 1e-12) / (ms × 1e-3)
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# ── 1. Triton kernel ──────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N_OUT": 32,  "BLOCK_C_OUT": 16}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_N_OUT": 64,  "BLOCK_C_OUT": 16}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_N_OUT": 128, "BLOCK_C_OUT": 16}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_N_OUT": 32,  "BLOCK_C_OUT": 32}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_N_OUT": 64,  "BLOCK_C_OUT": 32}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_N_OUT": 128, "BLOCK_C_OUT": 32}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_N_OUT": 16,  "BLOCK_C_OUT": 16}, num_stages=2, num_warps=2),
        triton.Config({"BLOCK_N_OUT": 32,  "BLOCK_C_OUT": 64}, num_stages=3, num_warps=8),
    ],
    key=["N_out", "C_in", "C_out"],
    # K is a constexpr — Triton compiles a separate specialised kernel per K value,
    # so it does not need to appear in the runtime key.
)
@triton.jit
def conv1d_kernel(
    x_ptr, w_ptr, y_ptr,
    B, C_in, C_out, N, N_out,
    stride_xb,  stride_xci, stride_xn,
    stride_wco, stride_wci, stride_wk,
    stride_yb,  stride_yco, stride_yn,
    K:          tl.constexpr,   # kernel size — constexpr enables k-loop unrolling
    BLOCK_N_OUT: tl.constexpr,
    BLOCK_C_OUT: tl.constexpr,
):
    pid_bc = tl.program_id(0)   # flat (b, c_out_block) index
    pid_n  = tl.program_id(1)   # tile index along N_out

    # Decode batch and output-channel block.
    # BLOCK_C_OUT is constexpr → tl.cdiv compiles to a multiply-shift (no division).
    num_cout_blocks = tl.cdiv(C_out, BLOCK_C_OUT)
    pid_b           = pid_bc // num_cout_blocks
    pid_cout_block  = pid_bc %  num_cout_blocks

    n_start    = pid_n * BLOCK_N_OUT
    offs_n     = n_start + tl.arange(0, BLOCK_N_OUT)
    mask_n     = offs_n < N_out

    cout_start = pid_cout_block * BLOCK_C_OUT
    offs_cout  = cout_start + tl.arange(0, BLOCK_C_OUT)
    mask_cout  = offs_cout < C_out

    # 2D accumulator: acc[i, j] = y[b, cout_start+i, n_start+j]
    acc = tl.zeros((BLOCK_C_OUT, BLOCK_N_OUT), dtype=tl.float32)

    for c_in in range(C_in):
        # tl.static_range unrolls the K iterations at compile time.
        # All K x-loads are independent (different offsets, no data dependency) — the
        # compiler can issue them simultaneously, hiding global-memory round-trip latency.
        for k in tl.static_range(K):
            x_val = tl.load(
                x_ptr + pid_b * stride_xb + c_in * stride_xci + (offs_n + k) * stride_xn,
                mask=mask_n,
                other=0.0,
            )
            w_val = tl.load(
                w_ptr + offs_cout * stride_wco + c_in * stride_wci + k * stride_wk,
                mask=mask_cout,
                other=0.0,
            )
            # Outer product: (BLOCK_C_OUT, 1) × (1, BLOCK_N_OUT) → (BLOCK_C_OUT, BLOCK_N_OUT)
            acc = acc + w_val[:, None].to(tl.float32) * x_val[None, :].to(tl.float32)

    y_offs = offs_cout[:, None] * stride_yco + offs_n[None, :] * stride_yn
    tl.store(
        y_ptr + pid_b * stride_yb + y_offs,
        acc.to(y_ptr.dtype.element_ty),
        mask=mask_cout[:, None] & mask_n[None, :],
    )


# ── 2. Python wrapper ─────────────────────────────────────────────────────────

def conv1d(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Direct 1D convolution — no padding, no bias.

    Args:
        x: (B, C_in, N)      input tensor on CUDA.
        w: (C_out, C_in, K)  weight tensor on CUDA.

    Returns:
        y: (B, C_out, N - K + 1), same dtype as x.
    """
    assert x.is_cuda and w.is_cuda, "Inputs must be on CUDA"
    x = x.contiguous()
    w = w.contiguous()
    assert x.ndim == 3, "x must be 3D: (B, C_in, N)"
    assert w.ndim == 3, "w must be 3D: (C_out, C_in, K)"

    B, C_in, N       = x.shape
    C_out, C_in_w, K = w.shape
    assert C_in == C_in_w, f"Channel mismatch: x has C_in={C_in}, w has C_in={C_in_w}"
    assert N >= K,         f"Signal length N={N} must be >= kernel size K={K}"

    N_out = N - K + 1
    y     = torch.empty((B, C_out, N_out), device=x.device, dtype=x.dtype)

    grid = lambda meta: (
        B * triton.cdiv(C_out, meta["BLOCK_C_OUT"]),
        triton.cdiv(N_out, meta["BLOCK_N_OUT"]),
    )
    conv1d_kernel[grid](
        x, w, y,
        B, C_in, C_out, N, N_out,
        x.stride(0), x.stride(1), x.stride(2),
        w.stride(0), w.stride(1), w.stride(2),
        y.stride(0), y.stride(1), y.stride(2),
        K=K,  # passed as constexpr — Triton JIT-compiles a separate kernel per K value
    )
    return y


# ── 3. Correctness tests ──────────────────────────────────────────────────────

def test_conv1d():
    configs = [
        (1,  1,   1,   16,   3),   # minimal — single channel
        (2,  4,   8,   64,   5),   # small multi-channel
        (4,  16,  32,  512,  7),   # larger channels
        (3,  5,   7,   100,  3),   # non-powers-of-2
        (1,  64,  64,  1024, 11),  # wider kernel
        (8,  32,  64,  2048, 3),   # larger batch
    ]
    for dtype in [torch.float32, torch.float16]:
        for B, C_in, C_out, N, K in configs:
            x   = torch.randn(B, C_in, N,     device="cuda", dtype=dtype)
            w   = torch.randn(C_out, C_in, K, device="cuda", dtype=dtype)
            ref = F.conv1d(x, w)
            got = conv1d(x, w)
            tol = dict(rtol=1e-2, atol=1e-2) if dtype == torch.float16 else dict(rtol=1e-3, atol=1e-3)
            torch.testing.assert_close(got, ref, **tol)

    print("test_conv1d: PASSED")


# ── 4. Benchmarks ─────────────────────────────────────────────────────────────
#
# Fix B=1, C_in=C_out=128, K=7 (representative of audio/WaveNet-style convolutions).
# Vary N (signal length) from 1K to 128K.
#
# Arithmetic intensity ≈ 0.5 FLOPs/byte (fp32) — well below T4's ridge point of ~25.
# Both kernels are HBM-bandwidth-limited; TFLOPS reflects memory throughput.
# At N≥4096 PyTorch (cuDNN) switches to implicit GEMM which reads x exactly once —
# expect PyTorch to stay ahead at large N; the gap should be ≤2-3× post-fix.

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[2**i for i in range(10, 18)],   # 1024 → 131072
        x_log=True,
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton", "PyTorch (F.conv1d)"],
        styles=[("blue", "-"), ("green", "--")],
        ylabel="TFLOPS",
        plot_name="conv1d",
        args={"B": 1, "C_in": 128, "C_out": 128, "K": 7},
    )
)
def benchmark_conv1d(B, C_in, C_out, K, N, provider):
    x         = torch.randn(B, C_in, N,     device="cuda", dtype=torch.float32)
    w         = torch.randn(C_out, C_in, K, device="cuda", dtype=torch.float32)
    N_out     = N - K + 1
    quantiles = [0.5, 0.2, 0.8]

    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: conv1d(x, w), warmup=25, rep=100, quantiles=quantiles
        )
    else:
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: F.conv1d(x, w), warmup=25, rep=100, quantiles=quantiles
        )

    tflops = 2 * B * C_out * N_out * C_in * K * 1e-12
    return tflops / (ms * 1e-3), tflops / (max_ms * 1e-3), tflops / (min_ms * 1e-3)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_conv1d()
    benchmark_conv1d.run(print_data=True, show_plots=True)
