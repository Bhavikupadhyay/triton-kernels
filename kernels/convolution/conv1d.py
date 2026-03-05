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

Algorithm — implicit GEMM via tl.dot:
  Convolution is equivalent to a batched matrix multiply:
    Y[b] = W_flat @ X_col[b]
  where  W_flat : (C_out, C_in × K)           — weight reshaped
         X_col  : (C_in × K, N_out)           — im2col view of x (virtualised, not materialised)

  Instead of materialising X_col (470 MB for C_in=128, K=7, N=131K), the kernel streams
  it one c_in slice at a time. For each c_in the slice contributes a (K, N_out) sub-block:
    X_col[c_in*K : (c_in+1)*K, n_out] = x[b, c_in, n_out + k]   for k=0..K-1

  This sub-block IS contiguous in the N dimension — x[b, c_in, n_start+k : n_start+k+BLOCK_N_OUT]
  for k=0..K-1 — so each row can be loaded as a single coalesced vector load.

  One Triton program per (b, c_out_block, n_out_block).
  Grid: (B × cdiv(C_out, BLOCK_C_OUT), cdiv(N_out, BLOCK_N_OUT))

  Inner loop (128 iterations over C_in, not 128*K = 896):
    W_tile : (BLOCK_C_OUT, K_PAD) from weight   — one K-wide row per c_out in block
    X_tile : (K_PAD, BLOCK_N_OUT) from x        — K rows of BLOCK_N_OUT consecutive elements
    acc    = tl.dot(W_tile, X_tile, acc=acc)    — (BLOCK_C_OUT, BLOCK_N_OUT) GEMM step

  K_PAD = max(16, next_power_of_2(K)) — tl.dot requires the inner dimension ≥ 16.
  Positions k≥K are zero-masked in both tiles so they don't contribute to the result.

  Why tl.dot beats the outer-product loop:
    - The outer product `w_val[:, None] * x_val[None, :]` requires Triton to distribute
      a (BLOCK_C_OUT,) and a (BLOCK_N_OUT,) vector across threads for every (c_in, k)
      iteration. At 896 iterations the broadcast overhead dominates.
    - tl.dot maps to hardware-optimised register blocking: each thread holds a small
      submatrix of acc and re-uses W_tile / X_tile elements from registers across the
      K_PAD inner steps. Memory traffic per iteration drops; ILP rises.
    - The C_in loop (128 iters) is software-pipelined via num_stages: next c_in's loads
      are issued while current c_in's dot product executes.

  For K=7 and K_PAD=16: 9/16 inner dot-product slots are zero — we "waste" 56% of
  the arithmetic in each tl.dot call. On T4 (fp32, no fp32 tensor cores) tl.dot still
  uses FP32 FMA units, but with optimal register blocking. This beats 896 serial loop
  iterations even with the padding waste.

  Remaining gap vs PyTorch (cuDNN implicit GEMM): cuDNN reads x exactly once from HBM
  across all C_out programs; our kernel reads x cdiv(C_out, BLOCK_C_OUT) times.
  Closing this fully requires a true implicit GEMM with shared HBM reads across the
  entire C_out dimension — beyond the scope of a direct-convolution baseline.

TFLOPS metric: (2 × B × C_out × N_out × C_in × K × 1e-12) / (ms × 1e-3)
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# ── 1. Triton kernel ──────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N_OUT": 16,  "BLOCK_C_OUT": 16}, num_stages=2, num_warps=2),
        triton.Config({"BLOCK_N_OUT": 32,  "BLOCK_C_OUT": 32}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_N_OUT": 64,  "BLOCK_C_OUT": 32}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_N_OUT": 128, "BLOCK_C_OUT": 32}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_N_OUT": 32,  "BLOCK_C_OUT": 64}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_N_OUT": 64,  "BLOCK_C_OUT": 64}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_N_OUT": 128, "BLOCK_C_OUT": 64}, num_stages=4, num_warps=8),
    ],
    key=["N_out", "C_in", "C_out"],
    # K and K_PAD are constexpr — Triton specialises a separate compiled kernel per
    # (K, K_PAD) pair; they do not need to appear in the runtime autotune key.
)
@triton.jit
def conv1d_kernel(
    x_ptr, w_ptr, y_ptr,
    B, C_in, C_out, N, N_out,
    stride_xb,  stride_xci, stride_xn,
    stride_wco, stride_wci, stride_wk,
    stride_yb,  stride_yco, stride_yn,
    K:           tl.constexpr,   # true kernel size
    K_PAD:       tl.constexpr,   # max(16, next_power_of_2(K)) — tl.dot inner-dim padding
    BLOCK_N_OUT: tl.constexpr,
    BLOCK_C_OUT: tl.constexpr,
):
    pid_bc = tl.program_id(0)   # flat (b, c_out_block) index
    pid_n  = tl.program_id(1)   # tile index along N_out

    # Decode batch and output-channel block.
    # BLOCK_C_OUT is constexpr → tl.cdiv compiles to a multiply-shift, not a division.
    num_cout_blocks = tl.cdiv(C_out, BLOCK_C_OUT)
    pid_b           = pid_bc // num_cout_blocks
    pid_cout_block  = pid_bc %  num_cout_blocks

    n_start    = pid_n * BLOCK_N_OUT
    offs_n     = n_start + tl.arange(0, BLOCK_N_OUT)
    mask_n     = offs_n < N_out

    cout_start = pid_cout_block * BLOCK_C_OUT
    offs_cout  = cout_start + tl.arange(0, BLOCK_C_OUT)
    mask_cout  = offs_cout < C_out

    # k_pad_offs covers [0, K_PAD); only [0, K) are real kernel positions.
    k_pad_offs = tl.arange(0, K_PAD)
    k_mask     = k_pad_offs < K   # False for zero-padding positions

    acc = tl.zeros((BLOCK_C_OUT, BLOCK_N_OUT), dtype=tl.float32)

    for c_in in range(C_in):
        # ── W_tile : (BLOCK_C_OUT, K_PAD) ─────────────────────────────────────
        # weight[offs_cout, c_in, k] zero-padded to K_PAD columns.
        # offs_cout is contiguous in memory (stride_wco is the leading stride),
        # so each row of W_tile is a coalesced load across output channels.
        w_tile = tl.load(
            w_ptr + offs_cout[:, None] * stride_wco
                  + c_in * stride_wci
                  + k_pad_offs[None, :] * stride_wk,
            mask=mask_cout[:, None] & k_mask[None, :],
            other=0.0,
        )

        # ── X_tile : (K_PAD, BLOCK_N_OUT) ─────────────────────────────────────
        # x[b, c_in, offs_n + k] for k=0..K-1 (rows) and offs_n (columns).
        # Row k is the slice x[b, c_in, n_start+k : n_start+k+BLOCK_N_OUT] —
        # BLOCK_N_OUT contiguous elements in the N dimension. ✓ coalesced.
        # k_mask[:, None] zeros out rows K..K_PAD-1 (padding); those rows also
        # have the corresponding W_tile columns zeroed, so they contribute 0 to acc.
        x_tile = tl.load(
            x_ptr + pid_b * stride_xb
                  + c_in * stride_xci
                  + (k_pad_offs[:, None] + offs_n[None, :]) * stride_xn,
            mask=k_mask[:, None] & mask_n[None, :],
            other=0.0,
        )

        # ── GEMM step ──────────────────────────────────────────────────────────
        # (BLOCK_C_OUT, K_PAD) × (K_PAD, BLOCK_N_OUT) → (BLOCK_C_OUT, BLOCK_N_OUT)
        # tl.dot uses hardware-optimal register blocking; acc is updated in-place.
        acc = tl.dot(w_tile, x_tile, acc=acc, out_dtype=tl.float32)

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

    # K_PAD: smallest value ≥ max(16, K) that is a power of 2.
    # tl.dot requires the inner (K_PAD) dimension to be ≥ 16 and a power of 2.
    K_PAD = max(16, triton.next_power_of_2(K))

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
        K=K,
        K_PAD=K_PAD,
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
# At these parameters arithmetic intensity ≈ 0.5 FLOPs/byte (fp32) — HBM-limited.
# PyTorch (cuDNN) switches from direct conv to implicit GEMM at N≈4096, reading x
# exactly once from HBM. Our kernel reads x cdiv(C_out, BLOCK_C_OUT) times — a
# structural gap that direct convolution cannot close without a full implicit GEMM.

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
