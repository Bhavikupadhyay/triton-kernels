"""
Kernel:   depthwise_conv2d
Category: convolution
Complexity: O(B × C × H_out × W_out × K²)
Memory bound: Yes — no cross-channel mixing; arithmetic intensity ≈ 2·K² / (K²·bytes + bytes)
PyTorch equivalent: torch.nn.functional.conv2d(x, weight, groups=C_in)
References:
  - MobileNets depthwise separable conv: https://arxiv.org/abs/1704.04861

Layout:
  x      : (B, C, H,     W)     — NCHW; one feature map per channel
  weight : (C, 1, K, K)         — one K×K filter per channel (groups=C_in PyTorch convention)
  y      : (B, C, H_out, W_out) — H_out = H - K + 1, W_out = W - K + 1

Algorithm — per-channel sliding window:

  Standard conv2d accumulates over C_in channels via a dot product (implicit GEMM).
  Depthwise has no cross-channel accumulation: output channel c depends only on
  input channel c. The inner loop reduces over K×K spatial positions only.

    y[b, c, h, w] = Σ_{kh,kw} weight[c, 0, kh, kw] × x[b, c, h+kh, w+kw]

  Grid (3D):
    axis 0: B × C                        — one program per (batch, channel) pair
    axis 1: cdiv(H_out, BLOCK_H)         — tiles over output rows
    axis 2: cdiv(W_out, BLOCK_W)         — tiles over output columns

  Per-program accumulator: (BLOCK_H, BLOCK_W) fp32 floats.

  Inner loops (runtime, not unrolled — same register-pressure reason as conv2d):
    for kh in range(K):
      for kw in range(K):
        acc += weight[c, kh, kw] × x[b, c, h_offs+kh, w_offs+kw]

  Load patterns:
    weight[c, kh, kw] — scalar broadcast (one float, reused for all BLOCK_H × BLOCK_W outputs)
    x[b, c, h_offs+kh, w_offs+kw] — (BLOCK_H, BLOCK_W) 2D tile; contiguous along W ✓

  Why no tl.dot: the weight tile is a scalar (1 value per (kh, kw)), not a matrix.
    The operation is a scalar-times-matrix FMA, not a matrix multiply.
    tl.dot requires (M, K) × (K, N) with K ≥ 16; our K dim = 1.

  Memory-bound analysis:
    Arithmetic intensity ≈ 2·K² FLOPs / ((K²+1)·dtype_bytes) ← very low
    At K=3: ~1.8 FLOPs/byte (fp32). T4 bandwidth-limited at all practical sizes.
    Both Triton and PyTorch are bandwidth-bound here; the comparison is pure BW efficiency.

TFLOPS metric: (2 × B × C × H_out × W_out × K² × 1e-12) / (ms × 1e-3)
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# ── 1. Triton kernel — per-channel sliding window ─────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 16}, num_stages=2, num_warps=2),
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 32}, num_stages=2, num_warps=2),
        triton.Config({"BLOCK_H": 32, "BLOCK_W": 16}, num_stages=2, num_warps=2),
        triton.Config({"BLOCK_H": 32, "BLOCK_W": 32}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 64}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_H": 64, "BLOCK_W": 16}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_H": 32, "BLOCK_W": 64}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_H": 64, "BLOCK_W": 32}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_H": 64, "BLOCK_W": 64}, num_stages=4, num_warps=8),
    ],
    key=["C", "H_out", "W_out"],
)
@triton.jit
def depthwise_conv2d_kernel(
    x_ptr, w_ptr, y_ptr,
    B, C, H, W, H_out, W_out,
    stride_xb,  stride_xc,  stride_xh,  stride_xw,
    stride_wc,  stride_wkh, stride_wkw,            # weight: (C, K, K) after squeeze(1)
    stride_yb,  stride_yc,  stride_yh,  stride_yw,
    K,                        # runtime int — avoids register explosion from unrolling
    BLOCK_H: tl.constexpr,   # output rows per program
    BLOCK_W: tl.constexpr,   # output columns per program
):
    pid_bc = tl.program_id(0)   # flat (batch, channel) index
    pid_h  = tl.program_id(1)   # H_out tile
    pid_w  = tl.program_id(2)   # W_out tile

    # Decode batch and channel from axis 0.
    pid_b = pid_bc // C
    pid_c = pid_bc %  C

    # ── Output spatial offsets ────────────────────────────────────────────────
    h_offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)   # BLOCK_H constexpr ✓
    w_offs = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)   # BLOCK_W constexpr ✓
    mask_h = h_offs < H_out
    mask_w = w_offs < W_out

    # ── Base pointers for this (b, c) slice ───────────────────────────────────
    x_base = x_ptr + pid_b * stride_xb + pid_c * stride_xc
    w_base = w_ptr + pid_c * stride_wc

    acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)

    # ── Inner loop over K×K kernel positions ─────────────────────────────────
    # weight[c, kh, kw] is a scalar — load once, broadcast over BLOCK_H × BLOCK_W.
    # x[b, c, h_offs+kh, w_offs+kw] is a (BLOCK_H, BLOCK_W) tile — contiguous along W ✓
    for kh in range(K):
        for kw in range(K):
            # Scalar weight: weight[c, 0, kh, kw] — one float per (kh, kw)
            w_val = tl.load(w_base + kh * stride_wkh + kw * stride_wkw)

            # 2D input tile: x[b, c, h_offs+kh, w_offs+kw]
            x_tile = tl.load(
                x_base
                + (h_offs[:, None] + kh) * stride_xh
                + (w_offs[None, :] + kw) * stride_xw,
                mask=mask_h[:, None] & mask_w[None, :],
                other=0.0,
            )

            # Scalar-times-matrix FMA: acc += w_val × x_tile
            acc += w_val * x_tile

    # ── Store output ──────────────────────────────────────────────────────────
    y_base = y_ptr + pid_b * stride_yb + pid_c * stride_yc
    tl.store(
        y_base
        + h_offs[:, None] * stride_yh
        + w_offs[None, :] * stride_yw,
        acc.to(y_ptr.dtype.element_ty),
        mask=mask_h[:, None] & mask_w[None, :],
    )


# ── 2. Python wrapper ─────────────────────────────────────────────────────────

def depthwise_conv2d(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Depthwise 2D convolution — one filter per channel, no padding, no bias.

    Equivalent to F.conv2d(x, w, groups=C_in).

    Args:
        x: (B, C, H, W)      input tensor on CUDA, NCHW layout.
        w: (C, 1, K, K)      weight tensor on CUDA, one K×K filter per channel.

    Returns:
        y: (B, C, H - K + 1, W - K + 1), same dtype as x.
    """
    assert x.is_cuda and w.is_cuda, "Inputs must be on CUDA"
    x = x.contiguous()
    w = w.contiguous()
    assert x.ndim == 4, "x must be 4D: (B, C, H, W)"
    assert w.ndim == 4, "w must be 4D: (C, 1, K, K)"

    B, C, H, W          = x.shape
    C_w, groups, KH, KW = w.shape
    assert C_w == C,    f"Channel mismatch: x C={C}, w C={C_w}"
    assert groups == 1, f"weight dim 1 must be 1, got {groups}"
    assert KH == KW,    f"Non-square kernel: KH={KH}, KW={KW}"
    assert H >= KH and W >= KW, f"Spatial ({H},{W}) must be >= kernel size {KH}"
    K = KH

    H_out = H - K + 1
    W_out = W - K + 1
    y     = torch.empty((B, C, H_out, W_out), device=x.device, dtype=x.dtype)

    # Squeeze the groups=1 dim so weight is (C, K, K) with natural strides.
    w_sq = w.squeeze(1)   # (C, K, K)

    grid = lambda meta: (
        B * C,
        triton.cdiv(H_out, meta["BLOCK_H"]),
        triton.cdiv(W_out, meta["BLOCK_W"]),
    )
    depthwise_conv2d_kernel[grid](
        x, w_sq, y,
        B, C, H, W, H_out, W_out,
        x.stride(0),   x.stride(1),   x.stride(2),   x.stride(3),
        w_sq.stride(0), w_sq.stride(1), w_sq.stride(2),
        y.stride(0),   y.stride(1),   y.stride(2),   y.stride(3),
        K=K,
    )
    return y


# ── 3. Correctness tests ──────────────────────────────────────────────────────

def test_depthwise_conv2d():
    configs = [
        # B, C, H, W, K
        (1,  1,   8,   8,  3),
        (2,  4,   16,  16, 3),
        (4,  16,  32,  32, 3),
        (1,  32,  56,  56, 3),   # MobileNet-style
        (1,  64,  56,  56, 3),
        (2,  32,  32,  32, 5),
    ]
    for dtype in [torch.float32, torch.float16]:
        for B, C, H, W, K in configs:
            x   = torch.randn(B, C, H, W,    device="cuda", dtype=dtype)
            wt  = torch.randn(C, 1, K, K,    device="cuda", dtype=dtype)
            ref = F.conv2d(x, wt, groups=C)
            got = depthwise_conv2d(x, wt)
            tol = dict(rtol=1e-2, atol=1e-2) if dtype == torch.float16 else dict(rtol=1e-3, atol=1e-3)
            torch.testing.assert_close(got, ref, **tol)

    print("test_depthwise_conv2d: PASSED")


# ── 4. Benchmarks ─────────────────────────────────────────────────────────────
#
# Fix B=1, C=128, K=3 (MobileNet-style depthwise).
# Vary H=W from 32 to 512.
#
# Arithmetic intensity ≈ 2·K² / ((K²+1)·4) ≈ 1.8 FLOPs/byte (fp32, K=3).
# T4 ridge point ~25 FLOPs/byte → both Triton and PyTorch are HBM-bandwidth-limited.
# Benchmark measures BW efficiency rather than compute efficiency.

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["H"],
        x_vals=[2**i for i in range(5, 10)],
        x_log=True,
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton (depthwise)", "PyTorch (F.conv2d groups=C)"],
        styles=[("blue", "-"), ("green", "--")],
        ylabel="TFLOPS",
        plot_name="depthwise_conv2d",
        args={"B": 1, "C": 128, "K": 3},
    )
)
def benchmark_depthwise_conv2d(B, C, K, H, provider):
    W     = H
    x     = torch.randn(B, C, H, W,  device="cuda", dtype=torch.float32)
    wt    = torch.randn(C, 1, K, K,  device="cuda", dtype=torch.float32)
    H_out = H - K + 1
    W_out = W - K + 1
    quantiles = [0.5, 0.2, 0.8]

    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: depthwise_conv2d(x, wt), warmup=25, rep=100, quantiles=quantiles
        )
    else:
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: F.conv2d(x, wt, groups=C), warmup=25, rep=100, quantiles=quantiles
        )

    tflops = 2 * B * C * H_out * W_out * K * K * 1e-12
    return tflops / (ms * 1e-3), tflops / (max_ms * 1e-3), tflops / (min_ms * 1e-3)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_depthwise_conv2d()
    benchmark_depthwise_conv2d.run(print_data=True, show_plots=True)
