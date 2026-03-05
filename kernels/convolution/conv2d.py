"""
Kernel:   conv2d
Category: convolution
Complexity: O(B × C_out × H_out × W_out × C_in × K²)
Memory bound: Yes at small channel count; compute-bound at large channel count
PyTorch equivalent: torch.nn.functional.conv2d(x, weight, padding=0)
References:
  - Implicit GEMM for convolutions: https://arxiv.org/abs/1812.00849

Layout:
  x      : (B, C_in,  H,     W)      — NCHW
  weight : (C_out, C_in, K, K)       — square kernel
  y      : (B, C_out, H_out, W_out)  — H_out = H - K + 1, W_out = W - K + 1

Algorithm — Implicit GEMM (conv1d extended to 2D):

  Flatten output spatial: N_sp = H_out × W_out
  Y[b, :, :] = W_flat(C_out, CK) @ X_col[b](CK, N_sp)
  where CK = C_in × K², and
    X_col[c_in*K² + kh*K + kw, h*W_out + w] = x[b, c_in, h + kh, w + kw]

  Grid:
    axis 0: cdiv(C_out, BLOCK_M)   — tiles over output channels
    axis 1: cdiv(N_sp,  BLOCK_N)   — tiles over flattened output spatial
    axis 2: B                       — one slice per batch element

  Spatial decode — once per program, amortised over all inner iterations:
    h_out[i] = flat[i] // W_out
    w_out[i] = flat[i] % W_out
  W_out is a runtime value → these are runtime integer divides, but they are
  BLOCK_N scalar ops executed once before the inner loops begin.

  Two-level inner loop (avoids derived-constexpr tl.arange — see conv1d):
    Outer : for c_tile in range(cdiv(C_in, C_PER_TILE))
    Inner : for kh in tl.static_range(K)   — K is constexpr → fully unrolled
            for kw in tl.static_range(K)

    Each (c_tile, kh, kw) step:
    ┌──────────────────────────────────────────────────────────────────────┐
    │ W_sub : (BLOCK_M, C_PER_TILE) ← weight[M_tile, c_tile, kh, kw]     │
    │ X_sub : (C_PER_TILE, BLOCK_N) ← x[b, c_tile, h_tile+kh, w_tile+kw] │
    │ acc   += tl.dot(W_sub, X_sub)                                       │
    └──────────────────────────────────────────────────────────────────────┘

  Total tl.dot calls per program: cdiv(C_in, C_PER_TILE) × K²
    e.g. C_in=64, C_PER_TILE=16, K=3 → 4 × 9 = 36 dot calls
    e.g. C_in=64, C_PER_TILE=32, K=3 → 2 × 9 = 18 dot calls

  Bounds on x access: h_out < H_out = H−K+1 and kh < K → h_out+kh ≤ H−1 < H.
  Same for w. No OOB once flat_offs is masked to N_sp — no extra bounds check needed.

  C_PER_TILE ≥ 16 (tl.dot inner-dim constraint on T4). Configs use {16, 32}.

TFLOPS metric: (2 × B × C_out × H_out × W_out × C_in × K² × 1e-12) / (ms × 1e-3)
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# ── 1. Triton kernel — implicit GEMM ─────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "C_PER_TILE": 16}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "C_PER_TILE": 16}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "C_PER_TILE": 16}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "C_PER_TILE": 16}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "C_PER_TILE": 32}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "C_PER_TILE": 32}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "C_PER_TILE": 32}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "C_PER_TILE": 32}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "C_PER_TILE": 16}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 256, "C_PER_TILE": 16}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "C_PER_TILE": 32}, num_stages=4, num_warps=8),
    ],
    key=["C_in", "C_out", "H_out", "W_out"],
)
@triton.jit
def conv2d_kernel(
    x_ptr, w_ptr, y_ptr,
    B, C_in, C_out, H, W, H_out, W_out,
    stride_xb,  stride_xci, stride_xh,  stride_xw,
    stride_wco, stride_wci, stride_wkh, stride_wkw,
    stride_yb,  stride_yco, stride_yh,  stride_yw,
    K:           tl.constexpr,   # kernel size (square: KH = KW = K)
    BLOCK_M:     tl.constexpr,   # tile over C_out
    BLOCK_N:     tl.constexpr,   # tile over N_sp = H_out * W_out
    C_PER_TILE:  tl.constexpr,   # C_in channels per inner tile; must be ≥ 16
):
    pid_m = tl.program_id(0)   # C_out tile
    pid_n = tl.program_id(1)   # spatial tile (flat H_out × W_out)
    pid_b = tl.program_id(2)   # batch

    # ── Output channel offsets ────────────────────────────────────────────────
    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # BLOCK_M constexpr ✓
    mask_m = m_offs < C_out

    # ── Flat spatial offsets → (h_out, w_out) ────────────────────────────────
    # Decoded once per program, reused across all (c_tile, kh, kw) iterations.
    flat_offs  = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # BLOCK_N constexpr ✓
    mask_n     = flat_offs < H_out * W_out
    h_out_offs = flat_offs // W_out   # (BLOCK_N,)
    w_out_offs = flat_offs % W_out    # (BLOCK_N,)

    # Per-tile C_in local offsets [0, 1, ..., C_PER_TILE-1]
    c_tile_offs = tl.arange(0, C_PER_TILE)   # C_PER_TILE constexpr ✓

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    num_c_tiles = tl.cdiv(C_in, C_PER_TILE)

    for c_tile in range(num_c_tiles):
        c_in_base = c_tile * C_PER_TILE
        c_in_offs = c_in_base + c_tile_offs   # (C_PER_TILE,) absolute c_in indices
        mask_c    = c_in_offs < C_in

        # K×K kernel positions — both loops fully unrolled (K is constexpr).
        for kh in tl.static_range(K):
            for kw in tl.static_range(K):

                # ── W_sub : (BLOCK_M, C_PER_TILE) ───────────────────────────
                # weight[m_offs[m], c_in_offs[c], kh, kw]
                w_sub = tl.load(
                    w_ptr
                    + m_offs[:, None]    * stride_wco
                    + c_in_offs[None, :] * stride_wci
                    + kh                 * stride_wkh
                    + kw                 * stride_wkw,
                    mask=mask_m[:, None] & mask_c[None, :],
                    other=0.0,
                )

                # ── X_sub : (C_PER_TILE, BLOCK_N) ───────────────────────────
                # x[b, c_in_offs[c], h_out_offs[n] + kh, w_out_offs[n] + kw]
                # h_out + kh ≤ H−1 and w_out + kw ≤ W−1 (proven in docstring).
                x_sub = tl.load(
                    x_ptr
                    + pid_b                        * stride_xb
                    + c_in_offs[:, None]           * stride_xci
                    + (h_out_offs[None, :] + kh)   * stride_xh
                    + (w_out_offs[None, :] + kw)   * stride_xw,
                    mask=mask_c[:, None] & mask_n[None, :],
                    other=0.0,
                )

                # ── GEMM step ────────────────────────────────────────────────
                # (BLOCK_M, C_PER_TILE) × (C_PER_TILE, BLOCK_N) → (BLOCK_M, BLOCK_N)
                acc += tl.dot(w_sub, x_sub, out_dtype=tl.float32)

    # ── Store output ──────────────────────────────────────────────────────────
    # Re-use the decoded (h_out_offs, w_out_offs) to scatter into y[b, m, h, w].
    y_offs = (
        m_offs[:, None]      * stride_yco
        + h_out_offs[None, :] * stride_yh
        + w_out_offs[None, :] * stride_yw
    )
    tl.store(
        y_ptr + pid_b * stride_yb + y_offs,
        acc.to(y_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_n[None, :],
    )


# ── 2. Python wrapper ─────────────────────────────────────────────────────────

def conv2d(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Implicit-GEMM 2D convolution — square kernel, no padding, no bias.

    Args:
        x: (B, C_in, H, W)       input tensor on CUDA, NCHW layout.
        w: (C_out, C_in, K, K)   weight tensor on CUDA, square kernel.

    Returns:
        y: (B, C_out, H - K + 1, W - K + 1), same dtype as x.
    """
    assert x.is_cuda and w.is_cuda, "Inputs must be on CUDA"
    x = x.contiguous()
    w = w.contiguous()
    assert x.ndim == 4, "x must be 4D: (B, C_in, H, W)"
    assert w.ndim == 4, "w must be 4D: (C_out, C_in, K, K)"

    B, C_in, H, W         = x.shape
    C_out, C_in_w, KH, KW = w.shape
    assert C_in == C_in_w, f"Channel mismatch: x C_in={C_in}, w C_in={C_in_w}"
    assert KH == KW,        f"Non-square kernel: KH={KH}, KW={KW}"
    assert H >= KH and W >= KW, f"Spatial ({H},{W}) must be >= kernel size {KH}"
    K = KH

    H_out = H - K + 1
    W_out = W - K + 1
    y     = torch.empty((B, C_out, H_out, W_out), device=x.device, dtype=x.dtype)

    N_sp = H_out * W_out
    grid = lambda meta: (
        triton.cdiv(C_out, meta["BLOCK_M"]),
        triton.cdiv(N_sp,  meta["BLOCK_N"]),
        B,
    )
    conv2d_kernel[grid](
        x, w, y,
        B, C_in, C_out, H, W, H_out, W_out,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        w.stride(0), w.stride(1), w.stride(2), w.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        K=K,
    )
    return y


# ── 3. Correctness tests ──────────────────────────────────────────────────────

def test_conv2d():
    configs = [
        # B, C_in, C_out, H, W, K
        (1, 1,   1,   8,   8,  3),
        (2, 4,   8,   16,  16, 3),
        (4, 16,  32,  32,  32, 3),
        (1, 3,   8,   28,  28, 5),   # MNIST-like
        (1, 64,  64,  56,  56, 3),   # ResNet conv2 block
        (2, 32,  64,  32,  32, 3),
    ]
    for dtype in [torch.float32, torch.float16]:
        for B, C_in, C_out, H, W, K in configs:
            x   = torch.randn(B, C_in, H, W,     device="cuda", dtype=dtype)
            wt  = torch.randn(C_out, C_in, K, K, device="cuda", dtype=dtype)
            ref = F.conv2d(x, wt)
            got = conv2d(x, wt)
            tol = dict(rtol=1e-2, atol=1e-2) if dtype == torch.float16 else dict(rtol=1e-3, atol=1e-3)
            torch.testing.assert_close(got, ref, **tol)

    print("test_conv2d: PASSED")


# ── 4. Benchmarks ─────────────────────────────────────────────────────────────
#
# Fix B=1, C_in=C_out=64, K=3 (ResNet-style 3×3 conv).
# Vary H=W (square images) from 32 to 512.
#
# N_sp = (H-2)² grows quadratically. At H=512, N_sp ≈ 260K spatial positions.
# Both kernels are compute-bound (AI ≫ ridge point) at these channel counts.

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["H"],
        x_vals=[2**i for i in range(5, 10)],
        x_log=True,
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton (implicit GEMM)", "PyTorch (F.conv2d)"],
        styles=[("blue", "-"), ("green", "--")],
        ylabel="TFLOPS",
        plot_name="conv2d",
        args={"B": 1, "C_in": 64, "C_out": 64, "K": 3},
    )
)
def benchmark_conv2d(B, C_in, C_out, K, H, provider):
    W     = H
    x     = torch.randn(B, C_in, H, W,     device="cuda", dtype=torch.float32)
    wt    = torch.randn(C_out, C_in, K, K, device="cuda", dtype=torch.float32)
    H_out = H - K + 1
    W_out = W - K + 1
    quantiles = [0.5, 0.2, 0.8]

    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: conv2d(x, wt), warmup=25, rep=100, quantiles=quantiles
        )
    else:
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: F.conv2d(x, wt), warmup=25, rep=100, quantiles=quantiles
        )

    tflops = 2 * B * C_out * H_out * W_out * C_in * K * K * 1e-12
    return tflops / (ms * 1e-3), tflops / (max_ms * 1e-3), tflops / (min_ms * 1e-3)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_conv2d()
    benchmark_conv2d.run(print_data=True, show_plots=True)
