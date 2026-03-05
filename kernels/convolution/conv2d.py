"""
Kernel:   conv2d
Category: convolution
Complexity: O(B × C_out × H_out × W_out × C_in × K²)
Memory bound: Yes at small channel count; compute-bound at large channel count
PyTorch equivalent: torch.nn.functional.conv2d(x, weight, padding=0)
References:
  - Implicit GEMM for convolutions: https://arxiv.org/abs/1812.00849

Layout:
  x      : (B, C_in,  H,     W)      — NCHW (unchanged)
  weight : (C_out, C_in, K, K)       — square kernel; transposed to (C_out, K, K, C_in)
                                        inside the wrapper before the kernel sees it
  y      : (B, C_out, H_out, W_out)  — H_out = H - K + 1, W_out = W - K + 1

Algorithm — 2D-tiled Implicit GEMM with weight transpose:

  Key insight 1 — weight transpose for coalesced W_sub loads:
    In the original (C_out, C_in, K, K) layout, weight[m, c, kh, kw] has
    stride_wci = K² = 9. Loading C_PER_TILE consecutive C_in indices hits elements
    9 floats apart — one cache line covers only ~3 elements → ~5× over-fetch.

    Fix: permute weight to (C_out, K, K, C_in) once in the wrapper (.permute(0,2,3,1)).
    Now stride_wci = 1. Loading C_PER_TILE consecutive C_in indices is a single
    contiguous run → one cache-line load per row of W_sub. ✓

  Key insight 2 — 2D spatial tiling for coalesced X_sub loads:
    Flattening H_out × W_out (as in conv1d) causes scatter whenever W_out is not a
    multiple of BLOCK_N. Fix: tile in 2D. One output row per program (pid_h ∈ [0,H_out)),
    BLOCK_W columns per program (pid_w). X load for fixed (c_in, kh, kw):
      x[b, c_in, pid_h + kh, w_offs : w_offs + BLOCK_W]   — always contiguous ✓

  Key insight 3 — K as runtime parameter:
    Declaring K as tl.constexpr and using tl.static_range(K) forces a separate kernel
    binary per K value. K=5 → 25 fully unrolled dot calls → large LLVM IR → 20-30 s
    compile time per autotune config. Making K a plain int eliminates per-K recompilation;
    the dynamic inner loop overhead for K=3 (9 iterations) is negligible vs dot time.

  Grid (3D):
    axis 0: cdiv(C_out, BLOCK_M)            — C_out tiles
    axis 1: H_out                            — one program per output row
    axis 2: B × cdiv(W_out, BLOCK_W)        — batch × W column tiles

  Inner loops:
    Outer : for c_tile in range(cdiv(C_in, C_PER_TILE))
    Inner : for kh in range(K)              — dynamic loop (K is runtime int)
            for kw in range(K)

    Each (c_tile, kh, kw) step:
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │ W_sub : (BLOCK_M, C_PER_TILE) ← w_t[m_offs, kh, kw, c_tile]  stride_wci=1 │
    │ X_sub : (C_PER_TILE, BLOCK_W) ← x[b, c_tile, pid_h+kh, w_offs+kw] ← contig│
    │ acc  += tl.dot(W_sub, X_sub)                                               │
    └─────────────────────────────────────────────────────────────────────────────┘

  C_PER_TILE ∈ {16, 32, 64}. For C_in=64, C_PER_TILE=64 → 1 c_tile × K² dot calls
  (vs 4 tiles × K² calls with C_PER_TILE=16) → 4× fewer, larger tensor-core ops.

  Bound proofs (no extra mask needed on x rows/cols):
    pid_h   < H_out = H−K+1, kh < K  →  pid_h + kh  ≤ H−1 < H  ✓
    w_offs  < W_out = W−K+1, kw < K  →  w_offs + kw ≤ W−1 < W  ✓

TFLOPS metric: (2 × B × C_out × H_out × W_out × C_in × K² × 1e-12) / (ms × 1e-3)
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# ── 1. Triton kernel — 2D-tiled implicit GEMM ────────────────────────────────

@triton.autotune(
    configs=[
        # C_PER_TILE=16 — handles any C_in; good for small channels
        triton.Config({"BLOCK_M": 64,  "BLOCK_W": 64,  "C_PER_TILE": 16}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 64,  "BLOCK_W": 128, "C_PER_TILE": 16}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_W": 64,  "C_PER_TILE": 16}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_W": 128, "C_PER_TILE": 16}, num_stages=4, num_warps=8),
        # C_PER_TILE=32 — 2× fewer tiles for C_in=32,64,128,...
        triton.Config({"BLOCK_M": 64,  "BLOCK_W": 64,  "C_PER_TILE": 32}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 64,  "BLOCK_W": 128, "C_PER_TILE": 32}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_W": 64,  "C_PER_TILE": 32}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_W": 128, "C_PER_TILE": 32}, num_stages=4, num_warps=8),
        # C_PER_TILE=64 — 1 c_tile for C_in=64 (4× fewer dot calls; larger, better-utilized tiles)
        triton.Config({"BLOCK_M": 64,  "BLOCK_W": 64,  "C_PER_TILE": 64}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 64,  "BLOCK_W": 128, "C_PER_TILE": 64}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_W": 128, "C_PER_TILE": 64}, num_stages=4, num_warps=8),
    ],
    key=["C_in", "C_out", "H_out", "W_out"],
)
@triton.jit
def conv2d_kernel(
    x_ptr, w_ptr, y_ptr,
    B, C_in, C_out, H, W, H_out, W_out,
    stride_xb,  stride_xci, stride_xh,  stride_xw,
    stride_wco, stride_wkh, stride_wkw, stride_wci,   # note: wci LAST (=1 after transpose)
    stride_yb,  stride_yco, stride_yh,  stride_yw,
    K,                        # runtime int — no constexpr; avoids per-K recompilation
    BLOCK_M:    tl.constexpr, # tile over C_out
    BLOCK_W:    tl.constexpr, # tile over W_out (output columns per program)
    C_PER_TILE: tl.constexpr, # C_in channels per inner tile; must be ≥ 16 (tl.dot constraint)
):
    pid_m  = tl.program_id(0)   # C_out tile
    pid_h  = tl.program_id(1)   # output row index ∈ [0, H_out)
    pid_bw = tl.program_id(2)   # batch × W tile (flat)

    # Decode batch and W-tile from axis 2.
    n_w_tiles = tl.cdiv(W_out, BLOCK_W)
    pid_b  = pid_bw // n_w_tiles
    pid_w  = pid_bw %  n_w_tiles

    # ── C_out offsets ─────────────────────────────────────────────────────────
    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # BLOCK_M constexpr ✓
    mask_m = m_offs < C_out

    # ── W output offsets ──────────────────────────────────────────────────────
    w_offs = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)   # BLOCK_W constexpr ✓
    mask_w = w_offs < W_out

    # ── C_in tile offsets ─────────────────────────────────────────────────────
    c_tile_offs = tl.arange(0, C_PER_TILE)              # C_PER_TILE constexpr ✓

    acc = tl.zeros((BLOCK_M, BLOCK_W), dtype=tl.float32)

    num_c_tiles = tl.cdiv(C_in, C_PER_TILE)

    for c_tile in range(num_c_tiles):
        c_in_base = c_tile * C_PER_TILE
        c_in_offs = c_in_base + c_tile_offs
        mask_c    = c_in_offs < C_in

        # K×K kernel positions — dynamic loops (K is runtime, not constexpr).
        for kh in range(K):
            for kw in range(K):

                # ── W_sub : (BLOCK_M, C_PER_TILE) ───────────────────────────
                # w_t[m_offs[m], kh, kw, c_in_offs[c]]
                # After permute(0,2,3,1): stride_wci = 1 → c_in_offs dim is contiguous ✓
                w_sub = tl.load(
                    w_ptr
                    + m_offs[:, None]    * stride_wco
                    + kh                 * stride_wkh
                    + kw                 * stride_wkw
                    + c_in_offs[None, :] * stride_wci,
                    mask=mask_m[:, None] & mask_c[None, :],
                    other=0.0,
                )

                # ── X_sub : (C_PER_TILE, BLOCK_W) ───────────────────────────
                # x[b, c_in_offs[c], pid_h + kh, w_offs[n] + kw]
                # w_offs + kw is a contiguous run of BLOCK_W integers (stride_xw=1) ✓
                x_sub = tl.load(
                    x_ptr
                    + pid_b              * stride_xb
                    + c_in_offs[:, None] * stride_xci
                    + (pid_h + kh)       * stride_xh
                    + (w_offs[None, :] + kw) * stride_xw,
                    mask=mask_c[:, None] & mask_w[None, :],
                    other=0.0,
                )

                # ── GEMM step ────────────────────────────────────────────────
                # (BLOCK_M, C_PER_TILE) × (C_PER_TILE, BLOCK_W) → (BLOCK_M, BLOCK_W)
                acc += tl.dot(w_sub, x_sub, out_dtype=tl.float32)

    # ── Store output ──────────────────────────────────────────────────────────
    y_offs = (
        m_offs[:, None]   * stride_yco
        + pid_h           * stride_yh
        + w_offs[None, :] * stride_yw
    )
    tl.store(
        y_ptr + pid_b * stride_yb + y_offs,
        acc.to(y_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_w[None, :],
    )


# ── 2. Python wrapper ─────────────────────────────────────────────────────────

def conv2d(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """2D-tiled implicit-GEMM convolution — square kernel, no padding, no bias.

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

    # Transpose weight to (C_out, K, K, C_in) so the C_in dimension is contiguous.
    # This makes W_sub loads in the kernel stride-1 along C_in → coalesced access.
    w_t = w.permute(0, 2, 3, 1).contiguous()   # (C_out, K, K, C_in)

    grid = lambda meta: (
        triton.cdiv(C_out, meta["BLOCK_M"]),
        H_out,                                          # one row per pid_h
        B * triton.cdiv(W_out, meta["BLOCK_W"]),
    )
    conv2d_kernel[grid](
        x, w_t, y,
        B, C_in, C_out, H, W, H_out, W_out,
        x.stride(0),   x.stride(1),   x.stride(2),   x.stride(3),
        w_t.stride(0), w_t.stride(1), w_t.stride(2), w_t.stride(3),
        y.stride(0),   y.stride(1),   y.stride(2),   y.stride(3),
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

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["H"],
        x_vals=[2**i for i in range(5, 10)],
        x_log=True,
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton (2D-tiled implicit GEMM)", "PyTorch (F.conv2d)"],
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
