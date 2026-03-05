"""
Kernel:   conv1d
Category: convolution
Complexity: O(B × C_out × N_out × C_in × K)
Memory bound: Yes — arithmetic intensity ≈ 2·K·C_in / (K·C_in·dtype_bytes + dtype_bytes)
PyTorch equivalent: torch.nn.functional.conv1d(x, weight, padding=0)
References:
  - Direct convolution: https://en.wikipedia.org/wiki/Convolution#Discrete_convolution

Layout:
  x      : (B, C_in,  N)      — batch, input channels, signal length
  weight : (C_out, C_in, K)   — output channels, input channels, kernel size
  y      : (B, C_out, N_out)  — N_out = N - K + 1  (valid convolution, no padding)

Algorithm — Implicit GEMM (two-level tiled matrix multiply, no im2col materialisation):

  Observation: conv1d is equivalent to a matrix multiply
    Y[b, :, :] = W_flat @ X_col[b]        (C_out, N_out) = (C_out, CK) @ (CK, N_out)
  where CK = C_in × K, W_flat = weight.view(C_out, CK), and
    X_col[b, c_in*K + k, n_out] = x[b, c_in, n_out + k]   (im2col virtual view)

  Instead of materialising X_col (7× memory blowup), we stream it tile by tile.

  Grid:
    axis 0: cdiv(C_out, BLOCK_M)   — tiles over output channels
    axis 1: cdiv(N_out, BLOCK_N)   — tiles over output positions
    axis 2: B                       — one slice per batch element

  Two-level inner loop over the CK dimension:
    Outer: c_tile in range(cdiv(C_in, C_PER_TILE))   — C_in in chunks of C_PER_TILE
    Inner: k in tl.static_range(K)                    — K positions, fully unrolled

    Each (c_tile, k) step:
    ┌──────────────────────────────────────────────────────────────────────┐
    │ W_sub : (BLOCK_M, C_PER_TILE) ← weight[M_tile, c_in_base:c_in_end, k] │
    │ X_sub : (C_PER_TILE, BLOCK_N) ← x[b, c_in_base:c_in_end, N_tile + k]  │
    │ acc   += tl.dot(W_sub, X_sub)                                       │
    └──────────────────────────────────────────────────────────────────────┘

  Why two levels instead of one BLOCK_K = C_PER_TILE × K loop:
    Triton's JIT traces functions symbolically. A value like BLOCK_K = K * C_PER_TILE
    computed inside the kernel body loses its constexpr status — even though both
    operands are tl.constexpr — because the variable assignment is evaluated at trace
    time in a context where _unwrap_if_constexpr may not produce a plain int.
    tl.arange(0, BLOCK_K) then raises "arguments must be of type tl.constexpr".
    Splitting into two loops uses only BLOCK_M, BLOCK_N, K, C_PER_TILE directly in
    tl.arange — all of which are explicit tl.constexpr kernel parameters.

  C_PER_TILE ≥ 16 constraint:
    tl.dot requires the inner (contraction) dimension to be ≥ 16 for Triton's hardware
    matmul path. C_PER_TILE is that inner dim, so autotune configs use {16, 32}.

  Total tl.dot calls per program: cdiv(C_in, C_PER_TILE) × K
    = (128/16) × 7 = 56  (C_PER_TILE=16)
    = (128/32) × 7 = 28  (C_PER_TILE=32)
  Each call: (BLOCK_M, C_PER_TILE) × (C_PER_TILE, BLOCK_N) → (BLOCK_M, BLOCK_N)

  Key properties vs previous outer-product approach:
  • X is read once per N_out tile regardless of C_out (no C_out fan-out amplification).
    Previous approach read x cdiv(C_out, BLOCK_C_OUT) times per N_tile.
  • W fits in L2 cache and is reused across N_out tiles.
  • tl.dot uses hardware-optimal register blocking: each thread holds a submatrix of
    acc (BLOCK_M × BLOCK_N) across all loop iterations — FMA units stay busy.
  • tl.static_range(K) unrolls the K loop at compile time → no runtime branch overhead;
    the compiler can schedule 7 consecutive load+dot blocks for software pipelining.

  Remaining gap vs cuDNN implicit GEMM:
  • cuDNN reuses W_flat across all B batch elements simultaneously; our kernel has one
    set of programs per batch element. For B=1 (our benchmark) this is irrelevant.
  • cuDNN also uses fp16 tensor cores on T4 (fp16 path only); we benchmark fp32 which
    uses FMA units. If benchmarked in fp16, Triton would access T4's tensor cores at
    ~65 TFLOPS theoretical → dramatically faster than the fp32 FMA path.

TFLOPS metric: (2 × B × C_out × N_out × C_in × K × 1e-12) / (ms × 1e-3)
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# ── 1. Triton kernel — implicit GEMM, two-level loop ─────────────────────────

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
    ],
    key=["C_in", "C_out", "N_out"],
)
@triton.jit
def conv1d_kernel(
    x_ptr, w_ptr, y_ptr,
    B, C_in, C_out, N, N_out,
    stride_xb,  stride_xci, stride_xn,
    stride_wco, stride_wci, stride_wk,
    stride_yb,  stride_yco, stride_yn,
    K:           tl.constexpr,   # kernel size
    BLOCK_M:     tl.constexpr,   # tile over C_out
    BLOCK_N:     tl.constexpr,   # tile over N_out
    C_PER_TILE:  tl.constexpr,   # C_in channels per inner tile; must be ≥ 16 (tl.dot constraint)
):
    pid_m = tl.program_id(0)  # C_out tile
    pid_n = tl.program_id(1)  # N_out tile
    pid_b = tl.program_id(2)  # batch

    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # BLOCK_M is constexpr ✓
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # BLOCK_N is constexpr ✓
    mask_m = m_offs < C_out
    mask_n = n_offs < N_out

    # Per-tile C_in local offsets [0, 1, ..., C_PER_TILE-1].
    # Computed once outside the loops and reused — C_PER_TILE is constexpr ✓.
    c_tile_offs = tl.arange(0, C_PER_TILE)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    num_c_tiles = tl.cdiv(C_in, C_PER_TILE)

    for c_tile in range(num_c_tiles):
        c_in_base = c_tile * C_PER_TILE
        c_in_offs = c_in_base + c_tile_offs   # absolute c_in indices for this tile
        mask_c    = c_in_offs < C_in

        # Inner loop over K — fully unrolled because K is constexpr.
        # tl.static_range(K) tells Triton to unroll at compile time.
        for k in tl.static_range(K):

            # ── W_sub : (BLOCK_M, C_PER_TILE) ───────────────────────────────
            # weight[m_offs[m], c_in_offs[c], k]
            w_sub = tl.load(
                w_ptr
                + m_offs[:, None]    * stride_wco
                + c_in_offs[None, :] * stride_wci
                + k                  * stride_wk,
                mask=mask_m[:, None] & mask_c[None, :],
                other=0.0,
            )

            # ── X_sub : (C_PER_TILE, BLOCK_N) ───────────────────────────────
            # x[b, c_in_offs[c], n_offs[n] + k]
            # n_offs + k < N is guaranteed: n_offs < N_out = N-K+1, k ≤ K-1 → n_offs+k ≤ N-1.
            x_sub = tl.load(
                x_ptr
                + pid_b              * stride_xb
                + c_in_offs[:, None] * stride_xci
                + (n_offs[None, :] + k) * stride_xn,
                mask=mask_c[:, None] & mask_n[None, :],
                other=0.0,
            )

            # ── GEMM step ────────────────────────────────────────────────────
            # (BLOCK_M, C_PER_TILE) × (C_PER_TILE, BLOCK_N) → (BLOCK_M, BLOCK_N)
            acc += tl.dot(w_sub, x_sub, out_dtype=tl.float32)

    y_offs = m_offs[:, None] * stride_yco + n_offs[None, :] * stride_yn
    tl.store(
        y_ptr + pid_b * stride_yb + y_offs,
        acc.to(y_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_n[None, :],
    )


# ── 2. Python wrapper ─────────────────────────────────────────────────────────

def conv1d(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Implicit-GEMM 1D convolution — no padding, no bias.

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
        triton.cdiv(C_out, meta["BLOCK_M"]),
        triton.cdiv(N_out, meta["BLOCK_N"]),
        B,
    )
    conv1d_kernel[grid](
        x, w, y,
        B, C_in, C_out, N, N_out,
        x.stride(0), x.stride(1), x.stride(2),
        w.stride(0), w.stride(1), w.stride(2),
        y.stride(0), y.stride(1), y.stride(2),
        K=K,
    )
    return y


# ── 3. Correctness tests ──────────────────────────────────────────────────────

def test_conv1d():
    configs = [
        (1,  1,   1,   16,   3),
        (2,  4,   8,   64,   5),
        (4,  16,  32,  512,  7),
        (3,  5,   7,   100,  3),
        (1,  64,  64,  1024, 11),
        (8,  32,  64,  2048, 3),
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
# Fix B=1, C_in=C_out=128, K=7 (WaveNet-style).
# Vary N from 1K to 128K.
#
# Arithmetic intensity ≈ 0.5 FLOPs/byte (fp32) — HBM-bandwidth-limited.
# Both Triton and PyTorch are memory-bound at large N; the gap reflects how many
# times each algorithm reads x from HBM (implicit GEMM reads x once; direct conv
# reads it K × cdiv(C_out, BLOCK_C_OUT) times — a structural 7–14× amplification).
#
# Triton uses FP32 FMA units on T4 (no fp32 tensor cores).
# PyTorch (cuDNN) uses the same units on the fp32 path.
# An fp16 benchmark would show a much larger Triton advantage (T4 has fp16 tensor cores).

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[2**i for i in range(10, 18)],
        x_log=True,
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton (implicit GEMM)", "PyTorch (F.conv1d)"],
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
