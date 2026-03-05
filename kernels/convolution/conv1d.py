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

Algorithm — Implicit GEMM (tiled matrix multiply without materialising im2col):

  Observation: conv1d is equivalent to a matrix multiply
    Y[b, :, :] = W_flat @ X_col[b]        (C_out, N_out) = (C_out, CK) @ (CK, N_out)
  where CK = C_in × K, W_flat = weight.view(C_out, CK), and
    X_col[b, c_in*K + k, n_out] = x[b, c_in, n_out + k]   (im2col virtual view)

  Instead of materialising X_col (7× memory blowup), we stream it tile by tile.

  Grid:
    axis 0: cdiv(C_out, BLOCK_M)   — tiles over output channels
    axis 1: cdiv(N_out, BLOCK_N)   — tiles over output positions
    axis 2: B                       — one slice per batch element

  Inner loop over BLOCK_K = C_PER_TILE × K sized tiles of the CK dimension:
    ┌──────────────────────────────────────────────────────────────────────┐
    │ W_tile : (BLOCK_M, BLOCK_K) ← weight[M_tile, c_in_base*K : c_in_end*K] │
    │ X_tile : (BLOCK_K, BLOCK_N) ← x[b, c_in_base:c_in_end, N_tile + k_off] │
    │ acc    += tl.dot(W_tile, X_tile)                                    │
    └──────────────────────────────────────────────────────────────────────┘

  BLOCK_K = C_PER_TILE × K is always a multiple of K. Within each tile:
    c_in_off[k_inner] = k_inner // K   — compile-time constant (K is constexpr)
    k_off[k_inner]    = k_inner % K    — compile-time constant

  This means each row k of X_tile is a contiguous BLOCK_N-wide load:
    x[b, c_in_base + c_in_off[k], N_tile_start + k_off[k] : N_tile_start + k_off[k] + BLOCK_N]

  Key properties vs previous outer-product approach:
  • X is read once per N_out tile regardless of C_out (no C_out fan-out amplification).
    Previous approach read x cdiv(C_out, BLOCK_C_OUT) times per N_tile.
  • W fits in L2 cache (128 × 128 × 7 × 4 = 0.45 MB) and is reused across N_out tiles.
  • tl.dot uses hardware-optimal register blocking: each thread holds a submatrix of
    acc (BLOCK_M × BLOCK_N) across all C_PER_TILE loop iterations — FMA units stay busy.
  • Software pipelining (num_stages) hides the memory latency of the next tile's loads
    behind computation of the current tile's dot product.

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


# ── 1. Triton kernel — implicit GEMM ─────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "C_PER_TILE": 4}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "C_PER_TILE": 4}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "C_PER_TILE": 4}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "C_PER_TILE": 4}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "C_PER_TILE": 8}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "C_PER_TILE": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "C_PER_TILE": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "C_PER_TILE": 8}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "C_PER_TILE": 4}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 256, "C_PER_TILE": 4}, num_stages=4, num_warps=8),
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
    C_PER_TILE:  tl.constexpr,   # C_in channels per K-inner tile; BLOCK_K = K × C_PER_TILE
):
    # BLOCK_K is a compile-time constant (both K and C_PER_TILE are constexpr)
    BLOCK_K = K * C_PER_TILE

    pid_m = tl.program_id(0)  # C_out tile
    pid_n = tl.program_id(1)  # N_out tile
    pid_b = tl.program_id(2)  # batch

    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = m_offs < C_out
    mask_n = n_offs < N_out

    # Per-position offsets within a BLOCK_K tile.
    # K is constexpr → // and % fold to compile-time constant arrays.
    k_arange      = tl.arange(0, BLOCK_K)   # [0, 1, ..., BLOCK_K-1]
    c_in_tile_rel = k_arange // K            # which c_in offset within the tile
    k_tile_rel    = k_arange % K             # which k offset within that c_in

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    num_k_tiles = tl.cdiv(C_in, C_PER_TILE)

    for k_tile in range(num_k_tiles):
        c_in_base = k_tile * C_PER_TILE

        # ── W_tile : (BLOCK_M, BLOCK_K) ─────────────────────────────────────
        # weight[m, c_in_base + c_in_tile_rel[k], k_tile_rel[k]]
        # For contiguous weight (stride_wco = C_in*K, stride_wci = K, stride_wk = 1)
        # this is weight viewed as (C_out, C_in*K) with contiguous k_inner columns.
        w_tile = tl.load(
            w_ptr
            + m_offs[:, None] * stride_wco
            + (c_in_base + c_in_tile_rel[None, :]) * stride_wci
            + k_tile_rel[None, :] * stride_wk,
            mask=mask_m[:, None] & ((c_in_base + c_in_tile_rel[None, :]) < C_in),
            other=0.0,
        )

        # ── X_tile : (BLOCK_K, BLOCK_N) ─────────────────────────────────────
        # x[b, c_in_base + c_in_tile_rel[k], n_offs[j] + k_tile_rel[k]]
        # Row k is x[b, c_in_abs, n_start + k_off : n_start + k_off + BLOCK_N] — contiguous.
        # Mask: c_in_abs < C_in is sufficient; n_offs + k_tile_rel < N is guaranteed
        # because n_offs < N_out = N - K + 1 and k_tile_rel ≤ K-1, so n_offs + k_tile_rel ≤ N-1.
        x_tile = tl.load(
            x_ptr
            + pid_b * stride_xb
            + (c_in_base + c_in_tile_rel[:, None]) * stride_xci
            + (k_tile_rel[:, None] + n_offs[None, :]) * stride_xn,
            mask=((c_in_base + c_in_tile_rel[:, None]) < C_in) & mask_n[None, :],
            other=0.0,
        )

        # ── GEMM step ────────────────────────────────────────────────────────
        # (BLOCK_M, BLOCK_K) × (BLOCK_K, BLOCK_N) → (BLOCK_M, BLOCK_N)
        acc += tl.dot(w_tile, x_tile, out_dtype=tl.float32)

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
