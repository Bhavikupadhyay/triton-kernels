"""
Kernel:   winograd_conv2d_fused
Category: convolution
Complexity: O(B × C_out × N_tiles × C_in) — same arithmetic as unfused, zero HBM intermediates
Memory bound: No — compute-bound at large C
PyTorch equivalent: torch.nn.functional.conv2d(x, weight, padding=0) with K=3
References:
  - Lavin & Gray, "Fast Algorithms for Convolutional Neural Networks", CVPR 2016
    https://arxiv.org/abs/1509.09308

Algorithm — Fused Winograd F(2,3):

  Unfused Winograd uses four separate kernel launches and two HBM-resident intermediate
  tensors (V_t and M_t). At B=1, H=256, C=64:
    V_t: (B=1, 16, N_tiles=16129, C_in=64) × 4 bytes ≈  66 MB written + read
    M_t: same size                                     ≈  66 MB written + read
    Total intermediate HBM traffic: ~264 MB per forward pass

  At T4's 320 GB/s peak that is ≥0.83 ms of unavoidable latency before any compute.

  This kernel fuses the input transform, pointwise accumulate, and output transform into a
  single Triton program. V_t and M_t never touch HBM; all 16 accumulator planes live in
  registers.

  Two-kernel structure:
    1. Weight transform (winograd_weight_transform_kernel): identical to the unfused version.
       U_t: (16, C_out, C_in) — ~256 KB for C=64, fits entirely in T4's 4 MB L2.
       Computed once; reused by all programs of kernel 2 via L2 cache.

    2. Fused kernel (winograd_fused_kernel):
       Grid: (cdiv(N_tiles, BLOCK_TILES), cdiv(C_out, BLOCK_CO), B)
       Each program holds 16 × (BLOCK_TILES, BLOCK_CO) fp32 accumulators in registers.
       Iterates over C_in; for each ci:
         a. Load 16 input scalars for all BLOCK_TILES tiles (gather from x, masked)
         b. Apply B^T × d × B to get 16 v values (BLOCK_TILES,)
         c. Load U_t[p, co_offs, ci] for all 16 positions (from L2-resident U_t)
         d. Outer-product accumulate: acc_p += v_p[:, None] * u_p[None, :]
       After C_in loop: apply A^T × acc × A, store 4 output elements per (tile, co).

  Layout:
    x      : (B, C_in,  H,     W)     — NCHW, K=3 only, no padding, no bias
    weight : (C_out, C_in, 3, 3)
    U_t    : (16, C_out, C_in)        — weight transform output (L2-resident)
    y      : (B, C_out, H-2, W-2)    — valid convolution output

TFLOPS metric: (2 × B × C_out × H_out × W_out × C_in × 9 × 1e-12) / (ms × 1e-3)
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# ── 1. Weight transform kernel (inline copy — no cross-file import) ───────────

@triton.jit
def winograd_weight_transform_kernel(
    w_ptr,     # (C_out, C_in, 3, 3)
    u_t_ptr,   # (16, C_out, C_in)
    C_in, C_out,
    stride_wco, stride_wci, stride_wkh, stride_wkw,
    stride_ut_p, stride_ut_co, stride_ut_ci,
):
    """Apply G × w × G^T to each (c_out, c_in) 3×3 weight patch → U_t[p, c_out, c_in]."""
    pid_co = tl.program_id(0)
    pid_ci = tl.program_id(1)

    if pid_co >= C_out or pid_ci >= C_in:
        return

    base = w_ptr + pid_co * stride_wco + pid_ci * stride_wci

    g00 = tl.load(base + 0 * stride_wkh + 0 * stride_wkw).to(tl.float32)
    g01 = tl.load(base + 0 * stride_wkh + 1 * stride_wkw).to(tl.float32)
    g02 = tl.load(base + 0 * stride_wkh + 2 * stride_wkw).to(tl.float32)
    g10 = tl.load(base + 1 * stride_wkh + 0 * stride_wkw).to(tl.float32)
    g11 = tl.load(base + 1 * stride_wkh + 1 * stride_wkw).to(tl.float32)
    g12 = tl.load(base + 1 * stride_wkh + 2 * stride_wkw).to(tl.float32)
    g20 = tl.load(base + 2 * stride_wkh + 0 * stride_wkw).to(tl.float32)
    g21 = tl.load(base + 2 * stride_wkh + 1 * stride_wkw).to(tl.float32)
    g22 = tl.load(base + 2 * stride_wkh + 2 * stride_wkw).to(tl.float32)

    # Intermediate = G × w  (4 rows × 3 cols)
    i00 = g00;                      i01 = g01;                      i02 = g02
    i10 = (g00 + g10 + g20) * 0.5; i11 = (g01 + g11 + g21) * 0.5; i12 = (g02 + g12 + g22) * 0.5
    i20 = (g00 - g10 + g20) * 0.5; i21 = (g01 - g11 + g21) * 0.5; i22 = (g02 - g12 + g22) * 0.5
    i30 = g20;                      i31 = g21;                      i32 = g22

    # U_t = intermediate × G^T  (4 rows × 4 cols), stored at U_t[p, pid_co, pid_ci]
    u_t_base = u_t_ptr + pid_co * stride_ut_co + pid_ci * stride_ut_ci

    tl.store(u_t_base +  0 * stride_ut_p, i00)
    tl.store(u_t_base +  1 * stride_ut_p, (i00 + i01 + i02) * 0.5)
    tl.store(u_t_base +  2 * stride_ut_p, (i00 - i01 + i02) * 0.5)
    tl.store(u_t_base +  3 * stride_ut_p, i02)

    tl.store(u_t_base +  4 * stride_ut_p, i10)
    tl.store(u_t_base +  5 * stride_ut_p, (i10 + i11 + i12) * 0.5)
    tl.store(u_t_base +  6 * stride_ut_p, (i10 - i11 + i12) * 0.5)
    tl.store(u_t_base +  7 * stride_ut_p, i12)

    tl.store(u_t_base +  8 * stride_ut_p, i20)
    tl.store(u_t_base +  9 * stride_ut_p, (i20 + i21 + i22) * 0.5)
    tl.store(u_t_base + 10 * stride_ut_p, (i20 - i21 + i22) * 0.5)
    tl.store(u_t_base + 11 * stride_ut_p, i22)

    tl.store(u_t_base + 12 * stride_ut_p, i30)
    tl.store(u_t_base + 13 * stride_ut_p, (i30 + i31 + i32) * 0.5)
    tl.store(u_t_base + 14 * stride_ut_p, (i30 - i31 + i32) * 0.5)
    tl.store(u_t_base + 15 * stride_ut_p, i32)


# ── 2. Fused input transform + accumulate + output transform kernel ───────────

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_TILES": 32, "BLOCK_CO": 32}, num_warps=4),
        triton.Config({"BLOCK_TILES": 16, "BLOCK_CO": 32}, num_warps=4),
        triton.Config({"BLOCK_TILES": 32, "BLOCK_CO": 16}, num_warps=4),
        triton.Config({"BLOCK_TILES": 16, "BLOCK_CO": 16}, num_warps=4),
        triton.Config({"BLOCK_TILES": 64, "BLOCK_CO": 16}, num_warps=8),
        triton.Config({"BLOCK_TILES": 16, "BLOCK_CO": 64}, num_warps=8),
    ],
    key=["N_tiles", "C_in", "C_out"],
)
@triton.jit
def winograd_fused_kernel(
    x_ptr,    # (B, C_in, H, W)
    u_t_ptr,  # (16, C_out, C_in) — weight-transformed, L2-resident
    y_ptr,    # (B, C_out, H_out, W_out)
    B, C_in, C_out, H, W, H_out, W_out, H_tiles, W_tiles, N_tiles,
    stride_xb,   stride_xci,  stride_xh,   stride_xw,
    stride_ut_p, stride_ut_co, stride_ut_ci,
    stride_yb,   stride_yco,  stride_yh,   stride_yw,
    BLOCK_TILES: tl.constexpr, BLOCK_CO: tl.constexpr,
):
    """Fused Winograd F(2,3): input transform + C_in accumulate + output transform.

    No V_t or M_t intermediates in HBM. All 16 × (BLOCK_TILES, BLOCK_CO) accumulators
    live in registers across the C_in loop.

    Grid: (cdiv(N_tiles, BLOCK_TILES), cdiv(C_out, BLOCK_CO), B)
    """
    pid_bt = tl.program_id(0)   # tile block index
    pid_co = tl.program_id(1)   # output channel block index
    pid_b  = tl.program_id(2)   # batch index

    t_offs  = pid_bt * BLOCK_TILES + tl.arange(0, BLOCK_TILES)   # (BLOCK_TILES,)
    co_offs = pid_co * BLOCK_CO   + tl.arange(0, BLOCK_CO)       # (BLOCK_CO,)

    tile_mask = t_offs < N_tiles    # (BLOCK_TILES,)
    co_mask   = co_offs < C_out     # (BLOCK_CO,)

    # Decode flat tile index → (h_tile, w_tile) → top-left of 4×4 input patch
    ht = t_offs // W_tiles          # (BLOCK_TILES,)
    wt = t_offs %  W_tiles          # (BLOCK_TILES,)
    h0 = ht * 2                     # (BLOCK_TILES,) — patch row 0 in x
    w0 = wt * 2                     # (BLOCK_TILES,) — patch col 0 in x

    # 16 accumulator planes — persistent across C_in loop, one per Winograd position
    # p = row*4 + col;  shape each: (BLOCK_TILES, BLOCK_CO)
    acc_0  = tl.zeros((BLOCK_TILES, BLOCK_CO), dtype=tl.float32)
    acc_1  = tl.zeros((BLOCK_TILES, BLOCK_CO), dtype=tl.float32)
    acc_2  = tl.zeros((BLOCK_TILES, BLOCK_CO), dtype=tl.float32)
    acc_3  = tl.zeros((BLOCK_TILES, BLOCK_CO), dtype=tl.float32)
    acc_4  = tl.zeros((BLOCK_TILES, BLOCK_CO), dtype=tl.float32)
    acc_5  = tl.zeros((BLOCK_TILES, BLOCK_CO), dtype=tl.float32)
    acc_6  = tl.zeros((BLOCK_TILES, BLOCK_CO), dtype=tl.float32)
    acc_7  = tl.zeros((BLOCK_TILES, BLOCK_CO), dtype=tl.float32)
    acc_8  = tl.zeros((BLOCK_TILES, BLOCK_CO), dtype=tl.float32)
    acc_9  = tl.zeros((BLOCK_TILES, BLOCK_CO), dtype=tl.float32)
    acc_10 = tl.zeros((BLOCK_TILES, BLOCK_CO), dtype=tl.float32)
    acc_11 = tl.zeros((BLOCK_TILES, BLOCK_CO), dtype=tl.float32)
    acc_12 = tl.zeros((BLOCK_TILES, BLOCK_CO), dtype=tl.float32)
    acc_13 = tl.zeros((BLOCK_TILES, BLOCK_CO), dtype=tl.float32)
    acc_14 = tl.zeros((BLOCK_TILES, BLOCK_CO), dtype=tl.float32)
    acc_15 = tl.zeros((BLOCK_TILES, BLOCK_CO), dtype=tl.float32)

    x_b_base  = x_ptr   + pid_b * stride_xb

    for ci in range(C_in):
        x_ci_base  = x_b_base  + ci * stride_xci
        u_ci_base  = u_t_ptr   + ci * stride_ut_ci

        # ── 1. Load 4×4 input patch for all BLOCK_TILES tiles simultaneously ──
        # d_rc: (BLOCK_TILES,) — row r, col c of the input patch
        # Gather load: each tile reads a different (h0+r, w0+c) position in x
        d_00 = tl.load(x_ci_base + (h0+0)*stride_xh + (w0+0)*stride_xw,
                       mask=tile_mask & ((h0+0)<H) & ((w0+0)<W), other=0.0).to(tl.float32)
        d_01 = tl.load(x_ci_base + (h0+0)*stride_xh + (w0+1)*stride_xw,
                       mask=tile_mask & ((h0+0)<H) & ((w0+1)<W), other=0.0).to(tl.float32)
        d_02 = tl.load(x_ci_base + (h0+0)*stride_xh + (w0+2)*stride_xw,
                       mask=tile_mask & ((h0+0)<H) & ((w0+2)<W), other=0.0).to(tl.float32)
        d_03 = tl.load(x_ci_base + (h0+0)*stride_xh + (w0+3)*stride_xw,
                       mask=tile_mask & ((h0+0)<H) & ((w0+3)<W), other=0.0).to(tl.float32)
        d_10 = tl.load(x_ci_base + (h0+1)*stride_xh + (w0+0)*stride_xw,
                       mask=tile_mask & ((h0+1)<H) & ((w0+0)<W), other=0.0).to(tl.float32)
        d_11 = tl.load(x_ci_base + (h0+1)*stride_xh + (w0+1)*stride_xw,
                       mask=tile_mask & ((h0+1)<H) & ((w0+1)<W), other=0.0).to(tl.float32)
        d_12 = tl.load(x_ci_base + (h0+1)*stride_xh + (w0+2)*stride_xw,
                       mask=tile_mask & ((h0+1)<H) & ((w0+2)<W), other=0.0).to(tl.float32)
        d_13 = tl.load(x_ci_base + (h0+1)*stride_xh + (w0+3)*stride_xw,
                       mask=tile_mask & ((h0+1)<H) & ((w0+3)<W), other=0.0).to(tl.float32)
        d_20 = tl.load(x_ci_base + (h0+2)*stride_xh + (w0+0)*stride_xw,
                       mask=tile_mask & ((h0+2)<H) & ((w0+0)<W), other=0.0).to(tl.float32)
        d_21 = tl.load(x_ci_base + (h0+2)*stride_xh + (w0+1)*stride_xw,
                       mask=tile_mask & ((h0+2)<H) & ((w0+1)<W), other=0.0).to(tl.float32)
        d_22 = tl.load(x_ci_base + (h0+2)*stride_xh + (w0+2)*stride_xw,
                       mask=tile_mask & ((h0+2)<H) & ((w0+2)<W), other=0.0).to(tl.float32)
        d_23 = tl.load(x_ci_base + (h0+2)*stride_xh + (w0+3)*stride_xw,
                       mask=tile_mask & ((h0+2)<H) & ((w0+3)<W), other=0.0).to(tl.float32)
        d_30 = tl.load(x_ci_base + (h0+3)*stride_xh + (w0+0)*stride_xw,
                       mask=tile_mask & ((h0+3)<H) & ((w0+0)<W), other=0.0).to(tl.float32)
        d_31 = tl.load(x_ci_base + (h0+3)*stride_xh + (w0+1)*stride_xw,
                       mask=tile_mask & ((h0+3)<H) & ((w0+1)<W), other=0.0).to(tl.float32)
        d_32 = tl.load(x_ci_base + (h0+3)*stride_xh + (w0+2)*stride_xw,
                       mask=tile_mask & ((h0+3)<H) & ((w0+2)<W), other=0.0).to(tl.float32)
        d_33 = tl.load(x_ci_base + (h0+3)*stride_xh + (w0+3)*stride_xw,
                       mask=tile_mask & ((h0+3)<H) & ((w0+3)<W), other=0.0).to(tl.float32)

        # ── 2. Input transform B^T × d × B → 16 v values, each (BLOCK_TILES,) ──
        # Step 1: i = B^T × d  (B^T rows: [d0-d2], [d1+d2], [-d1+d2], [d1-d3])
        i_00 = d_00 - d_20;   i_01 = d_01 - d_21;   i_02 = d_02 - d_22;   i_03 = d_03 - d_23
        i_10 = d_10 + d_20;   i_11 = d_11 + d_21;   i_12 = d_12 + d_22;   i_13 = d_13 + d_23
        i_20 = -d_10 + d_20;  i_21 = -d_11 + d_21;  i_22 = -d_12 + d_22;  i_23 = -d_13 + d_23
        i_30 = d_10 - d_30;   i_31 = d_11 - d_31;   i_32 = d_12 - d_32;   i_33 = d_13 - d_33

        # Step 2: v = i × B  (same B^T ops applied across rows of i)
        v_00 = i_00 - i_02;   v_01 = i_01 + i_02;   v_02 = -i_01 + i_02;  v_03 = i_01 - i_03
        v_10 = i_10 - i_12;   v_11 = i_11 + i_12;   v_12 = -i_11 + i_12;  v_13 = i_11 - i_13
        v_20 = i_20 - i_22;   v_21 = i_21 + i_22;   v_22 = -i_21 + i_22;  v_23 = i_21 - i_23
        v_30 = i_30 - i_32;   v_31 = i_31 + i_32;   v_32 = -i_31 + i_32;  v_33 = i_31 - i_33

        # ── 3. Load U_t[p, co_offs, ci] — (BLOCK_CO,) per position ─────────────
        # U_t layout: (16, C_out, C_in) → stride_ut_p, stride_ut_co, stride_ut_ci
        # co_offs selects the output channel slice; ci is the current input channel
        u_0  = tl.load(u_ci_base +  0*stride_ut_p + co_offs*stride_ut_co, mask=co_mask, other=0.0)
        u_1  = tl.load(u_ci_base +  1*stride_ut_p + co_offs*stride_ut_co, mask=co_mask, other=0.0)
        u_2  = tl.load(u_ci_base +  2*stride_ut_p + co_offs*stride_ut_co, mask=co_mask, other=0.0)
        u_3  = tl.load(u_ci_base +  3*stride_ut_p + co_offs*stride_ut_co, mask=co_mask, other=0.0)
        u_4  = tl.load(u_ci_base +  4*stride_ut_p + co_offs*stride_ut_co, mask=co_mask, other=0.0)
        u_5  = tl.load(u_ci_base +  5*stride_ut_p + co_offs*stride_ut_co, mask=co_mask, other=0.0)
        u_6  = tl.load(u_ci_base +  6*stride_ut_p + co_offs*stride_ut_co, mask=co_mask, other=0.0)
        u_7  = tl.load(u_ci_base +  7*stride_ut_p + co_offs*stride_ut_co, mask=co_mask, other=0.0)
        u_8  = tl.load(u_ci_base +  8*stride_ut_p + co_offs*stride_ut_co, mask=co_mask, other=0.0)
        u_9  = tl.load(u_ci_base +  9*stride_ut_p + co_offs*stride_ut_co, mask=co_mask, other=0.0)
        u_10 = tl.load(u_ci_base + 10*stride_ut_p + co_offs*stride_ut_co, mask=co_mask, other=0.0)
        u_11 = tl.load(u_ci_base + 11*stride_ut_p + co_offs*stride_ut_co, mask=co_mask, other=0.0)
        u_12 = tl.load(u_ci_base + 12*stride_ut_p + co_offs*stride_ut_co, mask=co_mask, other=0.0)
        u_13 = tl.load(u_ci_base + 13*stride_ut_p + co_offs*stride_ut_co, mask=co_mask, other=0.0)
        u_14 = tl.load(u_ci_base + 14*stride_ut_p + co_offs*stride_ut_co, mask=co_mask, other=0.0)
        u_15 = tl.load(u_ci_base + 15*stride_ut_p + co_offs*stride_ut_co, mask=co_mask, other=0.0)

        # ── 4. Outer-product accumulate: acc_p += v_p[:, None] * u_p[None, :] ──
        # v_p:  (BLOCK_TILES,) — input-transformed value for position p, across tiles
        # u_p:  (BLOCK_CO,)    — weight-transformed value for position p, across output channels
        # acc_p: (BLOCK_TILES, BLOCK_CO) — accumulated result
        acc_0  += v_00[:, None] * u_0 [None, :]
        acc_1  += v_01[:, None] * u_1 [None, :]
        acc_2  += v_02[:, None] * u_2 [None, :]
        acc_3  += v_03[:, None] * u_3 [None, :]
        acc_4  += v_10[:, None] * u_4 [None, :]
        acc_5  += v_11[:, None] * u_5 [None, :]
        acc_6  += v_12[:, None] * u_6 [None, :]
        acc_7  += v_13[:, None] * u_7 [None, :]
        acc_8  += v_20[:, None] * u_8 [None, :]
        acc_9  += v_21[:, None] * u_9 [None, :]
        acc_10 += v_22[:, None] * u_10[None, :]
        acc_11 += v_23[:, None] * u_11[None, :]
        acc_12 += v_30[:, None] * u_12[None, :]
        acc_13 += v_31[:, None] * u_13[None, :]
        acc_14 += v_32[:, None] * u_14[None, :]
        acc_15 += v_33[:, None] * u_15[None, :]

    # ── 5. Output transform A^T × acc × A → 4 output elements per (tile, co) ──
    # acc_p ≡ acc[row][col] where p = row*4 + col
    # A^T (2×4): rows [1,1,1,0] and [0,1,-1,-1]
    # Step 1: oi = A^T × acc  (applies A^T to rows of acc 4×4 block)
    oi_00 = acc_0  + acc_4  + acc_8
    oi_01 = acc_1  + acc_5  + acc_9
    oi_02 = acc_2  + acc_6  + acc_10
    oi_03 = acc_3  + acc_7  + acc_11
    oi_10 = acc_4  - acc_8  - acc_12
    oi_11 = acc_5  - acc_9  - acc_13
    oi_12 = acc_6  - acc_10 - acc_14
    oi_13 = acc_7  - acc_11 - acc_15

    # Step 2: y = oi × A  (applies A^T row ops to columns of oi)
    y_00 = oi_00 + oi_01 + oi_02
    y_01 = oi_01 - oi_02 - oi_03
    y_10 = oi_10 + oi_11 + oi_12
    y_11 = oi_11 - oi_12 - oi_13

    # ── 6. Store 4 output elements per (tile, co) with boundary masks ─────────
    # Output Y[pid_b, co_offs, ht*2+dr, wt*2+dc] for (dr, dc) ∈ {0,1}²
    y_b_base = y_ptr + pid_b * stride_yb
    # h0 = ht*2 (already computed), w0 = wt*2 (already computed)

    # y_00: (ht*2,   wt*2)
    tl.store(
        y_b_base + co_offs[None, :] * stride_yco
                 + (h0[:, None] + 0) * stride_yh
                 + (w0[:, None] + 0) * stride_yw,
        y_00,
        mask=tile_mask[:, None] & co_mask[None, :]
             & ((h0 + 0 < H_out)[:, None]) & ((w0 + 0 < W_out)[:, None]),
    )
    # y_01: (ht*2,   wt*2+1)
    tl.store(
        y_b_base + co_offs[None, :] * stride_yco
                 + (h0[:, None] + 0) * stride_yh
                 + (w0[:, None] + 1) * stride_yw,
        y_01,
        mask=tile_mask[:, None] & co_mask[None, :]
             & ((h0 + 0 < H_out)[:, None]) & ((w0 + 1 < W_out)[:, None]),
    )
    # y_10: (ht*2+1, wt*2)
    tl.store(
        y_b_base + co_offs[None, :] * stride_yco
                 + (h0[:, None] + 1) * stride_yh
                 + (w0[:, None] + 0) * stride_yw,
        y_10,
        mask=tile_mask[:, None] & co_mask[None, :]
             & ((h0 + 1 < H_out)[:, None]) & ((w0 + 0 < W_out)[:, None]),
    )
    # y_11: (ht*2+1, wt*2+1)
    tl.store(
        y_b_base + co_offs[None, :] * stride_yco
                 + (h0[:, None] + 1) * stride_yh
                 + (w0[:, None] + 1) * stride_yw,
        y_11,
        mask=tile_mask[:, None] & co_mask[None, :]
             & ((h0 + 1 < H_out)[:, None]) & ((w0 + 1 < W_out)[:, None]),
    )


# ── 3. Python wrapper ─────────────────────────────────────────────────────────

def winograd_conv2d_fused(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Fused Winograd F(2,3) 2D convolution — K=3 only, no padding, no bias.

    Two kernel launches: weight transform (once) + fused input/accumulate/output.
    V_t and M_t intermediates never materialise in HBM.

    Args:
        x: (B, C_in, H, W)       input on CUDA, fp32. H, W ≥ 4.
        w: (C_out, C_in, 3, 3)   weight on CUDA, fp32.

    Returns:
        y: (B, C_out, H-2, W-2), fp32.
    """
    assert x.is_cuda and w.is_cuda, "Inputs must be on CUDA"
    x = x.contiguous().to(torch.float32)
    w = w.contiguous().to(torch.float32)
    assert x.ndim == 4, "x must be 4D: (B, C_in, H, W)"
    assert w.ndim == 4, "w must be 4D: (C_out, C_in, 3, 3)"
    B, C_in, H, W    = x.shape
    C_out, _, KH, KW = w.shape
    assert KH == 3 and KW == 3, f"winograd_conv2d_fused only supports K=3; got {KH}×{KW}"
    assert H >= 4 and W >= 4,   f"Input spatial dims must be ≥ 4; got {H}×{W}"

    H_out   = H - 2
    W_out   = W - 2
    H_tiles = triton.cdiv(H_out, 2)
    W_tiles = triton.cdiv(W_out, 2)
    N_tiles = H_tiles * W_tiles

    U_t = torch.empty((16, C_out, C_in), device=x.device, dtype=torch.float32)
    Y   = torch.empty((B, C_out, H_out, W_out), device=x.device, dtype=torch.float32)

    # Kernel 1: weight transform — (C_out, C_in) grid → U_t: (16, C_out, C_in)
    winograd_weight_transform_kernel[(C_out, C_in)](
        w, U_t,
        C_in, C_out,
        w.stride(0), w.stride(1), w.stride(2), w.stride(3),
        U_t.stride(0), U_t.stride(1), U_t.stride(2),
    )

    # Kernel 2: fused input transform + accumulate + output transform
    grid = lambda meta: (
        triton.cdiv(N_tiles, meta["BLOCK_TILES"]),
        triton.cdiv(C_out,   meta["BLOCK_CO"]),
        B,
    )
    winograd_fused_kernel[grid](
        x, U_t, Y,
        B, C_in, C_out, H, W, H_out, W_out, H_tiles, W_tiles, N_tiles,
        x.stride(0),   x.stride(1),   x.stride(2),   x.stride(3),
        U_t.stride(0), U_t.stride(1), U_t.stride(2),
        Y.stride(0),   Y.stride(1),   Y.stride(2),   Y.stride(3),
    )
    return Y


# ── 4. Correctness tests ──────────────────────────────────────────────────────

def test_winograd_conv2d_fused():
    print("Testing winograd_conv2d_fused...")
    configs = [
        # B, C_in, C_out, H, W
        (1,  1,   1,   4,   4),    # minimal: 4×4 → 2×2 output, 1 tile
        (1,  3,   8,   8,   8),    # 6×6 output, 3×3 tiles
        (2,  4,   8,   16,  16),   # 14×14 output, 7×7 tiles
        (1,  16,  32,  32,  32),
        (2,  32,  64,  32,  32),
        (1,  64,  64,  56,  56),   # ResNet-style
        (1,  3,   8,   9,   9),    # H_out=7 → 4 H-tiles (partial last row)
        (1,  4,   4,   11,  11),   # H_out=9 → 5 H-tiles
    ]
    for B, C_in, C_out, H, W in configs:
        x  = torch.randn(B, C_in, H, W,     device="cuda", dtype=torch.float32)
        wt = torch.randn(C_out, C_in, 3, 3, device="cuda", dtype=torch.float32)
        ref = F.conv2d(x, wt, padding=0)
        got = winograd_conv2d_fused(x, wt)
        torch.testing.assert_close(got, ref, atol=1e-3, rtol=1e-3)
        print(f"  B={B} C_in={C_in} C_out={C_out} {H}×{W} → {H-2}×{W-2}  "
              f"max_err={(got - ref).abs().max():.2e}  PASS")
    print("All tests passed.")


# ── 5. Benchmark ──────────────────────────────────────────────────────────────

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["H"],
        x_vals=[2**i for i in range(5, 10)],
        x_log=True,
        line_arg="provider",
        line_vals=["direct", "winograd", "winograd_fused", "torch"],
        line_names=[
            "Triton direct conv2d",
            "Triton Winograd (unfused)",
            "Triton Winograd (fused)",
            "PyTorch (F.conv2d)",
        ],
        styles=[("blue", "-"), ("orange", "-"), ("red", "-"), ("green", ":")],
        ylabel="TFLOPS",
        plot_name="winograd_conv2d_fused",
        args={"B": 1, "C_in": 64, "C_out": 64, "K": 3},
    )
)
def benchmark_winograd_conv2d_fused(B, C_in, C_out, K, H, provider):
    from kernels.convolution.conv2d import conv2d
    from kernels.convolution.winograd_conv2d import winograd_conv2d

    W     = H
    x     = torch.randn(B, C_in, H, W,     device="cuda", dtype=torch.float32)
    wt    = torch.randn(C_out, C_in, K, K, device="cuda", dtype=torch.float32)
    H_out = H - K + 1
    W_out = W - K + 1
    quantiles = [0.5, 0.2, 0.8]

    if provider == "direct":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: conv2d(x, wt), warmup=25, rep=100, quantiles=quantiles
        )
    elif provider == "winograd":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: winograd_conv2d(x, wt), warmup=25, rep=100, quantiles=quantiles
        )
    elif provider == "winograd_fused":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: winograd_conv2d_fused(x, wt), warmup=25, rep=100, quantiles=quantiles
        )
    else:
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: F.conv2d(x, wt), warmup=25, rep=100, quantiles=quantiles
        )

    tflops = 2 * B * C_out * H_out * W_out * C_in * K * K * 1e-12
    return tflops / (ms * 1e-3), tflops / (max_ms * 1e-3), tflops / (min_ms * 1e-3)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_winograd_conv2d_fused()
    import os
    os.makedirs("benchmarks/results/convolution", exist_ok=True)
    benchmark_winograd_conv2d_fused.run(
        print_data=True,
        show_plots=True,
        save_path="benchmarks/results/convolution",
    )
