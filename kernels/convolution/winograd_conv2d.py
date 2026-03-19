"""
Kernel:   winograd_conv2d
Category: convolution
Complexity: O(B × 16 × C_out × N_tiles × C_in) — GEMM per position via tl.dot
Memory bound: No — Winograd reduces multiply count ~2.25× vs direct conv for K=3
PyTorch equivalent: torch.nn.functional.conv2d(x, weight, padding=0) with K=3
References:
  - Lavin & Gray, "Fast Algorithms for Convolutional Neural Networks", CVPR 2016
    https://arxiv.org/abs/1509.09308

Algorithm — Winograd F(2,3):

  Direct K=3 conv produces a 2×2 output patch from a 4×4 input patch using 9×4=36
  multiplications. Winograd F(2,3) reduces this to 16 pointwise multiplications:

  1. Weight transform:  U_t = G × g × G^T       (3×3 → 4×4, once per forward call)
  2. Input transform:   V_t = B^T × d × B       (4×4 input patch → 4×4 transform)
  3. GEMM per (b, p):  M_t[b, p] = U_t[p] @ V_t[b, p]  (C_out × N_tiles, tl.dot)
  4. Output transform:  Y = A^T × m × A         (4×4 transform → 2×2 output patch)

  Transformation matrices (Lavin & Gray 2015, exact rational values):

    B^T (4×4) — input transform:
      [[ 1,  0, -1,  0],
       [ 0,  1,  1,  0],
       [ 0, -1,  1,  0],
       [ 0,  1,  0, -1]]

    G (4×3) — weight transform:
      [[ 1,    0,   0  ],
       [ 1/2,  1/2, 1/2],
       [ 1/2, -1/2, 1/2],
       [ 0,    0,   1  ]]

    A^T (2×4) — output transform:
      [[ 1,  1,  1,  0],
       [ 0,  1, -1, -1]]

  Layout:
    x      : (B, C_in,  H,     W)     — NCHW, K=3 only, no padding, no bias
    weight : (C_out, C_in, 3, 3)
    y      : (B, C_out, H-2, W-2)     — valid convolution

    H_tiles = cdiv(H_out, 2),  W_tiles = cdiv(W_out, 2)
    Each tile produces a 2×2 output patch. Partial boundary tiles use mask= on store.

  GEMM-friendly intermediate layout (position-outermost for contiguous tl.dot access):
    U_t : (16, C_out, C_in)      — for each p: U_t[p, :, :] is a contiguous (C_out, C_in) slice
    V_t : (B, 16, C_in, N_tiles) — for (b, p): V_t[b, p, :, :] is contiguous (C_in, N_tiles)
    M_t : (B, 16, C_out, N_tiles)

  Step 3 GEMM: M_t[b, p, :, :] = U_t[p, :, :] @ V_t[b, p, :, :]
    = (C_out, C_in) @ (C_in, N_tiles) → (C_out, N_tiles).
  16 × B independent GEMMs, each dispatched as one Triton program group.
  M_t is permuted to (B, C_out, N_tiles, 16) before the output transform kernel.

TFLOPS metric: (2 × B × C_out × H_out × W_out × C_in × 9 × 1e-12) / (ms × 1e-3)
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# ── 1. Weight transform kernel ────────────────────────────────────────────────

@triton.jit
def winograd_weight_transform_kernel(
    w_ptr,     # (C_out, C_in, 3, 3)
    u_t_ptr,   # (16, C_out, C_in)
    C_in, C_out,
    stride_wco, stride_wci, stride_wkh, stride_wkw,
    stride_ut_p, stride_ut_co, stride_ut_ci,
):
    """Apply G × w × G^T to each (c_out, c_in) 3×3 weight patch → U_t[p, c_out, c_in].

    All inner-loop helpers are inlined — Triton JIT does not support nested def.
    """
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
    # G row 0: [1, 0, 0]         → i_r = w_col
    # G row 1: [1/2, 1/2, 1/2]  → i_r = (w[0] + w[1] + w[2]) * 0.5
    # G row 2: [1/2,-1/2, 1/2]  → i_r = (w[0] - w[1] + w[2]) * 0.5
    # G row 3: [0, 0, 1]         → i_r = w[2]
    i00 = g00;                      i01 = g01;                      i02 = g02
    i10 = (g00 + g10 + g20) * 0.5; i11 = (g01 + g11 + g21) * 0.5; i12 = (g02 + g12 + g22) * 0.5
    i20 = (g00 - g10 + g20) * 0.5; i21 = (g01 - g11 + g21) * 0.5; i22 = (g02 - g12 + g22) * 0.5
    i30 = g20;                      i31 = g21;                      i32 = g22

    # U_t = intermediate × G^T  (4 rows × 4 cols), stored at U_t[p, pid_co, pid_ci]
    # G^T col ops applied to each row a,b,c of intermediate:
    #   u[r][0] = a; u[r][1] = (a+b+c)*0.5; u[r][2] = (a-b+c)*0.5; u[r][3] = c
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


# ── 2. Input transform kernel ─────────────────────────────────────────────────

@triton.jit
def winograd_input_transform_kernel(
    x_ptr,    # (B, C_in, H, W)
    v_t_ptr,  # (B, 16, C_in, N_tiles)
    C_in, H, W, H_tiles, W_tiles,
    stride_xb, stride_xci, stride_xh, stride_xw,
    stride_vt_b, stride_vt_p, stride_vt_ci, stride_vt_n,
):
    """Apply B^T × d × B to each 4×4 input patch → V_t[b, p, c_in, tile_idx].

    tl.load with mask+other handles boundary zero-padding without branching.
    """
    pid_bci = tl.program_id(0)   # flat (b, c_in)
    pid_ht  = tl.program_id(1)
    pid_wt  = tl.program_id(2)

    pid_b  = pid_bci // C_in
    pid_ci = pid_bci %  C_in

    h0 = pid_ht * 2
    w0 = pid_wt * 2

    x_base = x_ptr + pid_b * stride_xb + pid_ci * stride_xci

    # Load 4×4 patch with boundary zero-padding (all 16 elements inlined)
    d00 = tl.load(x_base + (h0+0)*stride_xh + (w0+0)*stride_xw, mask=((h0+0)<H)&((w0+0)<W), other=0.0).to(tl.float32)
    d01 = tl.load(x_base + (h0+0)*stride_xh + (w0+1)*stride_xw, mask=((h0+0)<H)&((w0+1)<W), other=0.0).to(tl.float32)
    d02 = tl.load(x_base + (h0+0)*stride_xh + (w0+2)*stride_xw, mask=((h0+0)<H)&((w0+2)<W), other=0.0).to(tl.float32)
    d03 = tl.load(x_base + (h0+0)*stride_xh + (w0+3)*stride_xw, mask=((h0+0)<H)&((w0+3)<W), other=0.0).to(tl.float32)
    d10 = tl.load(x_base + (h0+1)*stride_xh + (w0+0)*stride_xw, mask=((h0+1)<H)&((w0+0)<W), other=0.0).to(tl.float32)
    d11 = tl.load(x_base + (h0+1)*stride_xh + (w0+1)*stride_xw, mask=((h0+1)<H)&((w0+1)<W), other=0.0).to(tl.float32)
    d12 = tl.load(x_base + (h0+1)*stride_xh + (w0+2)*stride_xw, mask=((h0+1)<H)&((w0+2)<W), other=0.0).to(tl.float32)
    d13 = tl.load(x_base + (h0+1)*stride_xh + (w0+3)*stride_xw, mask=((h0+1)<H)&((w0+3)<W), other=0.0).to(tl.float32)
    d20 = tl.load(x_base + (h0+2)*stride_xh + (w0+0)*stride_xw, mask=((h0+2)<H)&((w0+0)<W), other=0.0).to(tl.float32)
    d21 = tl.load(x_base + (h0+2)*stride_xh + (w0+1)*stride_xw, mask=((h0+2)<H)&((w0+1)<W), other=0.0).to(tl.float32)
    d22 = tl.load(x_base + (h0+2)*stride_xh + (w0+2)*stride_xw, mask=((h0+2)<H)&((w0+2)<W), other=0.0).to(tl.float32)
    d23 = tl.load(x_base + (h0+2)*stride_xh + (w0+3)*stride_xw, mask=((h0+2)<H)&((w0+3)<W), other=0.0).to(tl.float32)
    d30 = tl.load(x_base + (h0+3)*stride_xh + (w0+0)*stride_xw, mask=((h0+3)<H)&((w0+0)<W), other=0.0).to(tl.float32)
    d31 = tl.load(x_base + (h0+3)*stride_xh + (w0+1)*stride_xw, mask=((h0+3)<H)&((w0+1)<W), other=0.0).to(tl.float32)
    d32 = tl.load(x_base + (h0+3)*stride_xh + (w0+2)*stride_xw, mask=((h0+3)<H)&((w0+2)<W), other=0.0).to(tl.float32)
    d33 = tl.load(x_base + (h0+3)*stride_xh + (w0+3)*stride_xw, mask=((h0+3)<H)&((w0+3)<W), other=0.0).to(tl.float32)

    # Step 1: I = B^T × d  (B^T rows: [d0-d2], [d1+d2], [-d1+d2], [d1-d3])
    i00 = d00-d20;  i01 = d01-d21;  i02 = d02-d22;  i03 = d03-d23
    i10 = d10+d20;  i11 = d11+d21;  i12 = d12+d22;  i13 = d13+d23
    i20 = -d10+d20; i21 = -d11+d21; i22 = -d12+d22; i23 = -d13+d23
    i30 = d10-d30;  i31 = d11-d31;  i32 = d12-d32;  i33 = d13-d33

    # Step 2: V = I × B  (same B^T ops applied across rows of I)
    v00 = i00-i02;  v01 = i01+i02;  v02 = -i01+i02; v03 = i01-i03
    v10 = i10-i12;  v11 = i11+i12;  v12 = -i11+i12; v13 = i11-i13
    v20 = i20-i22;  v21 = i21+i22;  v22 = -i21+i22; v23 = i21-i23
    v30 = i30-i32;  v31 = i31+i32;  v32 = -i31+i32; v33 = i31-i33

    tile_idx = pid_ht * W_tiles + pid_wt
    v_t_base = v_t_ptr + pid_b * stride_vt_b + pid_ci * stride_vt_ci + tile_idx * stride_vt_n

    tl.store(v_t_base +  0 * stride_vt_p, v00)
    tl.store(v_t_base +  1 * stride_vt_p, v01)
    tl.store(v_t_base +  2 * stride_vt_p, v02)
    tl.store(v_t_base +  3 * stride_vt_p, v03)
    tl.store(v_t_base +  4 * stride_vt_p, v10)
    tl.store(v_t_base +  5 * stride_vt_p, v11)
    tl.store(v_t_base +  6 * stride_vt_p, v12)
    tl.store(v_t_base +  7 * stride_vt_p, v13)
    tl.store(v_t_base +  8 * stride_vt_p, v20)
    tl.store(v_t_base +  9 * stride_vt_p, v21)
    tl.store(v_t_base + 10 * stride_vt_p, v22)
    tl.store(v_t_base + 11 * stride_vt_p, v23)
    tl.store(v_t_base + 12 * stride_vt_p, v30)
    tl.store(v_t_base + 13 * stride_vt_p, v31)
    tl.store(v_t_base + 14 * stride_vt_p, v32)
    tl.store(v_t_base + 15 * stride_vt_p, v33)


# ── 3. Dot kernel (GEMM) ──────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 64}, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 32}, num_warps=8),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8),
    ],
    key=["C_in", "C_out", "N_tiles"],
)
@triton.jit
def winograd_dot_kernel(
    u_t_ptr,  # (16, C_out, C_in)
    v_t_ptr,  # (B, 16, C_in, N_tiles)
    m_t_ptr,  # (B, 16, C_out, N_tiles)
    C_in, C_out, N_tiles, B,
    stride_ut_p,  stride_ut_co, stride_ut_ci,
    stride_vt_b,  stride_vt_p,  stride_vt_ci, stride_vt_n,
    stride_mt_b,  stride_mt_p,  stride_mt_co, stride_mt_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    M_t[b, p, :, :] = U_t[p, :, :] @ V_t[b, p, :, :]
      = (C_out, C_in) @ (C_in, N_tiles) → (C_out, N_tiles)

    Grid: (16 * B, cdiv(C_out, BLOCK_M), cdiv(N_tiles, BLOCK_N))
    """
    pid_pb = tl.program_id(0)
    pid_m  = tl.program_id(1)
    pid_n  = tl.program_id(2)

    p = pid_pb // B
    b = pid_pb %  B

    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # (BLOCK_M,)
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # (BLOCK_N,)
    mask_m = m_offs < C_out
    mask_n = n_offs < N_tiles

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, C_in, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)   # (BLOCK_K,)
        mask_k = k_offs < C_in

        # Load U_t[p, m_offs, k_offs] — shape (BLOCK_M, BLOCK_K)
        u_tile = tl.load(
            u_t_ptr + p * stride_ut_p
                    + m_offs[:, None] * stride_ut_co
                    + k_offs[None, :] * stride_ut_ci,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        )
        # Load V_t[b, p, k_offs, n_offs] — shape (BLOCK_K, BLOCK_N)
        v_tile = tl.load(
            v_t_ptr + b * stride_vt_b
                    + p * stride_vt_p
                    + k_offs[:, None] * stride_vt_ci
                    + n_offs[None, :] * stride_vt_n,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0,
        )
        acc = tl.dot(u_tile, v_tile, acc)

    tl.store(
        m_t_ptr + b * stride_mt_b
                + p * stride_mt_p
                + m_offs[:, None] * stride_mt_co
                + n_offs[None, :] * stride_mt_n,
        acc,
        mask=mask_m[:, None] & mask_n[None, :],
    )


# ── 4. Output transform kernel ────────────────────────────────────────────────

@triton.jit
def winograd_output_transform_kernel(
    m_ptr,  # (B, C_out, H_tiles, W_tiles, 16)
    y_ptr,  # (B, C_out, H_out, W_out)
    C_out, H_out, W_out, H_tiles, W_tiles,
    stride_mb,  stride_mco, stride_mht, stride_mwt, stride_mp,
    stride_yb,  stride_yco, stride_yh,  stride_yw,
):
    """Apply A^T × m × A to each (b, c_out, h_tile, w_tile) 4×4 M patch → 2×2 Y patch."""
    pid_bco = tl.program_id(0)
    pid_ht  = tl.program_id(1)
    pid_wt  = tl.program_id(2)

    pid_b  = pid_bco // C_out
    pid_co = pid_bco %  C_out

    m_base = m_ptr + pid_b * stride_mb + pid_co * stride_mco + pid_ht * stride_mht + pid_wt * stride_mwt

    m00 = tl.load(m_base +  0 * stride_mp)
    m01 = tl.load(m_base +  1 * stride_mp)
    m02 = tl.load(m_base +  2 * stride_mp)
    m03 = tl.load(m_base +  3 * stride_mp)
    m10 = tl.load(m_base +  4 * stride_mp)
    m11 = tl.load(m_base +  5 * stride_mp)
    m12 = tl.load(m_base +  6 * stride_mp)
    m13 = tl.load(m_base +  7 * stride_mp)
    m20 = tl.load(m_base +  8 * stride_mp)
    m21 = tl.load(m_base +  9 * stride_mp)
    m22 = tl.load(m_base + 10 * stride_mp)
    m23 = tl.load(m_base + 11 * stride_mp)
    m30 = tl.load(m_base + 12 * stride_mp)
    m31 = tl.load(m_base + 13 * stride_mp)
    m32 = tl.load(m_base + 14 * stride_mp)
    m33 = tl.load(m_base + 15 * stride_mp)

    # Step 1: I = A^T × m  (A^T rows: [m0+m1+m2], [m1-m2-m3])
    i00 = m00+m10+m20; i01 = m01+m11+m21; i02 = m02+m12+m22; i03 = m03+m13+m23
    i10 = m10-m20-m30; i11 = m11-m21-m31; i12 = m12-m22-m32; i13 = m13-m23-m33

    # Step 2: Y = I × A  (A col ops = A^T row ops on columns of I)
    y00 = i00+i01+i02
    y01 = i01-i02-i03
    y10 = i10+i11+i12
    y11 = i11-i12-i13

    y_base = y_ptr + pid_b * stride_yb + pid_co * stride_yco
    h0 = pid_ht * 2
    w0 = pid_wt * 2

    # Store 2×2 output patch with boundary mask for partial last tiles
    tl.store(y_base + (h0+0)*stride_yh + (w0+0)*stride_yw, y00, mask=((h0+0)<H_out)&((w0+0)<W_out))
    tl.store(y_base + (h0+0)*stride_yh + (w0+1)*stride_yw, y01, mask=((h0+0)<H_out)&((w0+1)<W_out))
    tl.store(y_base + (h0+1)*stride_yh + (w0+0)*stride_yw, y10, mask=((h0+1)<H_out)&((w0+0)<W_out))
    tl.store(y_base + (h0+1)*stride_yh + (w0+1)*stride_yw, y11, mask=((h0+1)<H_out)&((w0+1)<W_out))


# ── 5. Python wrapper ─────────────────────────────────────────────────────────

def winograd_conv2d(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Winograd F(2,3) 2D convolution — K=3 only, no padding, no bias.

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
    assert KH == 3 and KW == 3, f"winograd_conv2d only supports K=3; got {KH}×{KW}"
    assert H >= 4 and W >= 4,   f"Input spatial dims must be ≥ 4; got {H}×{W}"

    H_out   = H - 2
    W_out   = W - 2
    H_tiles = triton.cdiv(H_out, 2)
    W_tiles = triton.cdiv(W_out, 2)
    N_tiles = H_tiles * W_tiles

    U_t = torch.empty((16, C_out, C_in),             device=x.device, dtype=torch.float32)
    V_t = torch.empty((B, 16, C_in, N_tiles),         device=x.device, dtype=torch.float32)
    M_t = torch.empty((B, 16, C_out, N_tiles),        device=x.device, dtype=torch.float32)
    Y   = torch.empty((B, C_out, H_out, W_out),       device=x.device, dtype=torch.float32)

    # Kernel 1: weight transform — (C_out, C_in) grid → U_t: (16, C_out, C_in)
    winograd_weight_transform_kernel[(C_out, C_in)](
        w, U_t,
        C_in, C_out,
        w.stride(0), w.stride(1), w.stride(2), w.stride(3),
        U_t.stride(0), U_t.stride(1), U_t.stride(2),   # stride_ut_p, stride_ut_co, stride_ut_ci
    )

    # Kernel 2: input transform — (B*C_in, H_tiles, W_tiles) grid → V_t: (B, 16, C_in, N_tiles)
    winograd_input_transform_kernel[(B * C_in, H_tiles, W_tiles)](
        x, V_t,
        C_in, H, W, H_tiles, W_tiles,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        V_t.stride(0), V_t.stride(1), V_t.stride(2), V_t.stride(3),   # vt_b, vt_p, vt_ci, vt_n
    )

    # Kernel 3: GEMM — grid (16*B, cdiv(C_out, BLOCK_M), cdiv(N_tiles, BLOCK_N))
    # M_t[b, p, :, :] = U_t[p, :, :] @ V_t[b, p, :, :]  (C_out, C_in) @ (C_in, N_tiles)
    grid_dot = lambda meta: (
        16 * B,
        triton.cdiv(C_out, meta["BLOCK_M"]),
        triton.cdiv(N_tiles, meta["BLOCK_N"]),
    )
    winograd_dot_kernel[grid_dot](
        U_t, V_t, M_t,
        C_in, C_out, N_tiles, B,
        U_t.stride(0), U_t.stride(1), U_t.stride(2),
        V_t.stride(0), V_t.stride(1), V_t.stride(2), V_t.stride(3),
        M_t.stride(0), M_t.stride(1), M_t.stride(2), M_t.stride(3),
    )

    # Permute M_t → (B, C_out, N_tiles, 16), then reshape for output transform kernel
    M    = M_t.permute(0, 2, 3, 1).contiguous()    # (B, C_out, N_tiles, 16)
    M_5d = M.view(B, C_out, H_tiles, W_tiles, 16)

    # Kernel 4: output transform — (B*C_out, H_tiles, W_tiles) grid (unchanged)
    winograd_output_transform_kernel[(B * C_out, H_tiles, W_tiles)](
        M_5d, Y,
        C_out, H_out, W_out, H_tiles, W_tiles,
        M_5d.stride(0), M_5d.stride(1), M_5d.stride(2), M_5d.stride(3), M_5d.stride(4),
        Y.stride(0),    Y.stride(1),    Y.stride(2),    Y.stride(3),
    )

    return Y


# ── 6. Correctness tests ──────────────────────────────────────────────────────

def test_winograd_conv2d():
    print("Testing winograd_conv2d...")
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
        got = winograd_conv2d(x, wt)
        torch.testing.assert_close(got, ref, atol=1e-3, rtol=1e-3)
        print(f"  B={B} C_in={C_in} C_out={C_out} {H}×{W} → {H-2}×{W-2}  "
              f"max_err={(got - ref).abs().max():.2e}  PASS")
    print("All tests passed.")


# ── 7. Benchmark ──────────────────────────────────────────────────────────────

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["H"],
        x_vals=[2**i for i in range(5, 10)],
        x_log=True,
        line_arg="provider",
        line_vals=["direct", "winograd", "torch"],
        line_names=["Triton direct conv2d", "Triton Winograd F(2,3)", "PyTorch (F.conv2d)"],
        styles=[("blue", "-"), ("orange", "-"), ("green", ":")],
        ylabel="TFLOPS",
        plot_name="winograd_conv2d",
        args={"B": 1, "C_in": 64, "C_out": 64, "K": 3},
    )
)
def benchmark_winograd_conv2d(B, C_in, C_out, K, H, provider):
    from kernels.convolution.conv2d import conv2d

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
    else:
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: F.conv2d(x, wt), warmup=25, rep=100, quantiles=quantiles
        )

    tflops = 2 * B * C_out * H_out * W_out * C_in * K * K * 1e-12
    return tflops / (ms * 1e-3), tflops / (max_ms * 1e-3), tflops / (min_ms * 1e-3)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_winograd_conv2d()
    import os
    os.makedirs("benchmarks/results/convolution", exist_ok=True)
    benchmark_winograd_conv2d.run(
        print_data=True,
        show_plots=True,
        save_path="benchmarks/results/convolution",
    )
