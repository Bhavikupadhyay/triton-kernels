"""
Kernel:   winograd_conv2d
Category: convolution
Complexity: O(B × C_out × H_tiles × W_tiles × C_in × 16) pointwise products
Memory bound: No — Winograd reduces multiply count ~2.25× vs direct conv for K=3
PyTorch equivalent: torch.nn.functional.conv2d(x, weight, padding=0) with K=3
References:
  - Lavin & Gray, "Fast Algorithms for Convolutional Neural Networks", CVPR 2016
    https://arxiv.org/abs/1509.09308

Algorithm — Winograd F(2,3):

  Direct K=3 conv produces a 2×2 output patch from a 4×4 input patch using 9×4=36
  multiplications. Winograd F(2,3) reduces this to 16 pointwise multiplications:

  1. Weight transform:  U = G × g × G^T       (3×3 → 4×4, once per forward call)
  2. Input transform:   V = B^T × d × B       (4×4 input patch → 4×4 transform)
  3. Pointwise product: M[pos] = sum_{c_in} V[c_in, pos] * U[c_out, c_in, pos]
  4. Output transform:  Y = A^T × m × A       (4×4 transform → 2×2 output patch)

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

  V indexing note:
    V has shape (B, C_in, H_tiles, W_tiles, 16), row-major. The flat tile index
    tile_idx = h_tile * W_tiles + w_tile satisfies:
      V[b, c, h, w, :] base = b*stride_vb + c*stride_vci + tile_idx * stride_vwt
    since stride_vwt = V.stride(3) = 16, and tile_idx*16 = h*W_tiles*16 + w*16
    = h*stride_vht + w*stride_vwt. The wrapper passes V.stride(3) as stride_vwt. ✓

TFLOPS metric: (2 × B × C_out × H_out × W_out × C_in × 9 × 1e-12) / (ms × 1e-3)
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# ── 1. Weight transform kernel ────────────────────────────────────────────────

@triton.jit
def winograd_weight_transform_kernel(
    w_ptr,   # (C_out, C_in, 3, 3)
    u_ptr,   # (C_out, C_in, 16)
    C_in, C_out,
    stride_wco, stride_wci, stride_wkh, stride_wkw,
    stride_uco, stride_uci, stride_up,
):
    """Apply G × w × G^T to each (c_out, c_in) 3×3 weight patch → 4×4 U patch.

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

    # U = intermediate × G^T  (4 rows × 4 cols)
    # G^T col ops applied to each row a,b,c of intermediate:
    #   u[r][0] = a; u[r][1] = (a+b+c)*0.5; u[r][2] = (a-b+c)*0.5; u[r][3] = c
    u_base = u_ptr + pid_co * stride_uco + pid_ci * stride_uci

    tl.store(u_base +  0 * stride_up, i00)
    tl.store(u_base +  1 * stride_up, (i00 + i01 + i02) * 0.5)
    tl.store(u_base +  2 * stride_up, (i00 - i01 + i02) * 0.5)
    tl.store(u_base +  3 * stride_up, i02)

    tl.store(u_base +  4 * stride_up, i10)
    tl.store(u_base +  5 * stride_up, (i10 + i11 + i12) * 0.5)
    tl.store(u_base +  6 * stride_up, (i10 - i11 + i12) * 0.5)
    tl.store(u_base +  7 * stride_up, i12)

    tl.store(u_base +  8 * stride_up, i20)
    tl.store(u_base +  9 * stride_up, (i20 + i21 + i22) * 0.5)
    tl.store(u_base + 10 * stride_up, (i20 - i21 + i22) * 0.5)
    tl.store(u_base + 11 * stride_up, i22)

    tl.store(u_base + 12 * stride_up, i30)
    tl.store(u_base + 13 * stride_up, (i30 + i31 + i32) * 0.5)
    tl.store(u_base + 14 * stride_up, (i30 - i31 + i32) * 0.5)
    tl.store(u_base + 15 * stride_up, i32)


# ── 2. Input transform kernel ─────────────────────────────────────────────────

@triton.jit
def winograd_input_transform_kernel(
    x_ptr,  # (B, C_in, H, W)
    v_ptr,  # (B, C_in, H_tiles, W_tiles, 16)
    C_in, H, W, H_tiles, W_tiles,
    stride_xb, stride_xci, stride_xh, stride_xw,
    stride_vb, stride_vci, stride_vht, stride_vwt, stride_vp,
):
    """Apply B^T × d × B to each 4×4 input patch → V[b,c,h_tile,w_tile,16].

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

    v_base = v_ptr + pid_b * stride_vb + pid_ci * stride_vci + pid_ht * stride_vht + pid_wt * stride_vwt

    tl.store(v_base +  0 * stride_vp, v00)
    tl.store(v_base +  1 * stride_vp, v01)
    tl.store(v_base +  2 * stride_vp, v02)
    tl.store(v_base +  3 * stride_vp, v03)
    tl.store(v_base +  4 * stride_vp, v10)
    tl.store(v_base +  5 * stride_vp, v11)
    tl.store(v_base +  6 * stride_vp, v12)
    tl.store(v_base +  7 * stride_vp, v13)
    tl.store(v_base +  8 * stride_vp, v20)
    tl.store(v_base +  9 * stride_vp, v21)
    tl.store(v_base + 10 * stride_vp, v22)
    tl.store(v_base + 11 * stride_vp, v23)
    tl.store(v_base + 12 * stride_vp, v30)
    tl.store(v_base + 13 * stride_vp, v31)
    tl.store(v_base + 14 * stride_vp, v32)
    tl.store(v_base + 15 * stride_vp, v33)


# ── 3. Dot kernel ─────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_C": 32},  num_warps=4),
        triton.Config({"BLOCK_C": 64},  num_warps=4),
        triton.Config({"BLOCK_C": 64},  num_warps=8),
        triton.Config({"BLOCK_C": 128}, num_warps=8),
    ],
    key=["C_in", "C_out", "N_tiles"],
)
@triton.jit
def winograd_dot_kernel(
    v_ptr,  # (B, C_in, H_tiles, W_tiles, 16)
    u_ptr,  # (C_out, C_in, 16)
    m_ptr,  # (B, C_out, N_tiles, 16)  — N_tiles = H_tiles * W_tiles (flat spatial)
    C_in, C_out, N_tiles,
    stride_vb,  stride_vci, stride_vwt, stride_vp,
    stride_uco, stride_uci, stride_up,
    stride_mb,  stride_mco, stride_mt,  stride_mp,
    BLOCK_C: tl.constexpr,
):
    """
    M[b, c_out, tile, p] = sum_{c_in} V[b, c_in, tile, p] * U[c_out, c_in, p]

    V layout: (B, C_in, H_tiles, W_tiles, 16). stride_vwt = V.stride(3) = 16.
    tile_idx * stride_vwt correctly addresses V[b, c, tile_idx, 0] because
    tile_idx = h*W_tiles + w, so tile_idx*16 = h*W_tiles*16 + w*16. ✓

    Grid: (B * N_tiles, cdiv(C_out, BLOCK_C))
    """
    pid_flat = tl.program_id(0)
    pid_co   = tl.program_id(1)

    pid_b    = pid_flat // N_tiles
    tile_idx = pid_flat %  N_tiles

    co_offs = pid_co * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_co = co_offs < C_out
    p_offs  = tl.arange(0, 16)

    acc = tl.zeros((BLOCK_C, 16), dtype=tl.float32)

    for c_in in range(C_in):
        v_base = v_ptr + pid_b * stride_vb + c_in * stride_vci + tile_idx * stride_vwt
        v_vals = tl.load(v_base + p_offs * stride_vp)   # (16,)

        u_base = u_ptr + c_in * stride_uci
        u_vals = tl.load(
            u_base + co_offs[:, None] * stride_uco + p_offs[None, :] * stride_up,
            mask=mask_co[:, None],
            other=0.0,
        )   # (BLOCK_C, 16)

        acc += u_vals * v_vals[None, :]

    m_base = m_ptr + pid_b * stride_mb + tile_idx * stride_mt
    tl.store(
        m_base + co_offs[:, None] * stride_mco + p_offs[None, :] * stride_mp,
        acc,
        mask=mask_co[:, None],
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

    U = torch.empty((C_out, C_in, 16),              device=x.device, dtype=torch.float32)
    V = torch.empty((B, C_in, H_tiles, W_tiles, 16), device=x.device, dtype=torch.float32)
    M = torch.empty((B, C_out, N_tiles, 16),         device=x.device, dtype=torch.float32)
    Y = torch.empty((B, C_out, H_out, W_out),        device=x.device, dtype=torch.float32)

    # Kernel 1: weight transform — (C_out, C_in) grid
    winograd_weight_transform_kernel[(C_out, C_in)](
        w, U,
        C_in, C_out,
        w.stride(0), w.stride(1), w.stride(2), w.stride(3),
        U.stride(0), U.stride(1), U.stride(2),
    )

    # Kernel 2: input transform — (B*C_in, H_tiles, W_tiles) grid
    winograd_input_transform_kernel[(B * C_in, H_tiles, W_tiles)](
        x, V,
        C_in, H, W, H_tiles, W_tiles,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3), V.stride(4),
    )

    # Kernel 3: dot product — (B*N_tiles, cdiv(C_out, BLOCK_C)) grid
    # Pass V.stride(3) as stride_vwt: tile_idx * stride_vwt = tile_idx * 16
    # correctly addresses V[b, c, tile_idx, 0] for the flat tile index. ✓
    grid_dot = lambda meta: (B * N_tiles, triton.cdiv(C_out, meta["BLOCK_C"]))
    winograd_dot_kernel[grid_dot](
        V, U, M,
        C_in, C_out, N_tiles,
        V.stride(0), V.stride(1), V.stride(3), V.stride(4),   # vb, vci, vwt(=16), vp
        U.stride(0), U.stride(1), U.stride(2),
        M.stride(0), M.stride(1), M.stride(2), M.stride(3),   # mb, mco, mt(=16), mp
    )

    # Kernel 4: output transform — (B*C_out, H_tiles, W_tiles) grid
    # Reshape M to 5D so it has clean per-dim strides for the output kernel.
    M_5d = M.view(B, C_out, H_tiles, W_tiles, 16)
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
        line_vals=["winograd", "direct", "torch"],
        line_names=["Triton Winograd F(2,3)", "Triton direct conv2d", "PyTorch (F.conv2d)"],
        styles=[("blue", "-"), ("red", "--"), ("green", ":")],
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

    if provider == "winograd":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: winograd_conv2d(x, wt), warmup=25, rep=100, quantiles=quantiles
        )
    elif provider == "direct":
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
    test_winograd_conv2d()
    import os
    os.makedirs("benchmarks/results/convolution", exist_ok=True)
    benchmark_winograd_conv2d.run(
        print_data=True,
        show_plots=True,
        save_path="benchmarks/results/convolution",
    )
