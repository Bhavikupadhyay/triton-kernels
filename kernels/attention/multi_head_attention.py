"""
Kernel: Multi-Head Attention (MHA)
Category: attention
Complexity: O(N^2 * d) per head, O(H * N^2 * d) total
Memory bound: No — compute bound at large N
PyTorch equivalent: F.scaled_dot_product_attention on head-split tensors (is_causal=True)
References: https://arxiv.org/abs/1706.03762 (Attention Is All You Need)
"""

import torch
import triton
import triton.language as tl
import math
import torch.nn.functional as F


# ── Kernel ────────────────────────────────────────────────────────────────────

@triton.jit
def mha_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr,
    # Inputs have shape (B, N, H*d) — packed-head layout.
    # stride_n = H*d (each sequence step skips over all heads).
    # Head h is accessed via offset head_id * BLOCK_D into the last dim.
    stride_qb, stride_qn,
    stride_kb, stride_kn,
    stride_vb, stride_vn,
    stride_ob, stride_on,
    N,
    scale,
    BLOCK_N: tl.constexpr,   # query tile size along N
    BLOCK_D: tl.constexpr,   # head dimension d (= H*d / H)
):
    """
    Causal multi-head attention over packed (B, N, H*d) tensors.

    Each program handles one (batch, head, query-tile) triple.
    - tl.program_id(1) selects the head; its contribution is an offset of
      head_id * BLOCK_D into the last dimension of Q/K/V.
    - stride_qn = H * BLOCK_D, so consecutive sequence positions in the
      same head are non-contiguous — the kernel handles this via explicit
      stride arithmetic rather than requiring a reshape + transpose.
    - Causal masking and online softmax follow the same pattern as sdpa.
    """
    batch_id = tl.program_id(0)
    head_id  = tl.program_id(1)
    tile_id  = tl.program_id(2)

    q_start  = tile_id * BLOCK_N
    q_offs   = q_start + tl.arange(0, BLOCK_N)   # [BLOCK_N]
    d_offs   = tl.arange(0, BLOCK_D)              # [BLOCK_D]
    q_mask   = q_offs < N

    # Offset into the packed head dimension for this head
    head_off = head_id * BLOCK_D

    q_base = q_ptr   + batch_id * stride_qb + head_off
    k_base = k_ptr   + batch_id * stride_kb + head_off
    v_base = v_ptr   + batch_id * stride_vb + head_off
    o_base = out_ptr + batch_id * stride_ob + head_off

    # Load Q tile: [BLOCK_N, BLOCK_D]
    # ptr[n, k] = base + n * stride_n + k   (k contiguous within one head)
    q = tl.load(
        q_base + q_offs[:, None] * stride_qn + d_offs[None, :],
        mask=q_mask[:, None],
        other=0.0,
    ).to(tl.float32)

    # Online softmax state
    m   = tl.full([BLOCK_N], float("-inf"), dtype=tl.float32)
    s   = tl.zeros([BLOCK_N], dtype=tl.float32)
    acc = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)

    causal_limit = q_start + BLOCK_N   # skip all-future K/V tiles

    for kv_start in range(0, causal_limit, BLOCK_N):
        kv_offs = kv_start + tl.arange(0, BLOCK_N)
        kv_mask = kv_offs < N

        k = tl.load(
            k_base + kv_offs[:, None] * stride_kn + d_offs[None, :],
            mask=kv_mask[:, None],
            other=0.0,
        ).to(tl.float32)

        v = tl.load(
            v_base + kv_offs[:, None] * stride_vn + d_offs[None, :],
            mask=kv_mask[:, None],
            other=0.0,
        ).to(tl.float32)

        # QK^T: [BLOCK_N, BLOCK_N]
        score_tile = tl.dot(q, tl.trans(k)) * scale

        # Causal mask: query at position i only attends to key j <= i
        causal_mask = q_offs[:, None] >= kv_offs[None, :]
        score_tile  = tl.where(causal_mask, score_tile, float("-inf"))
        score_tile  = tl.where(kv_mask[None, :], score_tile, float("-inf"))

        # Online softmax update
        tile_max = tl.max(score_tile, axis=1)
        new_m    = tl.maximum(m, tile_max)
        alpha    = tl.exp(m - new_m)
        s        = s * alpha
        acc      = acc * alpha[:, None]

        exp_scores = tl.exp(score_tile - new_m[:, None])
        acc += tl.dot(exp_scores, v)
        s   += tl.sum(exp_scores, axis=1)
        m    = new_m

    out = acc / s[:, None]

    tl.store(
        o_base + q_offs[:, None] * stride_on + d_offs[None, :],
        out,
        mask=q_mask[:, None],
    )


# ── Wrapper ───────────────────────────────────────────────────────────────────

def multi_head_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    H: int,
) -> torch.Tensor:
    """
    Causal multi-head attention over packed tensors.

    Args:
        q, k, v: Tensors of shape (B, N, H*d), fp32 or fp16.
                 d = (H*d) // H must be a power of 2 and <= 128.
        H: number of attention heads.

    Returns:
        out: Tensor of shape (B, N, H*d), fp32.
    """
    assert q.is_cuda and k.is_cuda and v.is_cuda, "Inputs must be CUDA tensors"
    assert q.shape == k.shape == v.shape, "q, k, v must have the same shape"

    q = q.contiguous().to(torch.float32)
    k = k.contiguous().to(torch.float32)
    v = v.contiguous().to(torch.float32)

    B, N, Hd = q.shape
    assert Hd % H == 0, f"H*d={Hd} must be divisible by H={H}"
    d = Hd // H
    assert d & (d - 1) == 0, f"head dim d={d} must be a power of 2"
    assert d <= 128, f"head dim d={d} exceeds BLOCK_D limit (128)"

    BLOCK_N = min(32, N)
    scale   = 1.0 / math.sqrt(d)
    out     = torch.empty_like(q)

    grid = (B, H, triton.cdiv(N, BLOCK_N))

    mha_kernel[grid](
        q, k, v, out,
        q.stride(0), q.stride(1),
        k.stride(0), k.stride(1),
        v.stride(0), v.stride(1),
        out.stride(0), out.stride(1),
        N,
        scale,
        BLOCK_N=BLOCK_N,
        BLOCK_D=d,
    )

    return out


# ── Reference helper ──────────────────────────────────────────────────────────

def _ref_mha(q, k, v, H):
    """PyTorch reference: head-split → causal SDPA → merge heads."""
    B, N, Hd = q.shape
    d = Hd // H
    q_h = q.view(B, N, H, d).permute(0, 2, 1, 3)   # (B, H, N, d)
    k_h = k.view(B, N, H, d).permute(0, 2, 1, 3)
    v_h = v.view(B, N, H, d).permute(0, 2, 1, 3)
    out_h = F.scaled_dot_product_attention(q_h, k_h, v_h, is_causal=True)
    return out_h.permute(0, 2, 1, 3).contiguous().view(B, N, Hd)


# ── Test ──────────────────────────────────────────────────────────────────────

def test_multi_head_attention():
    print("Testing multi_head_attention...")

    torch.manual_seed(0)
    for N in [64, 128, 256, 512]:
        for H, d in [(4, 32), (8, 64)]:
            B = 2
            Hd = H * d
            q = torch.randn(B, N, Hd, device="cuda", dtype=torch.float32)
            k = torch.randn(B, N, Hd, device="cuda", dtype=torch.float32)
            v = torch.randn(B, N, Hd, device="cuda", dtype=torch.float32)

            ref = _ref_mha(q, k, v, H)
            got = multi_head_attention(q, k, v, H)

            torch.testing.assert_close(got, ref, atol=1e-3, rtol=1e-3)
            print(f"  N={N:4d}  H={H}  d={d}  max_err={(got - ref).abs().max():.2e}  PASS")

    # Packed layout: verify head 0 and head 1 slices independently
    B, N, H, d = 1, 128, 4, 32
    q = torch.randn(B, N, H * d, device="cuda")
    k = torch.randn(B, N, H * d, device="cuda")
    v = torch.randn(B, N, H * d, device="cuda")
    got = multi_head_attention(q, k, v, H)
    ref = _ref_mha(q, k, v, H)
    for h in range(H):
        got_h = got[:, :, h*d:(h+1)*d]
        ref_h = ref[:, :, h*d:(h+1)*d]
        torch.testing.assert_close(got_h, ref_h, atol=1e-3, rtol=1e-3)
    print("  Per-head slice correctness  PASS")

    print("All tests passed.")


# ── Benchmark ─────────────────────────────────────────────────────────────────

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[64, 128, 256, 512, 1024],
        x_log=True,
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton MHA (causal)", "torch MHA (causal)"],
        styles=[("blue", "-"), ("green", "--")],
        ylabel="TFLOPS",
        plot_name="mha_benchmark",
        args={"B": 4, "H": 8, "d": 64},
    )
)
def benchmark_multi_head_attention(N, B, H, d, provider):
    Hd = H * d
    q = torch.randn(B, N, Hd, device="cuda", dtype=torch.float32)
    k = torch.randn(B, N, Hd, device="cuda", dtype=torch.float32)
    v = torch.randn(B, N, Hd, device="cuda", dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]

    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: multi_head_attention(q, k, v, H),
            warmup=25, rep=100, quantiles=quantiles,
        )
    else:
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: _ref_mha(q, k, v, H),
            warmup=25, rep=100, quantiles=quantiles,
        )

    # Causal-adjusted: 2 * B * H * N^2 * d FLOPs (lower triangle, two matmuls)
    tflops = lambda ms: (2 * B * H * N * N * d * 1e-12) / (ms * 1e-3)
    return tflops(ms), tflops(max_ms), tflops(min_ms)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_multi_head_attention()
    import os
    os.makedirs("benchmarks/results/attention", exist_ok=True)
    benchmark_multi_head_attention.run(
        print_data=True,
        show_plots=True,
        save_path="benchmarks/results/attention",
    )
