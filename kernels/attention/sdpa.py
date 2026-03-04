"""
Kernel: Scaled Dot-Product Attention with Causal Masking (SDPA)
Category: attention
Complexity: O(N^2 * d) compute, O(N^2) memory (lower triangle only)
Memory bound: No — compute bound at large N and d
PyTorch equivalent: torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
References: https://arxiv.org/abs/1706.03762 (Attention Is All You Need)
"""

import torch
import triton
import triton.language as tl
import math


# ── Kernel ────────────────────────────────────────────────────────────────────

@triton.jit
def sdpa_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    N, d,
    scale,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Causal scaled dot-product attention. Query at position i attends only to
    keys 0..i (lower triangle of the score matrix).

    Optimisation over naive_attention:
    - K/V tiles entirely in the future (kv_start >= q_start + BLOCK_N) are
      skipped — their softmax contribution is zero.
    - K/V tiles that cross the causal diagonal get a per-element mask applied.
    - K/V tiles entirely in the past need no masking at all.

    On average across all query tiles this skips ~half the K/V iterations,
    roughly halving wall-clock time vs. full attention at the same N.
    """
    batch_id = tl.program_id(0)
    head_id  = tl.program_id(1)
    tile_id  = tl.program_id(2)

    q_start = tile_id * BLOCK_N
    q_offs  = q_start + tl.arange(0, BLOCK_N)   # [BLOCK_N]
    d_offs  = tl.arange(0, BLOCK_D)              # [BLOCK_D]
    q_mask  = q_offs < N

    q_base = q_ptr + batch_id * stride_qb + head_id * stride_qh
    k_base = k_ptr + batch_id * stride_kb + head_id * stride_kh
    v_base = v_ptr + batch_id * stride_vb + head_id * stride_vh
    o_base = out_ptr + batch_id * stride_ob + head_id * stride_oh

    # Load Q tile: [BLOCK_N, BLOCK_D]
    q = tl.load(
        q_base + q_offs[:, None] * stride_qn + d_offs[None, :] * stride_qd,
        mask=q_mask[:, None],
        other=0.0,
    ).to(tl.float32)

    # Online softmax accumulators
    m   = tl.full([BLOCK_N], float("-inf"), dtype=tl.float32)
    s   = tl.zeros([BLOCK_N], dtype=tl.float32)
    acc = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)

    # Only iterate over K/V tiles that are not entirely in the future.
    # q_offs range is [q_start, q_start + BLOCK_N).
    # A K/V tile starting at kv_start is entirely future when:
    #   kv_start > max(q_offs) = q_start + BLOCK_N - 1
    # i.e. kv_start >= q_start + BLOCK_N.
    # We iterate kv_start in [0, q_start + BLOCK_N) — the causal bound.
    causal_limit = q_start + BLOCK_N  # exclusive upper bound for kv_start

    for kv_start in range(0, causal_limit, BLOCK_N):
        kv_offs = kv_start + tl.arange(0, BLOCK_N)
        kv_mask = kv_offs < N

        # Load K tile: [BLOCK_N, BLOCK_D]
        k = tl.load(
            k_base + kv_offs[:, None] * stride_kn + d_offs[None, :] * stride_kd,
            mask=kv_mask[:, None],
            other=0.0,
        ).to(tl.float32)

        # Load V tile: [BLOCK_N, BLOCK_D]
        v = tl.load(
            v_base + kv_offs[:, None] * stride_vn + d_offs[None, :] * stride_vd,
            mask=kv_mask[:, None],
            other=0.0,
        ).to(tl.float32)

        # QK^T: [BLOCK_N, BLOCK_N]
        score_tile = tl.dot(q, tl.trans(k)) * scale

        # ── Causal mask ──────────────────────────────────────────────────
        # score[i, j] is valid iff q_offs[i] >= kv_offs[j],
        # i.e. the query position is >= the key position.
        causal_mask = q_offs[:, None] >= kv_offs[None, :]   # [BLOCK_N, BLOCK_N]
        score_tile = tl.where(causal_mask, score_tile, float("-inf"))

        # Padding mask for out-of-bounds keys
        score_tile = tl.where(kv_mask[None, :], score_tile, float("-inf"))

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

    # Normalise and store
    out = acc / s[:, None]
    tl.store(
        o_base + q_offs[:, None] * stride_on + d_offs[None, :] * stride_od,
        out,
        mask=q_mask[:, None],
    )


# ── Wrapper ───────────────────────────────────────────────────────────────────

def sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """
    Compute causal scaled dot-product attention.

    Args:
        q, k, v: Tensors of shape (B, H, N, d), fp32 or fp16.
                 d must be a power of 2 and <= 128.

    Returns:
        out: Tensor of shape (B, H, N, d), fp32.
    """
    assert q.is_cuda and k.is_cuda and v.is_cuda, "Inputs must be CUDA tensors"
    assert q.shape == k.shape == v.shape, "q, k, v must have the same shape"

    q = q.contiguous().to(torch.float32)
    k = k.contiguous().to(torch.float32)
    v = v.contiguous().to(torch.float32)

    B, H, N, d = q.shape
    assert d & (d - 1) == 0, f"d must be a power of 2, got {d}"
    assert d <= 128, f"d={d} exceeds BLOCK_D limit (128)"

    BLOCK_N = min(32, N)
    BLOCK_D = d
    scale   = 1.0 / math.sqrt(d)
    out     = torch.empty_like(q)

    grid = (B, H, triton.cdiv(N, BLOCK_N))

    sdpa_kernel[grid](
        q, k, v, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        N, d,
        scale,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
    )

    return out


# ── Test ──────────────────────────────────────────────────────────────────────

def test_sdpa():
    print("Testing sdpa (causal)...")

    torch.manual_seed(0)
    for N in [64, 128, 256, 512]:
        for d in [32, 64]:
            B, H = 2, 4
            q = torch.randn(B, H, N, d, device="cuda", dtype=torch.float32)
            k = torch.randn(B, H, N, d, device="cuda", dtype=torch.float32)
            v = torch.randn(B, H, N, d, device="cuda", dtype=torch.float32)

            ref = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=True
            )
            got = sdpa(q, k, v)

            torch.testing.assert_close(got, ref, atol=1e-3, rtol=1e-3)
            print(f"  N={N:4d}  d={d:3d}  max_err={(got - ref).abs().max():.2e}  PASS")

    # First token: query at position 0 only attends to key 0.
    # Output should equal V[0] regardless of Q and K values.
    N, d = 128, 64
    q = torch.randn(1, 1, N, d, device="cuda")
    k = torch.randn(1, 1, N, d, device="cuda")
    v = torch.randn(1, 1, N, d, device="cuda")
    got = sdpa(q, k, v)
    # Row 0 of output must equal V[0] (only one key attended)
    torch.testing.assert_close(got[0, 0, 0], v[0, 0, 0], atol=1e-3, rtol=1e-3)
    print("  First-token test (output[0] == V[0])  PASS")

    print("All tests passed.")


# ── Benchmark ─────────────────────────────────────────────────────────────────

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[64, 128, 256, 512, 1024],
        x_log=True,
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton SDPA (causal)", "torch SDPA (causal)"],
        styles=[("blue", "-"), ("green", "--")],
        ylabel="TFLOPS",
        plot_name="sdpa_benchmark",
        args={"B": 4, "H": 8, "d": 64},
    )
)
def benchmark_sdpa(N, B, H, d, provider):
    q = torch.randn(B, H, N, d, device="cuda", dtype=torch.float32)
    k = torch.randn(B, H, N, d, device="cuda", dtype=torch.float32)
    v = torch.randn(B, H, N, d, device="cuda", dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]

    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: sdpa(q, k, v), warmup=25, rep=100, quantiles=quantiles
        )
    else:
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=True
            ),
            warmup=25, rep=100, quantiles=quantiles,
        )

    # Use causal-adjusted FLOP count: lower-triangle only ≈ N²/2 per matmul,
    # two matmuls (QK^T and scores·V) → 2 * B * H * N² * d total FLOPs.
    tflops = lambda ms: (2 * B * H * N * N * d * 1e-12) / (ms * 1e-3)
    return tflops(ms), tflops(max_ms), tflops(min_ms)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_sdpa()
    import os
    os.makedirs("benchmarks/results/attention", exist_ok=True)
    benchmark_sdpa.run(
        print_data=True,
        show_plots=True,
        save_path="benchmarks/results/attention",
    )
