"""
Kernel: Flash Attention v2
Category: attention
Complexity: O(N^2 * d) compute, O(N * d) memory
Memory bound: No — compute bound; HBM reads scale as O(N * d)
PyTorch equivalent: F.scaled_dot_product_attention(q, k, v, is_causal=True)
References: https://arxiv.org/abs/2307.08691 (Dao, FlashAttention-2, 2023)
"""

import torch
import triton
import triton.language as tl
import math
import torch.nn.functional as F


# ── Kernel ────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=4),
    ],
    key=["N", "d"],
)
@triton.jit
def flash_attention_v2_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    N, d,
    scale,
    BLOCK_M: tl.constexpr,   # Q tile size  — autotuned
    BLOCK_N: tl.constexpr,   # K/V tile size — autotuned
    BLOCK_D: tl.constexpr,   # head dim — fixed at compile time
):
    """
    Flash Attention v2 forward.

    Differences from v1:
    1. Autotuning: BLOCK_M and BLOCK_N are tuned independently. Larger tiles
       (up to 64×64) improve compute intensity and warp utilisation vs the
       fixed BLOCK_N=32 in v1.

    2. Split K/V loop: the single inner loop over K/V tiles is split into two:
       - Loop A: tiles entirely in the past (kv_start + BLOCK_N <= q_start).
         No causal mask check — every element is valid.
       - Loop B: diagonal tiles (q_start <= kv_start < q_start + BLOCK_M).
         Causal mask applied per element.
       At large N, Loop A dominates (e.g. at q_start=2048, BLOCK_N=64: 32
       past tiles vs 1–2 diagonal tiles). Eliminating the mask branch from the
       majority of iterations reduces non-matmul overhead.

    3. Grid uses a lambda to pick up BLOCK_M from the autotuner config.
    """
    batch_id = tl.program_id(0)
    head_id  = tl.program_id(1)
    tile_id  = tl.program_id(2)

    q_start = tile_id * BLOCK_M
    q_offs  = q_start + tl.arange(0, BLOCK_M)
    d_offs  = tl.arange(0, BLOCK_D)
    q_mask  = q_offs < N

    q_base = q_ptr + batch_id * stride_qb + head_id * stride_qh
    k_base = k_ptr + batch_id * stride_kb + head_id * stride_kh
    v_base = v_ptr + batch_id * stride_vb + head_id * stride_vh
    o_base = out_ptr + batch_id * stride_ob + head_id * stride_oh

    # Load Q tile once — held in registers for both K/V loops
    q = tl.load(
        q_base + q_offs[:, None] * stride_qn + d_offs[None, :] * stride_qd,
        mask=q_mask[:, None],
        other=0.0,
    ).to(tl.float32)

    m   = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    s   = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # ── Loop A: fully-past K/V tiles — no causal mask needed ─────────────────
    # kv_start + BLOCK_N <= q_start  ↔  all keys come before all queries.
    for kv_start in range(0, q_start, BLOCK_N):
        kv_offs = kv_start + tl.arange(0, BLOCK_N)
        kv_mask = kv_offs < N

        k = tl.load(
            k_base + kv_offs[:, None] * stride_kn + d_offs[None, :] * stride_kd,
            mask=kv_mask[:, None],
            other=0.0,
        ).to(tl.float32)

        v = tl.load(
            v_base + kv_offs[:, None] * stride_vn + d_offs[None, :] * stride_vd,
            mask=kv_mask[:, None],
            other=0.0,
        ).to(tl.float32)

        score_tile = tl.dot(q, tl.trans(k)) * scale
        # Only padding mask — no causal mask required
        score_tile = tl.where(kv_mask[None, :], score_tile, float("-inf"))

        tile_max = tl.max(score_tile, axis=1)
        new_m    = tl.maximum(m, tile_max)
        alpha    = tl.exp(m - new_m)
        s        = s * alpha
        acc      = acc * alpha[:, None]

        exp_scores = tl.exp(score_tile - new_m[:, None])
        acc += tl.dot(exp_scores, v)
        s   += tl.sum(exp_scores, axis=1)
        m    = new_m

    # ── Loop B: diagonal K/V tiles — causal mask required ────────────────────
    # kv_start in [q_start, q_start + BLOCK_M) — the diagonal crossing region.
    # Future tiles (kv_start >= q_start + BLOCK_M) are skipped entirely.
    for kv_start in range(q_start, q_start + BLOCK_M, BLOCK_N):
        kv_offs = kv_start + tl.arange(0, BLOCK_N)
        kv_mask = kv_offs < N

        k = tl.load(
            k_base + kv_offs[:, None] * stride_kn + d_offs[None, :] * stride_kd,
            mask=kv_mask[:, None],
            other=0.0,
        ).to(tl.float32)

        v = tl.load(
            v_base + kv_offs[:, None] * stride_vn + d_offs[None, :] * stride_vd,
            mask=kv_mask[:, None],
            other=0.0,
        ).to(tl.float32)

        score_tile = tl.dot(q, tl.trans(k)) * scale

        # Causal mask: query i only attends to key j <= i
        causal_mask = q_offs[:, None] >= kv_offs[None, :]
        score_tile  = tl.where(causal_mask, score_tile, float("-inf"))
        score_tile  = tl.where(kv_mask[None, :], score_tile, float("-inf"))

        tile_max = tl.max(score_tile, axis=1)
        new_m    = tl.maximum(m, tile_max)
        alpha    = tl.exp(m - new_m)
        s        = s * alpha
        acc      = acc * alpha[:, None]

        exp_scores = tl.exp(score_tile - new_m[:, None])
        acc += tl.dot(exp_scores, v)
        s   += tl.sum(exp_scores, axis=1)
        m    = new_m

    # Single normalisation write — one HBM store per Q tile
    out = acc / s[:, None]
    tl.store(
        o_base + q_offs[:, None] * stride_on + d_offs[None, :] * stride_od,
        out,
        mask=q_mask[:, None],
    )


# ── Wrapper ───────────────────────────────────────────────────────────────────

def flash_attention_v2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """
    Flash Attention v2: causal attention with autotuned tile sizes.

    Args:
        q, k, v: Tensors of shape (B, H, N, d), fp32 or fp16.
                 d must be a power of 2 and <= 128.
                 N can be arbitrarily large.

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

    scale = 1.0 / math.sqrt(d)
    out   = torch.empty_like(q)

    # Grid lambda: BLOCK_M comes from the autotuner config at launch time
    grid = lambda meta: (B, H, triton.cdiv(N, meta["BLOCK_M"]))

    flash_attention_v2_kernel[grid](
        q, k, v, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        N, d,
        scale,
        BLOCK_D=d,
    )

    return out


# ── Test ──────────────────────────────────────────────────────────────────────

def test_flash_attention_v2():
    print("Testing flash_attention_v2...")

    torch.manual_seed(0)
    for N in [64, 128, 256, 512, 1024, 2048, 4096]:
        for d in [32, 64]:
            B, H = 2, 4
            q = torch.randn(B, H, N, d, device="cuda", dtype=torch.float32)
            k = torch.randn(B, H, N, d, device="cuda", dtype=torch.float32)
            v = torch.randn(B, H, N, d, device="cuda", dtype=torch.float32)

            ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            got = flash_attention_v2(q, k, v)

            torch.testing.assert_close(got, ref, atol=1e-3, rtol=1e-3)
            print(f"  N={N:5d}  d={d}  max_err={(got - ref).abs().max():.2e}  PASS")

    print("All tests passed.")


# ── Benchmark ─────────────────────────────────────────────────────────────────

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[128, 256, 512, 1024, 2048, 4096],
        x_log=True,
        line_arg="provider",
        line_vals=["triton_v2", "triton_v1", "torch"],
        line_names=["Triton Flash v2", "Triton Flash v1", "torch SDPA (causal)"],
        styles=[("blue", "-"), ("red", "--"), ("green", ":")],
        ylabel="TFLOPS",
        plot_name="flash_attention_v2_benchmark",
        args={"B": 4, "H": 8, "d": 64},
    )
)
def benchmark_flash_attention_v2(N, B, H, d, provider):
    from kernels.attention.flash_attention_v1 import flash_attention_v1

    q = torch.randn(B, H, N, d, device="cuda", dtype=torch.float32)
    k = torch.randn(B, H, N, d, device="cuda", dtype=torch.float32)
    v = torch.randn(B, H, N, d, device="cuda", dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]

    if provider == "triton_v2":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: flash_attention_v2(q, k, v),
            warmup=25, rep=100, quantiles=quantiles,
        )
    elif provider == "triton_v1":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: flash_attention_v1(q, k, v),
            warmup=25, rep=100, quantiles=quantiles,
        )
    else:
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True),
            warmup=25, rep=100, quantiles=quantiles,
        )

    tflops = lambda ms: (2 * B * H * N * N * d * 1e-12) / (ms * 1e-3)
    return tflops(ms), tflops(max_ms), tflops(min_ms)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_flash_attention_v2()
    import os
    os.makedirs("benchmarks/results/attention", exist_ok=True)
    benchmark_flash_attention_v2.run(
        print_data=True,
        show_plots=True,
        save_path="benchmarks/results/attention",
    )
