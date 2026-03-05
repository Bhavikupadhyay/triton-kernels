"""
Kernel: Flash Attention v2 (fp16 / Tensor Core)
Category: attention
Complexity: O(N^2 * d) compute, O(N * d) memory
Memory bound: No — compute bound on tensor cores
PyTorch equivalent: F.scaled_dot_product_attention(q, k, v, is_causal=True)
References: https://arxiv.org/abs/2307.08691 (Dao, FlashAttention-2, 2023)
            https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html
"""

import torch
import triton
import triton.language as tl
import math
import torch.nn.functional as F


# ── Kernel ────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        # Larger blocks: better compute intensity once data fits in SMEM.
        # fp16 halves memory vs fp32, so BLOCK_M=128 is now practical.
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 32}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 32,  "BLOCK_N": 32}, num_warps=4, num_stages=2),
    ],
    key=["N", "d"],
)
@triton.jit
def flash_attention_v2_fp16_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    N, d,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Flash Attention v2 with fp16 tensor core matmuls.

    Precision strategy:
    - Q, K, V loaded and kept as fp16 in registers.
    - tl.dot(fp16, fp16) on T4 (SM75) routes through Tensor Cores and
      accumulates into fp32 automatically.
    - Softmax state (m, s, acc) stays fp32 throughout for numerical stability.
    - exp_scores cast to fp16 before the second matmul (scores · V) to keep
      both operands in fp16 for Tensor Core dispatch.
    - Output cast back to fp16 before the final store.

    Structural changes vs fp32 v2:
    - BLOCK_M=128 is viable (fp16 halves SMEM vs fp32 v2's max BLOCK_M=64).
    - num_stages=2/4 enables Triton's software pipelining: memory loads for
      the next K/V tile are issued while the current tile's matmul executes.
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

    # Load Q tile in fp16 — stays fp16 for the QK^T matmul
    q = tl.load(
        q_base + q_offs[:, None] * stride_qn + d_offs[None, :] * stride_qd,
        mask=q_mask[:, None],
        other=0.0,
    ).to(tl.float16)

    # Softmax accumulators in fp32
    m   = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    s   = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # ── Loop A: fully-past K/V tiles — no causal mask ─────────────────────
    for kv_start in range(0, q_start, BLOCK_N):
        kv_offs = kv_start + tl.arange(0, BLOCK_N)
        kv_mask = kv_offs < N

        k = tl.load(
            k_base + kv_offs[:, None] * stride_kn + d_offs[None, :] * stride_kd,
            mask=kv_mask[:, None],
            other=0.0,
        ).to(tl.float16)

        v = tl.load(
            v_base + kv_offs[:, None] * stride_vn + d_offs[None, :] * stride_vd,
            mask=kv_mask[:, None],
            other=0.0,
        ).to(tl.float16)

        # fp16 QK^T → fp32 (Tensor Core path on T4)
        score_tile = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * scale
        score_tile = tl.where(kv_mask[None, :], score_tile, float("-inf"))

        tile_max = tl.max(score_tile, axis=1)
        new_m    = tl.maximum(m, tile_max)
        alpha    = tl.exp(m - new_m)
        s        = s * alpha
        acc      = acc * alpha[:, None]

        exp_scores = tl.exp(score_tile - new_m[:, None])   # fp32

        # fp16 scores · V → fp32 (Tensor Core path on T4)
        acc += tl.dot(exp_scores.to(tl.float16), v, out_dtype=tl.float32)
        s   += tl.sum(exp_scores, axis=1)
        m    = new_m

    # ── Loop B: diagonal K/V tiles — causal mask required ─────────────────
    for kv_start in range(q_start, q_start + BLOCK_M, BLOCK_N):
        kv_offs = kv_start + tl.arange(0, BLOCK_N)
        kv_mask = kv_offs < N

        k = tl.load(
            k_base + kv_offs[:, None] * stride_kn + d_offs[None, :] * stride_kd,
            mask=kv_mask[:, None],
            other=0.0,
        ).to(tl.float16)

        v = tl.load(
            v_base + kv_offs[:, None] * stride_vn + d_offs[None, :] * stride_vd,
            mask=kv_mask[:, None],
            other=0.0,
        ).to(tl.float16)

        score_tile  = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * scale
        causal_mask = q_offs[:, None] >= kv_offs[None, :]
        score_tile  = tl.where(causal_mask, score_tile, float("-inf"))
        score_tile  = tl.where(kv_mask[None, :], score_tile, float("-inf"))

        tile_max = tl.max(score_tile, axis=1)
        new_m    = tl.maximum(m, tile_max)
        alpha    = tl.exp(m - new_m)
        s        = s * alpha
        acc      = acc * alpha[:, None]

        exp_scores = tl.exp(score_tile - new_m[:, None])
        acc += tl.dot(exp_scores.to(tl.float16), v, out_dtype=tl.float32)
        s   += tl.sum(exp_scores, axis=1)
        m    = new_m

    # Normalise, cast to fp16, write once
    out = (acc / s[:, None]).to(tl.float16)
    tl.store(
        o_base + q_offs[:, None] * stride_on + d_offs[None, :] * stride_od,
        out,
        mask=q_mask[:, None],
    )


# ── Wrapper ───────────────────────────────────────────────────────────────────

def flash_attention_v2_fp16(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """
    Flash Attention v2 with fp16 tensor core matmuls.

    Args:
        q, k, v: Tensors of shape (B, H, N, d), fp16 or fp32.
                 Inputs are cast to fp16 internally; output matches input dtype.
                 d must be a power of 2 and <= 128.

    Returns:
        out: Tensor of shape (B, H, N, d), same dtype as input.
    """
    assert q.is_cuda and k.is_cuda and v.is_cuda, "Inputs must be CUDA tensors"
    assert q.shape == k.shape == v.shape, "q, k, v must have the same shape"

    orig_dtype = q.dtype
    q = q.contiguous().to(torch.float16)
    k = k.contiguous().to(torch.float16)
    v = v.contiguous().to(torch.float16)

    B, H, N, d = q.shape
    assert d & (d - 1) == 0, f"d must be a power of 2, got {d}"
    assert d <= 128, f"d={d} exceeds BLOCK_D limit (128)"

    scale = 1.0 / math.sqrt(d)
    out   = torch.empty_like(q)   # fp16

    grid = lambda meta: (B, H, triton.cdiv(N, meta["BLOCK_M"]))

    flash_attention_v2_fp16_kernel[grid](
        q, k, v, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        N, d,
        scale,
        BLOCK_D=d,
    )

    return out.to(orig_dtype)


# ── Test ──────────────────────────────────────────────────────────────────────

def test_flash_attention_v2_fp16():
    print("Testing flash_attention_v2_fp16...")

    torch.manual_seed(0)
    for N in [64, 128, 256, 512, 1024, 2048, 4096]:
        for d in [32, 64]:
            B, H = 2, 4
            # Test with fp16 input
            q = torch.randn(B, H, N, d, device="cuda", dtype=torch.float16)
            k = torch.randn(B, H, N, d, device="cuda", dtype=torch.float16)
            v = torch.randn(B, H, N, d, device="cuda", dtype=torch.float16)

            ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            got = flash_attention_v2_fp16(q, k, v)

            # fp16 needs slightly looser tolerance
            torch.testing.assert_close(got, ref, atol=1e-2, rtol=1e-2)
            print(f"  N={N:5d}  d={d}  max_err={(got - ref).abs().max():.2e}  PASS")

    # Test fp32 input → fp32 output path
    q32 = torch.randn(2, 4, 256, 64, device="cuda", dtype=torch.float32)
    k32 = torch.randn(2, 4, 256, 64, device="cuda", dtype=torch.float32)
    v32 = torch.randn(2, 4, 256, 64, device="cuda", dtype=torch.float32)
    ref32 = F.scaled_dot_product_attention(q32, k32, v32, is_causal=True)
    got32 = flash_attention_v2_fp16(q32, k32, v32)
    assert got32.dtype == torch.float32, "output dtype must match input"
    torch.testing.assert_close(got32, ref32, atol=1e-2, rtol=1e-2)
    print("  fp32 input → fp32 output  PASS")

    print("All tests passed.")


# ── Benchmark ─────────────────────────────────────────────────────────────────

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[128, 256, 512, 1024, 2048, 4096],
        x_log=True,
        line_arg="provider",
        line_vals=["triton_fp16", "triton_fp32_v2", "torch"],
        line_names=["Triton Flash v2 (fp16)", "Triton Flash v2 (fp32)", "torch SDPA (causal)"],
        styles=[("blue", "-"), ("red", "--"), ("green", ":")],
        ylabel="TFLOPS",
        plot_name="flash_attention_v2_fp16_benchmark",
        args={"B": 4, "H": 8, "d": 64},
    )
)
def benchmark_flash_attention_v2_fp16(N, B, H, d, provider):
    from kernels.attention.flash_attention_v2 import flash_attention_v2

    q16 = torch.randn(B, H, N, d, device="cuda", dtype=torch.float16)
    k16 = torch.randn(B, H, N, d, device="cuda", dtype=torch.float16)
    v16 = torch.randn(B, H, N, d, device="cuda", dtype=torch.float16)
    q32 = q16.float()
    k32 = k16.float()
    v32 = v16.float()
    quantiles = [0.5, 0.2, 0.8]

    if provider == "triton_fp16":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: flash_attention_v2_fp16(q16, k16, v16),
            warmup=25, rep=100, quantiles=quantiles,
        )
    elif provider == "triton_fp32_v2":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: flash_attention_v2(q32, k32, v32),
            warmup=25, rep=100, quantiles=quantiles,
        )
    else:
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: F.scaled_dot_product_attention(q16, k16, v16, is_causal=True),
            warmup=25, rep=100, quantiles=quantiles,
        )

    tflops = lambda ms: (2 * B * H * N * N * d * 1e-12) / (ms * 1e-3)
    return tflops(ms), tflops(max_ms), tflops(min_ms)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_flash_attention_v2_fp16()
    import os
    os.makedirs("benchmarks/results/attention", exist_ok=True)
    benchmark_flash_attention_v2_fp16.run(
        print_data=True,
        show_plots=True,
        save_path="benchmarks/results/attention",
    )
