"""
Kernel: Flash Attention v2 (fp16)
Category: attention
Complexity: O(N^2 * d) compute, O(N * d) memory
Memory bound: No — compute bound; HBM reads scale as O(N * d)
PyTorch equivalent: F.scaled_dot_product_attention(q, k, v, is_causal=True)
References: https://arxiv.org/abs/2307.08691 (Dao, FlashAttention-2, 2023)

Difference from flash_attention_v2.py:
  - Accepts and operates on fp16 tensors natively.
  - K is pre-transposed in the wrapper to (B, H, d, N), eliminating in-kernel
    tl.trans. tl.trans on Turing (SM75) can violate HMMA fragment layout.
  - tl.dot uses accumulator form: tl.dot(fp16, fp16, fp32_acc) → HMMA on T4
    (65 TFLOPS peak vs 8.1 TFLOPS for the fp32 SIMT path). out_dtype= form
    may cause Triton to upcast inputs to fp32 before the matmul.
  - Softmax state (running max m, sum s) and accumulator (acc) remain in fp32
    for numerical stability; only the tl.dot operands are fp16.
  - num_stages=1 configs added: Turing has limited async copy support;
    num_stages=2/3 adds shared memory pressure and can cause register spills.
"""

import torch
import triton
import triton.language as tl
import math
import torch.nn.functional as F


# ── Kernel ────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        # num_stages=1 first — Turing has limited async copy; stages add SMEM pressure
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64},  num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 32},  num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64},  num_warps=8, num_stages=1),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64},  num_warps=8, num_stages=1),
        # num_stages=2 — may help at large N if SMEM budget allows
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64},  num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 32,  "BLOCK_N": 64},  num_warps=4, num_stages=2),
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
    Flash Attention v2 forward with fp16 tensor core path.

    K is pre-transposed in the wrapper to (B, H, d, N) and loaded directly
    as (BLOCK_D, BLOCK_N) — no in-kernel tl.trans. tl.trans on Turing (SM75)
    can violate HMMA fragment layout, preventing tensor core dispatch.

    tl.dot uses accumulator form: tl.dot(fp16, fp16, fp32_acc). This form
    directly maps to mma.sync PTX → HMMA on T4. Softmax running state (m, s)
    and output accumulator (acc) stay in fp32 throughout.

    Loop structure mirrors flash_attention_v2:
    - Loop A: fully-past tiles (kv_start + BLOCK_N <= q_start) — no causal mask.
    - Loop B: diagonal tiles — causal mask applied per element.
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

    # Load Q tile as fp16 — held in registers for both loops
    q = tl.load(
        q_base + q_offs[:, None] * stride_qn + d_offs[None, :] * stride_qd,
        mask=q_mask[:, None],
        other=0.0,
    ).to(tl.float16)

    m   = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    s   = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # ── Loop A: fully-past K/V tiles — no causal mask needed ─────────────────
    for kv_start in range(0, q_start, BLOCK_N):
        kv_offs = kv_start + tl.arange(0, BLOCK_N)
        kv_mask = kv_offs < N

        # K pre-transposed to (B, H, d, N) in wrapper — load as (BLOCK_D, BLOCK_N)
        kt = tl.load(
            k_base + d_offs[:, None] * stride_kn + kv_offs[None, :] * stride_kd,
            mask=kv_mask[None, :],
            other=0.0,
        ).to(tl.float16)

        v = tl.load(
            v_base + kv_offs[:, None] * stride_vn + d_offs[None, :] * stride_vd,
            mask=kv_mask[:, None],
            other=0.0,
        ).to(tl.float16)

        # Accumulator form → HMMA guaranteed on T4
        qk = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        score_tile = tl.dot(q, kt, qk) * scale
        score_tile = tl.where(kv_mask[None, :], score_tile, float("-inf"))

        tile_max = tl.max(score_tile, axis=1)
        new_m    = tl.maximum(m, tile_max)
        alpha    = tl.exp(m - new_m)
        s        = s * alpha
        acc      = acc * alpha[:, None]

        exp_scores = tl.exp(score_tile - new_m[:, None])
        acc = tl.dot(exp_scores.to(tl.float16), v, acc)
        s   += tl.sum(exp_scores, axis=1)
        m    = new_m

    # ── Loop B: diagonal K/V tiles — causal mask required ────────────────────
    for kv_start in range(q_start, q_start + BLOCK_M, BLOCK_N):
        kv_offs = kv_start + tl.arange(0, BLOCK_N)
        kv_mask = kv_offs < N

        # K pre-transposed to (B, H, d, N) in wrapper — load as (BLOCK_D, BLOCK_N)
        kt = tl.load(
            k_base + d_offs[:, None] * stride_kn + kv_offs[None, :] * stride_kd,
            mask=kv_mask[None, :],
            other=0.0,
        ).to(tl.float16)

        v = tl.load(
            v_base + kv_offs[:, None] * stride_vn + d_offs[None, :] * stride_vd,
            mask=kv_mask[:, None],
            other=0.0,
        ).to(tl.float16)

        # Accumulator form → HMMA guaranteed on T4
        qk = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        score_tile = tl.dot(q, kt, qk) * scale

        causal_mask = q_offs[:, None] >= kv_offs[None, :]
        score_tile  = tl.where(causal_mask, score_tile, float("-inf"))
        score_tile  = tl.where(kv_mask[None, :], score_tile, float("-inf"))

        tile_max = tl.max(score_tile, axis=1)
        new_m    = tl.maximum(m, tile_max)
        alpha    = tl.exp(m - new_m)
        s        = s * alpha
        acc      = acc * alpha[:, None]

        exp_scores = tl.exp(score_tile - new_m[:, None])
        acc = tl.dot(exp_scores.to(tl.float16), v, acc)
        s   += tl.sum(exp_scores, axis=1)
        m    = new_m

    # Normalise and store as fp16
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
    Flash Attention v2 with fp16 tensor core path.

    Args:
        q, k, v: Tensors of shape (B, H, N, d), fp16.
                 d must be a power of 2 and <= 128.

    Returns:
        out: Tensor of shape (B, H, N, d), fp16.
    """
    assert q.is_cuda and k.is_cuda and v.is_cuda, "Inputs must be CUDA tensors"
    assert q.shape == k.shape == v.shape, "q, k, v must have the same shape"
    assert q.dtype == torch.float16, f"Expected fp16 inputs, got {q.dtype}"

    q   = q.contiguous()
    k_t = k.transpose(-2, -1).contiguous()   # (B, H, d, N) — no in-kernel tl.trans needed
    v   = v.contiguous()

    B, H, N, d = q.shape
    assert d & (d - 1) == 0, f"d must be a power of 2, got {d}"
    assert d <= 128, f"d={d} exceeds BLOCK_D limit (128)"

    scale = 1.0 / math.sqrt(d)
    out   = torch.empty_like(q)

    grid = lambda meta: (B, H, triton.cdiv(N, meta["BLOCK_M"]))

    flash_attention_v2_fp16_kernel[grid](
        q, k_t, v, out,
        q.stride(0),   q.stride(1),   q.stride(2),   q.stride(3),
        k_t.stride(0), k_t.stride(1), k_t.stride(2), k_t.stride(3),
        v.stride(0),   v.stride(1),   v.stride(2),   v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        N, d,
        scale,
        BLOCK_D=d,
    )

    return out


# ── Test ──────────────────────────────────────────────────────────────────────

def test_flash_attention_v2_fp16():
    print("Testing flash_attention_v2_fp16...")

    torch.manual_seed(0)
    for N in [64, 128, 256, 512, 1024, 2048, 4096]:
        for d in [32, 64]:
            B, H = 2, 4
            q = torch.randn(B, H, N, d, device="cuda", dtype=torch.float16)
            k = torch.randn(B, H, N, d, device="cuda", dtype=torch.float16)
            v = torch.randn(B, H, N, d, device="cuda", dtype=torch.float16)

            ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            got = flash_attention_v2_fp16(q, k, v)

            # fp16 tolerance: accumulation rounding → slightly looser than fp32
            torch.testing.assert_close(got, ref, atol=1e-2, rtol=1e-2)
            print(f"  N={N:5d}  d={d}  max_err={(got.float() - ref.float()).abs().max():.2e}  PASS")

    print("All tests passed.")


# ── Benchmark ─────────────────────────────────────────────────────────────────

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[128, 256, 512, 1024, 2048, 4096, 8192],
        x_log=True,
        line_arg="provider",
        line_vals=["triton_fp16", "triton_fp32", "torch"],
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
    quantiles = [0.5, 0.2, 0.8]

    if provider == "triton_fp16":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: flash_attention_v2_fp16(q16, k16, v16),
            warmup=25, rep=100, quantiles=quantiles,
        )
    elif provider == "triton_fp32":
        q32 = q16.float(); k32 = k16.float(); v32 = v16.float()
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: flash_attention_v2(q32, k32, v32),
            warmup=25, rep=100, quantiles=quantiles,
        )
    else:
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: F.scaled_dot_product_attention(q16, k16, v16, is_causal=True),
            warmup=25, rep=100, quantiles=quantiles,
        )

    tflops = lambda ms: (4 * B * H * N * N * d * 1e-12) / (ms * 1e-3)
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
