"""
Kernel: Naive Attention
Category: attention
Complexity: O(N^2 * d) compute, O(N^2) memory
Memory bound: No — compute bound at large N and d
PyTorch equivalent: torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
References: https://arxiv.org/abs/1706.03762 (Attention Is All You Need)
"""

import torch
import triton
import triton.language as tl
import math


# ── Kernel ────────────────────────────────────────────────────────────────────

@triton.jit
def naive_attention_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    N, d,
    scale,                          # = 1 / sqrt(d), passed as float
    BLOCK_N: tl.constexpr,          # tile size along N (query rows per program)
    BLOCK_D: tl.constexpr,          # head dimension (must equal d, power of 2)
):
    """
    Each program computes BLOCK_N rows of the output for one (batch, head) pair.

    Design:
    - One program per (batch, head, query-tile).
    - The inner loop iterates over all K key/value rows to build the full
      attention score row, applies online-safe softmax, then accumulates V.
    - O(N^2) memory: scores are materialised in registers (BLOCK_N × N), so
      N must be small enough to fit (N <= 2048 for BLOCK_N=1 on a T4).
    - No causal masking — full attention.
    """
    batch_id = tl.program_id(0)
    head_id  = tl.program_id(1)
    tile_id  = tl.program_id(2)

    # Row range this program owns
    q_start = tile_id * BLOCK_N
    q_offs  = q_start + tl.arange(0, BLOCK_N)   # [BLOCK_N]
    d_offs  = tl.arange(0, BLOCK_D)              # [BLOCK_D]
    q_mask  = q_offs < N

    # Base pointers for this (batch, head)
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

    # ── Compute full attention scores for this query tile ─────────────────
    # scores[i, j] = dot(Q[i], K[j]) * scale
    # We accumulate scores across all key positions, then softmax, then
    # multiply by V.  Because N fits in registers for small N, we hold the
    # full [BLOCK_N, N] score matrix in registers.

    # Pass 1: accumulate scores and track per-row max (for numerical stability)
    scores = tl.zeros([BLOCK_N, 1], dtype=tl.float32)  # placeholder shape
    # We can't dynamically size a register array by N at runtime.
    # Instead we loop over K tiles and do online softmax (max + sum tracked).

    # Online softmax state
    m = tl.full([BLOCK_N], float("-inf"), dtype=tl.float32)  # running max
    s = tl.zeros([BLOCK_N], dtype=tl.float32)                # running sum of exp
    acc = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)     # running weighted V

    # Loop over key/value tiles
    for kv_start in range(0, N, BLOCK_N):
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

        # QK^T for this tile: [BLOCK_N, BLOCK_N]
        # q: [BLOCK_N, BLOCK_D], k: [BLOCK_N, BLOCK_D]
        # score_tile[i, j] = sum_d q[i,d] * k[j,d]
        score_tile = tl.dot(q, tl.trans(k)) * scale  # [BLOCK_N, BLOCK_N]

        # Mask out-of-bounds keys
        score_tile = tl.where(
            kv_mask[None, :],
            score_tile,
            float("-inf"),
        )

        # Online softmax update
        tile_max = tl.max(score_tile, axis=1)          # [BLOCK_N]
        new_m = tl.maximum(m, tile_max)                # [BLOCK_N]

        # Rescale running sum and accumulator for the new max
        alpha = tl.exp(m - new_m)                      # [BLOCK_N]
        s = s * alpha                                   # [BLOCK_N]
        acc = acc * alpha[:, None]                      # [BLOCK_N, BLOCK_D]

        # Exponentiated scores for this tile
        exp_scores = tl.exp(score_tile - new_m[:, None])  # [BLOCK_N, BLOCK_N]

        # Accumulate weighted V
        acc += tl.dot(exp_scores, v)                   # [BLOCK_N, BLOCK_D]
        s += tl.sum(exp_scores, axis=1)                # [BLOCK_N]
        m = new_m

    # Normalise
    out = acc / s[:, None]  # [BLOCK_N, BLOCK_D]

    # Write output
    tl.store(
        o_base + q_offs[:, None] * stride_on + d_offs[None, :] * stride_od,
        out,
        mask=q_mask[:, None],
    )


# ── Wrapper ───────────────────────────────────────────────────────────────────

def naive_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """
    Compute scaled dot-product attention (no causal mask).

    Args:
        q, k, v: Tensors of shape (B, H, N, d), fp32 or fp16.
                 d must be a power of 2; N must be divisible by BLOCK_N.
                 N <= 2048 recommended (register pressure).

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
    scale = 1.0 / math.sqrt(d)

    out = torch.empty_like(q)

    grid = (B, H, triton.cdiv(N, BLOCK_N))

    naive_attention_kernel[grid](
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

def test_naive_attention():
    print("Testing naive_attention...")

    torch.manual_seed(0)
    for N in [64, 128, 256, 512]:
        for d in [32, 64]:
            B, H = 2, 4
            q = torch.randn(B, H, N, d, device="cuda", dtype=torch.float32)
            k = torch.randn(B, H, N, d, device="cuda", dtype=torch.float32)
            v = torch.randn(B, H, N, d, device="cuda", dtype=torch.float32)

            ref = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            got = naive_attention(q, k, v)

            torch.testing.assert_close(got, ref, atol=1e-3, rtol=1e-3)
            print(f"  N={N:4d}  d={d:3d}  max_err={(got - ref).abs().max():.2e}  PASS")

    print("All tests passed.")


# ── Benchmark ─────────────────────────────────────────────────────────────────

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[64, 128, 256, 512, 1024],
        x_log=True,
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton Naive Attention", "torch SDPA"],
        styles=[("blue", "-"), ("green", "--")],
        ylabel="TFLOPS",
        plot_name="naive_attention_benchmark",
        args={"B": 4, "H": 8, "d": 64},
    )
)
def benchmark_naive_attention(N, B, H, d, provider):
    q = torch.randn(B, H, N, d, device="cuda", dtype=torch.float32)
    k = torch.randn(B, H, N, d, device="cuda", dtype=torch.float32)
    v = torch.randn(B, H, N, d, device="cuda", dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]

    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: naive_attention(q, k, v), warmup=25, rep=100, quantiles=quantiles
        )
    else:
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v),
            warmup=25, rep=100, quantiles=quantiles,
        )

    # 4 * B * H * N^2 * d flops (2 matmuls: QK^T and scores*V, each 2*N^2*d)
    tflops = lambda ms: (4 * B * H * N * N * d * 1e-12) / (ms * 1e-3)
    return tflops(ms), tflops(max_ms), tflops(min_ms)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_naive_attention()
    import os
    os.makedirs("benchmarks/results/attention", exist_ok=True)
    benchmark_naive_attention.run(
        print_data=True,
        show_plots=True,
        save_path="benchmarks/results/attention",
    )
