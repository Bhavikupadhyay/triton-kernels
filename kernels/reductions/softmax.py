"""
Kernel:   softmax
Category: reductions
Complexity: O(M × N)
Memory bound: Yes
PyTorch equivalent: torch.nn.functional.softmax(x, dim=-1)
References:
  - Triton softmax tutorial: https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html
  - Online softmax: https://arxiv.org/abs/1805.02867

How it works:
One Triton program per row. Each program:
  1. Loads the full row (padded to BLOCK_SIZE with -inf)
  2. Finds the row max (numerical stability — prevents exp overflow)
  3. Computes exp(x - max) for each element
  4. Sums the exps
  5. Divides each exp by the sum and writes the result

BLOCK_SIZE is set to next_power_of_2(N) at launch so the entire row fits
in one program. This avoids a multi-pass loop and fuses all five steps into
a single kernel — one read pass and one write pass, hitting HBM exactly twice.

HBM traffic: read M×N + write M×N = 2×M×N bytes.
Metric: GB/s = (2 × M × N × element_size × 1e-9) / (ms × 1e-3)

Input shape: (M, N) — M rows, each of length N.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# ── 1. Triton kernel ──────────────────────────────────────────────────────────

@triton.jit
def softmax_kernel(
    x_ptr, out_ptr,
    n_cols: int,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols

    # Load row; pad out-of-bounds positions with -inf so they don't affect max/sum
    x = tl.load(x_ptr + row * n_cols + offs, mask=mask, other=-float("inf"))

    # Upcast to fp32 for numerical stability
    x = x.to(tl.float32)

    # Subtract row max for numerical stability
    row_max = tl.max(x, axis=0)
    x = x - row_max

    # Exp, sum, normalise
    x_exp = tl.math.exp(x)
    exp_sum = tl.sum(x_exp, axis=0)
    out = x_exp / exp_sum

    tl.store(out_ptr + row * n_cols + offs, out, mask=mask)


# ── 2. Python wrapper ─────────────────────────────────────────────────────────

def softmax(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "Input must be on CUDA"
    assert x.is_contiguous(), "Input must be contiguous"
    assert x.ndim == 2, "Input must be 2D (M rows × N cols)"
    M, N = x.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    out = torch.empty_like(x)
    softmax_kernel[(M,)](x, out, N, BLOCK_SIZE=BLOCK_SIZE)
    return out


# ── 3. Correctness tests ──────────────────────────────────────────────────────

def test_softmax():
    # Test varying row widths (including non-powers-of-2) and batch sizes
    configs = [
        (1, 128), (1, 127), (128, 128), (512, 256),
        (1024, 512), (128, 1024), (64, 4096),
    ]
    dtypes = [torch.float32]  # kernel always outputs fp32

    for dtype in dtypes:
        for M, N in configs:
            x = torch.randn(M, N, device="cuda", dtype=dtype)
            ref = F.softmax(x, dim=-1)
            got = softmax(x)
            torch.testing.assert_close(got, ref, rtol=1e-3, atol=1e-3)

    print("test_softmax: PASSED")


# ── 4. Benchmarks ─────────────────────────────────────────────────────────────
#
# Fix M=4096 rows, vary N (column width).
# This tests how softmax scales with row width — the practically interesting
# dimension (e.g. sequence length in attention, vocab size in language models).

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[2**i for i in range(7, 15)],   # 128 → 16384 cols
        x_log=True,
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton", "PyTorch"],
        styles=[("blue", "-"), ("green", "--")],
        ylabel="GB/s",
        plot_name="softmax",
        args={"M": 4096},
    )
)
def benchmark_softmax(M, N, provider):
    x = torch.randn(M, N, device="cuda", dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]

    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: softmax(x), warmup=25, rep=100, quantiles=quantiles
        )
    else:
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: F.softmax(x, dim=-1), warmup=25, rep=100, quantiles=quantiles
        )

    # Read M×N + write M×N
    gb = 2 * M * N * x.element_size() * 1e-9
    return gb / (ms * 1e-3), gb / (max_ms * 1e-3), gb / (min_ms * 1e-3)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_softmax()
    benchmark_softmax.run(print_data=True, show_plots=True)
