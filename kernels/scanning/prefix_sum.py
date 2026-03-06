"""
Kernel:   prefix_sum
Category: scanning
Complexity: O(n)
Memory bound: Yes
PyTorch equivalent: torch.cumsum(x, dim=0)
References:
  - Hillis & Steele, "Data Parallel Algorithms" (1986)
  - Blelloch, "Prefix Sums and Their Applications" (1990):
      https://www.cs.cmu.edu/~blelloch/papers/Ble93.pdf

How it works:
prefix_sum (cumsum) has a sequential dependency: y[i] = y[i-1] + x[i]. You can't
split x across independent blocks without knowing each block's carry-in first.

The solution is three kernel passes:

  Pass 1 — local scan + block totals  (num_blocks programs):
    Each block loads BLOCK_SIZE elements, computes a local INCLUSIVE prefix scan
    using tl.associative_scan (Kogge-Stone / Hillis-Steele in hardware), writes
    the scanned values to out, and writes its total (sum of all its elements) to
    a block_sums scratch buffer.

  Pass 2 — scan the block totals  (1 program):
    A single block runs an inclusive prefix scan over block_sums, producing
    carry_inclusive[i] = block_sums[0] + ... + block_sums[i].

  Pass 3 — add carry  (num_blocks programs):
    Each block loads carry_inclusive[block_id - 1] (0 for block 0) and adds it
    to every element in its pass-1 output, turning local scans into global ones.

Correctness sketch (n=6, BLOCK_SIZE=2, x=[1,2,3,4,5,6]):
  Pass 1 local scans:   [1,3] | [3,7] | [5,11]
  block_sums:           [3, 7, 11]
  Pass 2 carry_incl:    [3, 10, 21]
  Pass 3 carry-in:      +0  | +3   | +10
  Final:                [1,3,6,10,15,21] ✓

BLOCK_SIZE is fixed at 1024. For n up to ~16 M (2^24), pass 2 handles at most
16384 block totals in a single block — well within register budget for a simple sum.

HBM traffic vs torch.cumsum:
  We make three passes over out (write in pass 1, read+write in pass 3) plus one
  pass reading x. That is ~4n element reads/writes internally. torch.cumsum reads
  and writes n elements each (2n). We report GB/s with the standard 2×n formula
  applied to both, so the wall-time comparison is fair but our kernel is doing
  more total HBM work — which is reflected in the numbers.
"""

import torch
import triton
import triton.language as tl


# ── Helper: associative combine function ─────────────────────────────────────

@triton.jit
def _add(a, b):
    return a + b


# ── 1. Triton kernels ─────────────────────────────────────────────────────────

@triton.jit
def prefix_sum_pass1(
    x_ptr, out_ptr, block_sums_ptr,
    n: int,
    BLOCK_SIZE: tl.constexpr,
):
    block_id = tl.program_id(0)
    offs = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n

    # Load row; OOB positions get 0.0 so they don't contribute to the scan or sum
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # Inclusive prefix scan within this block (Kogge-Stone internally)
    local_scan = tl.associative_scan(x, axis=0, combine_fn=_add)
    tl.store(out_ptr + offs, local_scan, mask=mask)

    # Block total: sum of all (real) elements — equals last element of local_scan
    # but tl.sum is cleaner than trying to index the last lane
    block_total = tl.sum(x, axis=0)
    tl.store(block_sums_ptr + block_id, block_total)


@triton.jit
def prefix_sum_pass2(
    block_sums_ptr, carry_ptr,
    num_blocks: int,
    BLOCK2: tl.constexpr,
):
    # Single block — inclusive prefix scan of the num_blocks block totals
    offs = tl.arange(0, BLOCK2)
    mask = offs < num_blocks
    s = tl.load(block_sums_ptr + offs, mask=mask, other=0.0)
    carry_inclusive = tl.associative_scan(s, axis=0, combine_fn=_add)
    tl.store(carry_ptr + offs, carry_inclusive, mask=mask)


@triton.jit
def prefix_sum_pass3(
    out_ptr, carry_ptr,
    n: int,
    BLOCK_SIZE: tl.constexpr,
):
    block_id = tl.program_id(0)
    offs = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n

    # carry-in = inclusive scan of block_sums up to (block_id - 1)
    # block 0 gets 0 (no prior blocks); mask prevents OOB load at block_id=0
    carry = tl.load(carry_ptr + block_id - 1, mask=(block_id > 0), other=0.0)

    local_scan = tl.load(out_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, local_scan + carry, mask=mask)


# ── 2. Python wrapper ─────────────────────────────────────────────────────────

BLOCK_SIZE = 1024

def prefix_sum(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "Input must be on CUDA"
    assert x.is_contiguous(), "Input must be contiguous"
    assert x.ndim == 1, "Input must be 1D"

    n = x.numel()
    num_blocks = triton.cdiv(n, BLOCK_SIZE)

    out = x.to(torch.float32)  # work in fp32; also makes a copy
    block_sums = torch.empty(num_blocks, device=x.device, dtype=torch.float32)

    # Pass 1: local inclusive scan per block + write block totals
    prefix_sum_pass1[(num_blocks,)](out, out, block_sums, n, BLOCK_SIZE=BLOCK_SIZE)

    if num_blocks > 1:
        # Pass 2: prefix scan over block totals → carry_inclusive
        BLOCK2 = triton.next_power_of_2(num_blocks)
        carry = torch.empty(num_blocks, device=x.device, dtype=torch.float32)
        prefix_sum_pass2[(1,)](block_sums, carry, num_blocks, BLOCK2=BLOCK2)

        # Pass 3: add carry-in to each block's local scan
        prefix_sum_pass3[(num_blocks,)](out, carry, n, BLOCK_SIZE=BLOCK_SIZE)

    return out


# ── 3. Correctness tests ──────────────────────────────────────────────────────

def test_prefix_sum():
    sizes = [
        1, 127, 128, 1023, 1024, 1025,   # edge cases around BLOCK_SIZE
        4096, 10000, 2**20, 2**20 + 7,   # multi-block, non-power-of-2
    ]
    for n in sizes:
        x = torch.rand(n, device="cuda", dtype=torch.float32)
        ref = torch.cumsum(x, dim=0)
        got = prefix_sum(x)
        torch.testing.assert_close(got, ref, rtol=1e-2, atol=1e-2)

    print("test_prefix_sum: PASSED")


# ── 4. Benchmarks ─────────────────────────────────────────────────────────────

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["n"],
        x_vals=[2**i for i in range(10, 25)],  # 1K → 16M
        x_log=True,
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton", "PyTorch"],
        styles=[("blue", "-"), ("green", "--")],
        ylabel="GB/s",
        plot_name="prefix_sum",
        args={},
    )
)
def benchmark_prefix_sum(n, provider):
    x = torch.rand(n, device="cuda", dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]

    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: prefix_sum(x), warmup=25, rep=100, quantiles=quantiles
        )
    else:
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch.cumsum(x, dim=0), warmup=25, rep=100, quantiles=quantiles
        )

    gb = 2 * n * x.element_size() * 1e-9
    return gb / (ms * 1e-3), gb / (max_ms * 1e-3), gb / (min_ms * 1e-3)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_prefix_sum()
    benchmark_prefix_sum.run(print_data=True, show_plots=True)
