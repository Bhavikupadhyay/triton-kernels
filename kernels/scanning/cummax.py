"""
Kernel:   cummax
Category: scanning
Complexity: O(n)
Memory bound: Yes
PyTorch equivalent: torch.cummax(x, dim=0).values
References:
  - Same parallel scan decomposition as prefix_sum — see prefix_sum.py for full
    algorithm explanation.

How it works:
cummax has the same cross-block dependency as prefix_sum: y[i] = max(y[i-1], x[i]).
Block i can't finalise until it knows the running max from all prior blocks.

Three-pass design (identical structure to prefix_sum, combine_fn = max):

  Pass 1 — local scan + block maximums  (num_blocks programs):
    Each block computes a LOCAL inclusive running-max scan of its BLOCK_SIZE
    elements using tl.associative_scan with _max, writes the scanned values to
    out, and writes its block maximum to block_maxes scratch.

  Pass 2 — scan the block maximums  (1 program):
    Single block computes inclusive prefix scan of block_maxes with _max.
    carry_inclusive[i] = max(block_maxes[0..i]).

  Pass 3 — propagate carry  (num_blocks programs):
    Each block loads carry_inclusive[block_id - 1] (-inf for block 0) and takes
    the element-wise max with its locally-scanned values.

Same 4n HBM traffic limitation as prefix_sum — see prefix_sum.py interpretation
for the full explanation.
"""

import torch
import triton
import triton.language as tl


# ── Helper: associative combine function ─────────────────────────────────────

@triton.jit
def _max(a, b):
    return tl.maximum(a, b)


# ── 1. Triton kernels ─────────────────────────────────────────────────────────

@triton.jit
def cummax_pass1(
    x_ptr, out_ptr, block_maxes_ptr,
    n: int,
    BLOCK_SIZE: tl.constexpr,
):
    block_id = tl.program_id(0)
    offs = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n

    # OOB positions get -inf so they don't affect the running max
    x = tl.load(x_ptr + offs, mask=mask, other=float("-inf")).to(tl.float32)

    local_scan = tl.associative_scan(x, axis=0, combine_fn=_max)
    tl.store(out_ptr + offs, local_scan, mask=mask)

    block_max = tl.max(x, axis=0)
    tl.store(block_maxes_ptr + block_id, block_max)


@triton.jit
def cummax_pass2(
    block_maxes_ptr, carry_ptr,
    num_blocks: int,
    BLOCK2: tl.constexpr,
):
    offs = tl.arange(0, BLOCK2)
    mask = offs < num_blocks
    s = tl.load(block_maxes_ptr + offs, mask=mask, other=float("-inf"))
    carry_inclusive = tl.associative_scan(s, axis=0, combine_fn=_max)
    tl.store(carry_ptr + offs, carry_inclusive, mask=mask)


@triton.jit
def cummax_pass3(
    out_ptr, carry_ptr,
    n: int,
    BLOCK_SIZE: tl.constexpr,
):
    block_id = tl.program_id(0)
    offs = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n

    # carry-in = max of all elements before this block (-inf for block 0)
    carry = tl.load(carry_ptr + block_id - 1, mask=(block_id > 0), other=float("-inf"))

    local_scan = tl.load(out_ptr + offs, mask=mask, other=float("-inf"))
    tl.store(out_ptr + offs, tl.maximum(local_scan, carry), mask=mask)


# ── 2. Python wrapper ─────────────────────────────────────────────────────────

BLOCK_SIZE = 1024

def cummax(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "Input must be on CUDA"
    assert x.is_contiguous(), "Input must be contiguous"
    assert x.ndim == 1, "Input must be 1D"

    n = x.numel()
    num_blocks = triton.cdiv(n, BLOCK_SIZE)

    out = x.to(torch.float32)
    block_maxes = torch.empty(num_blocks, device=x.device, dtype=torch.float32)

    cummax_pass1[(num_blocks,)](out, out, block_maxes, n, BLOCK_SIZE=BLOCK_SIZE)

    if num_blocks > 1:
        BLOCK2 = triton.next_power_of_2(num_blocks)
        carry = torch.empty(num_blocks, device=x.device, dtype=torch.float32)
        cummax_pass2[(1,)](block_maxes, carry, num_blocks, BLOCK2=BLOCK2)
        cummax_pass3[(num_blocks,)](out, carry, n, BLOCK_SIZE=BLOCK_SIZE)

    return out


# ── 3. Correctness tests ──────────────────────────────────────────────────────

def test_cummax():
    sizes = [
        1, 127, 128, 1023, 1024, 1025,
        4096, 10000, 2**20, 2**20 + 7,
    ]
    for n in sizes:
        x = torch.randn(n, device="cuda", dtype=torch.float32)
        ref = torch.cummax(x, dim=0).values
        got = cummax(x)
        torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-5)

    print("test_cummax: PASSED")


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
        plot_name="cummax",
        args={},
    )
)
def benchmark_cummax(n, provider):
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]

    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: cummax(x), warmup=25, rep=100, quantiles=quantiles
        )
    else:
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch.cummax(x, dim=0).values, warmup=25, rep=100, quantiles=quantiles
        )

    gb = 2 * n * x.element_size() * 1e-9
    return gb / (ms * 1e-3), gb / (max_ms * 1e-3), gb / (min_ms * 1e-3)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_cummax()
    benchmark_cummax.run(print_data=True, show_plots=True)
