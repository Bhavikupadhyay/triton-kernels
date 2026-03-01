"""
Kernel:   reduce_sum
Category: reductions
Complexity: O(n)
Memory bound: Yes
PyTorch equivalent: torch.sum(x)
References:
  - Parallel reduction: https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf

How it works:
Each program block loads BLOCK_SIZE elements and reduces them to a single
scalar with tl.sum (a warp-level tree reduction). The partial sums are
written to a small scratch buffer (one entry per block). A second kernel
then sums those partials into the final scalar.

Two-pass design:
  Pass 1 — grid of ceil(n / BLOCK_SIZE) blocks, each producing one partial sum
  Pass 2 — single block sums the partials; BLOCK_SIZE = next_power_of_2(num_blocks)

Memory traffic: 1 read of x (n × dtype bytes) + negligible scratch I/O.
Metric: GB/s = (n × element_size × 1e-9) / (ms × 1e-3)

Expected behaviour: matches torch.sum within floating-point associativity
differences; should reach near-peak memory bandwidth at large n.
"""

import torch
import triton
import triton.language as tl


# ── 1. Triton kernels ─────────────────────────────────────────────────────────

@triton.jit
def reduce_sum_kernel(
    x_ptr,
    partial_ptr,
    n: int,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    partial = tl.sum(x, axis=0)
    tl.store(partial_ptr + pid, partial)


@triton.jit
def reduce_partials_kernel(
    partial_ptr,
    out_ptr,
    num_blocks: int,
    BLOCK_SIZE: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < num_blocks
    x = tl.load(partial_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr, tl.sum(x, axis=0))


# ── 2. Python wrapper ─────────────────────────────────────────────────────────

def reduce_sum(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "Input must be on CUDA"
    assert x.is_contiguous(), "Input must be contiguous"
    n = x.numel()
    BLOCK_SIZE = 1024
    num_blocks = triton.cdiv(n, BLOCK_SIZE)

    # Pass 1: one partial sum per block
    partials = torch.empty(num_blocks, device=x.device, dtype=torch.float32)
    reduce_sum_kernel[(num_blocks,)](x, partials, n, BLOCK_SIZE=BLOCK_SIZE)

    # Pass 2: sum the partials in a single block.
    # next_power_of_2 gives a valid tl.constexpr that covers all partials.
    block2 = triton.next_power_of_2(num_blocks)
    out = torch.empty(1, device=x.device, dtype=torch.float32)
    reduce_partials_kernel[(1,)](partials, out, num_blocks, BLOCK_SIZE=block2)

    return out[0].to(x.dtype)


# ── 3. Correctness tests ──────────────────────────────────────────────────────

def test_reduce_sum():
    sizes = [1, 127, 128, 1024, 1025, 65536, 100_000, 1_000_000]
    dtypes = [torch.float32, torch.float16]

    for dtype in dtypes:
        for n in sizes:
            x = torch.randn(n, device="cuda", dtype=dtype)
            ref = torch.sum(x)
            got = reduce_sum(x)
            # Reductions accumulate fp error; scale tolerance with sqrt(n)
            torch.testing.assert_close(
                got.float(), ref.float(),
                rtol=1e-3, atol=1e-3 * (n ** 0.5),
            )

    print("test_reduce_sum: PASSED")


# ── 4. Benchmarks ─────────────────────────────────────────────────────────────

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["n"],
        x_vals=[2**i for i in range(12, 28)],
        x_log=True,
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton", "PyTorch"],
        styles=[("blue", "-"), ("green", "--")],
        ylabel="GB/s",
        plot_name="reduce_sum",
        args={},
    )
)
def benchmark_reduce_sum(n, provider):
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]

    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: reduce_sum(x), warmup=25, rep=100, quantiles=quantiles
        )
    else:
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch.sum(x), warmup=25, rep=100, quantiles=quantiles
        )

    # 1 read of x; output is a scalar (negligible)
    gb = n * x.element_size() * 1e-9
    return gb / (ms * 1e-3), gb / (max_ms * 1e-3), gb / (min_ms * 1e-3)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_reduce_sum()
    benchmark_reduce_sum.run(print_data=True, show_plots=True)
