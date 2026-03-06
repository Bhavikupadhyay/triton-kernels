"""
Kernel:   max_min
Category: reductions
Complexity: O(n)
Memory bound: Yes
PyTorch equivalents:
  torch.max(x)   / torch.argmax(x)
  torch.min(x)   / torch.argmin(x)
References:
  - Parallel reduction: https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf

How it works:
Same two-pass design as reduce_sum, but each block tracks both the extreme
value AND its index. Triton's tl.max / tl.min reduce the value; a paired
tl.argmax / tl.argmin recovers the index within the block. The pass-1
scratch buffer stores (value, index) pairs; pass 2 finds the global extreme.

Two-pass design:
  Pass 1 — grid of ceil(n / BLOCK_SIZE) blocks; each emits one (value, index) pair
  Pass 2 — single block finds the global extreme across all pairs

Memory traffic: 1 read of x (n × dtype bytes) + negligible scratch I/O.
Metric: GB/s = (n × element_size × 1e-9) / (ms × 1e-3)
"""

import torch
import triton
import triton.language as tl


# ── 1. Triton kernels ─────────────────────────────────────────────────────────

@triton.jit
def argmax_kernel(
    x_ptr,
    val_ptr,
    idx_ptr,
    n: int,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=-float("inf"))
    local_max = tl.max(x, axis=0)
    local_idx = tl.argmax(x, axis=0)
    tl.store(val_ptr + pid, local_max)
    tl.store(idx_ptr + pid, pid * BLOCK_SIZE + local_idx)


@triton.jit
def argmax_partials_kernel(
    val_ptr,
    idx_ptr,
    out_val_ptr,
    out_idx_ptr,
    num_blocks: int,
    BLOCK_SIZE: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < num_blocks
    vals = tl.load(val_ptr + offs, mask=mask, other=-float("inf"))
    idxs = tl.load(idx_ptr + offs, mask=mask, other=0)
    global_max = tl.max(vals, axis=0)
    # Recover the index: among all blocks whose val equals global_max, take argmax
    local_idx = tl.argmax(vals, axis=0)
    tl.store(out_val_ptr, global_max)
    tl.store(out_idx_ptr, tl.load(idx_ptr + local_idx))


@triton.jit
def argmin_kernel(
    x_ptr,
    val_ptr,
    idx_ptr,
    n: int,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=float("inf"))
    local_min = tl.min(x, axis=0)
    local_idx = tl.argmin(x, axis=0)
    tl.store(val_ptr + pid, local_min)
    tl.store(idx_ptr + pid, pid * BLOCK_SIZE + local_idx)


@triton.jit
def argmin_partials_kernel(
    val_ptr,
    idx_ptr,
    out_val_ptr,
    out_idx_ptr,
    num_blocks: int,
    BLOCK_SIZE: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < num_blocks
    vals = tl.load(val_ptr + offs, mask=mask, other=float("inf"))
    idxs = tl.load(idx_ptr + offs, mask=mask, other=0)
    global_min = tl.min(vals, axis=0)
    local_idx = tl.argmin(vals, axis=0)
    tl.store(out_val_ptr, global_min)
    tl.store(out_idx_ptr, tl.load(idx_ptr + local_idx))


# ── 2. Python wrappers ────────────────────────────────────────────────────────

def _reduce(val_kernel, part_kernel, x: torch.Tensor, fill: float):
    assert x.is_cuda, "Input must be on CUDA"
    assert x.is_contiguous(), "Input must be contiguous"
    n = x.numel()
    BLOCK_SIZE = 1024
    num_blocks = triton.cdiv(n, BLOCK_SIZE)
    block2 = triton.next_power_of_2(num_blocks)

    part_vals = torch.empty(num_blocks, device=x.device, dtype=torch.float32)
    part_idxs = torch.empty(num_blocks, device=x.device, dtype=torch.int64)
    val_kernel[(num_blocks,)](x, part_vals, part_idxs, n, BLOCK_SIZE=BLOCK_SIZE)

    out_val = torch.empty(1, device=x.device, dtype=torch.float32)
    out_idx = torch.empty(1, device=x.device, dtype=torch.int64)
    part_kernel[(1,)](part_vals, part_idxs, out_val, out_idx, num_blocks, BLOCK_SIZE=block2)

    return out_val[0].to(x.dtype), out_idx[0]


def argmax(x: torch.Tensor):
    return _reduce(argmax_kernel, argmax_partials_kernel, x, fill=-float("inf"))


def argmin(x: torch.Tensor):
    return _reduce(argmin_kernel, argmin_partials_kernel, x, fill=float("inf"))


# ── 3. Correctness tests ──────────────────────────────────────────────────────

def test_max_min():
    sizes = [1, 127, 128, 1024, 1025, 65536, 100_000, 1_000_000]
    dtypes = [torch.float32, torch.float16]

    for dtype in dtypes:
        for n in sizes:
            x = torch.randn(n, device="cuda", dtype=dtype)

            got_val, got_idx = argmax(x)
            ref_val = torch.max(x)
            ref_idx = torch.argmax(x)
            torch.testing.assert_close(got_val.float(), ref_val.float(), rtol=1e-3, atol=1e-3)
            assert got_idx == ref_idx, f"argmax index mismatch: got {got_idx}, ref {ref_idx}"

            got_val, got_idx = argmin(x)
            ref_val = torch.min(x)
            ref_idx = torch.argmin(x)
            torch.testing.assert_close(got_val.float(), ref_val.float(), rtol=1e-3, atol=1e-3)
            assert got_idx == ref_idx, f"argmin index mismatch: got {got_idx}, ref {ref_idx}"

    print("test_max_min: PASSED")


# ── 4. Benchmarks ─────────────────────────────────────────────────────────────

def _make_benchmark(triton_fn, torch_fn, name):
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
            plot_name=name,
            args={},
        )
    )
    def bench(n, provider):
        x = torch.randn(n, device="cuda", dtype=torch.float32)
        quantiles = [0.5, 0.2, 0.8]
        fn = triton_fn if provider == "triton" else torch_fn
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: fn(x), warmup=25, rep=100, quantiles=quantiles
        )
        gb = n * x.element_size() * 1e-9
        return gb / (ms * 1e-3), gb / (max_ms * 1e-3), gb / (min_ms * 1e-3)

    return bench


benchmark_argmax = _make_benchmark(argmax, torch.max, "argmax")
benchmark_argmin = _make_benchmark(argmin, torch.min, "argmin")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_max_min()
    benchmark_argmax.run(print_data=True, show_plots=True)
    benchmark_argmin.run(print_data=True, show_plots=True)
