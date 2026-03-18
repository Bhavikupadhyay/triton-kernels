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
  - Merrill & Garland, "Single-pass Parallel Prefix Scan with Decoupled Look-back" (2016):
      https://research.nvidia.com/publication/2016-03_single-pass-parallel-prefix-scan-decoupled-look-back

Algorithm — single-pass decoupled look-back:

  The original three-pass design does ~4n HBM traffic:
    Pass 1 writes the local inclusive scan to out (n writes).
    Pass 3 reads out and rewrites every element with carry added (2n more).
  torch.cumsum (via CUB) does ~2n: read x once, write out once.

  Decoupled look-back closes this gap with one kernel launch:

  State per block stored in global scratch (stays in L2, << n elements):
    flags[i]    — int32: FLAG_INVALID(0) → FLAG_PARTIAL(1) → FLAG_INCLUSIVE(2)
    prefixes[i] — float32: first the local block total, then the inclusive prefix

  A global atomic counter (counter_ptr) assigns each program its block_id in
  execution order. Because blocks can be scheduled in any order by the GPU, the
  counter is the only guarantee that block_id i is processed by some SM before
  block_id i+1 spins on it (all launched blocks eventually get an SM).

  Each block (block_id assigned via atomic counter):
    1. Loads its BLOCK_SIZE chunk of x, computes a local inclusive scan in
       registers via tl.associative_scan. Also computes the block total.
    2. Writes local_total to prefixes[block_id].
    3. Atomically sets flags[block_id] = FLAG_PARTIAL with sem='release'.
       The release ordering ensures the store in step 2 is globally visible
       before any other thread can observe the flag transition.
    4. Look-back loop (from block_id-1 downward):
       a. Spin on flags[look_back_id] with sem='acquire' until != FLAG_INVALID.
          The acquire ordering ensures that once the flag is visible, the
          corresponding prefixes[] value is also visible (no extra fence needed).
       b. Add prefixes[look_back_id] to aggregate carry.
       c. If flag == FLAG_INCLUSIVE: this predecessor already accumulated all
          earlier carry — stop here.
          If flag == FLAG_PARTIAL: only the local total is published — keep
          looking back one step further.
    5. Writes local_total + aggregate to prefixes[block_id].
    6. Sets flags[block_id] = FLAG_INCLUSIVE with sem='release'.
    7. Adds aggregate carry to the local scan and stores to out.

  HBM traffic: read x (n reads) + write out (n writes) = 2n. Matches CUB.
  Scratch buffers are num_blocks << n elements and reside entirely in L2.

  Memory ordering note:
    sem='release' on the flag write (steps 3, 6) ensures the prefix store
    (steps 2, 5) is ordered before it. sem='acquire' on the flag spin-read
    (step 4a) ensures the prefix load (step 4b) is ordered after it. Together
    these replace __threadfence() from the equivalent CUDA C implementation.
    Requires Triton >= 2.1 on sm_70+ (T4 is sm_75). ✓
"""

import torch
import triton
import triton.language as tl


# ── Constants ─────────────────────────────────────────────────────────────────

BLOCK_SIZE    = 1024
FLAG_INVALID  = 0
FLAG_PARTIAL  = 1
FLAG_INCLUSIVE = 2


# ── Helper ────────────────────────────────────────────────────────────────────

@triton.jit
def _add(a, b):
    return a + b


# ── 1. Triton kernel ──────────────────────────────────────────────────────────

@triton.jit
def prefix_sum_dlb_kernel(
    x_ptr, out_ptr,
    flags_ptr, prefixes_ptr,
    counter_ptr,
    n: int,
    BLOCK_SIZE: tl.constexpr,
):
    # ── Claim a block ID in execution order ───────────────────────────────────
    # atomic_add returns the old value; blocks get IDs 0, 1, 2, … in the order
    # they reach this instruction. All IDs are eventually assigned because every
    # launched CUDA block is guaranteed to run.
    block_id = tl.atomic_add(counter_ptr, 1, sem='relaxed')

    # ── Load chunk and compute local inclusive scan ───────────────────────────
    offs        = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask        = offs < n
    x           = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    local_scan  = tl.associative_scan(x, axis=0, combine_fn=_add)
    local_total = tl.sum(x, axis=0)

    # ── Publish partial prefix (local total only) ─────────────────────────────
    tl.store(prefixes_ptr + block_id, local_total)
    # 'release': all stores above this point are globally visible before any
    # thread can observe flags[block_id] transition away from FLAG_INVALID.
    # Literal 1 = FLAG_PARTIAL (Triton JIT cannot read non-constexpr globals).
    tl.atomic_xchg(flags_ptr + block_id, 1, sem='release')

    # ── Look-back: accumulate carry from predecessor blocks ───────────────────
    aggregate    = 0.0
    look_back_id = block_id - 1

    while look_back_id >= 0:
        # Spin until predecessor has published at least a partial prefix.
        # 'acquire': once we observe f != FLAG_INVALID, the corresponding
        # prefixes[] value written before the release flag is also visible.
        # Literal 0 = FLAG_INVALID, 2 = FLAG_INCLUSIVE
        f = tl.atomic_add(flags_ptr + look_back_id, 0, sem='acquire')
        while f == 0:
            f = tl.atomic_add(flags_ptr + look_back_id, 0, sem='acquire')

        # Safe to load now — ordered after the acquire above.
        val        = tl.load(prefixes_ptr + look_back_id)
        aggregate += val

        # tl.where keeps look_back_id as a Triton tensor in all branches,
        # satisfying the SSA phi-node type constraint for the while loop.
        # f==2 (FLAG_INCLUSIVE): set to -1 → outer while exits next check.
        # f==1 (FLAG_PARTIAL):   decrement → keep looking back.
        look_back_id = tl.where(f == 2, -1, look_back_id - 1)

    # ── Publish inclusive prefix ──────────────────────────────────────────────
    tl.store(prefixes_ptr + block_id, local_total + aggregate)
    # Literal 2 = FLAG_INCLUSIVE
    tl.atomic_xchg(flags_ptr + block_id, 2, sem='release')

    # ── Write final output ────────────────────────────────────────────────────
    tl.store(out_ptr + offs, local_scan + aggregate, mask=mask)


# ── 2. Python wrapper ─────────────────────────────────────────────────────────

def prefix_sum(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda,         "Input must be on CUDA"
    assert x.is_contiguous(), "Input must be contiguous"
    assert x.ndim == 1,       "Input must be 1D"

    n          = x.numel()
    num_blocks = triton.cdiv(n, BLOCK_SIZE)

    out      = torch.empty(n,          device=x.device, dtype=torch.float32)
    flags    = torch.zeros(num_blocks, device=x.device, dtype=torch.int32)
    prefixes = torch.empty(num_blocks, device=x.device, dtype=torch.float32)
    counter  = torch.zeros(1,          device=x.device, dtype=torch.int32)

    prefix_sum_dlb_kernel[(num_blocks,)](
        x.float(), out,
        flags, prefixes, counter,
        n, BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


# ── 3. Correctness tests ──────────────────────────────────────────────────────

def test_prefix_sum():
    sizes = [
        1, 127, 128, 1023, 1024, 1025,   # edge cases around BLOCK_SIZE
        4096, 10000, 2**20, 2**20 + 7,   # multi-block, non-power-of-2
        2**23, 2**24,                     # large — stress-tests the look-back
    ]
    for n in sizes:
        x   = torch.rand(n, device="cuda", dtype=torch.float32)
        ref = torch.cumsum(x, dim=0)
        got = prefix_sum(x)
        torch.testing.assert_close(got, ref, rtol=1e-2, atol=1e-2)
        print(f"  n={n}: PASSED")

    print("test_prefix_sum: ALL PASSED")


# ── 4. Benchmarks ─────────────────────────────────────────────────────────────

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["n"],
        x_vals=[2**i for i in range(10, 25)],  # 1K → 16M
        x_log=True,
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton (DLB)", "PyTorch"],
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
