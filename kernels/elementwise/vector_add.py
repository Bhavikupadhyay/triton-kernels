"""
Kernel:   vector_add
Category: elementwise
Complexity: O(n)
Memory bound: Yes
PyTorch equivalent: a + b
References:
  - Triton tutorial: https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html
"""

import torch
import triton
import triton.language as tl


# ── 1. Triton kernel ──────────────────────────────────────────────────────────

@triton.jit
def vector_add_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    n: int,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n

    a = tl.load(a_ptr + offs, mask=mask)
    b = tl.load(b_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, a + b, mask=mask)


# ── 2. Python wrapper ─────────────────────────────────────────────────────────

def vector_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.is_cuda and b.is_cuda, "Inputs must be on CUDA"
    assert a.is_contiguous() and b.is_contiguous(), "Inputs must be contiguous"
    assert a.shape == b.shape, "Inputs must have the same shape"

    n = a.numel()
    out = torch.empty_like(a)

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    vector_add_kernel[grid](a, b, out, n, BLOCK_SIZE=BLOCK_SIZE)
    return out


# ── 3. Correctness test ───────────────────────────────────────────────────────

def test_vector_add():
    sizes = [1, 127, 128, 1024, 1025, 65536, 100_000]
    dtypes = [torch.float32, torch.float16]

    for dtype in dtypes:
        for n in sizes:
            a = torch.randn(n, device="cuda", dtype=dtype)
            b = torch.randn(n, device="cuda", dtype=dtype)
            out = vector_add(a, b)
            expected = a + b
            torch.testing.assert_close(out, expected)

    print("test_vector_add: PASSED")


# ── 4. Benchmark ──────────────────────────────────────────────────────────────

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["n"],
        x_vals=[2**i for i in range(12, 28)],  # 4K to 128M elements
        x_log=True,
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton", "PyTorch"],
        styles=[("blue", "-"), ("green", "--")],
        ylabel="GB/s",
        plot_name="vector_add",
        args={},
    )
)
def benchmark_vector_add(n, provider):
    dtype = torch.float32
    a = torch.randn(n, device="cuda", dtype=dtype)
    b = torch.randn(n, device="cuda", dtype=dtype)

    quantiles = [0.5, 0.2, 0.8]

    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: vector_add(a, b), warmup=25, rep=100, quantiles=quantiles
        )
    else:
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: a + b, warmup=25, rep=100, quantiles=quantiles
        )

    # 3 tensors × n elements × 4 bytes (fp32)
    gb = 3 * n * a.element_size() * 1e-9
    return gb / (ms * 1e-3), gb / (max_ms * 1e-3), gb / (min_ms * 1e-3)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_vector_add()
    benchmark_vector_add.run(print_data=True, show_plots=True)
