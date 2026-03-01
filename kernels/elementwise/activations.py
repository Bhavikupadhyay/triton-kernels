"""
Kernel:   activations
Category: elementwise
Complexity: O(n)
Memory bound: Yes
PyTorch equivalents:
  torch.relu(x)
  torch.nn.functional.gelu(x, approximate='tanh')
  torch.nn.functional.silu(x)
References:
  - GELU: https://arxiv.org/abs/1606.08415
  - SiLU/Swish: https://arxiv.org/abs/1710.05941

Note on GELU: we implement the tanh approximation
  0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
which matches torch.nn.functional.gelu(x, approximate='tanh').
The exact erf-based version differs by <0.1% in practice.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# ── 1. Triton kernels ─────────────────────────────────────────────────────────

@triton.jit
def relu_kernel(x_ptr, out_ptr, n: int, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, tl.where(x > 0.0, x, 0.0), mask=mask)


@triton.jit
def gelu_kernel(x_ptr, out_ptr, n: int, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    # tanh approximation: matches F.gelu(x, approximate='tanh')
    c = 0.7978845608028654  # sqrt(2 / pi)
    inner = c * (x + 0.044715 * x * x * x)
    out = 0.5 * x * (1.0 + tl.math.tanh(inner))
    tl.store(out_ptr + offs, out, mask=mask)


@triton.jit
def silu_kernel(x_ptr, out_ptr, n: int, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    # SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
    out = x / (1.0 + tl.math.exp(-x))
    tl.store(out_ptr + offs, out, mask=mask)


# ── 2. Python wrappers ────────────────────────────────────────────────────────

def _launch(kernel, x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "Input must be on CUDA"
    assert x.is_contiguous(), "Input must be contiguous"
    n = x.numel()
    out = torch.empty_like(x)
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    kernel[grid](x, out, n, BLOCK_SIZE=BLOCK_SIZE)
    return out


def relu(x: torch.Tensor) -> torch.Tensor:
    return _launch(relu_kernel, x)


def gelu(x: torch.Tensor) -> torch.Tensor:
    return _launch(gelu_kernel, x)


def silu(x: torch.Tensor) -> torch.Tensor:
    return _launch(silu_kernel, x)


# ── 3. Correctness tests ──────────────────────────────────────────────────────

def test_activations():
    sizes = [1, 127, 128, 1024, 1025, 65536, 100_000]
    dtypes = [torch.float32, torch.float16]

    for dtype in dtypes:
        for n in sizes:
            x = torch.randn(n, device="cuda", dtype=dtype)

            torch.testing.assert_close(relu(x), torch.relu(x))
            torch.testing.assert_close(
                gelu(x), F.gelu(x, approximate="tanh"),
                rtol=1e-3, atol=1e-3,
            )
            torch.testing.assert_close(silu(x), F.silu(x))

    print("test_activations: PASSED")


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
        # 2 tensors × n elements × 4 bytes (1 read, 1 write)
        gb = 2 * n * x.element_size() * 1e-9
        return gb / (ms * 1e-3), gb / (max_ms * 1e-3), gb / (min_ms * 1e-3)

    return bench


benchmark_relu = _make_benchmark(relu, torch.relu, "relu")
benchmark_gelu = _make_benchmark(gelu, lambda x: F.gelu(x, approximate="tanh"), "gelu")
benchmark_silu = _make_benchmark(silu, F.silu, "silu")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_activations()
    benchmark_relu.run(print_data=True, show_plots=True)
    benchmark_gelu.run(print_data=True, show_plots=True)
    benchmark_silu.run(print_data=True, show_plots=True)
