"""
Kernel:   fused_elementwise
Category: elementwise
Complexity: O(n)
Memory bound: Yes
PyTorch equivalent: activation(x + bias)  [two separate ops]
References:
  - Dao et al. FlashAttention (motivation for fusion): https://arxiv.org/abs/2205.14135

The key idea: fusing bias_add + activation into one kernel eliminates the
intermediate tensor write/read between the two ops.

HBM traffic comparison for out = activation(x + bias):
  Unfused (2 kernels): read x, read bias, WRITE tmp, READ tmp, write out  → 5n bytes
  Fused   (1 kernel):  read x, read bias,                   write out     → 3n bytes

Expected speedup at large n (bandwidth-limited): ~5/3 ≈ 1.67×
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# ── 1. Triton kernels ─────────────────────────────────────────────────────────

@triton.jit
def fused_bias_relu_kernel(
    x_ptr, bias_ptr, out_ptr, n: int, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    b = tl.load(bias_ptr + offs, mask=mask)
    z = x + b
    tl.store(out_ptr + offs, tl.where(z > 0.0, z, 0.0), mask=mask)


@triton.jit
def fused_bias_gelu_kernel(
    x_ptr, bias_ptr, out_ptr, n: int, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    b = tl.load(bias_ptr + offs, mask=mask)
    z = (x + b).to(tl.float32)
    c = 0.7978845608028654  # sqrt(2 / pi)
    inner = c * (z + 0.044715 * z * z * z)
    tanh_inner = 2.0 / (1.0 + tl.math.exp(-2.0 * inner)) - 1.0
    out = 0.5 * z * (1.0 + tanh_inner)
    tl.store(out_ptr + offs, out.to(x.dtype), mask=mask)


@triton.jit
def fused_bias_silu_kernel(
    x_ptr, bias_ptr, out_ptr, n: int, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    b = tl.load(bias_ptr + offs, mask=mask)
    z = (x + b).to(tl.float32)
    out = z / (1.0 + tl.math.exp(-z))
    tl.store(out_ptr + offs, out.to(x.dtype), mask=mask)


# ── 2. Python wrappers ────────────────────────────────────────────────────────

def _launch(kernel, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and bias.is_cuda, "Inputs must be on CUDA"
    assert x.is_contiguous() and bias.is_contiguous(), "Inputs must be contiguous"
    assert x.shape == bias.shape, "x and bias must have the same shape"
    n = x.numel()
    out = torch.empty_like(x)
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    kernel[grid](x, bias, out, n, BLOCK_SIZE=BLOCK_SIZE)
    return out


def fused_bias_relu(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return _launch(fused_bias_relu_kernel, x, bias)


def fused_bias_gelu(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return _launch(fused_bias_gelu_kernel, x, bias)


def fused_bias_silu(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return _launch(fused_bias_silu_kernel, x, bias)


# ── 3. Correctness tests ──────────────────────────────────────────────────────

def test_fused_elementwise():
    sizes = [1, 127, 128, 1024, 1025, 65536, 100_000]
    dtypes = [torch.float32, torch.float16]

    for dtype in dtypes:
        for n in sizes:
            x    = torch.randn(n, device="cuda", dtype=dtype)
            bias = torch.randn(n, device="cuda", dtype=dtype)

            torch.testing.assert_close(
                fused_bias_relu(x, bias),
                torch.relu(x + bias),
            )
            torch.testing.assert_close(
                fused_bias_gelu(x, bias),
                F.gelu(x + bias, approximate="tanh"),
                rtol=1e-3, atol=1e-3,
            )
            torch.testing.assert_close(
                fused_bias_silu(x, bias),
                F.silu(x + bias),
            )

    print("test_fused_elementwise: PASSED")


# ── 4. Benchmarks ─────────────────────────────────────────────────────────────
#
# Three-way comparison:
#   - Triton fused:        our hand-written kernel (1 pass, 3n bytes)
#   - PyTorch unfused:     two separate ops (2 passes, 5n bytes)
#   - torch.compile fused: TorchInductor auto-fuses into a Triton kernel
#
# Metric: effective GB/s using minimum possible data movement (3n bytes).
# Unfused PyTorch moves 5n bytes; its effective GB/s will be ~3/5 of fused.

# torch.compile fuses the bias_add + activation into one kernel automatically.
_compiled_bias_relu = torch.compile(lambda x, b: torch.relu(x + b))
_compiled_bias_gelu = torch.compile(lambda x, b: F.gelu(x + b, approximate="tanh"))
_compiled_bias_silu = torch.compile(lambda x, b: F.silu(x + b))


def _make_benchmark(triton_fn, torch_fn, compiled_fn, name):
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["n"],
            x_vals=[2**i for i in range(12, 28)],
            x_log=True,
            line_arg="provider",
            line_vals=["triton_fused", "torch_compiled", "torch_unfused"],
            line_names=["Triton (fused)", "torch.compile (fused)", "PyTorch (unfused)"],
            styles=[("blue", "-"), ("orange", "-."), ("green", "--")],
            ylabel="Effective GB/s",
            plot_name=name,
            args={},
        )
    )
    def bench(n, provider):
        x    = torch.randn(n, device="cuda", dtype=torch.float32)
        bias = torch.randn(n, device="cuda", dtype=torch.float32)
        quantiles = [0.5, 0.2, 0.8]

        if provider == "triton_fused":
            fn = lambda: triton_fn(x, bias)
        elif provider == "torch_compiled":
            fn = lambda: compiled_fn(x, bias)
        else:
            fn = lambda: torch_fn(x + bias)

        ms, min_ms, max_ms = triton.testing.do_bench(
            fn, warmup=25, rep=100, quantiles=quantiles
        )

        # 3 tensors × n elements × 4 bytes: 2 reads (x, bias) + 1 write (out)
        gb = 3 * n * x.element_size() * 1e-9
        return gb / (ms * 1e-3), gb / (max_ms * 1e-3), gb / (min_ms * 1e-3)

    return bench


benchmark_fused_bias_relu = _make_benchmark(
    fused_bias_relu, torch.relu, _compiled_bias_relu, "fused_bias_relu"
)
benchmark_fused_bias_gelu = _make_benchmark(
    fused_bias_gelu, lambda x: F.gelu(x, approximate="tanh"), _compiled_bias_gelu, "fused_bias_gelu"
)
benchmark_fused_bias_silu = _make_benchmark(
    fused_bias_silu, F.silu, _compiled_bias_silu, "fused_bias_silu"
)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_fused_elementwise()
    benchmark_fused_bias_relu.run(print_data=True, show_plots=True)
    benchmark_fused_bias_gelu.run(print_data=True, show_plots=True)
    benchmark_fused_bias_silu.run(print_data=True, show_plots=True)
