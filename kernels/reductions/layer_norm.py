"""
Kernel:   layer_norm
Category: reductions
Complexity: O(M × N)
Memory bound: Yes
PyTorch equivalent: torch.nn.functional.layer_norm(x, normalized_shape, weight, bias)
References:
  - Ba et al. "Layer Normalization" (2016): https://arxiv.org/abs/1607.06450
  - Triton layer norm tutorial: https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html

How it works:
One Triton program per row. Each program:
  1. Loads the full row (padded to BLOCK_SIZE with 0.0)
  2. Computes the row mean: μ = Σxᵢ / N
  3. Computes the row variance: σ² = Σ(xᵢ - μ)² / N
     — OOB (padded) positions are zeroed in the diff to avoid counting (-μ)² for
       non-existent elements
  4. Normalises: x̂ᵢ = (xᵢ - μ) / √(σ² + ε)
  5. Applies the affine transform: yᵢ = γᵢ · x̂ᵢ + βᵢ  (γ = weight, β = bias)
  6. Writes the output

BLOCK_SIZE = next_power_of_2(N) so the entire row fits in one program — the same
strategy as softmax, fusing all steps into a single kernel.

HBM traffic: read x (M×N) + read weight (N) + read bias (N) + write y (M×N).
For large M the weight/bias reads (N elements each) are negligible, so we account
for 2×M×N bytes.
Metric: GB/s = (2 × M × N × element_size × 1e-9) / (ms × 1e-3)

Input shape: (M, N) — M rows, each of length N (the "normalised shape" is N).
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# ── 1. Triton kernel ──────────────────────────────────────────────────────────

@triton.jit
def layer_norm_kernel(
    x_ptr, out_ptr,
    weight_ptr, bias_ptr,
    n_cols: int,
    eps: float,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols

    # Load row; pad OOB positions with 0.0 so they don't skew the mean
    x = tl.load(x_ptr + row * n_cols + offs, mask=mask, other=0.0)
    x = x.to(tl.float32)

    # Mean — OOB zeros contribute 0 to the sum, so dividing by n_cols is correct
    mean = tl.sum(x, axis=0) / n_cols

    # Variance — zero OOB positions in diff so (-mean)^2 isn't counted for them
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / n_cols

    # Normalise
    inv_std = 1.0 / tl.math.sqrt(var + eps)
    x_hat = diff * inv_std

    # Affine transform: y = weight * x_hat + bias
    weight = tl.load(weight_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    bias   = tl.load(bias_ptr   + offs, mask=mask, other=0.0).to(tl.float32)
    out = weight * x_hat + bias

    tl.store(out_ptr + row * n_cols + offs, out, mask=mask)


# ── 2. Python wrapper ─────────────────────────────────────────────────────────

def layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    assert x.is_cuda and weight.is_cuda and bias.is_cuda, "All tensors must be on CUDA"
    assert x.is_contiguous() and weight.is_contiguous() and bias.is_contiguous(), \
        "All tensors must be contiguous"
    assert x.ndim == 2, "Input must be 2D (M rows × N cols)"
    M, N = x.shape
    assert weight.shape == (N,), f"weight must be shape ({N},)"
    assert bias.shape   == (N,), f"bias must be shape ({N},)"

    BLOCK_SIZE = triton.next_power_of_2(N)
    out = torch.empty_like(x)
    layer_norm_kernel[(M,)](x, out, weight, bias, N, eps, BLOCK_SIZE=BLOCK_SIZE)
    return out


# ── 3. Correctness tests ──────────────────────────────────────────────────────

def test_layer_norm():
    configs = [
        (1, 128), (1, 127), (128, 256), (512, 512),
        (1024, 1024), (64, 4096),
    ]
    dtypes = [torch.float32]  # kernel always outputs fp32

    for dtype in dtypes:
        for M, N in configs:
            x      = torch.randn(M, N, device="cuda", dtype=dtype)
            weight = torch.randn(N,    device="cuda", dtype=dtype)
            bias   = torch.randn(N,    device="cuda", dtype=dtype)
            ref = F.layer_norm(x, (N,), weight, bias)
            got = layer_norm(x, weight, bias)
            torch.testing.assert_close(got, ref, rtol=1e-3, atol=1e-3)

    print("test_layer_norm: PASSED")


# ── 4. Benchmarks ─────────────────────────────────────────────────────────────
#
# Fix M=1024 rows, vary N (hidden dimension / sequence width).
# This mirrors transformer usage: layer norm is applied over the hidden dimension
# (e.g. N=512, 1024, 2048, 4096) across a batch of token positions.

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[2**i for i in range(7, 14)],   # 128 → 8192 cols
        x_log=True,
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton", "PyTorch"],
        styles=[("blue", "-"), ("green", "--")],
        ylabel="GB/s",
        plot_name="layer_norm",
        args={"M": 1024},
    )
)
def benchmark_layer_norm(M, N, provider):
    x      = torch.randn(M, N, device="cuda", dtype=torch.float32)
    weight = torch.ones( N,    device="cuda", dtype=torch.float32)
    bias   = torch.zeros(N,    device="cuda", dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]

    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: layer_norm(x, weight, bias), warmup=25, rep=100, quantiles=quantiles
        )
    else:
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: F.layer_norm(x, (N,), weight, bias), warmup=25, rep=100, quantiles=quantiles
        )

    # Read M×N (x) + write M×N (out); weight/bias reads (N each) negligible vs M×N
    gb = 2 * M * N * x.element_size() * 1e-9
    return gb / (ms * 1e-3), gb / (max_ms * 1e-3), gb / (min_ms * 1e-3)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_layer_norm()
    benchmark_layer_norm.run(print_data=True, show_plots=True)
