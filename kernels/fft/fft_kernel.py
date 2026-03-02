"""
Kernel: FFT (Cooley-Tukey Radix-2 DIT)
Category: fft
Complexity: O(N log N) per row
Memory bound: No — compute bound at large N
PyTorch equivalent: torch.fft.fft(x)
References: https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm
            https://triton-lang.org/main/getting-started/tutorials/
"""

import torch
import triton
import triton.language as tl
import math


# ── Kernel ────────────────────────────────────────────────────────────────────

@triton.jit
def fft_kernel(
    re_ptr, im_ptr,          # input:  real and imaginary parts (fp32, [B, N])
    out_re_ptr, out_im_ptr,  # output: real and imaginary parts (fp32, [B, N])
    N: tl.constexpr,         # FFT size — must be power of 2, <= 8192
    BLOCK_SIZE: tl.constexpr,  # == N
):
    """
    One program per batch row. All log2(N) butterfly stages in global memory.

    Loop strategy: tl.static_range(13) unrolls 13 iterations at compile time.
    Each iteration uses a Python-level `if (1 << b) < N:` guard — since both
    sides are Python ints at trace time (literal shift of a static loop var,
    constexpr N), Triton evaluates the branch at compile time and emits code
    only for active stages. No constexpr arithmetic inside the kernel needed.
    """
    batch_id = tl.program_id(0)

    # Base pointers for this batch row
    base = batch_id * N
    offs = tl.arange(0, BLOCK_SIZE)  # [0, N)
    mask = offs < N

    # Load inputs as fp32
    re = tl.load(re_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    im = tl.load(im_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)

    # ── Bit-reversal permutation ──────────────────────────────────────────────
    # N is a Python int inside @triton.jit (proven: N.to() fails with 'int' error).
    # N.bit_length() - 1 == log2(N) for powers of 2, entirely Python at compile time.
    # tl.constexpr(python_int) satisfies tl.static_range's isinstance(end, constexpr).
    LOG2_N = N.bit_length() - 1                    # Python int at compile time
    rev = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    src = offs.to(tl.int32)

    for b in tl.static_range(tl.constexpr(LOG2_N)):  # exactly log2(N) iterations
        bit_val = src & 1
        rev = (rev << 1) | bit_val
        src = src >> 1

    # Gather: re_br[rev[i]] = re[i]  →  re_br[i] = re[rev^{-1}[i]]
    # Since bit-reversal is its own inverse, re_br[i] = re[rev[i]].
    re_br = tl.load(re_ptr + base + rev, mask=mask, other=0.0).to(tl.float32)
    im_br = tl.load(im_ptr + base + rev, mask=mask, other=0.0).to(tl.float32)

    # ── Butterfly stages ──────────────────────────────────────────────────────
    # Stage s has butterfly span = 2^(s+1), half-span = 2^s.
    # All N butterflies in a stage run in parallel (one lane per element).
    # Lanes communicate through global memory: each stage reads out_{re,im}
    # written by the previous stage. This costs 2 × log2(N) HBM round-trips
    # total — more than cuFFT's shared-memory approach, but correct.
    #
    # Write bit-reversed input as stage-0 input.
    tl.store(out_re_ptr + base + offs, re_br, mask=mask)
    tl.store(out_im_ptr + base + offs, im_br, mask=mask)

    # Butterfly loop: exactly log2(N) iterations (no if-guard needed)
    for s in tl.static_range(tl.constexpr(LOG2_N)):
        half = 1 << s          # Python int (s from static_range)
        span = half << 1       # Python int

        pos = offs % span
        is_top = pos < half
        partner = tl.where(is_top, offs + half, offs - half)

        # Twiddle angle: compute scale as a pure Python float BEFORE any tensor ops.
        # half and span are Python ints at this point → no tensor conversion needed.
        k = tl.where(is_top, pos, pos - half).to(tl.float32)
        scale = -2.0 * math.pi / span   # Python float / Python int → Python float
        angle = scale * k               # Python float × fp32 tensor → fp32 tensor
        w_re = tl.cos(angle)
        w_im = tl.sin(angle)

        # Load this stage's values
        cur_re = tl.load(out_re_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        cur_im = tl.load(out_im_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        par_re = tl.load(out_re_ptr + base + partner, mask=mask, other=0.0).to(tl.float32)
        par_im = tl.load(out_im_ptr + base + partner, mask=mask, other=0.0).to(tl.float32)

        # Twiddle * partner
        tw_re = w_re * par_re - w_im * par_im
        tw_im = w_re * par_im + w_im * par_re

        # Butterfly: top = cur + tw,  bottom = cur - tw
        new_re = tl.where(is_top, cur_re + tw_re, cur_re - tw_re)
        new_im = tl.where(is_top, cur_im + tw_im, cur_im - tw_im)

        tl.store(out_re_ptr + base + offs, new_re, mask=mask)
        tl.store(out_im_ptr + base + offs, new_im, mask=mask)


# ── Wrapper ───────────────────────────────────────────────────────────────────

def fft(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the 1-D FFT of each row of x using Triton.

    Args:
        x: Real-valued input tensor of shape (B, N) or (N,), fp32 or fp16.
           N must be a power of 2, N <= 8192.

    Returns:
        Complex-valued tensor of shape (..., N), dtype=torch.complex64.
    """
    squeeze = x.dim() == 1
    if squeeze:
        x = x.unsqueeze(0)

    assert x.is_cuda, "Input must be a CUDA tensor"
    x = x.contiguous().to(torch.float32)

    B, N = x.shape
    assert N > 0 and (N & (N - 1)) == 0, f"N must be a power of 2, got {N}"
    assert N <= 8192, f"N={N} exceeds max supported size (8192)"

    re = x.clone()
    im = torch.zeros_like(x)
    out_re = torch.empty_like(x)
    out_im = torch.empty_like(x)

    grid = (B,)
    fft_kernel[grid](
        re, im,
        out_re, out_im,
        N=N,
        BLOCK_SIZE=N,
    )

    result = torch.complex(out_re, out_im)
    return result.squeeze(0) if squeeze else result


# ── Test ──────────────────────────────────────────────────────────────────────

def test_fft():
    print("Testing fft_kernel...")

    # Power-of-2 sizes, including non-trivial ones
    for N in [64, 128, 256, 512, 1024, 2048, 4096]:
        x = torch.randn(16, N, device="cuda", dtype=torch.float32)

        ref = torch.fft.fft(x)          # PyTorch reference (cuFFT)
        got = fft(x)

        torch.testing.assert_close(
            got.real, ref.real, atol=1e-3, rtol=1e-3,
            msg=f"Real mismatch at N={N}"
        )
        torch.testing.assert_close(
            got.imag, ref.imag, atol=1e-3, rtol=1e-3,
            msg=f"Imag mismatch at N={N}"
        )
        print(f"  N={N:5d}  max_err_re={( got.real - ref.real).abs().max():.2e}"
              f"  max_err_im={(got.imag - ref.imag).abs().max():.2e}  PASS")

    # Single vector (1-D input)
    x1d = torch.randn(256, device="cuda")
    ref1d = torch.fft.fft(x1d)
    got1d = fft(x1d)
    torch.testing.assert_close(got1d.real, ref1d.real, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(got1d.imag, ref1d.imag, atol=1e-3, rtol=1e-3)
    print("  1-D input (N=256)  PASS")

    print("All tests passed.")


# ── Benchmark ─────────────────────────────────────────────────────────────────

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[64, 128, 256, 512, 1024, 2048, 4096],
        x_log=True,
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton FFT", "torch.fft.fft (cuFFT)"],
        styles=[("blue", "-"), ("green", "--")],
        ylabel="GFLOPS",
        plot_name="fft_benchmark",
        args={"B": 64},
    )
)
def benchmark_fft(N, B, provider):
    x = torch.randn(B, N, device="cuda", dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]

    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: fft(x), warmup=25, rep=100, quantiles=quantiles
        )
    else:
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch.fft.fft(x), warmup=25, rep=100, quantiles=quantiles
        )

    # GFLOPS = 5 * N * log2(N) * B / (ms * 1e-3) / 1e9
    gflops = lambda ms: (5 * B * N * math.log2(N) * 1e-9) / (ms * 1e-3)
    return gflops(ms), gflops(max_ms), gflops(min_ms)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_fft()
    benchmark_fft.run(
        print_data=True,
        show_plots=True,
        save_path="benchmarks/results/fft",
    )
