"""
Kernel: FFT (Cooley-Tukey Radix-2 DIT)
Category: fft
Complexity: O(N log N) per row
Memory bound: No — compute bound at large N
PyTorch equivalent: torch.fft.fft(x)
References: https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm
"""

import torch
import triton
import triton.language as tl
import math


# ── Kernel ────────────────────────────────────────────────────────────────────

@triton.jit
def fft_kernel(
    re_ptr, im_ptr,            # in-place buffer (bit-reversed input on entry)
    N,                          # row length (runtime int)
    LOG2_N: tl.constexpr,      # log2(N); controls compile-time loop unroll count
    BLOCK_SIZE: tl.constexpr,  # == N
):
    """
    All log2(N) Cooley-Tukey butterfly stages in a single kernel launch.

    Design rationale:
    - for s in range(LOG2_N) is compile-time unrolled (LOG2_N is tl.constexpr),
      so s, half, and angle_scale are Python-level constants at JIT trace time.
    - In-place operation: each stage loads all N elements, computes butterflies,
      stores all N results back to the same buffer.
    - tl.debug_barrier() between stages is bar.sync 0 in PTX (__syncthreads__) —
      required for stages where half >= 32 (partners span warp boundaries).
    - vs per-stage-kernel design: eliminates log2(N)-1 kernel launch round-trips
      (~5us each). Intermediate data hits L2 cache for N <= ~256K complex fp32.
    """
    batch_id = tl.program_id(0)
    base = batch_id * N
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    for s in range(LOG2_N):                      # compile-time unrolled
        half = 1 << s                             # Python int at trace time
        span = half << 1                          # Python int
        angle_scale = -2.0 * math.pi / span      # Python float (constexpr)

        pos     = offs % span
        is_top  = pos < half
        partner = tl.where(is_top, offs + half, offs - half)

        k     = tl.where(is_top, pos, pos - half).to(tl.float32)
        angle = angle_scale * k
        w_re  = tl.cos(angle)
        w_im  = tl.sin(angle)

        cur_re = tl.load(re_ptr + base + offs,    mask=mask, other=0.0).to(tl.float32)
        cur_im = tl.load(im_ptr + base + offs,    mask=mask, other=0.0).to(tl.float32)
        par_re = tl.load(re_ptr + base + partner, mask=mask, other=0.0).to(tl.float32)
        par_im = tl.load(im_ptr + base + partner, mask=mask, other=0.0).to(tl.float32)

        even_re = tl.where(is_top, cur_re, par_re)
        even_im = tl.where(is_top, cur_im, par_im)
        odd_re  = tl.where(is_top, par_re, cur_re)
        odd_im  = tl.where(is_top, par_im, cur_im)

        tw_re = w_re * odd_re - w_im * odd_im
        tw_im = w_re * odd_im + w_im * odd_re

        new_re = tl.where(is_top, even_re + tw_re, even_re - tw_re)
        new_im = tl.where(is_top, even_im + tw_im, even_im - tw_im)

        tl.store(re_ptr + base + offs, new_re, mask=mask)
        tl.store(im_ptr + base + offs, new_im, mask=mask)
        tl.debug_barrier()


# ── Wrapper ───────────────────────────────────────────────────────────────────

def fft(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the 1-D DFT of each row of x using Triton.

    Args:
        x: Real-valued tensor of shape (B, N) or (N,), fp32 or fp16.
           N must be a power of 2, N <= 32768.

    Returns:
        Complex tensor of shape (..., N), dtype=torch.complex64.
    """
    squeeze = x.dim() == 1
    if squeeze:
        x = x.unsqueeze(0)

    assert x.is_cuda, "Input must be a CUDA tensor"
    x = x.contiguous().to(torch.float32)

    B, N = x.shape
    assert N > 0 and (N & (N - 1)) == 0, f"N must be a power of 2, got {N}"
    assert N <= 32768, f"N={N} exceeds max supported size (32768)"

    log2_n = int(math.log2(N))

    # Bit-reversal permutation (CPU, vectorised) — unchanged from prior design
    indices = torch.arange(N, dtype=torch.int64)
    rev = torch.zeros(N, dtype=torch.int64)
    tmp = indices.clone()
    for _ in range(log2_n):
        rev = (rev << 1) | (tmp & 1)
        tmp = tmp >> 1

    # Apply bit-reversal to input; imaginary part starts at zero
    buf_re = x[:, rev].contiguous().clone()
    buf_im = torch.zeros(B, N, device=x.device)

    # Single kernel launch — all stages run inside the kernel
    fft_kernel[(B,)](buf_re, buf_im, N, LOG2_N=log2_n, BLOCK_SIZE=N)

    result = torch.complex(buf_re, buf_im)
    return result.squeeze(0) if squeeze else result


# ── Test ──────────────────────────────────────────────────────────────────────

def test_fft():
    print("Testing fft (fft_kernel — single launch, all stages)...")

    for N in [64, 128, 256, 512, 1024, 2048, 4096, 8192]:
        x = torch.randn(16, N, device="cuda", dtype=torch.float32)
        ref = torch.fft.fft(x)
        got = fft(x)
        torch.testing.assert_close(got.real, ref.real, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(got.imag, ref.imag, atol=1e-3, rtol=1e-3)
        print(f"  N={N:5d}  max_err_re={(got.real - ref.real).abs().max():.2e}"
              f"  max_err_im={(got.imag - ref.imag).abs().max():.2e}  PASS")

    # 1-D input
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
        x_vals=[64, 128, 256, 512, 1024, 2048, 4096, 8192],
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
