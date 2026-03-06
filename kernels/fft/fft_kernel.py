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
def butterfly_stage_kernel(
    in_re_ptr, in_im_ptr,     # read-only input for this stage
    out_re_ptr, out_im_ptr,   # write-only output for this stage (separate buffer)
    N,                        # row length (runtime int)
    half: tl.constexpr,       # half-span = 2^s for stage s; passed from Python
    angle_scale: tl.constexpr,  # = -2*pi / span; passed from Python as float
    BLOCK_SIZE: tl.constexpr,   # == N
):
    """
    One Cooley-Tukey butterfly stage. Called log2(N) times from the wrapper.

    Design rationale:
    - All loop-bound arithmetic (half, angle_scale) is computed in pure Python
      in the wrapper and passed as tl.constexpr — no JIT-level math needed.
    - Separate in/out buffers (ping-pong) avoid the cross-warp race condition
      that arises when half >= 32 and adjacent butterfly pairs span warp boundaries.
    - The kernel is recompiled once per unique (half, BLOCK_SIZE) pair, so for
      N=4096 there are 12 compiled variants — one per stage.
    """
    batch_id = tl.program_id(0)
    base = batch_id * N
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    # Which butterfly group and which half is this element in?
    span = half + half                                   # = 2^(s+1), constexpr
    pos = offs % span                                    # position within group
    is_top = pos < half                                  # top (even) half?
    partner = tl.where(is_top, offs + half, offs - half)

    # Twiddle factor W = exp(-2πi * k / span)
    k = tl.where(is_top, pos, pos - half).to(tl.float32)
    angle = angle_scale * k   # constexpr float × fp32 tensor → fp32 tensor
    w_re = tl.cos(angle)
    w_im = tl.sin(angle)

    # Load from in-buffer (all loads before any stores → no race between stages)
    cur_re = tl.load(in_re_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    cur_im = tl.load(in_im_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    par_re = tl.load(in_re_ptr + base + partner, mask=mask, other=0.0).to(tl.float32)
    par_im = tl.load(in_im_ptr + base + partner, mask=mask, other=0.0).to(tl.float32)

    # Resolve even (top-half) and odd (bottom-half) for this butterfly pair.
    # Top element:    cur=even, par=odd.
    # Bottom element: cur=odd,  par=even.
    # Twiddle always multiplies the odd element, so we must select correctly.
    even_re = tl.where(is_top, cur_re, par_re)
    even_im = tl.where(is_top, cur_im, par_im)
    odd_re  = tl.where(is_top, par_re, cur_re)
    odd_im  = tl.where(is_top, par_im, cur_im)

    # Complex multiply: twiddle × odd
    tw_re = w_re * odd_re - w_im * odd_im
    tw_im = w_re * odd_im + w_im * odd_re

    # Butterfly: top = even + tw,  bottom = even - tw
    new_re = tl.where(is_top, even_re + tw_re, even_re - tw_re)
    new_im = tl.where(is_top, even_im + tw_im, even_im - tw_im)

    # Write to out-buffer (disjoint from in-buffer → no intra-stage race)
    tl.store(out_re_ptr + base + offs, new_re, mask=mask)
    tl.store(out_im_ptr + base + offs, new_im, mask=mask)


# ── Wrapper ───────────────────────────────────────────────────────────────────

def fft(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the 1-D DFT of each row of x using Triton.

    Args:
        x: Real-valued tensor of shape (B, N) or (N,), fp32 or fp16.
           N must be a power of 2, N <= 8192.

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
    assert N <= 8192, f"N={N} exceeds max supported size (8192)"

    log2_n = int(math.log2(N))

    # ── Bit-reversal permutation (CPU, vectorised) ────────────────────────────
    # The bit-reversal for a given N is a fixed permutation — compute it once
    # on CPU and apply with PyTorch advanced indexing.
    indices = torch.arange(N, dtype=torch.int64)
    rev = torch.zeros(N, dtype=torch.int64)
    tmp = indices.clone()
    for _ in range(log2_n):
        rev = (rev << 1) | (tmp & 1)
        tmp = tmp >> 1

    # Apply bit-reversal: buf0 starts as the bit-reversed input
    buf = [
        [x[:, rev].contiguous().clone(), torch.zeros(B, N, device=x.device)],
        [torch.empty(B, N, device=x.device), torch.empty(B, N, device=x.device)],
    ]

    # ── log2(N) butterfly stages ──────────────────────────────────────────────
    # Each stage reads from buf[s % 2] and writes to buf[(s+1) % 2].
    # All arithmetic on half/angle_scale is done in Python (never inside JIT).
    grid = (B,)
    for s in range(log2_n):
        src = s % 2
        dst = (s + 1) % 2
        half = 1 << s
        span = half << 1
        angle_scale = -2.0 * math.pi / span  # Python float, passed as constexpr

        butterfly_stage_kernel[grid](
            buf[src][0], buf[src][1],
            buf[dst][0], buf[dst][1],
            N,
            half=half,
            angle_scale=angle_scale,
            BLOCK_SIZE=N,
        )

    final_re, final_im = buf[log2_n % 2]
    result = torch.complex(final_re, final_im)
    return result.squeeze(0) if squeeze else result


# ── Test ──────────────────────────────────────────────────────────────────────

def test_fft():
    print("Testing fft (butterfly_stage_kernel)...")

    for N in [64, 128, 256, 512, 1024, 2048, 4096]:
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
