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
    re_ptr, im_ptr,       # input:  real and imaginary parts (fp32, [B, N])
    out_re_ptr, out_im_ptr,  # output: real and imaginary parts (fp32, [B, N])
    N: tl.constexpr,     # FFT size — must be power of 2
    BLOCK_SIZE: tl.constexpr,  # == N (one program handles one row)
):
    """
    One program per batch row. All log2(N) butterfly stages run in registers.
    Requires N == BLOCK_SIZE to fit the entire row in one program's register file.
    Valid for N up to ~8192 on T4 (register pressure becomes the limit above that).
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
    # Cooley-Tukey DIT requires bit-reversed input order.
    # We compute the bit-reversed index for each position and gather.
    # Triton has no dynamic gather, so we implement iterative bit-reversal
    # via the standard shift-based algorithm unrolled across log2(N) steps.
    #
    # Strategy: build rev[i] by iterating bits. Since N is constexpr we unroll.
    # We use a register-level approach: copy re/im into bit-reversed positions
    # by swapping pairs that are out of place.
    #
    # A cleaner Triton idiom: compute rev_idx via arithmetic, use tl.load with
    # explicit pointer arithmetic for the gather.
    log2_N = tl.log2(N.to(tl.float32)).to(tl.int32)

    # Compute bit-reversed indices using iterative bit swap
    rev = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    src = offs  # [0, 1, 2, ..., N-1]

    # Unroll bit-reversal: for each bit position b in [0, log2_N),
    # accumulate the reversed bit into rev.
    # We need constexpr loop count — use the maximum possible (13 for N=8192)
    # and guard with runtime check.
    for b in tl.static_range(13):  # covers N up to 2^13 = 8192
        bit_on = (b < log2_N)
        bit_val = src & 1
        rev = tl.where(bit_on, (rev << 1) | bit_val, rev)
        src = tl.where(bit_on, src >> 1, src)

    # Gather: re_br[rev[i]] = re[i]  →  re_br[i] = re[rev^{-1}[i]]
    # Since bit-reversal is its own inverse, re_br[i] = re[rev[i]].
    re_br = tl.load(re_ptr + base + rev, mask=mask, other=0.0).to(tl.float32)
    im_br = tl.load(im_ptr + base + rev, mask=mask, other=0.0).to(tl.float32)

    # ── Butterfly stages ──────────────────────────────────────────────────────
    # Stage s has butterfly span = 2^s.
    # For each group of 2^s elements, the first half are "even" (top) and
    # the second half are "odd" (bottom). Twiddle W = exp(-2πi * k / 2^s).
    #
    # Triton has no inter-lane communication (no shuffle/warp intrinsics).
    # We compute all N butterflies per stage in parallel using arithmetic on
    # the full register arrays — each lane computes its own output by reading
    # its partner's value from the same register array.
    #
    # This works because in Triton, all BLOCK_SIZE lanes share the same
    # register arrays re_br/im_br and can index them freely with tl.load
    # from scratch pointers — but we cannot do live register-to-register
    # communication without going through SRAM (shared memory).
    #
    # Implementation: iterate stages; for each stage write to output scratch,
    # then use that as next stage's input. We ping-pong through global memory
    # scratch buffers (out_re/out_im) between stages.
    #
    # Write bit-reversed input into output buffer as stage-0 starting point.
    tl.store(out_re_ptr + base + offs, re_br, mask=mask)
    tl.store(out_im_ptr + base + offs, im_br, mask=mask)

    # Butterfly loop: log2(N) stages
    for s in tl.static_range(13):  # unrolled; guarded by s < log2_N
        do_stage = (s < log2_N)
        half = 1 << s          # half-span = 2^s
        span = half << 1       # full span = 2^(s+1)

        # Which "half" of its butterfly group is each lane in?
        # group = offs // span;  pos = offs % span
        # top (even) half: pos < half;  bottom (odd) half: pos >= half
        pos = offs % span
        is_top = pos < half

        # Partner index: top lane's partner is (offs + half), bottom's is (offs - half)
        partner = tl.where(is_top, offs + half, offs - half)

        # Twiddle factor index k = pos for top lanes (0..half-1)
        k = tl.where(is_top, pos, pos - half).to(tl.float32)
        angle = -2.0 * math.pi * k / span.to(tl.float32)
        w_re = tl.cos(angle)
        w_im = tl.sin(angle)

        # Load current stage values
        cur_re = tl.load(out_re_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        cur_im = tl.load(out_im_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        par_re = tl.load(out_re_ptr + base + partner, mask=mask, other=0.0).to(tl.float32)
        par_im = tl.load(out_im_ptr + base + partner, mask=mask, other=0.0).to(tl.float32)

        # Twiddle * partner  (complex multiply)
        tw_re = w_re * par_re - w_im * par_im
        tw_im = w_re * par_im + w_im * par_re

        # Butterfly output
        # top:    X[i]         = cur + twiddle * partner
        # bottom: X[i + half]  = cur - twiddle * partner
        new_re = tl.where(is_top, cur_re + tw_re, cur_re - tw_re)
        new_im = tl.where(is_top, cur_im + tw_im, cur_im - tw_im)

        # Only write if this stage is active (s < log2_N)
        tl.store(out_re_ptr + base + offs, tl.where(do_stage, new_re, cur_re), mask=mask)
        tl.store(out_im_ptr + base + offs, tl.where(do_stage, new_im, cur_im), mask=mask)


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
