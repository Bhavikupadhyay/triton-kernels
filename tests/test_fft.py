"""Tests for kernels/fft/fft_kernel.py — run on Colab (requires CUDA)."""

import torch
import pytest
from kernels.fft.fft_kernel import fft


@pytest.mark.parametrize("N", [64, 128, 256, 512, 1024, 2048, 4096])
@pytest.mark.parametrize("B", [1, 8, 32])
def test_fft_batched(N, B):
    x = torch.randn(B, N, device="cuda", dtype=torch.float32)
    ref = torch.fft.fft(x)
    got = fft(x)
    torch.testing.assert_close(got.real, ref.real, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(got.imag, ref.imag, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("N", [64, 256, 1024])
def test_fft_1d(N):
    x = torch.randn(N, device="cuda", dtype=torch.float32)
    ref = torch.fft.fft(x)
    got = fft(x)
    torch.testing.assert_close(got.real, ref.real, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(got.imag, ref.imag, atol=1e-3, rtol=1e-3)


def test_fft_identity_dc():
    """DC component of FFT of all-ones is N; all others are 0."""
    N = 512
    x = torch.ones(1, N, device="cuda", dtype=torch.float32)
    got = fft(x)
    assert abs(got[0, 0].real - N) < 0.1
    assert got[0, 1:].abs().max() < 1e-2


def test_fft_single_freq():
    """FFT of a pure sinusoid has energy only at ±k."""
    N, k = 512, 3
    t = torch.arange(N, device="cuda", dtype=torch.float32)
    x = torch.cos(2 * torch.pi * k * t / N).unsqueeze(0)
    got = fft(x)
    # Energy at bin k and N-k; all others near zero
    mag = got[0].abs()
    assert mag[k] > N / 2 - 1.0
    assert mag[N - k] > N / 2 - 1.0
    mask = torch.ones(N, dtype=torch.bool, device="cuda")
    mask[k] = False
    mask[N - k] = False
    assert mag[mask].max() < 1.0
