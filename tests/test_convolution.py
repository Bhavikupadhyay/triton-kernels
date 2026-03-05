"""Tests for kernels/convolution/ — run on Colab (requires CUDA)."""

import torch
import pytest

from kernels.convolution.conv1d import conv1d
# Uncomment as kernels are implemented:
# from kernels.convolution.conv2d import conv2d
# from kernels.convolution.depthwise_conv2d import depthwise_conv2d

import torch.nn.functional as F


# ── conv1d ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("B,C_in,C_out,N,K", [
    (1,  1,   1,   16,   3),
    (2,  4,   8,   64,   5),
    (4,  16,  32,  512,  7),
    (3,  5,   7,   100,  3),
    (1,  64,  64,  1024, 11),
    (8,  32,  64,  2048, 3),
])
def test_conv1d(B, C_in, C_out, N, K, dtype):
    x   = torch.randn(B, C_in, N,     device="cuda", dtype=dtype)
    w   = torch.randn(C_out, C_in, K, device="cuda", dtype=dtype)
    ref = F.conv1d(x, w)
    got = conv1d(x, w)
    tol = dict(rtol=1e-2, atol=1e-2) if dtype == torch.float16 else dict(rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(got, ref, **tol)
