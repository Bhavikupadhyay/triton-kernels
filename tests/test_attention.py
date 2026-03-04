"""Tests for kernels/attention/ — run on Colab (requires CUDA)."""

import torch
import pytest
from kernels.attention.naive_attention import naive_attention


@pytest.mark.parametrize("N", [64, 128, 256, 512])
@pytest.mark.parametrize("d", [32, 64])
@pytest.mark.parametrize("B,H", [(1, 1), (2, 4)])
def test_naive_attention_shapes(N, d, B, H):
    torch.manual_seed(0)
    q = torch.randn(B, H, N, d, device="cuda", dtype=torch.float32)
    k = torch.randn(B, H, N, d, device="cuda", dtype=torch.float32)
    v = torch.randn(B, H, N, d, device="cuda", dtype=torch.float32)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    got = naive_attention(q, k, v)
    torch.testing.assert_close(got, ref, atol=1e-3, rtol=1e-3)


def test_naive_attention_identity_v():
    """When V = I (one-hot), output row i = softmax(scores)[i, :] exactly."""
    N, d = 64, 64
    torch.manual_seed(1)
    q = torch.randn(1, 1, N, d, device="cuda", dtype=torch.float32)
    k = torch.randn(1, 1, N, d, device="cuda", dtype=torch.float32)
    v = torch.eye(N, device="cuda", dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    got = naive_attention(q, k, v)
    torch.testing.assert_close(got, ref, atol=1e-3, rtol=1e-3)


def test_naive_attention_uniform_scores():
    """When all Q=K, scores are uniform; output should equal mean of V rows."""
    N, d = 128, 32
    q = torch.ones(1, 1, N, d, device="cuda", dtype=torch.float32)
    k = torch.ones(1, 1, N, d, device="cuda", dtype=torch.float32)
    v = torch.randn(1, 1, N, d, device="cuda", dtype=torch.float32)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    got = naive_attention(q, k, v)
    torch.testing.assert_close(got, ref, atol=1e-3, rtol=1e-3)
    # Each output row should be identical (uniform attention)
    assert got[0, 0].std(dim=0).max() < 1e-4
