"""Tests for kernels/attention/ — run on Colab (requires CUDA)."""

import torch
import pytest
from kernels.attention.naive_attention import naive_attention
from kernels.attention.sdpa import sdpa


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


# ── sdpa (causal) ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("N", [64, 128, 256, 512])
@pytest.mark.parametrize("d", [32, 64])
@pytest.mark.parametrize("B,H", [(1, 1), (2, 4)])
def test_sdpa_shapes(N, d, B, H):
    torch.manual_seed(0)
    q = torch.randn(B, H, N, d, device="cuda", dtype=torch.float32)
    k = torch.randn(B, H, N, d, device="cuda", dtype=torch.float32)
    v = torch.randn(B, H, N, d, device="cuda", dtype=torch.float32)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    got = sdpa(q, k, v)
    torch.testing.assert_close(got, ref, atol=1e-3, rtol=1e-3)


def test_sdpa_first_token():
    """Position 0 attends only to key 0; output[0] must equal V[0]."""
    N, d = 128, 64
    torch.manual_seed(2)
    q = torch.randn(1, 1, N, d, device="cuda", dtype=torch.float32)
    k = torch.randn(1, 1, N, d, device="cuda", dtype=torch.float32)
    v = torch.randn(1, 1, N, d, device="cuda", dtype=torch.float32)
    got = sdpa(q, k, v)
    torch.testing.assert_close(got[0, 0, 0], v[0, 0, 0], atol=1e-3, rtol=1e-3)


def test_sdpa_last_token_matches_full():
    """The last token attends to all positions — its output must match full attention."""
    N, d = 128, 64
    torch.manual_seed(3)
    q = torch.randn(1, 1, N, d, device="cuda", dtype=torch.float32)
    k = torch.randn(1, 1, N, d, device="cuda", dtype=torch.float32)
    v = torch.randn(1, 1, N, d, device="cuda", dtype=torch.float32)
    causal_out  = sdpa(q, k, v)
    full_out    = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    torch.testing.assert_close(
        causal_out[0, 0, -1], full_out[0, 0, -1], atol=1e-3, rtol=1e-3
    )
