"""Tests for kernels/attention/ — run on Colab (requires CUDA)."""

import torch
import pytest
from kernels.attention.naive_attention import naive_attention
from kernels.attention.sdpa import sdpa
from kernels.attention.multi_head_attention import multi_head_attention, _ref_mha


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


# ── multi_head_attention ──────────────────────────────────────────────────────

@pytest.mark.parametrize("N", [64, 128, 256, 512])
@pytest.mark.parametrize("H,d", [(4, 32), (8, 64)])
@pytest.mark.parametrize("B", [1, 2])
def test_mha_shapes(N, H, d, B):
    torch.manual_seed(0)
    Hd = H * d
    q = torch.randn(B, N, Hd, device="cuda", dtype=torch.float32)
    k = torch.randn(B, N, Hd, device="cuda", dtype=torch.float32)
    v = torch.randn(B, N, Hd, device="cuda", dtype=torch.float32)
    ref = _ref_mha(q, k, v, H)
    got = multi_head_attention(q, k, v, H)
    torch.testing.assert_close(got, ref, atol=1e-3, rtol=1e-3)


def test_mha_per_head_slices():
    """Each head slice of the output must match the reference independently."""
    B, N, H, d = 1, 128, 4, 32
    torch.manual_seed(1)
    q = torch.randn(B, N, H * d, device="cuda", dtype=torch.float32)
    k = torch.randn(B, N, H * d, device="cuda", dtype=torch.float32)
    v = torch.randn(B, N, H * d, device="cuda", dtype=torch.float32)
    got = multi_head_attention(q, k, v, H)
    ref = _ref_mha(q, k, v, H)
    for h in range(H):
        torch.testing.assert_close(
            got[:, :, h*d:(h+1)*d], ref[:, :, h*d:(h+1)*d], atol=1e-3, rtol=1e-3
        )


def test_mha_matches_sdpa_single_head():
    """With H=1, MHA output must match SDPA on the equivalent (B,1,N,d) input."""
    import torch.nn.functional as F
    B, N, d = 2, 128, 64
    q = torch.randn(B, N, d, device="cuda", dtype=torch.float32)
    k = torch.randn(B, N, d, device="cuda", dtype=torch.float32)
    v = torch.randn(B, N, d, device="cuda", dtype=torch.float32)
    mha_out = multi_head_attention(q, k, v, H=1)
    sdpa_ref = F.scaled_dot_product_attention(
        q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1), is_causal=True
    ).squeeze(1)
    torch.testing.assert_close(mha_out, sdpa_ref, atol=1e-3, rtol=1e-3)
from kernels.attention.flash_attention_v1 import flash_attention_v1


# ── flash_attention_v1 ────────────────────────────────────────────────────────

@pytest.mark.parametrize("N", [64, 128, 256, 512, 1024, 2048])
@pytest.mark.parametrize("d", [32, 64])
@pytest.mark.parametrize("B,H", [(1, 1), (2, 4)])
def test_flash_v1_shapes(N, d, B, H):
    import torch.nn.functional as F
    torch.manual_seed(0)
    q = torch.randn(B, H, N, d, device="cuda", dtype=torch.float32)
    k = torch.randn(B, H, N, d, device="cuda", dtype=torch.float32)
    v = torch.randn(B, H, N, d, device="cuda", dtype=torch.float32)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    got = flash_attention_v1(q, k, v)
    torch.testing.assert_close(got, ref, atol=1e-3, rtol=1e-3)


def test_flash_v1_large_n():
    """Flash attention should handle N=4096 without OOM (no N^2 materialisation)."""
    import torch.nn.functional as F
    B, H, N, d = 1, 4, 4096, 64
    q = torch.randn(B, H, N, d, device="cuda", dtype=torch.float32)
    k = torch.randn(B, H, N, d, device="cuda", dtype=torch.float32)
    v = torch.randn(B, H, N, d, device="cuda", dtype=torch.float32)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    got = flash_attention_v1(q, k, v)
    torch.testing.assert_close(got, ref, atol=1e-3, rtol=1e-3)


def test_flash_v1_first_token():
    """Position 0 attends only to itself; output[0] must equal V[0]."""
    import torch.nn.functional as F
    B, H, N, d = 1, 1, 256, 64
    torch.manual_seed(2)
    q = torch.randn(B, H, N, d, device="cuda")
    k = torch.randn(B, H, N, d, device="cuda")
    v = torch.randn(B, H, N, d, device="cuda")
    got = flash_attention_v1(q, k, v)
    torch.testing.assert_close(got[0, 0, 0], v[0, 0, 0], atol=1e-3, rtol=1e-3)
from kernels.attention.flash_attention_v2 import flash_attention_v2


# ── flash_attention_v2 ────────────────────────────────────────────────────────

@pytest.mark.parametrize("N", [64, 128, 256, 512, 1024, 2048, 4096])
@pytest.mark.parametrize("d", [32, 64])
@pytest.mark.parametrize("B,H", [(1, 1), (2, 4)])
def test_flash_v2_shapes(N, d, B, H):
    torch.manual_seed(0)
    q = torch.randn(B, H, N, d, device="cuda", dtype=torch.float32)
    k = torch.randn(B, H, N, d, device="cuda", dtype=torch.float32)
    v = torch.randn(B, H, N, d, device="cuda", dtype=torch.float32)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    got = flash_attention_v2(q, k, v)
    torch.testing.assert_close(got, ref, atol=1e-3, rtol=1e-3)


def test_flash_v2_matches_v1():
    """v2 output must match v1 output exactly (same algorithm, better tiling)."""
    from kernels.attention.flash_attention_v1 import flash_attention_v1
    B, H, N, d = 2, 4, 512, 64
    torch.manual_seed(1)
    q = torch.randn(B, H, N, d, device="cuda", dtype=torch.float32)
    k = torch.randn(B, H, N, d, device="cuda", dtype=torch.float32)
    v = torch.randn(B, H, N, d, device="cuda", dtype=torch.float32)
    got_v1 = flash_attention_v1(q, k, v)
    got_v2 = flash_attention_v2(q, k, v)
    torch.testing.assert_close(got_v2, got_v1, atol=1e-3, rtol=1e-3)


def test_flash_v2_first_token():
    """Position 0 attends only to itself; output[0] must equal V[0]."""
    B, H, N, d = 1, 1, 256, 64
    torch.manual_seed(2)
    q = torch.randn(B, H, N, d, device="cuda")
    k = torch.randn(B, H, N, d, device="cuda")
    v = torch.randn(B, H, N, d, device="cuda")
    got = flash_attention_v2(q, k, v)
    torch.testing.assert_close(got[0, 0, 0], v[0, 0, 0], atol=1e-3, rtol=1e-3)
from kernels.attention.flash_attention_v2_fp16 import flash_attention_v2_fp16


# ── flash_attention_v2_fp16 ───────────────────────────────────────────────────

@pytest.mark.parametrize("N", [64, 128, 256, 512, 1024, 2048, 4096])
@pytest.mark.parametrize("d", [32, 64])
@pytest.mark.parametrize("B,H", [(1, 1), (2, 4)])
def test_flash_v2_fp16_shapes(N, d, B, H):
    torch.manual_seed(0)
    q = torch.randn(B, H, N, d, device="cuda", dtype=torch.float16)
    k = torch.randn(B, H, N, d, device="cuda", dtype=torch.float16)
    v = torch.randn(B, H, N, d, device="cuda", dtype=torch.float16)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    got = flash_attention_v2_fp16(q, k, v)
    torch.testing.assert_close(got, ref, atol=1e-2, rtol=1e-2)


def test_flash_v2_fp16_dtype_passthrough():
    """fp32 input must produce fp32 output; fp16 input must produce fp16 output."""
    B, H, N, d = 1, 1, 128, 64
    for dtype in [torch.float16, torch.float32]:
        q = torch.randn(B, H, N, d, device="cuda", dtype=dtype)
        k = torch.randn(B, H, N, d, device="cuda", dtype=dtype)
        v = torch.randn(B, H, N, d, device="cuda", dtype=dtype)
        out = flash_attention_v2_fp16(q, k, v)
        assert out.dtype == dtype, f"expected {dtype}, got {out.dtype}"


def test_flash_v2_fp16_matches_fp32_v2():
    """fp16 and fp32 v2 outputs should agree within fp16 tolerance."""
    from kernels.attention.flash_attention_v2 import flash_attention_v2
    B, H, N, d = 2, 4, 512, 64
    torch.manual_seed(1)
    q = torch.randn(B, H, N, d, device="cuda", dtype=torch.float32)
    k = torch.randn(B, H, N, d, device="cuda", dtype=torch.float32)
    v = torch.randn(B, H, N, d, device="cuda", dtype=torch.float32)
    out_fp32 = flash_attention_v2(q, k, v)
    out_fp16 = flash_attention_v2_fp16(q, k, v)
    torch.testing.assert_close(out_fp16, out_fp32, atol=1e-2, rtol=1e-2)


def test_flash_v2_fp16_large_n():
    """Flash attention fp16 should handle N=8192 without OOM."""
    B, H, N, d = 1, 4, 8192, 64
    q = torch.randn(B, H, N, d, device="cuda", dtype=torch.float16)
    k = torch.randn(B, H, N, d, device="cuda", dtype=torch.float16)
    v = torch.randn(B, H, N, d, device="cuda", dtype=torch.float16)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    got = flash_attention_v2_fp16(q, k, v)
    torch.testing.assert_close(got, ref, atol=1e-2, rtol=1e-2)
