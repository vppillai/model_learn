import torch
from slm.config import TOY
from slm.model import RMSNorm, build_rope_cache, apply_rope, Attention, SwiGLU


def test_rmsnorm_shape_and_scale():
    torch.manual_seed(0)
    n = RMSNorm(TOY.d_model, eps=TOY.rms_norm_eps)
    x = torch.randn(2, 5, TOY.d_model)
    out = n(x)
    assert out.shape == x.shape
    # default weight=1 => rms of each row ~ 1
    rms = out.pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)


def test_rope_preserves_shape_and_norm():
    torch.manual_seed(0)
    cos, sin = build_rope_cache(TOY.head_dim, TOY.context_len, TOY.rope_theta)
    x = torch.randn(1, TOY.n_heads, 7, TOY.head_dim)
    out = apply_rope(x, cos[:7], sin[:7])
    assert out.shape == x.shape
    # rotation preserves vector norm per (head, position)
    assert torch.allclose(out.norm(dim=-1), x.norm(dim=-1), atol=1e-4)


def test_attention_is_causal():
    torch.manual_seed(0)
    attn = Attention(TOY)
    cos, sin = build_rope_cache(TOY.head_dim, TOY.context_len, TOY.rope_theta)
    x = torch.randn(1, 6, TOY.d_model)
    out_full = attn(x, cos[:6], sin[:6])
    # changing the LAST token must not change earlier outputs (causality)
    x2 = x.clone()
    x2[0, -1] += 10.0
    out2 = attn(x2, cos[:6], sin[:6])
    assert torch.allclose(out_full[0, :-1], out2[0, :-1], atol=1e-5)


def test_swiglu_shape():
    torch.manual_seed(0)
    mlp = SwiGLU(TOY)
    x = torch.randn(2, 4, TOY.d_model)
    assert mlp(x).shape == x.shape
