"""Lab 02 — print attention weights for one short sequence.
Run: PYTHONPATH=src python labs/lab02_attention_peek.py
"""
import torch
from slm.config import TOY
from slm.model import Attention, build_rope_cache

torch.manual_seed(0)
attn = Attention(TOY)
cos, sin = build_rope_cache(TOY.head_dim, TOY.context_len, TOY.rope_theta)
x = torch.randn(1, 5, TOY.d_model)

# Re-run attention internals to expose the weights (mirrors Attention.forward):
import torch.nn.functional as F
c = TOY
q = attn.q_proj(x).view(1, 5, c.n_heads, c.head_dim).transpose(1, 2)
k = attn.k_proj(x).view(1, 5, c.n_kv_heads, c.head_dim).transpose(1, 2)
from slm.model import apply_rope
q, k = apply_rope(q, cos[:5], sin[:5]), apply_rope(k, cos[:5], sin[:5])
scores = (q @ k.transpose(-2, -1)) * (c.head_dim ** -0.5)
scores = scores + torch.full((5, 5), float("-inf")).triu(1)
w = F.softmax(scores, dim=-1)[0, 0]  # head 0
print("Attention weights (row = query position, col = key position), head 0:")
for i, row in enumerate(w):
    print(f"  q{i}: " + " ".join(f"{p:.2f}" for p in row))
print("\nNote the upper triangle is 0.00 — that's the causal mask: a token")
print("can only attend to itself and earlier tokens.")
