import torch
from slm.config import TOY
from slm.model import LlamaSLM
from slm.export_hf import to_hf_config, _copy_weights_into_hf


def test_hf_roundtrip_matches_handbuilt():
    torch.manual_seed(0)
    src = LlamaSLM(TOY).eval()
    from transformers import LlamaForCausalLM
    hf = LlamaForCausalLM(to_hf_config(TOY)).eval()
    _copy_weights_into_hf(src, hf)
    idx = torch.randint(0, TOY.vocab_size, (1, 12))
    with torch.no_grad():
        a = src(idx)
        b = hf(idx).logits
    assert torch.allclose(a, b, atol=1e-4), (a - b).abs().max().item()
