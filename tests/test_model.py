import torch
from slm.config import TOY
from slm.model import LlamaSLM


def test_forward_logits_shape():
    torch.manual_seed(0)
    m = LlamaSLM(TOY)
    idx = torch.randint(0, TOY.vocab_size, (2, 10))
    logits = m(idx)
    assert logits.shape == (2, 10, TOY.vocab_size)


def test_embeddings_are_tied():
    m = LlamaSLM(TOY)
    assert m.lm_head.weight is m.embed_tokens.weight


def test_generate_extends_sequence():
    torch.manual_seed(0)
    m = LlamaSLM(TOY)
    idx = torch.zeros((1, 3), dtype=torch.long)
    out = m.generate(idx, max_new_tokens=5)
    assert out.shape == (1, 8)


def test_untrained_forward_is_valid_and_input_dependent():
    # An untrained model does NOT produce a near-uniform distribution: tied
    # embeddings + the pre-norm residual stream make it strongly favor
    # echoing back the most recently seen token (confirmed empirically —
    # see DEVLOG.md 2026-07-07). That's expected, architecture-driven
    # behavior, not garbage. The real "wiring works" sanity check is that
    # the output is a valid probability distribution and actually depends
    # on the input (i.e. the forward pass isn't silently constant/dead).
    torch.manual_seed(0)
    m = LlamaSLM(TOY)
    idx_a = torch.randint(0, TOY.vocab_size, (1, 4))
    idx_b = torch.randint(0, TOY.vocab_size, (1, 4))
    probs_a = torch.softmax(m(idx_a)[0, -1], dim=-1)
    probs_b = torch.softmax(m(idx_b)[0, -1], dim=-1)
    assert torch.allclose(probs_a.sum(), torch.tensor(1.0), atol=1e-4)
    assert not torch.allclose(probs_a, probs_b)
