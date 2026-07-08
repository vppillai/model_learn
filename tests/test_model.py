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


def test_untrained_forward_is_finite_and_input_dependent():
    # Sanity of the wiring (not the weights): the forward pass must produce
    # finite logits and actually depend on the input — i.e. it is not
    # silently constant/dead or emitting NaNs.
    torch.manual_seed(0)
    m = LlamaSLM(TOY)
    idx_a = torch.randint(0, TOY.vocab_size, (1, 4))
    idx_b = torch.randint(0, TOY.vocab_size, (1, 4))
    logits_a = m(idx_a)[0, -1]
    logits_b = m(idx_b)[0, -1]
    assert torch.isfinite(logits_a).all()          # no NaN/inf
    assert not torch.allclose(logits_a, logits_b)  # depends on input


def test_untrained_model_echoes_last_token():
    # An untrained model does NOT produce a near-uniform distribution. Tied
    # embeddings + the pre-norm residual stream carry the last token's
    # embedding straight to the (tied) lm_head, so its self-dot-product
    # |embed(last)|^2 dominates every other token's logit — the model's
    # top prediction is the token it just saw. This is expected,
    # architecture-driven behavior, verified quantitatively in DEVLOG.md
    # (top logit ~= |embed(last)|^2). Training (Task 5) overwrites it.
    torch.manual_seed(0)
    m = LlamaSLM(TOY)
    idx = torch.randint(0, TOY.vocab_size, (1, 6))
    last = idx[0, -1].item()
    logits = m(idx)[0, -1]
    assert logits.argmax().item() == last
