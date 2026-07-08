import torch
from slm.tokenizer import train_tokenizer
from slm.data import tokenize_texts, get_batch

LINES = open("tests/fixtures/tiny_stories.txt").read().splitlines()


def _tok(tmp_path):
    return train_tokenizer(LINES * 20, vocab_size=300,
                           save_path=str(tmp_path / "t.json"))


def test_tokenize_inserts_eot(tmp_path):
    tok = _tok(tmp_path)
    stream = tokenize_texts(tok, LINES[:3])
    assert 0 in stream  # <|endoftext|> id 0 separates docs
    assert all(isinstance(i, int) for i in stream)


def test_batch_shapes_and_shift(tmp_path):
    tok = _tok(tmp_path)
    stream = tokenize_texts(tok, LINES * 5)
    x, y = get_batch(stream, batch_size=4, context_len=8, seed=0)
    assert x.shape == (4, 8) and y.shape == (4, 8)
    assert x.dtype == torch.long
    # y is x shifted by one: y[:, :-1] aligns with x[:, 1:]
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_batch_is_deterministic_with_seed(tmp_path):
    tok = _tok(tmp_path)
    stream = tokenize_texts(tok, LINES * 5)
    a = get_batch(stream, 4, 8, seed=0)[0]
    b = get_batch(stream, 4, 8, seed=0)[0]
    assert torch.equal(a, b)
