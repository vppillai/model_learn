import random
from slm.tokenizer import train_tokenizer, load_tokenizer, encode, decode

CORPUS = [
    "Once upon a time there was a little cat.",
    "The cat liked to play in the sun every day.",
    "One day the cat met a happy dog and they became friends.",
] * 50


def test_roundtrip_and_vocab(tmp_path):
    random.seed(0)
    path = str(tmp_path / "tok.json")
    tok = train_tokenizer(CORPUS, vocab_size=300, save_path=path)
    assert tok.get_vocab_size() <= 300
    ids = encode(tok, "the cat liked the sun")
    assert isinstance(ids, list) and all(isinstance(i, int) for i in ids)
    assert decode(tok, ids).replace(" ", "") == "thecatlikedthesun".replace(" ", "")


def test_load_after_save(tmp_path):
    path = str(tmp_path / "tok.json")
    train_tokenizer(CORPUS, vocab_size=300, save_path=path)
    tok = load_tokenizer(path)
    assert encode(tok, "happy dog") == encode(load_tokenizer(path), "happy dog")


def test_endoftext_special_token(tmp_path):
    path = str(tmp_path / "tok.json")
    tok = train_tokenizer(CORPUS, vocab_size=300, save_path=path)
    assert tok.token_to_id("<|endoftext|>") == 0
