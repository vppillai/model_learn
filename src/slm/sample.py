import torch
from slm.train import load_checkpoint
from slm.tokenizer import load_tokenizer, encode, decode


def generate_text(ckpt_path, tok_path, prompt="", max_new_tokens=120,
                  temperature=0.8, top_k=40) -> str:
    model, _ = load_checkpoint(ckpt_path)
    tok = load_tokenizer(tok_path)
    ids = encode(tok, prompt) if prompt else [0]
    idx = torch.tensor([ids], dtype=torch.long)
    out = model.generate(idx, max_new_tokens, temperature=temperature, top_k=top_k)
    return decode(tok, out[0].tolist())


if __name__ == "__main__":
    import sys
    ck, tk = sys.argv[1], sys.argv[2]
    prompt = sys.argv[3] if len(sys.argv) > 3 else "Once upon a time"
    print(generate_text(ck, tk, prompt))
