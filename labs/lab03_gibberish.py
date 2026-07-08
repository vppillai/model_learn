"""Lab 03 — an UNTRAINED model already runs; it just talks nonsense.
This is the key insight: architecture (wiring) works instantly; only the
weights (the learned numbers) are missing.

Also watch for the tied-embedding "echo" bias (see CONCEPTS.md /
DEVLOG.md 2026-07-07): an untrained model's dominant move is to repeat the
most recently seen token, so generation tends to degenerate into repeating
whatever token came last — not random noise, a specific architectural
property that training will overwrite.

Run: PYTHONPATH=src python labs/lab03_gibberish.py
"""
import torch
from slm.config import TOY
from slm.model import LlamaSLM
from slm.tokenizer import train_tokenizer, encode, decode

lines = open("tests/fixtures/tiny_stories.txt").read().splitlines()
tok = train_tokenizer(lines * 20, vocab_size=TOY.vocab_size, save_path="/tmp/lab03.json")

torch.manual_seed(0)
model = LlamaSLM(TOY)

print("=== seeded with a real prompt: 'Once upon a time' ===")
prompt_ids = encode(tok, "Once upon a time")
start = torch.tensor([prompt_ids], dtype=torch.long)
out = model.generate(start, max_new_tokens=30)
print(decode(tok, out[0].tolist()))
print(
    "\n^ Gibberish — but it generated end-to-end! Notice it degenerates into\n"
    "repeating one word: that's the tied-embedding echo bias in action, not\n"
    "randomness. Training (Task 5) will overwrite this."
)

print("\n=== seeded with <|endoftext|> (id 0) — the worst-case trigger ===")
start0 = torch.zeros((1, 1), dtype=torch.long)
out0 = model.generate(start0, max_new_tokens=20)
ids0 = out0[0].tolist()
print("raw token ids:", ids0)
print(
    "decoded:", repr(decode(tok, ids0)),
    "(empty — decode() hides <|endoftext|>, and the model just echoes it forever)",
)
