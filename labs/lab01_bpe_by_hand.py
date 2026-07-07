"""Lab 01 — watch BPE merges form on a single paragraph.

Run: PYTHONPATH=src python labs/lab01_bpe_by_hand.py
"""
from slm.tokenizer import train_tokenizer, encode

para = (
    "the cat sat on the mat. the cat sat on the hat. "
    "the rat sat on the cat."
)

for vsz in (260, 270, 290, 320):
    tok = train_tokenizer([para] * 20, vocab_size=vsz, save_path="/tmp/lab01.json")
    ids = encode(tok, "the cat sat")
    pieces = [tok.id_to_token(i) for i in ids]
    print(f"vocab={vsz:>4}  'the cat sat' -> {pieces}")

print("\nNotice: as the vocab grows, common chunks like 'the ' and 'cat'")
print("become single tokens. That merging IS what BPE training does.")
