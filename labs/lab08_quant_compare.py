"""Lab 08 — compare output + size across quantization levels.
Run: python labs/lab08_quant_compare.py
"""
import os, subprocess

LLAMA = os.path.expanduser("~/llama.cpp/build/bin/llama-cli")
for q in ("f16", "Q8_0", "Q4_K_M"):
    path = f"export/tinystories-slm-{q}.gguf"
    if not os.path.exists(path):
        print(f"=== {q}: (missing {path}) ===\n")
        continue
    size_mb = os.path.getsize(path) / 1e6
    # -st (single-turn) makes llama-cli exit after one generation instead of
    # dropping into an interactive REPL (which would hang subprocess.run).
    out = subprocess.run([LLAMA, "-m", path, "-p", "Once upon a time",
                          "-n", "40", "--seed", "0", "-st", "--simple-io",
                          "--no-warmup", "--no-display-prompt"],
                         capture_output=True, text=True, timeout=120,
                         stdin=subprocess.DEVNULL).stdout
    # Strip llama-cli's banner/stats: keep the text after the prompt echo and
    # before the timing line, so we compare actual generation across quants.
    story = out
    if "> Once upon a time" in story:
        story = story.split("> Once upon a time", 1)[1]
    story = story.split("[ Prompt:", 1)[0].strip()
    print(f"=== {q}  ({size_mb:.1f} MB) ===\nOnce upon a time{story[:280]}\n")
print("Smaller quant = smaller file + faster, with some quality loss.")
