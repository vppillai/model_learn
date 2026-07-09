#!/usr/bin/env bash
set -euo pipefail
# Convert the exported HF model to GGUF, then quantize.
#
# Prereqs (one-time):
#   git clone --depth 1 https://github.com/ggerganov/llama.cpp ~/llama.cpp
#   uv pip install gguf sentencepiece protobuf          # converter deps
#   cd ~/llama.cpp && uv run cmake -B build -DLLAMA_CURL=OFF \
#     && uv run cmake --build build -j 4                # -j 4: avoid OOM on small boxes
#
# Run from the model_learn repo root:  bash scripts/convert_to_gguf.sh
LLAMA_CPP="${LLAMA_CPP:-$HOME/llama.cpp}"
SRC="export/tinystories-slm"
OUT="export"
QUANT="$LLAMA_CPP/build/bin/llama-quantize"

# 1. Export to GGUF IR (f16). The converter is pure Python; run it with the
#    project venv (which has gguf/torch/transformers/safetensors).
uv run python "$LLAMA_CPP/convert_hf_to_gguf.py" "$SRC" \
  --outfile "$OUT/tinystories-slm-f16.gguf" --outtype f16

# 2. Optimization pass: quantize f16 -> Q8_0 (near-lossless) and Q4_K_M (small).
"$QUANT" "$OUT/tinystories-slm-f16.gguf" "$OUT/tinystories-slm-Q8_0.gguf"   Q8_0
"$QUANT" "$OUT/tinystories-slm-f16.gguf" "$OUT/tinystories-slm-Q4_K_M.gguf" Q4_K_M

echo "Wrote f16, Q8_0, Q4_K_M GGUFs to $OUT/"
ls -lh "$OUT"/tinystories-slm-*.gguf
