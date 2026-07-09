---
license: mit
datasets:
  - roneneldan/TinyStories
language:
  - en
pipeline_tag: text-generation
tags:
  - llama
  - tinystories
  - educational
  - from-scratch
---

# tinystories-slm

A ~13.8M-parameter, Llama-architecture small language model **hand-built from
scratch** and trained on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories).
It generates short, simple children's stories.

This model exists to *teach*, not to be useful: it is the artifact of a
learning project that builds the full "download → build → run" LLM pipeline by
making one. See the source and developer notes at
<https://github.com/vppillai/model_learn>.

## What it is

- **Architecture:** decoder-only Llama (RoPE, RMSNorm, SwiGLU, tied
  embeddings), written from scratch in plain PyTorch, then exported to the
  standard `LlamaForCausalLM` format — so it loads with stock `transformers`
  and converts to GGUF with no custom code.
- **Size:** 13,767,552 parameters. `d_model=384`, 6 layers, 6 heads,
  `head_dim=64`, `ffn_hidden=1024`, vocab 8192, context 512.
- **Tokenizer:** byte-level BPE trained on TinyStories; `<|endoftext|>` is
  token id 0 (also used as BOS/EOS/PAD).
- **Training:** ~200k TinyStories (~44M tokens), 20k steps, AdamW,
  cosine LR schedule (peak 6e-4) with warmup, batch size 64, on a single
  Colab T4 GPU. Local eval loss ≈ 1.81 (perplexity ≈ 6.1).

## Usage

```python
import torch
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast

model = LlamaForCausalLM.from_pretrained("vysakhpillai/tinystories-slm").eval()
tok = PreTrainedTokenizerFast.from_pretrained("vysakhpillai/tinystories-slm")

ids = tok("Once upon a time", return_tensors="pt").input_ids
out = model.generate(ids, max_new_tokens=120, do_sample=True, temperature=0.8, top_k=40)
print(tok.decode(out[0], skip_special_tokens=True))
```

Or, from the source repo, on CPU:

```bash
PYTHONPATH=src python -m slm.sample checkpoints/small.pt checkpoints/small_tok.json "Once upon a time"
```

## Example output

> Once upon a time, there was a little boy named Timmy. He loved to play with
> his toy cars and trucks. One day, Timmy's mom asked him to help her with the
> garage. They were going to build a big car with blocks. Timmy was excited to
> help and started building his car.

## Intended use & limitations

- **Intended use:** education — understanding how LLMs are built, trained,
  packaged, and run. A worked example, not a product.
- **Limitations:** tiny; only produces simple TinyStories-style children's
  prose. It has no knowledge, no instruction-following, and no safety tuning.
  Expect occasional repetition and invented words. Not suitable for any real
  application.
