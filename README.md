# model_learn — Build a Language Model From Scratch to Ollama

A hand-built, Llama-shaped small language model (~14M params) trained on
TinyStories, packaged to GGUF, and run on CPU via Ollama — built to learn the
full **download → build → run** pipeline by making one.

**Phase 1 is complete.** The model runs locally on CPU:

```bash
ollama run tinystories-slm "Once upon a time"
```
> Once upon a time, there was a little girl named Lily. She loved to play
> outside in the park. One day, she saw a big tree and wanted to climb it...

## What's here

- `src/slm/` — the model, tokenizer, data pipeline, training loop, HF export,
  all written from scratch in plain PyTorch (Llama architecture: RoPE, RMSNorm,
  SwiGLU, tied embeddings).
- `labs/` — small runnable side-experiments that open the hood on each concept.
- `CONCEPTS.md` — a plain-language glossary, one entry per idea met in the code.
- `DEVLOG.md` — the dated build narrative, including every dead-end and gotcha.
- `REPRODUCE.md` — build the whole thing yourself, milestone by milestone.
- `notebooks/` — the Colab notebook for the `small` GPU training run.
- `docs/superpowers/specs/` — the design spec.

## The milestone ladder (all reached)

tokenizer → batches → untrained gibberish → overfit one batch → toy CPU run →
coherent stories (Colab GPU) → published to HF Hub → GGUF (f16/Q8_0/Q4_K_M) →
running in Ollama.

Start with `REPRODUCE.md` to build it from scratch.
