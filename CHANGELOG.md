# Changelog

## Unreleased
- Task 1: project scaffold, `ModelConfig` (toy/small), pytest wiring, CPU-only torch pin.
- Task 2: BPE tokenizer (train/load/encode/decode) + Lab 01 (watch merges form).
- Task 3: data pipeline (tokenize stream + shifted batches) + fixture.
- Task 4a: model components (RMSNorm, RoPE, causal attention, SwiGLU).
- Task 4b: assemble LlamaSLM + generate; untrained-gibberish milestone (Labs 02-03); fixed flawed high-entropy test assertion (tied-embedding echo bias).
- Task 5: training loop (AdamW, warmup+cosine LR, grad clip, checkpoints), sample.py, Lab 05; overfit-one-batch + toy-training milestones. Fixed train_tokenizer to create its save dir; load_tinystories streams when limited.
