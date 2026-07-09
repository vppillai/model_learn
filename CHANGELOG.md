# Changelog

## Unreleased
- Task 1: project scaffold, `ModelConfig` (toy/small), pytest wiring, CPU-only torch pin.
- Task 2: BPE tokenizer (train/load/encode/decode) + Lab 01 (watch merges form).
- Task 3: data pipeline (tokenize stream + shifted batches) + fixture.
- Task 4a: model components (RMSNorm, RoPE, causal attention, SwiGLU).
- Task 4b: assemble LlamaSLM + generate; untrained-gibberish milestone (Labs 02-03); fixed flawed high-entropy test assertion (tied-embedding echo bias).
- Task 5: training loop (AdamW, warmup+cosine LR, grad clip, checkpoints), sample.py, Lab 05; overfit-one-batch + toy-training milestones. Fixed train_tokenizer to create its save dir; load_tinystories streams when limited.
- Task 6 (Half A): Colab notebook + runbook for the `small` GPU run. Made training device-portable (train() moves batches to the model's device) and scale-safe (tensorize the stream once, not per step). Run itself pending (handoff).
- Task 6 (complete): coherent-stories milestone. Trained `small` (13.8M params) on Colab T4; local eval loss 1.81 / perplexity 6.1 (vs toy ~4.3/74). Fixed Colab pip clobbering numpy (minimal install) and load_checkpoint failing on GPU-saved checkpoints (map_location="cpu").
- Task 7: export to HF `LlamaForCausalLM` (round-trip verified, max diff 9.5e-06) + push to Hub. Fixed exported special-token ids (eos=0) so generation stops at story end. Published private repo vysakhpillai/tinystories-slm.
