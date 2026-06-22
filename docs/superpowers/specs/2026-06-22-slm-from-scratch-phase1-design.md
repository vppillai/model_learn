# Design Spec — `model_learn` Phase 1: A Story-Telling SLM From Scratch to Ollama

- **Status:** Approved for planning
- **Date:** 2026-06-22
- **Author:** vpillai + Claude (explanatory mode)
- **Scope of this spec:** Phase 1 only (the base model). Phases 2 and 3 are sketched for context but each gets its own spec → plan → build cycle.

---

## 1. Purpose & Learning Goal

Build genuine, first-principles intuition for how downloadable LLMs actually work by **making one ourselves** end-to-end: train a small language model (SLM) from scratch, package it in the industry-standard format, publish it to Hugging Face, and run it locally on **CPU** via llama.cpp / Ollama.

The guiding analogy throughout: **a downloaded model is almost exactly like downloaded software.**

| Software concept | Model equivalent |
|---|---|
| Compiled binary | Weights (`.safetensors` / `.gguf`) — the learned numbers |
| Source you build against | Architecture code + `config.json` |
| Character encoding | Tokenizer |
| Runtime / VM | Inference engine (llama.cpp / Ollama) |
| Compile step | Training (run data through an optimizer until the numbers are useful) |

The deliverable is **as much the journey as the model**: reproduction-grade developer notes that later become a series of blog articles.

**Audience constraint:** the author is not a mathematician and explicitly wants to avoid heavy math. Explanations favor intuition, data flow, and runnable demonstration over derivations.

---

## 2. Roadmap (3 phases = 3+ articles)

| Phase | Article | Learns | End artifact |
|---|---|---|---|
| **1. Base model** *(this spec)* | "From zero to a story-telling model on your CPU" | Tokenization, the transformer, pretraining, packaging, GGUF, CPU inference | A TinyStories model on HF Hub that runs in Ollama |
| **2. Chat model** | "Teaching it to follow instructions" | Fine-tuning, chat templates, base-vs-instruct distinction | A tiny instruct/chat model |
| **3. Deepening** *(optional)* | "What's really in a `.gguf`" + "Mixture of Experts, gently" | Quantization internals; MoE intuition | A GGUF teardown + a small 4-expert MoE side-experiment |

Each phase builds on the prior one's artifact. Architecture choices in Phase 1 are made so Phases 2–3 follow cleanly (standard Llama shape → fine-tuning and GGUF "just work"; the feed-forward block is the future MoE "expert").

---

## 3. Design Principles (non-negotiables)

1. **Hand-build, but Llama-shaped.** Write the model and training loop ourselves in plain PyTorch for understanding, but make the architecture bit-for-bit a standard Llama so the final artifact needs **no custom runtime** — it is consumed by tools the world already uses (HF Transformers, llama.cpp, Ollama).
2. **Reuse standard libraries as building blocks, then open the hood.** Use real libraries (`tokenizers`, `transformers`, `datasets`) productively, and probe each with a small **Lab** so none stays a black box.
3. **Always something to show.** Every step yields a demoable intermediate result (the "milestone ladder"). No waiting until the end.
4. **Config-scalable code.** The exact same code runs a 2-minute `toy` job on local CPU and a multi-hour `small` job on a Colab GPU — only a config changes.
5. **Concept-on-contact teaching.** No new code without (a) a one-line "why this exists," (b) a runnable Lab, and (c) a `CONCEPTS.md` entry.
6. **Reproduction-first docs, blog-later.** Notes are optimized for fidelity/reproducibility, not narrative. The blog is a separate downstream task written in the author's own style from these notes.

---

## 4. Tech Stack (with rationale)

- **Python 3.13 + PyTorch** — CPU build locally, GPU on Colab; same code both places.
- **Hugging Face `tokenizers`** — train a real BPE tokenizer (Lab opens the hood on BPE merges).
- **Hugging Face `transformers`** — used *only at packaging time* to wrap hand-trained weights as `LlamaForCausalLM` so the artifact is standard and converts to GGUF cleanly.
- **Hugging Face `datasets`** — pull `roneneldan/TinyStories` (public).
- **llama.cpp** — `convert_hf_to_gguf.py` + quantizer + CPU inference engine.
- **Ollama** — friendly local runner for the final demo.
- **matplotlib** — loss-curve plots.

---

## 5. Dataset

- **`roneneldan/TinyStories`** (public on HF Hub): ~2M short, simple stories using a small vocabulary. Chosen because research showed even ~1–10M-parameter models trained on it produce *coherent* children's stories — ideal for a CPU-friendly, genuinely-works teaching target.
- **`toy` subset:** a small slice (a few MB) so the full pipeline runs in minutes locally.
- **`small` run:** full dataset (or a large subset) on Colab.

---

## 6. The Model (hand-built, Llama-shaped, decoder-only)

Components we write ourselves in plain PyTorch (each introduced with a "why" + a Lab):

- **Token embeddings** — lookup table mapping token id → vector.
- **RoPE** (rotary positional encoding) — how the model encodes word order.
- **RMSNorm** — a simpler normalization than classic LayerNorm.
- **Causal self-attention** — standard multi-head attention with a causal mask ("only look at previous tokens").
- **SwiGLU feed-forward block** — the per-token "thinking" layer; also the future MoE *expert*.
- **Tied input/output embeddings** + final linear projection to vocabulary logits.

**Phase-1 simplifications (deliberate, to keep the first model readable):**
- Standard multi-head attention (**no Grouped-Query Attention** in Phase 1 — GQA becomes a Lab).
- **No Mixture of Experts** in Phase 1 (MoE becomes a Phase-3 Lab/article).

These are the documented judgment calls; revisit only if a later phase needs them earlier.

### 6.1 Two configs, one codebase

These are **starting defaults**; exact `small` numbers are finalized right before the Colab run after observing local iteration speed. Parameter counts are approximate.

| Field | `toy` (local CPU, minutes) | `small` (Colab, final target ~12–15M params) |
|---|---|---|
| layers | 2 | 6 |
| hidden dim (`d_model`) | 64 | 384 |
| attention heads | 4 | 6 |
| head dim | 16 | 64 |
| FFN hidden (SwiGLU) | 128 | ~1024 |
| context length | 128 | 512 |
| vocab size | 2,048 | 8,192 |
| tied embeddings | yes | yes |

The `small` target (~12–15M params) produces coherent stories, trains in ~1–2 hrs on a free Colab GPU, yields a few-MB GGUF, and runs instantly on CPU.

---

## 7. The Milestone Ladder (showable increments)

Each rung is independently demoable and gets a `DEVLOG.md` entry. This is the spine of the project.

1. **Tokenizer works** → encode/decode a sentence live; show token count.
2. **Data pipeline works** → print a real batch of token IDs + tensor shapes.
3. **Model runs untrained** → generates gibberish → *proves wiring; separates architecture from weights.*
4. **Overfit one batch** → loss → ~0, model parrots one story back → *proves learning works.*
5. **Toy training run (local CPU)** → loss curve bends down; samples get less random.
6. **Real training run (Colab)** → coherent little stories.
7. **Packaged** → on HF Hub with an auto-generated model card.
8. **Converted** → `.gguf` file; quantized (compare Q8_0 vs Q4_K_M).
9. **Running in Ollama** → `ollama run <model>` tells a story. 🎉

---

## 8. Training Strategy

Hand-written training loop so every part is visible:

- AdamW optimizer; cosine learning-rate schedule with warmup; gradient clipping.
- Next-token cross-entropy loss.
- Periodic checkpointing.
- A **sample-generation callback** that prints a story every N steps (watch it learn).
- Loss logged to CSV → matplotlib plot.
- Mixed precision on GPU (Colab), fp32 on CPU.

**Workflow:** develop/debug on `toy` locally (fast intermediate results) → run `small` on a provided Colab notebook → download the checkpoint → continue packaging + inference locally on CPU. This also produces the "CPU vs GPU" tradeoff story for the blog.

---

## 9. Packaging → GGUF → Run (the "build & ship" half)

1. Map trained tensors into `LlamaForCausalLM` with a matching config; save tokenizer; `save_pretrained`.
2. `push_to_hub` with an auto-generated model card — the "publish" moment.
3. `convert_hf_to_gguf.py` → a single `.gguf` file; then quantize (produce Q8_0 and Q4_K_M to compare quality/size).
4. Write an Ollama `Modelfile`; `ollama create` + `ollama run`.
5. **Lab:** hex-peek the GGUF header + metadata; compare quantized output quality.

---

## 10. Documentation Model (reproduction-first, blog-later, decoupled)

Three distinct artifacts:

1. **`DEVLOG.md`** — dated, chronological record of *what we did and why*: decisions + rationale, commands run, configs used, observations, metrics, and **dead-ends/gotchas**. Primary raw material.
2. **`REPRODUCE.md`** — a clean, ordered **"build it yourself from scratch"** guide distilled from the devlog: exact steps, environment setup, commands, expected output at each milestone. Serves the goal of letting someone follow along and recreate the work *without* cloning the repo.
3. **`CHANGELOG.md`** — terse "what changed when."

Plus the teaching artifacts:
- **`CONCEPTS.md`** — a growing plain-language primer/glossary; one entry per term met in the code (embedding, logit, attention, RoPE, RMSNorm, SwiGLU, quantization, GGUF, MoE…), cross-linked to the Lab that demonstrates it.

The **blog articles are a separate, later task**, generated from `DEVLOG.md` + `REPRODUCE.md` using writing constructs the author will provide. Nothing is pre-styled as reader-facing prose now.

---

## 11. Labs (side-experiments — the active-learning mechanism)

Small, runnable scripts/notebooks that open the hood. Initial set (grows as needed):

- `lab01_bpe_by_hand` — train BPE on one paragraph; watch merges form.
- `lab02_attention_peek` — print/visualize attention weights on a 5-token sentence.
- `lab03_embeddings` — nearest-neighbor lookups to show what an embedding "knows."
- `lab04_rope` — visualize how RoPE rotates position information.
- `lab05_sampling` — temperature / top-k effects on generation.
- `lab06_gibberish_to_coherent` — sample the same prompt across checkpoints.
- `lab07_gguf_teardown` — hex-peek the `.gguf`; read its header/metadata.
- `lab08_quant_compare` — Q4 vs Q8 vs fp16 output quality/size.
- *(Phase 3)* `labXX_tiny_moe` — swap the FFN for a 4-expert MoE; watch routing.

---

## 12. Repository Layout

```
model_learn/
├── README.md            # what this is + quickstart
├── DEVLOG.md            # dated narrative record — raw material for the blog
├── REPRODUCE.md         # clean "build it yourself" follow-along guide
├── CHANGELOG.md         # terse "what changed when"
├── CONCEPTS.md          # growing plain-language glossary, linked to Labs
├── requirements.txt
├── .gitignore
├── src/slm/
│   ├── config.py        # dataclass configs (toy vs small)
│   ├── tokenizer.py     # train/load BPE
│   ├── data.py          # load + tokenize + batch TinyStories
│   ├── model.py         # hand-built Llama-style transformer
│   ├── train.py         # training loop + sampling + checkpoints
│   ├── sample.py        # generate from a checkpoint
│   └── export_hf.py     # map weights -> LlamaForCausalLM, push to hub
├── labs/                # the side-experiments
├── notebooks/
│   └── colab_train.ipynb
├── scripts/
│   ├── convert_to_gguf.sh
│   └── Modelfile
├── data/                # gitignored
├── checkpoints/         # gitignored
└── docs/superpowers/specs/
```

---

## 13. Verification & Testing

Each milestone has a concrete pass/fail check, keeping "it works" honest:

- **Init:** untrained model produces gibberish of the correct shape (sanity of wiring).
- **Overfit:** loss on a single batch drops toward ~0 within a few hundred steps.
- **Training:** validation loss decreases; sample text becomes less random over checkpoints.
- **Packaging:** weights reload into `LlamaForCausalLM` and produce identical logits to the hand-built model on a fixed input (round-trip equivalence).
- **GGUF:** converted model loads in llama.cpp and produces coherent output close to the PyTorch model.

Plus light unit tests: model forward-pass output shape; tokenizer encode→decode round-trip.

---

## 14. Compute & Environment

- **Local dev:** OrbStack Linux VM (aarch64), 14 CPU cores, ~11 GiB RAM, no NVIDIA GPU, ~33 GB free disk. Runs `toy` configs and all Labs comfortably.
- **Heavy training:** Google Colab free GPU for the `small` run; download checkpoint afterward.
- **Inference/packaging:** local CPU.

---

## 15. Out of Scope (Phase 1)

- Instruction tuning / chat templates (Phase 2).
- Mixture of Experts; GGUF quantization deep-dive (Phase 3).
- GQA, multi-GPU/distributed training, advanced optimizers, RLHF.
- The actual blog articles (separate downstream task using author-provided writing constructs).

---

## 16. Open Decisions (resolved with defaults, revisit before Colab run)

- **Final `small` config numbers** — defaults in §6.1; tune after seeing local iteration speed.
- **Vocab size** — default 8,192 for `small`; may adjust based on tokenizer quality vs embedding-table size.
- **Quantization levels to publish** — default Q8_0 + Q4_K_M for comparison.
```
