# Phase 1 Self-Study Course — Authoring Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Author a single self-contained Markdown course, `docs/phase1-course.md`, that teaches
the entire Phase 1 arc (tokenizer → model → training → HF export → GGUF → Ollama) as a
structured self-study course with annotated real code, for use as a NotebookLM tutoring source.

**Architecture:** One master document. Front matter (overview + how-to-use-in-NotebookLM +
level + TOC), then 10 modules in build order, then a glossary. Every module follows the same
six-section template. Content is *reconciled from the current repo* (real `src/slm/*.py`,
`DEVLOG.md`, `CONCEPTS.md`, the Phase 1 plan) and fully inlined — no path/Lab cross-references
that NotebookLM can't resolve.

**Tech Stack:** Markdown only. No code changes. Verification uses `grep` and manual cross-checks
against `DEVLOG.md`/`CHANGELOG.md`. Source of truth for embedded code is the live `src/slm/*.py`.

## Global Constraints

- **Single deliverable:** this plan only *creates* `docs/phase1-course.md`. Do not modify `src/`,
  `tests/`, or existing docs. If reconciliation surfaces a genuine error in an existing doc, note
  it in the task's commit body or DEVLOG — do not silently edit it.
- **Self-contained:** no reference names a repo file/Lab/date by path expecting the reader to open
  it. Inline the concept, its code, and its observed output together. (Naming a file as *"this is
  `src/slm/model.py`"* while showing its contents is fine; *"see Lab 03"* without inlining is not.)
- **Code matches current source:** embed code read from the live `src/slm/*.py` this session — NOT
  the Phase 1 plan's pre-implementation snippets (e.g. use the clean `lr_at`, never the convoluted
  draft the plan tells you to replace).
- **Numbers match the record:** every cited figure matches `DEVLOG.md`/`CHANGELOG.md`. Canonical
  values: `SMALL` = 13.8M params; toy run loss ≈ 62 → ≈ 4.3; `small` eval loss 1.81 / perplexity
  6.1; round-trip max logit diff 9.5e-06; GGUF f16 27MB / Q8_0 15MB / Q4_K_M 11MB; vocab TOY 2048 /
  SMALL 8192; special-token ids bos=eos=pad=0; tokenizer hash `fe391dc4`→"gpt-2"; Ollama 0.31.2.
- **Level:** intuition-first, no heavy math. Match the register of the existing `CONCEPTS.md`.
- **Module template (every module, in order):** (1) Learning objectives, (2) Frame, (3) Annotated
  code walkthrough, (4) What we observed, (5) Gotchas & design decisions, (6) Checkpoint (2–4 quiz
  questions + one "explain it back" prompt).
- **Commit cadence:** one commit per task, message prefixed `docs:`, co-author trailer
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

## File Structure

- Create: `docs/phase1-course.md` — the entire deliverable. All tasks append to this one file.
- Read-only sources (never modified): `src/slm/{config,tokenizer,data,model,train,sample,export_hf}.py`,
  `labs/*.py`, `CONCEPTS.md`, `DEVLOG.md`, `REPRODUCE.md`, `CHANGELOG.md`, and
  `docs/superpowers/plans/2026-06-22-slm-phase1-base-model.md`.

## Per-task verification (the "test" for a prose deliverable)

Each task ends by confirming, for the modules it authored:
- All six template sections present.
- Embedded code was copied from the live `src/slm/*.py` read during the task (not the old plan).
- Every number cross-checks against `DEVLOG.md`/`CHANGELOG.md`.
- No dangling references: `grep -nE '\b(Lab [0-9]|see DEVLOG|see CONCEPTS|src/slm/|labs/|2026-07-07)\b' docs/phase1-course.md`
  returns only lines where the referenced content is inlined right there (a filename used as a
  label above its own embedded code is OK; a bare "see X" pointer is not).

---

### Task 1: Scaffold the document + Module 0 (front matter)

**Files:**
- Create: `docs/phase1-course.md`

**Interfaces:**
- Produces: the document skeleton — H1 title, the "How this course works" template description,
  the "How to use this in NotebookLM" note, level/prereqs, a full table of contents linking all 10
  module headings + glossary, and Module 0's big-picture map. Later tasks append modules under the
  TOC's headings.

- [ ] **Step 1: Read the source material for the overview**

Read `README.md`, the Phase 1 plan's Goal/Architecture (lines 1–20), and `CONCEPTS.md` entry
"model compiler" for the framing of the end-to-end pipeline.

- [ ] **Step 2: Write the front matter into `docs/phase1-course.md`**

Author these sections:
- `# Phase 1 — Build a Language Model From Scratch to Ollama: A Self-Study Course`
- **How this course works:** one paragraph naming the six-section template so the reader (and
  NotebookLM) know the rhythm of every module.
- **How to use this in NotebookLM:** state that answers are grounded in this document; give
  concrete prompts — *"Act as my tutor and quiz me on Module 4,"* *"Check my understanding of
  RoPE — ask me to explain it, then correct me,"* *"Give me the audio overview of the training
  module,"* *"I'll answer the Module 6 checkpoint; grade my answers."*
- **Who this is for / level:** intuition-first, Python literacy assumed, no heavy math.
- **Table of contents:** Markdown links to Modules 1–10 + Glossary.
- **Module 0 — The big picture:** the "the ecosystem lets you *download → build → run* a model;
  this course *builds one* end-to-end, then hands it to the same runtimes everyone else uses"
  framing. Include a one-line map of the 10 modules as a pipeline.

- [ ] **Step 3: Verify structure**

Run: `grep -nE '^#{1,3} ' docs/phase1-course.md`
Expected: H1 title, and H2 headings for the front-matter sections + Module 0. TOC lists Modules
1–10 + Glossary.

- [ ] **Step 4: Commit**

```bash
git add docs/phase1-course.md
git commit -m "docs: scaffold Phase 1 course + Module 0 (overview, NotebookLM how-to)"
```

---

### Task 2: Modules 1–2 (Environment & reproducibility; Tokenizer & BPE)

**Files:**
- Modify: `docs/phase1-course.md` (append Modules 1 and 2)

**Interfaces:**
- Consumes: the six-section template established in Task 1.
- Produces: `## Module 1 — Environment & Reproducibility` and `## Module 2 — Tokenizer & BPE`.

- [ ] **Step 1: Read the live sources**

Read `src/slm/tokenizer.py` (whole), `labs/lab01_bpe_by_hand.py`, `pyproject.toml`, and the
`CONCEPTS.md` entries: virtual environment, lockfile, tokenizer, token/token id, BPE, vocabulary,
special token, byte-level/`Ġ`. Read `DEVLOG.md` Task 1 + Task 2 entries for gotchas.

- [ ] **Step 2: Author Module 1 — Environment & Reproducibility**

Six sections. Frame: why an isolated, lockfile-backed env matters for reproducibility. Annotated
code: the `pyproject.toml` deps block + the `[tool.uv.sources]` CPU-torch pin. What we observed:
`uv sync` writes `uv.lock`; `uv export` makes the Colab `requirements.txt`. Gotchas: (1) default
torch wheel pulled ~2GB unused CUDA on this no-GPU box → pinned to the CPU index; (2) that pin
leaked into the Colab-bound `requirements.txt` → fixed with `uv export --no-emit-package torch`.
Checkpoint questions:
- "What does `uv.lock` give you that `pyproject.toml`'s `torch>=2.5` does not?"
- "Why would the CPU torch pin have broken the Colab GPU run if left in `requirements.txt`?"
- Explain-it-back: "In your own words, what is a virtual environment and why use one here?"

- [ ] **Step 3: Author Module 2 — Tokenizer & BPE**

Six sections. Frame: neural nets see only numbers; the tokenizer is the mandatory text↔int
bridge; BPE = repeatedly merge the most frequent adjacent pair. Annotated code: embed the real
`train_tokenizer`/`load_tokenizer`/`encode`/`decode` from `src/slm/tokenizer.py`, annotating the
ByteLevel pre-tokenizer, the `<|endoftext|>` special token pinned to id 0, and byte-level decode.
What we observed: inline Lab 01's output pattern — as `vocab_size` grows (260→320), `"the cat sat"`
collapses from many char tokens into fewer merged tokens; explain the `Ġ` space-prefix symbol.
Gotchas: byte-level means no "unknown token"; `train_tokenizer` had to create its save dir.
Checkpoint questions:
- "Why is nothing ever 'unrepresentable' with a byte-level BPE tokenizer?"
- "What does `Ġcat` mean and how does it differ from `cat`?"
- "Why is `<|endoftext|>` fixed to id 0?"
- Explain-it-back: "Walk through what BPE training actually does, step by step."

- [ ] **Step 4: Verify**

Run: `grep -nE '^## Module [12] ' docs/phase1-course.md` (expect both). Confirm the embedded
tokenizer code matches `src/slm/tokenizer.py`. Run the dangling-ref grep from the Global section.

- [ ] **Step 5: Commit**

```bash
git add docs/phase1-course.md
git commit -m "docs: course Modules 1-2 (environment, tokenizer/BPE)"
```

---

### Task 3: Module 3 (Data pipeline)

**Files:**
- Modify: `docs/phase1-course.md` (append Module 3)

**Interfaces:**
- Consumes: template; Module 2's tokenizer.
- Produces: `## Module 3 — Data Pipeline: Batching and the Shift-by-One Target`.

- [ ] **Step 1: Read the live sources**

Read `src/slm/data.py` (whole), `tests/test_data.py`, and `CONCEPTS.md` entries: context
window/sequence length, batch, next-token target (shift-by-one). Read the `DEVLOG.md` Task 3 entry
(the decode-`x[0]`-vs-`y[0]` demo showing text slid forward by one token).

- [ ] **Step 2: Author Module 3**

Six sections. Frame: language modeling's supervised signal — for input window `x`, target `y` is
the same window shifted one left, so one sequence yields a training example at every position.
Annotated code: embed `tokenize_texts` (EOT id 0 between docs) and `get_batch` (seeded, returns
`(x, y)` with `y = x` shifted), annotating the shift and the deterministic seed. What we observed:
inline the DEVLOG demo — decoding `x[0]` and `y[0]` from a real batch gives the same text slid
forward by exactly one token; batch shapes `(batch_size, context_len)`. Gotchas: `load_tinystories`
streams the real dataset and is deliberately never used in unit tests (tests use a tiny fixture).
Checkpoint questions:
- "Why does one input sequence give you a prediction target at *every* position, not just the end?"
- "What exactly is the relationship between `x` and `y` in a batch?"
- Explain-it-back: "Why must the unit tests never download the real TinyStories dataset?"

- [ ] **Step 3: Verify**

Run: `grep -nE '^## Module 3 ' docs/phase1-course.md`. Confirm embedded code matches
`src/slm/data.py`. Run the dangling-ref grep.

- [ ] **Step 4: Commit**

```bash
git add docs/phase1-course.md
git commit -m "docs: course Module 3 (data pipeline, shift-by-one target)"
```

---

### Task 4: Module 4 (Model internals — RMSNorm, RoPE, attention, SwiGLU)

**Files:**
- Modify: `docs/phase1-course.md` (append Module 4)

**Interfaces:**
- Consumes: template.
- Produces: `## Module 4 — Model Internals: RMSNorm, RoPE, Causal Attention, SwiGLU`.

- [ ] **Step 1: Read the live sources**

Read `src/slm/model.py` (the component classes/functions: `RMSNorm`, `build_rope_cache`,
`rotate_half`, `apply_rope`, `Attention`, `SwiGLU`), `labs/lab02_attention_peek.py`,
`tests/test_model_components.py`, and `CONCEPTS.md` entries: d_model, RMSNorm, RoPE, attention/
causal mask, SwiGLU/feed-forward.

- [ ] **Step 2: Author Module 4**

Six sections; this is dense, so give each component its own annotated sub-part.
- **Frame:** a token is a `d_model`-length vector; RMSNorm keeps magnitudes healthy; RoPE injects
  order by *rotating* q/k (rotation preserves length, encodes *relative* distance); attention is
  the only place tokens exchange info (Q/K/V + causal mask); SwiGLU is per-token "private thinking."
- **Annotated code:** embed `RMSNorm` (float32 trick), `build_rope_cache`+`rotate_half`+`apply_rope`
  (HF-matching convention), `Attention` (q/k/v/o projections, RoPE applied, scaled scores, `triu`
  causal mask, softmax, reshape), `SwiGLU` (gate/up/down, SiLU gate). All bias-free.
- **What we observed:** inline Lab 02's causal-attention matrix (upper triangle = 0.00, i.e. a token
  attends only to itself and earlier tokens); note `test_rope_preserves_shape_and_norm` (rotation
  keeps vector norm) and `test_attention_is_causal` (perturbing the last token leaves earlier
  outputs bit-for-bit identical).
- **Gotchas & decisions:** why RMSNorm over LayerNorm (skips mean-subtraction, cheaper); no GQA in
  Phase 1 (`n_kv_heads == n_heads`); the float32 RMSNorm trick is required for HF numerical match.
- **Checkpoint:**
  - "Why does rotating a query/key vector (RoPE) encode *relative* position rather than absolute?"
  - "What does the causal mask prevent, and why is that necessary at generation time?"
  - "Which sub-layer lets tokens exchange information, and which processes each token privately?"
  - Explain-it-back: "Trace one token vector through a block: norm → attention → norm → SwiGLU."

- [ ] **Step 3: Verify**

Run: `grep -nE '^## Module 4 ' docs/phase1-course.md`. Confirm embedded code matches
`src/slm/model.py`. Run the dangling-ref grep.

- [ ] **Step 4: Commit**

```bash
git add docs/phase1-course.md
git commit -m "docs: course Module 4 (model internals: RMSNorm/RoPE/attention/SwiGLU)"
```

---

### Task 5: Module 5 (Assembly + generation + echo-bias deep dive)

**Files:**
- Modify: `docs/phase1-course.md` (append Module 5)

**Interfaces:**
- Consumes: template; Module 4's components.
- Produces: `## Module 5 — Assembling the Model, Generation, and the Untrained "Echo Bias"`.

- [ ] **Step 1: Read the live sources**

Read `src/slm/model.py` (`Block`, `LlamaSLM`, `generate`), `labs/lab03_gibberish.py`,
`tests/test_model.py`, and `CONCEPTS.md` entries: embedding table, how the embedding table learns
despite being context-blind, pre-norm residual block, logits, autoregressive generation, weight
tying, tied-embedding "echo" bias. Read the full `DEVLOG.md` 2026-07-07 echo-bias investigation.

- [ ] **Step 2: Author Module 5**

- **Frame:** stack blocks with pre-norm residuals (`x = x + attn(norm(x))`), a final norm, and a
  tied `lm_head`; generation is autoregressive (predict → sample → append → repeat).
- **Annotated code:** embed `Block`, `LlamaSLM.__init__`/`forward` (tie `lm_head.weight` to
  `embed_tokens.weight`; RoPE cache as a non-persistent buffer), and `generate` (temperature,
  top_k, context cropping).
- **What we observed:** inline a Lab 03 gibberish sample — an *untrained* model already runs
  end-to-end; only the weights are missing. The pipeline works instantly; quality doesn't.
- **Deep dive — the echo bias (the real detour):** before any training, tied-embedding + pre-norm
  models strongly favor re-predicting the most recent token. Cause: residuals keep the input
  token's embedding direction dominant in the final hidden state, and because `lm_head` reuses that
  same matrix, that token's self-dot-product logit dwarfs all cross-dot-product logits (~30–40 logit
  gap → `max_prob ≈ 1.0`). Confirmed across TOY/SMALL, multiple seeds, varied inputs; vanishes when
  `tie_embeddings=False`. This is why `test_untrained_output_is_high_entropy` was wrong and had to
  be replaced — a live debugging finding, not a bug. Inline the before/after logic.
- **Gotchas & decisions:** weight tying halves the biggest parameter block but is the root of the
  echo bias; training overwrites it.
- **Checkpoint:**
  - "An untrained model produces gibberish but still runs. What does that tell you about the split
    between architecture and weights?"
  - "Explain the mechanism of the echo bias — why does weight tying + residuals cause it?"
  - "Why was the original high-entropy assertion in the test *wrong*?"
  - Explain-it-back: "Describe one full step of autoregressive generation."

- [ ] **Step 3: Verify**

Run: `grep -nE '^## Module 5 ' docs/phase1-course.md`. Confirm embedded code matches
`src/slm/model.py`. Run the dangling-ref grep.

- [ ] **Step 4: Commit**

```bash
git add docs/phase1-course.md
git commit -m "docs: course Module 5 (assembly, generation, echo-bias deep dive)"
```

---

### Task 6: Modules 6–7 (Training; the real Colab run)

**Files:**
- Modify: `docs/phase1-course.md` (append Modules 6 and 7)

**Interfaces:**
- Consumes: template; the assembled `LlamaSLM`; `get_batch`.
- Produces: `## Module 6 — Training` and `## Module 7 — The Real Run (small on Colab)`.

- [ ] **Step 1: Read the live sources**

Read `src/slm/train.py` (whole — use the *clean* `lr_at`, `train`, checkpoint save/load,
`plot_loss`), `src/slm/sample.py`, `labs/lab05_sampling.py`, `tests/test_train.py`,
`notebooks/` (the Colab notebook, if present) and `CONCEPTS.md` entries: loss/cross-entropy,
optimizer/AdamW, LR schedule, gradient clipping, checkpoint, overfitting-one-batch, device
portability, temperature/top-k. Read `DEVLOG.md` Tasks 5–6.

- [ ] **Step 2: Author Module 6 — Training**

- **Frame:** loss = how wrong (cross-entropy, penalizes confident-and-wrong); AdamW updates each
  param; warmup+cosine schedule; grad clip as a safety cap; overfit-one-batch as the "does learning
  work at all" proof.
- **Annotated code:** embed the clean `lr_at` (linear warmup → peak at `warmup_steps` → cosine
  decay), the `train` loop (cross-entropy over reshaped logits, per-step LR set, zero_grad/backward/
  clip/step), and `save_checkpoint`/`load_checkpoint` (config stored as plain dict →
  `weights_only=True`; `map_location="cpu"`). Embed the temperature/top-k reshaping from Lab 05.
- **What we observed:** overfit-one-batch drives loss < 1.0 (proof of learning); the toy run loss
  ≈ 62 → ≈ 4.3 over ~800 steps on CPU (~2 min); samples drift gibberish → story-shaped. Note the
  uniform-guess baseline `ln(vocab_size)` ≈ 7.6 and why the untrained model *starts above* it (~62)
  due to the echo bias (confidently wrong, not merely uncertain).
- **Gotchas & decisions:** the plan shipped a deliberately convoluted `lr_at` draft to be replaced —
  the clean version is authoritative; `weights_only=True` avoids the unpickle code-execution risk;
  `load_checkpoint` needs `map_location="cpu"` to load GPU-saved checkpoints on this box.
- **Checkpoint:**
  - "Why ramp the learning rate up (warmup) instead of starting at the peak?"
  - "What does overfitting a single batch prove, and why do it before a real dataset?"
  - "Why store the config as a plain dict and load with `weights_only=True`?"
  - Explain-it-back: "Why does an untrained model score loss ~62 when uniform guessing is only ~7.6?"

- [ ] **Step 3: Author Module 7 — The Real Run**

- **Frame:** same `train()`/`LlamaSLM`, just the `SMALL` config on a GPU — device-portable code
  means TOY→SMALL is a config swap, not a rewrite.
- **Annotated code:** the `SMALL` config (vocab 8192, d_model 384, 6 layers/heads, head_dim 64,
  ffn 1024, ctx 512 → 13.8M params) and the device-portability line in `train()` (batches moved to
  the model's device).
- **What we observed:** trained on Colab T4; downloaded to `checkpoints/small.pt` + `small_tok.json`;
  local eval loss **1.81 / perplexity 6.1** (vs toy ≈ 4.3 / 74); generates coherent TinyStories
  prose — the coherence milestone.
- **Gotchas & decisions:** Colab pip clobbered numpy → minimal `pip install datasets tokenizers`;
  GPU-saved checkpoint failed to load on CPU → `map_location="cpu"`; scaling deferred (TinyStories
  saturates at tens-of-M params; bigger hurts the instant-CPU demo).
- **Checkpoint:**
  - "What made scaling from TOY to SMALL a config change rather than a code change?"
  - "What does perplexity 6.1 mean in plain terms?"
  - Explain-it-back: "Why did we finish Phase 1 at ~14M params instead of going bigger?"

- [ ] **Step 4: Verify**

Run: `grep -nE '^## Module [67] ' docs/phase1-course.md` (expect both). Confirm embedded code
matches `src/slm/train.py`/`config.py`; confirm 13.8M / 1.81 / 6.1 match `DEVLOG.md`. Run the
dangling-ref grep.

- [ ] **Step 5: Commit**

```bash
git add docs/phase1-course.md
git commit -m "docs: course Modules 6-7 (training loop, the real Colab run)"
```

---

### Task 7: Modules 8–9 (HF packaging + round-trip; GGUF/quantization/Ollama)

**Files:**
- Modify: `docs/phase1-course.md` (append Modules 8 and 9)

**Interfaces:**
- Consumes: template; the trained checkpoint.
- Produces: `## Module 8 — Packaging to Hugging Face Format` and `## Module 9 — GGUF, Quantization, and Ollama`.

- [ ] **Step 1: Read the live sources**

Read `src/slm/export_hf.py` (whole), `tests/test_export.py`, `scripts/convert_to_gguf.sh`,
`scripts/Modelfile`, `labs/lab07_gguf_teardown.py`, `labs/lab08_quant_compare.py`, and
`CONCEPTS.md` entries: HF format, round-trip equivalence, config.json, safetensors, model card,
GGUF, quantization, Q8_0 vs Q4_K_M, inference engine, Modelfile. Read `DEVLOG.md` Tasks 7–8.

- [ ] **Step 2: Author Module 8 — Packaging to HF Format**

- **Frame:** the HF directory (`config.json` + `model.safetensors` + tokenizer files) is the
  "standard container" the whole ecosystem opens with zero custom code. Analogy: hand-built weights
  = the compiled binary; HF format = repackaging into the container every runtime already knows.
- **Annotated code:** embed the weight-copy from `LlamaSLM` into stock `LlamaForCausalLM`, the
  `to_hf_config` (bos=eos=pad=0 — the fix so generation stops at story end), and `push`.
- **What we observed:** round-trip test — hand-built vs official `LlamaForCausalLM` give identical
  logits, **max abs diff 9.5e-06** (pure float noise) → the architecture is bit-for-bit Llama.
  Published to private HF repo `vysakhpillai/tinystories-slm`.
- **Gotchas & decisions:** exported config first had Llama's default eos=2 → generation never
  stopped → fixed to 0; HF username `vysakhpillai` ≠ GitHub `vppillai`; **security:** a write HF
  token was pasted in chat and must be revoked (stored at `~/.cache/huggingface/token`).
- **Checkpoint:**
  - "Why does the round-trip logit diff of ~1e-5 matter — what does it prove?"
  - "What breaks downstream if the exported `config.json` keeps Llama's default eos=2?"
  - Explain-it-back: "Why can our hand-built model load into `transformers` with no custom code?"

- [ ] **Step 3: Author Module 9 — GGUF, Quantization, and Ollama**

- **Frame:** GGUF = the single self-describing file llama.cpp/Ollama consume (header + tensors +
  metadata); quantization = lower precision to shrink + speed up CPU inference; the inference engine
  loads the GGUF and runs the forward pass with optimized CPU kernels.
- **Annotated code:** embed `scripts/convert_to_gguf.sh` (HF → f16 GGUF → quantize) and
  `scripts/Modelfile` (FROM + temperature/top_k/stop + TEMPLATE); annotate the Lab 07 header read.
- **What we observed:** f16 27MB → Q8_0 15MB → Q4_K_M 11MB; all three still tell coherent stories
  (Lab 08); Lab 07 header: Q8_0 = 56 tensors / 34 metadata entries; on this tiny model 36/56 tensors
  fell back to higher precision under Q4_K_M (benign, shape-driven); `ollama run tinystories-slm`
  generates coherent stories on CPU.
- **Gotchas & decisions:** llama.cpp built with `-j 4` to avoid OOM; custom tokenizer hash
  `fe391dc4` registered as "gpt-2" in `convert_hf_to_gguf.py`; Ollama 0.31.2 needs `ollama serve`
  started manually + zstd + an **absolute** `FROM` path.
- **Checkpoint:**
  - "What does 'self-describing' mean for a GGUF file, and why does it need no separate config?"
  - "Why does the same prompt+seed produce slightly different text across quant levels?"
  - "What is the Modelfile's job on top of the GGUF?"
  - Explain-it-back: "Trace the file from `small.pt` all the way to `ollama run`."

- [ ] **Step 4: Verify**

Run: `grep -nE '^## Module [89] ' docs/phase1-course.md` (expect both). Confirm embedded code
matches `src/slm/export_hf.py` and the `scripts/`; confirm 9.5e-06 and the GGUF sizes match
`DEVLOG.md`/`CHANGELOG.md`. Run the dangling-ref grep.

- [ ] **Step 5: Commit**

```bash
git add docs/phase1-course.md
git commit -m "docs: course Modules 8-9 (HF export/round-trip, GGUF/quant/Ollama)"
```

---

### Task 8: Module 10 (transferable mental model) + Glossary

**Files:**
- Modify: `docs/phase1-course.md` (append Module 10 and the Glossary)

**Interfaces:**
- Consumes: template; all prior modules (the glossary consolidates their concepts).
- Produces: `## Module 10 — The Transferable Mental Model: Model Compilers` and `## Glossary`.

- [ ] **Step 1: Read the live sources**

Read the `CONCEPTS.md` entry "model compiler / deep-learning compiler" in full, plus the Phase 1
plan's Phase-4 framing (spec/plan mentions of Tenstorrent MLIR/Metalium). Then read *all* of
`CONCEPTS.md` for the glossary consolidation.

- [ ] **Step 2: Author Module 10 — The Transferable Mental Model**

- **Frame:** the GGUF path IS a miniature model compiler. Naming its stages is the model that
  generalizes to any target hardware.
- **Annotated mapping (not code — the pipeline mapping):**
  `convert_hf_to_gguf.py` = **export to IR**; `llama-quantize` (f16→Q8_0→Q4_K_M) = an
  **optimization pass**; llama.cpp/Ollama picking CPU kernels at load = **codegen + runtime**.
- **What we observed / where it generalizes:** the same shape appears in ONNX Runtime, TVM,
  MLIR/IREE, TensorRT, OpenVINO, MLC-LLM — and in vendor stacks like Tenstorrent's MLIR/Metalium.
  Once "convert a model for a target" clicks, deploying HF models on custom hardware is just:
  export → IR → lower/optimize → hardware engine. (Phase 4 expands this.)
- **Gotchas & decisions:** this is the deliberate bridge from Phase 1's CPU/Ollama path to
  vpillai's day-job hardware stack.
- **Checkpoint:**
  - "Map each of the three GGUF-path stages to its model-compiler role."
  - "Why is `llama-quantize` an 'optimization pass' rather than a format change?"
  - Explain-it-back: "State the four-step mental model for deploying a model onto new hardware."

- [ ] **Step 3: Author the Glossary**

Consolidate the ~40 `CONCEPTS.md` entries into an alphabetized `## Glossary`, deduplicated, each
2–4 sentences, intuition-first. Strip in-repo path references (or convert them to "see Module N").
This is the quick-lookup companion to the narrative modules.

- [ ] **Step 4: Verify**

Run: `grep -nE '^## (Module 10|Glossary)' docs/phase1-course.md` (expect both). Spot-check that
every glossary term also appears in a module. Run the dangling-ref grep.

- [ ] **Step 5: Commit**

```bash
git add docs/phase1-course.md
git commit -m "docs: course Module 10 (model-compiler model) + Glossary"
```

---

### Task 9: Final integration pass

**Files:**
- Modify: `docs/phase1-course.md` (TOC finalization, cross-cut consistency, read-through)

**Interfaces:**
- Consumes: the whole document.
- Produces: a finished, self-contained, internally consistent course.

- [ ] **Step 1: Finalize the table of contents**

Confirm the Task 1 TOC links resolve to every real `## Module N`/`## Glossary` heading (fix any
drift in titles introduced while authoring).

- [ ] **Step 2: Dangling-reference sweep**

Run: `grep -nE '\b(Lab [0-9]|see DEVLOG|see CONCEPTS|labs/|2026-07-07)\b' docs/phase1-course.md`
Expected: every hit is a filename/label sitting directly above its own inlined code/output — no
bare "see X" pointers. Inline or rephrase any that remain.

- [ ] **Step 3: Number-consistency check**

Verify against `DEVLOG.md`/`CHANGELOG.md`: 13.8M params; toy loss ≈ 62 → ≈ 4.3; small eval loss
1.81 / ppl 6.1; round-trip diff 9.5e-06; GGUF 27/15/11 MB; vocab 2048/8192; eos=bos=pad=0. Fix any
mismatch.

- [ ] **Step 4: Self-contained read-through**

Read the whole file top-to-bottom as if you had never seen the repo. Confirm each module has all
six template sections and that no explanation depends on opening an external file. Fix gaps inline.

- [ ] **Step 5: Commit**

```bash
git add docs/phase1-course.md
git commit -m "docs: finalize Phase 1 self-study course (TOC, consistency, read-through)"
```

---

## Self-Review (author's check against the spec)

**Spec coverage:**
- Single combined master doc → Task 1 creates it; all tasks append to the one file. ✓
- Structured self-study course (objectives, ordered path, checkpoints) → six-section template with
  checkpoints in every module task. ✓
- Annotated real code → each module task's Step 1 reads the live `src/slm/*.py` before embedding. ✓
- No dangling references → per-task dangling-ref grep + Task 9 Step 2 sweep. ✓
- Observed reality embedded → every module has a "What we observed" section with canonical numbers. ✓
- Hybrid structure, pipeline order, deep dives → Modules 1–10 in build order; deep dives in Modules
  5 (echo bias) and 4 (RoPE). ✓
- Glossary → Task 8 Step 3. ✓
- Level intuition-first → Global Constraints + every Frame section. ✓
- Out of scope (no src/test/doc edits; flag errors) → Global Constraints. ✓
- Numbers match record → Global Constraints canonical list + Task 9 Step 3. ✓

**Placeholder scan:** no "TBD/TODO"; checkpoint questions are written out verbatim, not deferred;
each module task enumerates the exact code to embed and outputs to inline. ✓

**Consistency:** module titles used in tasks match the TOC/verification greps; canonical numbers are
stated once in Global Constraints and reused verbatim. ✓
