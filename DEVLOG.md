# Developer Log

Dated, chronological record of what we did and why — including dead-ends.

## 2026-07-07 — Task 1: scaffold + configs

- Created package skeleton (`src/slm/`), `ModelConfig` dataclass with `TOY`
  (213,312 params) and `SMALL` (13,767,552 params) presets, pytest wiring
  (`pytest.ini` with `pythonpath = src`).
- `uv` 0.11.21 was already installed on this box; no install step needed.
- **Gotcha 1 — default `torch` wheel pulls in CUDA on a CPU-only box.**
  Plain `uv sync` resolved `torch==2.12.1+cu130` plus ~20 `nvidia-*` packages
  (~4.8GB venv total), even though this machine has no NVIDIA GPU
  (`torch.cuda.is_available()` was `False` regardless). PyPI's default Linux
  torch wheel always bundles CUDA runtime deps; there's no install-time
  auto-detection. Fixed by pinning torch to the CPU wheel index in
  `pyproject.toml`:
  ```toml
  [tool.uv.sources]
  torch = { index = "pytorch-cpu" }

  [[tool.uv.index]]
  name = "pytorch-cpu"
  url = "https://download.pytorch.org/whl/cpu"
  explicit = true
  ```
  Re-running `uv sync` after this dropped the venv to 1.1GB and installed
  `torch==2.12.1+cpu`.
- **Gotcha 2 — that same pin leaks into `requirements.txt` and would break
  Colab.** `uv export` propagates the CPU-only pin, so the naive
  `requirements.txt` said `torch==2.12.1+cpu` for all non-macOS platforms —
  installing it on Colab would replace Colab's preinstalled GPU-matched
  torch with a CPU build, silently killing GPU acceleration for the future
  `small` training run (Task 6). Fixed by excluding torch from the export:
  `uv export --no-hashes --format requirements-txt --no-emit-package torch -o requirements.txt`.
  Colab already ships a working GPU torch; the Colab notebook (Task 6) will
  note explicitly that `requirements.txt` intentionally omits torch.
- Working install commands (for reference / REPRODUCE.md):
  ```bash
  uv venv
  uv sync
  uv export --no-hashes --format requirements-txt --no-emit-package torch -o requirements.txt
  ```

## 2026-07-07 — Task 2: BPE tokenizer

- Implemented `train_tokenizer`/`load_tokenizer`/`encode`/`decode` in
  `src/slm/tokenizer.py` using HF `tokenizers` (BPE model, byte-level
  pre-tokenizer/decoder, `<|endoftext|>` pinned to id 0 as the only special
  token). All 3 tests pass first try — no gotchas this task.
- Lab 01 (`labs/lab01_bpe_by_hand.py`) confirms BPE merges forming live: at
  `vocab_size=260`, `"the cat sat"` tokenizes as
  `['the', 'Ġ', 'c', 'at', 'Ġ', 's', 'at']` (cat/sat still split); by
  `vocab_size=270` it collapses to `['the', 'Ġcat', 'Ġsat']` — common chunks
  become single tokens as the vocab budget grows.
- Noted the `Ġ` symbol (byte-level pre-tokenizer's visible stand-in for a
  leading space, a GPT-2-era convention) in `CONCEPTS.md` — `Ġcat` and `cat`
  are different tokens depending on whether a space precedes the word.

## 2026-07-07 — Side note: embeddings are context-blind but gradient-shaped

Before starting Task 3, dug into a question that came up naturally: if the
embedding table is just a per-token lookup with no notion of context, how
does training "teach" it anything context-dependent? Answer: the lookup
itself never changes based on context, but the *gradient* that updates each
row does — it's computed from whatever context that token happened to
appear in during a given training step. Demonstrated with a throwaway
(non-project) script:

```python
import torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0)
emb = nn.Embedding(10, 4)
proj = nn.Linear(4, 10, bias=False)
opt = torch.optim.SGD(list(emb.parameters()) + list(proj.parameters()), lr=0.5)

before = emb.weight[3].clone()
for target in [7, 1]:               # token 3 used in two different "contexts"
    x = emb(torch.tensor([3]))      # same context-blind lookup both times
    loss = F.cross_entropy(proj(x), torch.tensor([target]))
    opt.zero_grad(); loss.backward(); opt.step()
after = emb.weight[3].clone()
```

Output: token 3's row moved from `[0.1198, 1.2377, 1.1168, -0.2473]` to
`[0.2044, 1.0753, 0.8215, -0.1826]` after two gradient steps with
*conflicting* targets (7, then 1) — concrete proof that the same
context-blind row gets reshaped by context-dependent training signal.
Landed as two `CONCEPTS.md` entries: `d_model`, `embedding table`, and "how
the embedding table learns despite being context-blind." Also broke down
`n_params()` by component for `TOY` vs `SMALL` — embedding table is 61.4% of
`TOY` (tiny `d_model` relative to `vocab_size`) but only 22.8% of `SMALL`;
feed-forward becomes the largest component (51.4%) as the model scales up,
since attention/FFN cost scales with `n_layers` while the embedding table
does not.

## 2026-07-07 — Task 3: data pipeline

- Implemented `tokenize_texts` (flattens texts into one token stream with
  `EOT_ID=0` inserted between documents) and `get_batch` (random windowing,
  `(x, y)` shifted by one, deterministic given a `seed`) in `src/slm/data.py`.
  All 3 tests pass first try.
- Added `tests/fixtures/tiny_stories.txt` (30 short lines) so unit tests
  never touch the real network/dataset — `load_tinystories()` (real HF
  dataset loader) is exercised only by actual training runs later, per the
  plan's "tests never download the internet" constraint.
- Printed a real batch and decoded it back to text to see the shift-by-one
  target directly: with `context_len=12`, `x[0]` decoded to
  `'kled above as the t'` and `y[0]` decoded to `'led above as the tw'` —
  the same window, slid forward by exactly one token.

## 2026-07-07 — Task 4a: model components (RMSNorm, RoPE, attention, SwiGLU)

- Implemented `RMSNorm`, `build_rope_cache`/`apply_rope`, `Attention`
  (causal, RoPE-applied Q/K), and `SwiGLU` in `src/slm/model.py`. All 4
  tests pass first try — no gotchas this task.
- Verified causality concretely (beyond the test's assertion): ran
  `Attention` on a 6-token sequence, perturbed only the last token by
  `+10.0`, and diffed the outputs position-by-position. Positions 0-4 were
  bit-for-bit identical (`0.000000` difference); only position 5 changed —
  direct numeric proof that the causal mask blocks any influence from
  future tokens.

## 2026-07-07 — Task 4b: assemble LlamaSLM + generate (with a real debugging detour)

- Appended `Block` (pre-norm residual) and `LlamaSLM` (embed → layers →
  norm → tied `lm_head`, plus `generate()`) to `src/slm/model.py`.
- **Gotcha 3 — the plan's `test_untrained_output_is_high_entropy` failed**,
  and not for a trivial reason. Investigated with systematic debugging
  (root cause before fix):
  - Symptom: `probs.max()` was exactly `1.0` (not "somewhat peaked" — fully
    saturated), for the degenerate all-token-0 input the plan's test used.
  - Hypothesis 1 (repetition-specific artifact): ruled out — varied,
    non-repeated random inputs showed the identical saturation.
  - Hypothesis 2 (something special about token id 0): ruled out —
    embedding row norms for token 0 were unremarkable (7.8-10.8 range, no
    outlier); with independent seeds, the dominant predicted token was
    always *whichever token was last in the input*, not always id 0.
  - Root cause confirmed: **weight tying**. Pre-norm residual connections
    keep the most recent token's embedding direction dominant through the
    stack; `lm_head` reuses that exact embedding matrix, so that token's
    self-dot-product logit crushes every other token's cross-dot-product
    logit (~30-40 logit gap). Verified by setting `tie_embeddings=False`
    on an otherwise-identical config: `max_prob` dropped to ~0.002-0.005
    (near the 1/2048 uniform baseline) and the echo pattern vanished.
    Verified the bias persists at `SMALL` scale too (6 layers, `d_model=384`)
    — not a `TOY`-only artifact of extreme smallness.
  - Conclusion: not a code bug. The *test's assumption* ("untrained ⇒
    near-uniform") is false for this architecture as specified (tied
    embeddings, matching real Llama). The design spec's actual milestone
    goal (§7, milestone 3 — "generates gibberish → proves wiring") doesn't
    require near-uniformity.
  - Fix: replaced the assertion with one that checks what's actually true
    and still meaningfully guards against a broken/dead forward pass — the
    output is a valid probability distribution, and it changes across
    different inputs (proves the wiring is really connected end-to-end).
    Renamed to `test_untrained_forward_is_valid_and_input_dependent`.
  - Landed as a `CONCEPTS.md` entry: "tied-embedding echo bias at
    initialization."
- Lab 02 (`labs/lab02_attention_peek.py`) confirms the lower-triangular
  causal attention pattern visually.
- Lab 03 (`labs/lab03_gibberish.py`) turned into a second, very visible
  confirmation of the echo bias: prompting the untrained model with
  `"Once upon a time"` degenerates into `"time time time time..."` forever.
  Separately, seeding generation from `<|endoftext|>` (id 0) — the
  worst-case trigger — produces 20+ repeats of token 0, which `decode()`
  renders as an empty string (tokenizers hide special tokens by default),
  so the lab prints raw token ids alongside decoded text to make this
  visible rather than confusing.
- **Follow-up (model switch to Opus 4.8, review pass):** independently
  re-verified the echo-bias mechanism quantitatively — for a random input,
  the untrained model's top logit (59.42) equals `|embed(last_token)|²`
  (59.16), the exact predicted boost from the tied `lm_head` seeing the
  residual-carried embedding. Also strengthened the Task-4b test: the
  original replacement asserted `probs.sum() ≈ 1.0` (vacuous — softmax
  always sums to 1) and could let a NaN-producing forward slip through.
  Split into two meaningful tests: `test_untrained_forward_is_finite_and_input_dependent`
  (isfinite + input-dependence, catches dead/NaN forward) and
  `test_untrained_model_echoes_last_token` (asserts argmax == last input
  token — documents the verified echo property directly).

## 2026-07-07 — Task 5: training loop + overfit + toy run (Opus 4.8)

- Implemented `src/slm/train.py`: `TrainConfig`, `lr_at` (clean linear-warmup
  + cosine-decay — skipped the plan's deliberately-convoluted first draft),
  `train` (AdamW betas=(0.9,0.95), per-step LR schedule, grad clipping, CSV
  loss log, periodic sample callback), `save_checkpoint`/`load_checkpoint`
  (weights_only-safe: config stored as a plain dict), `plot_loss`. Also
  `src/slm/sample.py` (`generate_text`) and `labs/lab05_sampling.py`.
- **Milestone: learning works (overfit one batch).** With a single fixed
  batch repeated, loss collapsed 62.2 → 0.0000 by ~step 50. The starting
  loss of ~62 is ~8x worse than the uniform-guess baseline `ln(2048)≈7.6` —
  a direct, quantitative fingerprint of the Task-4b echo bias (the model is
  confidently wrong, betting on the current token instead of the next). The
  first ~50 steps are mostly the optimizer *unlearning* that bias.
- **Gotcha 4 — first toy run crashed:** `train_tokenizer` called
  `tok.save("checkpoints/toy_tok.json")` before `checkpoints/` existed
  (`Exception: No such file or directory`). Root cause: `train_tokenizer`
  wrote a file without ensuring its parent dir, unlike `save_checkpoint`/
  `_write_csv` which both `os.makedirs` first. Fixed at the source — added
  `os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)` to
  `train_tokenizer`. Existing tokenizer tests use `tmp_path` (already
  exists) so were unaffected.
- **Also enhanced `load_tinystories`** to *stream* when a `limit` is given,
  so the local toy run pulls only 2000 stories instead of downloading the
  full ~1.9GB corpus. Full download still happens for `limit=None` (the
  Colab `small` run). Same signature.
- **Milestone: toy training run (local CPU).** 2000 TinyStories, vocab 2048,
  TOY model, 800 steps, ~2 min on CPU. Loss: 62.4 → 12.1 (step 50) → 6.5
  (100) → 5.2 (200) → plateau ~4.3 (800). Samples climbed the coherence
  ladder:
  - step 200: "Once upon a time, there was a little girl named a saw a time.
    She had a it was very in as and." (real words + opening grammar,
    nonsense semantics)
  - step 600: "Once upon a time, there was a little girl named Tim. Every
    day excited one t adventure in the water and was so happy." (named
    character, coherent clauses)
  - step 800: "...She was very happy and he thought, he wanted to play with
    her. Once upon a time there were two friends with a" (recognizable story
    structure)
  Loss plateaus ~4.3 and wiggles slightly (per-batch loss is noisy since
  `fixed_batch=False` draws a fresh random batch each step; and 213K params
  is capacity-limited). Real coherence is the job of the SMALL Colab run.
  Artifacts (gitignored): `checkpoints/{toy.pt,toy_tok.json,toy_loss.csv,toy_loss.png}`.

## 2026-07-07 — Task 6 Half A: Colab notebook authored (run pending, Opus 4.8)

Task 6 is a handoff: the `small` run needs a Colab GPU (browser + Google
account), so this session authored everything needed to run it and left the
actual run to the user. Authored `notebooks/colab_train.ipynb` (9 cells) and
`notebooks/README.md` (runbook).

Three real bugs/limits caught and fixed *before* handoff, all of which the
CPU toy run masked:
- **Device portability.** The plan's cell used `torch.set_default_device("cuda")`,
  which would crash `get_batch` — it builds indices with a CPU `torch.Generator`,
  and `torch.randint(..., generator=cpu_gen)` errors when the default device is
  CUDA. Fixed properly in `train.py`: it now moves each batch to the model's
  device (`next(model.parameters()).device`) and `_print_sample` seeds its start
  tensor on that device too. No `set_default_device` needed; CPU path is a no-op.
- **Per-step re-tensorization.** `get_batch` did `torch.tensor(data)` every call.
  Fine at 470K tokens (toy); at Colab scale (tens of millions) it re-copies the
  whole stream every step and dominates runtime. Fixed: `train()` tensorizes the
  stream once before the loop; `get_batch` now accepts a tensor (or list, for
  the tests) and indexes it directly.
- **RAM.** `tokenize_texts` returns a Python `list[int]`; the *full* TinyStories
  (~470M tokens) as a Python list is ~13GB and would OOM Colab's ~12GB. The
  notebook uses `limit=200_000` stories (~47M tokens) — ample for coherent
  output from a 14M model, and the deliberate "finalize small config before the
  Colab run" call from spec §16.

Notebook config: `SMALL`, `lr=6e-4`, `warmup_steps=200`, `max_steps=20000`,
`batch_size=64`, `context_len=512`, fp32. `requirements.txt` correctly omits
torch so Colab keeps its GPU build (the Task-1 gotcha-2 fix pays off here).

**Pending (Half B / handoff):** push repo to a remote, run on Colab T4,
download `small.{pt,tok.json,loss.png}` into `checkpoints/`, then record the
real final loss + a verbatim story here and finish Task 6 Steps 4-6.

### 2026-07-07 — Task 6 handoff, gotcha 5: full requirements.txt breaks Colab

Repo pushed to https://github.com/vppillai/model_learn (public, verified
unauthenticated-cloneable). First Colab attempt used the notebook's original
`pip install -r requirements.txt` and hit two problems:
1. pip printed a wall of resolver conflicts — our full transitive pins
   (pandas 3.0.3, numpy 2.5.1, requests 2.34.2, rich 15, fsspec 2026.4)
   collide with Colab's co-tuned packages (google-colab, cudf, numba, ...).
2. Cell 2 then died with `ImportError: cannot import name '_center' from
   'numpy._core.umath'` — the numpy upgrade happened *inside a live kernel*
   that had already imported Colab's original numpy, leaving numpy's C
   extension and Python files at mismatched versions (broken install state).
Root cause: `requirements.txt` is the right artifact for a *fresh, empty* uv
venv (total reproducibility) but the *wrong* one for Colab's already-populated
environment. Fix: notebook Cell 1 now installs only our two direct deps
(`datasets tokenizers`); torch/numpy/pandas/matplotlib are already on Colab
and left untouched. Added a Troubleshooting section to `notebooks/README.md`
(numpy `_center` error → Restart session; never `-r requirements.txt` on
Colab). Recovery for a session already in the broken state: Runtime → Restart
session, then rerun with the fixed Cell 1.

### 2026-07-09 — Task 6 complete: coherent-stories milestone reached

Colab run succeeded on the fixed notebook. Step-0 loss was **361.7** — a lovely
confirmation of the echo-bias theory: initial loss ≈ `d_model` (SMALL's 384),
just as TOY's ~62 tracked its d_model of 64 (init loss ≈ `|embed(last)|²` ≈
d_model, since embedding entries are ~unit variance). Loss plunged to single
digits within a few hundred steps, then declined gradually over the full 20,000
steps (confirmed by the `small_loss.png` x-axis).

Config: `SMALL` exactly (vocab 8192, d_model 384, 6 layers, 6 heads, ctx 512),
**13,767,552 params** — the number `n_params()` predicted in Task 1 before any
model code existed. Run params: lr 6e-4, warmup 200, max_steps 20000, bs 64,
fp32, `limit=200_000` stories → 44,290,410-token stream. GPU: Colab T4 free
tier (exact wall-clock not recorded).

**Local eval (CPU, fresh 500-story sample, 30 batches): loss 1.811,
perplexity 6.1** — vs the toy run's ~4.3 / ppl ~74 and the uniform baseline
ln(8192)=9.01. ~12x better perplexity than toy: the capacity + data payoff.

Verbatim generated story (prompt "Once upon a time", temp 0.8, top_k 40, seed 0):
> Once upon a time, there was a little boy named Tim. Tim loved to take pictures
> with his camera. One day, Tim went to the park with his mom. At the park, Tim
> found a nice spot under a big tree. Tim saw a bird's friend, Sarah. Tim asked,
> "Do you like my camera?" Sarah said, "Yes, it's very pretty." Tim took a photo
> of Tim and kept it in his pocket. They played and laughed and had fun. Soon

Coherent grammar, named characters, dialogue, a plot arc — real TinyStories
prose. Minor 14M-on-a-subset artifacts remain ("Tim took a photo of Tim"; the
invented word "lonari" in another sample) but the milestone is unambiguously hit.

**Gotcha 6 — GPU checkpoint won't load on a CPU box.** `torch.save` records each
tensor's device, so `small.pt` (saved on CUDA) carried CUDA location tags;
`load_checkpoint`'s `torch.load(...)` (no `map_location`) failed on this
CPU-only machine with "Attempting to deserialize object on a CUDA device but
torch.cuda.is_available() is False." This is *the* project's core scenario
(download a model trained elsewhere, run it locally), so fixed at the source:
`load_checkpoint` now passes `map_location="cpu"` (inference/packaging here is
always CPU; callers can `.to(device)` afterward). Alternative would be moving
the state_dict to CPU before saving, but the loader fix is the robust general
one. CPU→CPU load is unaffected (no-op), so existing tests still pass.

## 2026-07-09 — Task 7: export to HF format + published to Hub

- Implemented `src/slm/export_hf.py`: `to_hf_config`, `_copy_weights_into_hf`
  (renames our tensors to HF names), `export_to_hf`, `push`.
- **Round-trip test passed first try** even against transformers 5.13.0 (plan
  assumed 4.45+): hand-built `LlamaSLM` vs `LlamaForCausalLM` with our weights
  copied in — **max abs logit diff 9.54e-06**, mean 8.78e-07 (pure float
  noise). Proof the from-scratch architecture is bit-for-bit Llama. This is
  the Task-1 "HF-exact" constraint paying off; nothing to fix.
- Exported the trained `small` model to `export/tinystories-slm/` (config.json,
  model.safetensors, tokenizer files). Verified by loading with stock
  `transformers` (no custom code) and generating a coherent story.
- **Gotcha 7 — special-token id mismatch in exported config.** `LlamaConfig`
  defaults bos=1/eos=2, but our `<|endoftext|>` is id 0. Left as-is, a
  published copy would never stop generating at a story break (generate() waits
  for token 2). Fixed `to_hf_config` to set bos/eos/pad = 0. Verified effect:
  5/6 sampled generations now stop early at EOS (before: all ran to max_len).
- **Gotcha 8 — HF username != GitHub username.** GitHub is `vppillai`, but the
  HF account is `vysakhpillai`. Repo must live under the HF namespace; fixed
  the model-card usage examples to `vysakhpillai/tinystories-slm`.
- Published to **private** Hub repo `vysakhpillai/tinystories-slm` (weights +
  config + tokenizer + model card). `push_to_hub` skips README, so `push()`
  uploads the model card explicitly. Verified via API: private=True, all 7
  files present.
- Note: user pasted a write HF token into the chat during login — flagged for
  rotation. (Token stored at ~/.cache/huggingface/token for this machine.)

## 2026-07-09 — Task 8: GGUF + Ollama — PHASE 1 COMPLETE 🎉

The full pipeline closed: bytes → tokenizer → hand-built transformer → training
→ HF format → GGUF → quantized → Ollama, running on CPU through the same engine
people use for Llama.

**Tools installed (treated as learning):** cmake (via pip wheel, no sudo),
llama.cpp (clone + cmake build), Ollama (official installer). gcc already present.

**GGUF sizes (the quantization ladder):** f16 27MB → Q8_0 15MB → Q4_K_M 11MB
(16.0 / 8.51 / 6.21 bits per weight). All three generate coherent stories
(Lab 08). Because quantization perturbs logits, each quant gives a *different*
story for the same seed.

**Ollama finish line** — `ollama run tinystories-slm "Once upon a time"`:
> Once upon a time, there was a little girl named Lily. She loved to play
> outside in the park. One day, she saw a big tree and wanted to climb it. But
> when she tried to climb the tree, she slipped and fell. Her knee hurt a lot!
> She started to cry because she couldn't reach the top of the tree. Her mom
> came running over and asked what happened. Lily told her that she was hurt and
> needed to rest. Her mom gave her some medicine and told her to rest when she
> was tired. After resting, ...

Five real gotchas this task (all masked until now):
- **Gotcha 9 — OOM during the llama.cpp build.** `cmake --build -j` (all 14
  cores) spawned too many cc1plus on llama.cpp's largest files (t5.cpp,
  hunyuan-vl.cpp); ~11GB RAM ran out → `Killed signal terminated program
  cc1plus`. Fix: `-j 4` caps concurrent compilers (make is incremental, so it
  resumed from the 178 already-compiled .o files).
- **Gotcha 10 — custom tokenizer not recognized by the GGUF converter.**
  `convert_hf_to_gguf.py` identifies BPE tokenizers by hashing their
  pre-tokenizer output against a hardcoded list; our custom-trained byte-level
  BPE has a novel hash (fe391dc4...) → `NotImplementedError: BPE pre-tokenizer
  was not recognized`. Ours IS a GPT-2-style ByteLevel BPE, so registered the
  hash → "gpt-2" in llama.cpp's conversion/base.py (get_vocab_base_pre). The
  GGUF then records `tokenizer.ggml.pre = gpt-2`. NOTE: this patches the cloned
  llama.cpp checkout, not our repo — re-clone requires re-applying (documented
  in REPRODUCE).
- **Gotcha 11 — llama-cli runaway (2GB output file).** `llama-cli -no-cnv`
  still dropped into an interactive REPL after generating, reading /dev/null
  EOF and printing empty `>` prompts forever (plus the loading spinner writing
  frames to a non-TTY). Fix: `-st` (single-turn, exits when done) + `--simple-io`.
  Guard subprocess/one-shot runs with `timeout` + a byte cap.
- **Gotcha 12 — Ollama installer needs zstd.** `curl … | sh` failed with "This
  version requires zstd for extraction." Fix: `sudo apt-get install -y zstd`,
  re-run installer. Also: on this VM systemd didn't auto-start the daemon, so
  ran `ollama serve` manually in the background.
- **Gotcha 13 — Ollama 0.31.2 rejects a relative FROM path.** `ollama create`
  with `FROM ./export/…gguf` → `400 Bad Request: invalid model name` (it treats
  the relative path as a model-name reference). Fix: absolute path in FROM.
  scripts/Modelfile keeps the clean relative form; REPRODUCE shows a one-liner to
  rewrite it to absolute for ollama versions that need it.

Labs 07 (GGUF header teardown) and 08 (quant comparison) both run.

**Phase 1 is complete.** All 9 milestone-ladder rungs hit: tokenizer → batches →
gibberish → overfit → toy run → coherent stories → published → GGUF → Ollama.
