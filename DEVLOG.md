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
