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
