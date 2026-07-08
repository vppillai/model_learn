# SLM Phase 1: Base Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Hand-build a small Llama-shaped language model, train it on TinyStories, package it to the standard Hugging Face format, convert it to GGUF, and run it locally on CPU via Ollama — learning the full "download → build → run" pipeline by making one.

**Architecture:** A decoder-only transformer written from scratch in plain PyTorch (token embeddings → RoPE + RMSNorm + causal self-attention + SwiGLU MLP, pre-norm Llama style → tied LM head). Two configs (`toy` for minutes-on-CPU dev, `small` ~14M params for the Colab run) run the *same* code. Standard libraries (`tokenizers`, `transformers`, `datasets`) are reused as building blocks and probed with runnable Labs. The hand-built weights are exported into `LlamaForCausalLM` so the artifact converts to GGUF with no custom runtime.

**Tech Stack:** Python 3.13 managed with `uv` (venv + `uv.lock`), PyTorch (CPU local / GPU Colab), Hugging Face `tokenizers` / `transformers` / `datasets`, `huggingface_hub`, matplotlib, pytest; llama.cpp + Ollama for the run step (each installed/built as a learning step).

## Global Constraints

- **Reproducibility-first docs:** every task updates `DEVLOG.md` (dated narrative) and, where steps are user-followable, `REPRODUCE.md`. `CHANGELOG.md` gets a terse line. No blog-styled prose.
- **Concept-on-contact:** any task introducing a new term adds a 3–5 sentence `CONCEPTS.md` entry cross-linked to the Lab that demonstrates it.
- **Determinism in tests:** every test sets `torch.manual_seed(0)` (and `random.seed(0)`) before any random op.
- **Tests never download the internet:** unit tests use a tiny inline text fixture, never the real TinyStories dataset. The real dataset is used only by the actual training runs (Tasks 5–6).
- **HF-exact architecture:** the hand-built model must match `transformers` `LlamaForCausalLM` numerically — same RMSNorm float32 trick, same `rotate_half` RoPE convention, bias-free linears, `rms_norm_eps = 1e-5`, `rope_theta = 10000`. This is enforced by the round-trip test in Task 7.
- **No GQA, no MoE in Phase 1:** `num_key_value_heads == num_attention_heads`. (Both are later Labs/phases.)
- **Environment via `uv`:** the venv + dependencies are managed by `uv` (`pyproject.toml` + committed `uv.lock`); a portable `requirements.txt` is generated with `uv export` for Colab. Run tests/scripts with the venv activated or prefixed by `uv run`. Installing/building tools (uv, llama.cpp, Ollama) is treated as part of the learning, not glossed over.
- **Package import path:** all library code lives under `src/slm/`; tests run from repo root with `PYTHONPATH=src` (set in `pytest.ini`).

---

## Execution Protocol — Learning Mode (READ FIRST, every session)

**This project's primary purpose is for vpillai to *learn*, not to generate a blog or ship code fast.** The blog is a later, separate task. Execution must therefore be a guided, hands-on teaching loop — never silent batch implementation. Run **every task** through this loop:

1. **Frame** — before any code, explain in plain language *what* we're about to build and *why it exists* (the concept). No heavy math; favor intuition, the software↔model analogy, and data flow.
2. **Clarify** — invite questions; answer them; surface assumptions. Do not proceed until the concept is clear.
3. **Predict (optional)** — ask "what do you expect to happen when we run this?" to prime understanding before seeing output.
4. **Do — vpillai drives** — present the step; vpillai writes/runs it following along (Claude shows the exact code/command). Look at the real output together. Claude does not race ahead through multiple steps unprompted.
5. **Check understanding** — short recap; ask vpillai to explain it back in their own words; a **1–2 question mini-quiz** where it adds value. If shaky, re-explain before moving on.
6. **Record** — update `DEVLOG.md` (what/why/gotchas), add/extend the `CONCEPTS.md` entry, tick the step's checkbox.
7. **Checkpoint** — confirm with vpillai before starting the next task.

**Pace:** one step at a time. Showable intermediate results are the point — pause and observe at each milestone rung. The explanatory `★ Insight` blocks stay on throughout.

## Resuming Across Sessions (multi-session continuity)

This work spans many sessions. To resume cleanly in a **fresh session**, do this in order:

1. Read the spec: `docs/superpowers/specs/2026-06-22-slm-from-scratch-phase1-design.md`.
2. Read this plan (especially this protocol + Global Constraints).
3. **Find current position:** the source of truth is the checkbox state in this file — the first unchecked `- [ ]` step is where we resume. Cross-check with the latest `DEVLOG.md` entry and `git log`.
4. Re-activate the environment (don't reinstall): `. .venv/bin/activate`. (`.venv/`, `data/`, `checkpoints/` are gitignored and stay local between sessions; `REPRODUCE.md` has setup if missing.)
5. Continue under the Learning-Mode loop above from the first unchecked step.

**Progress-tracking rule:** as each step completes, (a) tick its `- [ ]` → `- [x]` in this file and (b) append to `DEVLOG.md`. These two, plus `git log`, are the durable cross-session state. Claude's per-project memory also records phase/status as a backup pointer, but the repo is authoritative.

**Kick-off phrase for a new session:** "Continue the model_learn project" — that signals Claude to run the resume steps above.

---

### Task 1: Project scaffold, configs, and doc skeleton

**Files:**
- Create: `pyproject.toml` (uv-managed dependencies) + `uv.lock` (committed lockfile)
- Create: `requirements.txt` (generated via `uv export`, for Colab / non-uv environments)
- Create: `pytest.ini`
- Create: `src/slm/__init__.py`
- Create: `src/slm/config.py`
- Create: `tests/test_config.py`
- Create: `README.md`, `DEVLOG.md`, `CHANGELOG.md`, `CONCEPTS.md`, `REPRODUCE.md` (stubs)

**Interfaces:**
- Produces: `ModelConfig` dataclass with fields `vocab_size:int, d_model:int, n_layers:int, n_heads:int, n_kv_heads:int, head_dim:int, ffn_hidden:int, context_len:int, rms_norm_eps:float, rope_theta:float, tie_embeddings:bool`; helper `ModelConfig.n_params() -> int`; module constants `TOY: ModelConfig` and `SMALL: ModelConfig`.

- [x] **Step 1: Install `uv`, then declare dependencies in `pyproject.toml`**

`uv` is a fast Python package + Python-version manager; we use it for a reproducible, **lockfile-backed** environment. *Installing the tool is itself part of the learning* — check first, install only if missing:
```bash
uv --version || curl -LsSf https://astral.sh/uv/install.sh | sh
```
Create `pyproject.toml` (the source of truth for dependencies — `uv` resolves and locks it):
```toml
[project]
name = "model-learn"
version = "0.1.0"
description = "Hand-built SLM from scratch to Ollama (learning project)"
requires-python = ">=3.13"
dependencies = [
    "torch>=2.5",
    "tokenizers>=0.20",
    "transformers>=4.45",
    "datasets>=3.0",
    "huggingface_hub>=0.25",
    "matplotlib>=3.9",
    "numpy>=1.26",
]

[dependency-groups]
dev = ["pytest>=8.0"]
```
No `[build-system]` table → `uv` treats this as a non-packaged "virtual" project: it installs the dependencies into the venv but does not try to build/install `model-learn` itself. Imports are handled by `pythonpath = src` (Step 2).

- [x] **Step 2: Create `pytest.ini`**

```ini
[pytest]
pythonpath = src
testpaths = tests
addopts = -q
```

- [x] **Step 3: Create the environment and install with `uv`**

Run:
```bash
uv venv          # creates .venv using an available Python 3.13
uv sync          # resolves + installs deps, writes uv.lock (commit it)
uv export --no-hashes --format requirements-txt -o requirements.txt  # portable list for Colab
```
Expected: `.venv/` created, `uv.lock` written, dependencies installed. Either `source .venv/bin/activate` once, or prefix commands with `uv run` (e.g. `uv run pytest`). If the `torch` aarch64/Py3.13 wheel fails to resolve, add the CPU index: `uv pip install torch --index-url https://download.pytorch.org/whl/cpu`, and record the working command in `DEVLOG.md`.

- [x] **Step 4: Write the failing test** in `tests/test_config.py`

```python
from slm.config import ModelConfig, TOY, SMALL


def test_toy_is_tiny_and_consistent():
    assert TOY.d_model == TOY.n_heads * TOY.head_dim
    assert TOY.n_kv_heads == TOY.n_heads  # no GQA in Phase 1
    assert TOY.n_params() < 1_000_000


def test_small_targets_about_14M_params():
    assert SMALL.d_model == SMALL.n_heads * SMALL.head_dim
    assert SMALL.n_kv_heads == SMALL.n_heads
    assert 10_000_000 < SMALL.n_params() < 16_000_000
```

- [x] **Step 5: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'slm.config'`.

- [x] **Step 6: Create `src/slm/__init__.py`** (empty file).

- [x] **Step 7: Implement `src/slm/config.py`**

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    ffn_hidden: int
    context_len: int
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    tie_embeddings: bool = True

    def n_params(self) -> int:
        """Approximate trainable parameter count (bias-free linears)."""
        embed = self.vocab_size * self.d_model  # tied: counted once
        attn = self.n_layers * (
            self.d_model * self.n_heads * self.head_dim          # q
            + 2 * self.d_model * self.n_kv_heads * self.head_dim  # k, v
            + self.n_heads * self.head_dim * self.d_model          # o
        )
        ffn = self.n_layers * (3 * self.d_model * self.ffn_hidden)
        norms = self.n_layers * 2 * self.d_model + self.d_model  # RMSNorm weights
        return embed + attn + ffn + norms


TOY = ModelConfig(
    vocab_size=2048, d_model=64, n_layers=2, n_heads=4, n_kv_heads=4,
    head_dim=16, ffn_hidden=128, context_len=128,
)

SMALL = ModelConfig(
    vocab_size=8192, d_model=384, n_layers=6, n_heads=6, n_kv_heads=6,
    head_dim=64, ffn_hidden=1024, context_len=512,
)
```

- [x] **Step 8: Run tests to verify they pass**

Run: `pytest tests/test_config.py -v`
Expected: PASS (2 passed).

- [x] **Step 9: Create doc stubs**

`README.md`:
```markdown
# model_learn — Build a Language Model From Scratch to Ollama

Phase 1: a hand-built, Llama-shaped SLM trained on TinyStories, packaged to
GGUF, run on CPU via Ollama. See `docs/superpowers/specs/` for the design and
`REPRODUCE.md` to build it yourself.
```

`DEVLOG.md`:
```markdown
# Developer Log

Dated, chronological record of what we did and why — including dead-ends.

## 2026-06-22 — Task 1: scaffold + configs
- Created package skeleton, `ModelConfig` with `toy`/`small`, pytest setup.
- (record the working torch install command here)
```

`CHANGELOG.md`:
```markdown
# Changelog

## Unreleased
- Task 1: project scaffold, `ModelConfig` (toy/small), pytest wiring.
```

`CONCEPTS.md`:
```markdown
# Concepts (plain-language, concept-on-contact)

Each entry: what it is, why it exists, and the Lab that shows it.

## parameter
A single learnable number in the model. "14M parameters" = 14 million such
numbers, adjusted during training. Lab: see `n_params()` in `src/slm/config.py`.
```

`REPRODUCE.md`:
```markdown
# Reproduce This From Scratch

Follow-along guide (distilled from DEVLOG). Build the whole thing yourself.

## 0. Environment (uv)
```bash
# install uv if needed: curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv && uv sync          # creates .venv, installs deps from uv.lock
source .venv/bin/activate   # or prefix commands with `uv run`
```
```

- [ ] **Step 10: Commit**

```bash
git add pyproject.toml uv.lock requirements.txt pytest.ini src tests README.md DEVLOG.md CHANGELOG.md CONCEPTS.md REPRODUCE.md
git commit -m "feat: project scaffold (uv env), ModelConfig (toy/small), docs skeleton"
```

---

### Task 2: BPE tokenizer (train + load) — *Milestone: tokenizer works*

**Files:**
- Create: `src/slm/tokenizer.py`
- Create: `tests/test_tokenizer.py`
- Create: `labs/lab01_bpe_by_hand.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `train_tokenizer(texts: Iterable[str], vocab_size: int, save_path: str) -> Tokenizer`; `load_tokenizer(path: str) -> Tokenizer`; `encode(tok, text: str) -> list[int]`; `decode(tok, ids: list[int]) -> str`. Special tokens: `<|endoftext|>` (id 0 reserved as both BOS-ish separator and EOS for stories).

- [x] **Step 1: Write the failing test** in `tests/test_tokenizer.py`

```python
import random
from slm.tokenizer import train_tokenizer, load_tokenizer, encode, decode

CORPUS = [
    "Once upon a time there was a little cat.",
    "The cat liked to play in the sun every day.",
    "One day the cat met a happy dog and they became friends.",
] * 50


def test_roundtrip_and_vocab(tmp_path):
    random.seed(0)
    path = str(tmp_path / "tok.json")
    tok = train_tokenizer(CORPUS, vocab_size=300, save_path=path)
    assert tok.get_vocab_size() <= 300
    ids = encode(tok, "the cat liked the sun")
    assert isinstance(ids, list) and all(isinstance(i, int) for i in ids)
    assert decode(tok, ids).replace(" ", "") == "thecatlikedthesun".replace(" ", "")


def test_load_after_save(tmp_path):
    path = str(tmp_path / "tok.json")
    train_tokenizer(CORPUS, vocab_size=300, save_path=path)
    tok = load_tokenizer(path)
    assert encode(tok, "happy dog") == encode(load_tokenizer(path), "happy dog")


def test_endoftext_special_token(tmp_path):
    path = str(tmp_path / "tok.json")
    tok = train_tokenizer(CORPUS, vocab_size=300, save_path=path)
    assert tok.token_to_id("<|endoftext|>") == 0
```

- [x] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tokenizer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'slm.tokenizer'`.

- [x] **Step 3: Implement `src/slm/tokenizer.py`**

```python
from typing import Iterable
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

EOT = "<|endoftext|>"


def train_tokenizer(texts: Iterable[str], vocab_size: int, save_path: str) -> Tokenizer:
    tok = Tokenizer(BPE(unk_token=None))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tok.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[EOT],
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(list(texts), trainer=trainer)
    tok.save(save_path)
    return tok


def load_tokenizer(path: str) -> Tokenizer:
    return Tokenizer.from_file(path)


def encode(tok: Tokenizer, text: str) -> list[int]:
    return tok.encode(text).ids


def decode(tok: Tokenizer, ids: list[int]) -> str:
    return tok.decode(ids)
```

- [x] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_tokenizer.py -v`
Expected: PASS (3 passed). Note: byte-level decode reconstructs spaces exactly; the roundtrip test strips spaces defensively in case of leading-space normalization.

- [x] **Step 5: Write the Lab** `labs/lab01_bpe_by_hand.py`

```python
"""Lab 01 — watch BPE merges form on a single paragraph.

Run: python labs/lab01_bpe_by_hand.py
"""
from slm.tokenizer import train_tokenizer, encode

para = (
    "the cat sat on the mat. the cat sat on the hat. "
    "the rat sat on the cat."
)

for vsz in (260, 270, 290, 320):
    tok = train_tokenizer([para] * 20, vocab_size=vsz, save_path="/tmp/lab01.json")
    ids = encode(tok, "the cat sat")
    pieces = [tok.id_to_token(i) for i in ids]
    print(f"vocab={vsz:>4}  'the cat sat' -> {pieces}")
print("\nNotice: as the vocab grows, common chunks like 'the ' and 'cat'")
print("become single tokens. That merging IS what BPE training does.")
```

- [x] **Step 6: Run the Lab and observe**

Run: `python labs/lab01_bpe_by_hand.py`
Expected: as vocab grows, `'the cat sat'` collapses from many character tokens into fewer, larger merged tokens.

- [x] **Step 7: Update docs**

Add to `CONCEPTS.md`: entries for **tokenizer**, **token / token id**, **BPE (byte-pair encoding)**, **vocabulary**, **special token (`<|endoftext|>`)**, each 3–5 sentences, cross-linking `labs/lab01_bpe_by_hand.py`. Add a `DEVLOG.md` dated entry and a `CHANGELOG.md` line. Add a "Milestone: tokenizer" section to `REPRODUCE.md` with the train/encode/decode commands.

- [ ] **Step 8: Commit**

```bash
git add src/slm/tokenizer.py tests/test_tokenizer.py labs/lab01_bpe_by_hand.py CONCEPTS.md DEVLOG.md CHANGELOG.md REPRODUCE.md
git commit -m "feat: BPE tokenizer with train/load/encode/decode + Lab 01"
```

---

### Task 3: Data pipeline (tokenize + batch) — *Milestone: real batches*

**Files:**
- Create: `src/slm/data.py`
- Create: `tests/test_data.py`
- Create: `tests/fixtures/tiny_stories.txt` (small local fixture, ~30 lines)

**Interfaces:**
- Consumes: `Tokenizer` from Task 2.
- Produces: `tokenize_texts(tok, texts: list[str]) -> list[int]` (flat stream with `<|endoftext|>` id 0 between docs); `get_batch(data: list[int], batch_size: int, context_len: int, seed: int) -> tuple[Tensor, Tensor]` returning `(x, y)` each shaped `(batch_size, context_len)`, where `y` is `x` shifted left by one; `load_tinystories(split: str, limit: int|None) -> list[str]` (real dataset loader, NOT used in tests).

- [x] **Step 1: Create the fixture** `tests/fixtures/tiny_stories.txt`

```
Once upon a time there was a little cat who loved the warm sun.
The cat played all day and slept all night under a big tree.
One day a happy dog came to play and they became best friends.
They ran across the green field and laughed in the bright light.
```
(repeat/extend to ~30 short lines of similar simple sentences.)

- [x] **Step 2: Write the failing test** in `tests/test_data.py`

```python
import torch
from slm.tokenizer import train_tokenizer
from slm.data import tokenize_texts, get_batch

LINES = open("tests/fixtures/tiny_stories.txt").read().splitlines()


def _tok(tmp_path):
    return train_tokenizer(LINES * 20, vocab_size=300,
                           save_path=str(tmp_path / "t.json"))


def test_tokenize_inserts_eot(tmp_path):
    tok = _tok(tmp_path)
    stream = tokenize_texts(tok, LINES[:3])
    assert 0 in stream  # <|endoftext|> id 0 separates docs
    assert all(isinstance(i, int) for i in stream)


def test_batch_shapes_and_shift(tmp_path):
    tok = _tok(tmp_path)
    stream = tokenize_texts(tok, LINES * 5)
    x, y = get_batch(stream, batch_size=4, context_len=8, seed=0)
    assert x.shape == (4, 8) and y.shape == (4, 8)
    assert x.dtype == torch.long
    # y is x shifted by one: y[:, :-1] aligns with x[:, 1:]
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_batch_is_deterministic_with_seed(tmp_path):
    tok = _tok(tmp_path)
    stream = tokenize_texts(tok, LINES * 5)
    a = get_batch(stream, 4, 8, seed=0)[0]
    b = get_batch(stream, 4, 8, seed=0)[0]
    assert torch.equal(a, b)
```

- [x] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_data.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'slm.data'`.

- [x] **Step 4: Implement `src/slm/data.py`**

```python
import torch
from tokenizers import Tokenizer

EOT_ID = 0


def tokenize_texts(tok: Tokenizer, texts: list[str]) -> list[int]:
    stream: list[int] = []
    for t in texts:
        stream.extend(tok.encode(t).ids)
        stream.append(EOT_ID)
    return stream


def get_batch(data: list[int], batch_size: int, context_len: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    t = torch.tensor(data, dtype=torch.long)
    max_start = len(t) - context_len - 1
    assert max_start > 0, "not enough tokens for one context window"
    starts = torch.randint(0, max_start, (batch_size,), generator=g)
    x = torch.stack([t[s : s + context_len] for s in starts])
    y = torch.stack([t[s + 1 : s + 1 + context_len] for s in starts])
    return x, y


def load_tinystories(split: str = "train", limit: int | None = None) -> list[str]:
    """Real dataset loader for training runs (not used in unit tests)."""
    from datasets import load_dataset
    ds = load_dataset("roneneldan/TinyStories", split=split)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    return [row["text"] for row in ds]
```

- [x] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_data.py -v`
Expected: PASS (3 passed).

- [x] **Step 6: Update docs**

`CONCEPTS.md`: add **context window / sequence length**, **batch**, **next-token target (the shift-by-one)**. Add `DEVLOG.md` + `CHANGELOG.md` entries. Note in `REPRODUCE.md` how to print a real batch.

- [ ] **Step 7: Commit**

```bash
git add src/slm/data.py tests/test_data.py tests/fixtures CONCEPTS.md DEVLOG.md CHANGELOG.md REPRODUCE.md
git commit -m "feat: data pipeline (tokenize stream + shifted batches) + fixture"
```

---

### Task 4a: Model components (RMSNorm, RoPE, attention, MLP)

**Files:**
- Create: `src/slm/model.py` (components only this task)
- Create: `tests/test_model_components.py`

**Interfaces:**
- Consumes: `ModelConfig` from Task 1.
- Produces: `class RMSNorm(nn.Module)`; functions `build_rope_cache(head_dim, seq_len, theta) -> (cos, sin)` and `apply_rope(x, cos, sin)`; `class Attention(nn.Module)` with `forward(x, cos, sin) -> Tensor`; `class SwiGLU(nn.Module)` with `forward(x) -> Tensor`. All linears bias-free. `rotate_half` matches HF Llama.

- [ ] **Step 1: Write the failing test** in `tests/test_model_components.py`

```python
import torch
from slm.config import TOY
from slm.model import RMSNorm, build_rope_cache, apply_rope, Attention, SwiGLU


def test_rmsnorm_shape_and_scale():
    torch.manual_seed(0)
    n = RMSNorm(TOY.d_model, eps=TOY.rms_norm_eps)
    x = torch.randn(2, 5, TOY.d_model)
    out = n(x)
    assert out.shape == x.shape
    # default weight=1 => rms of each row ~ 1
    rms = out.pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)


def test_rope_preserves_shape_and_norm():
    torch.manual_seed(0)
    cos, sin = build_rope_cache(TOY.head_dim, TOY.context_len, TOY.rope_theta)
    x = torch.randn(1, TOY.n_heads, 7, TOY.head_dim)
    out = apply_rope(x, cos[:7], sin[:7])
    assert out.shape == x.shape
    # rotation preserves vector norm per (head, position)
    assert torch.allclose(out.norm(dim=-1), x.norm(dim=-1), atol=1e-4)


def test_attention_is_causal():
    torch.manual_seed(0)
    attn = Attention(TOY)
    cos, sin = build_rope_cache(TOY.head_dim, TOY.context_len, TOY.rope_theta)
    x = torch.randn(1, 6, TOY.d_model)
    out_full = attn(x, cos[:6], sin[:6])
    # changing the LAST token must not change earlier outputs (causality)
    x2 = x.clone()
    x2[0, -1] += 10.0
    out2 = attn(x2, cos[:6], sin[:6])
    assert torch.allclose(out_full[0, :-1], out2[0, :-1], atol=1e-5)


def test_swiglu_shape():
    torch.manual_seed(0)
    mlp = SwiGLU(TOY)
    x = torch.randn(2, 4, TOY.d_model)
    assert mlp(x).shape == x.shape
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_model_components.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'slm.model'`.

- [ ] **Step 3: Implement components in `src/slm/model.py`**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from slm.config import ModelConfig


class RMSNorm(nn.Module):
    """Llama RMSNorm: normalize by root-mean-square in float32, then scale."""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (self.weight * x.to(dtype))


def build_rope_cache(head_dim: int, seq_len: int, theta: float):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    pos = torch.arange(seq_len).float()
    freqs = torch.outer(pos, inv_freq)            # (seq, head_dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)       # (seq, head_dim)
    return emb.cos(), emb.sin()


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x, cos, sin):
    # x: (B, n_heads, T, head_dim); cos/sin: (T, head_dim)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return (x * cos) + (rotate_half(x) * sin)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=False)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        c = self.cfg
        q = self.q_proj(x).view(B, T, c.n_heads, c.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, c.n_kv_heads, c.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, c.n_kv_heads, c.head_dim).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        scores = (q @ k.transpose(-2, -1)) * (c.head_dim ** -0.5)
        mask = torch.full((T, T), float("-inf"), device=x.device).triu(1)
        scores = scores + mask
        attn = F.softmax(scores, dim=-1)
        out = attn @ v                                  # (B, n_heads, T, head_dim)
        out = out.transpose(1, 2).reshape(B, T, c.n_heads * c.head_dim)
        return self.o_proj(out)


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.up_proj = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.down_proj = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_model_components.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Update docs**

`CONCEPTS.md`: **embedding** (placeholder note that the table comes in 4b), **RMSNorm**, **RoPE / positional encoding**, **attention / causal mask**, **SwiGLU / feed-forward**, **logits** (note: in 4b). Add `DEVLOG.md` + `CHANGELOG.md` lines.

- [ ] **Step 6: Commit**

```bash
git add src/slm/model.py tests/test_model_components.py CONCEPTS.md DEVLOG.md CHANGELOG.md
git commit -m "feat: model components (RMSNorm, RoPE, causal attention, SwiGLU)"
```

---

### Task 4b: Assemble the model + generation — *Milestone: untrained gibberish*

**Files:**
- Modify: `src/slm/model.py` (append Block + LlamaSLM + generate)
- Create: `tests/test_model.py`
- Create: `labs/lab02_attention_peek.py`
- Create: `labs/lab03_gibberish.py`

**Interfaces:**
- Consumes: components from Task 4a; `ModelConfig`.
- Produces: `class Block(nn.Module)` (pre-norm: `x = x + attn(norm1(x)); x = x + mlp(norm2(x))`); `class LlamaSLM(nn.Module)` with attributes `embed_tokens, layers, norm, lm_head` (tied), `forward(idx) -> logits (B,T,vocab)`, and `@torch.no_grad() generate(idx, max_new_tokens, temperature=1.0, top_k=None) -> idx`.

- [ ] **Step 1: Write the failing test** in `tests/test_model.py`

```python
import torch
from slm.config import TOY
from slm.model import LlamaSLM


def test_forward_logits_shape():
    torch.manual_seed(0)
    m = LlamaSLM(TOY)
    idx = torch.randint(0, TOY.vocab_size, (2, 10))
    logits = m(idx)
    assert logits.shape == (2, 10, TOY.vocab_size)


def test_embeddings_are_tied():
    m = LlamaSLM(TOY)
    assert m.lm_head.weight is m.embed_tokens.weight


def test_generate_extends_sequence():
    torch.manual_seed(0)
    m = LlamaSLM(TOY)
    idx = torch.zeros((1, 3), dtype=torch.long)
    out = m.generate(idx, max_new_tokens=5)
    assert out.shape == (1, 8)


def test_untrained_output_is_high_entropy():
    # An untrained model's next-token distribution should be near-uniform-ish:
    # no single token dominates. This is the "gibberish" sanity check.
    torch.manual_seed(0)
    m = LlamaSLM(TOY)
    idx = torch.zeros((1, 4), dtype=torch.long)
    probs = torch.softmax(m(idx)[0, -1], dim=-1)
    assert probs.max().item() < 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_model.py -v`
Expected: FAIL with `ImportError: cannot import name 'LlamaSLM'`.

- [ ] **Step 3: Append to `src/slm/model.py`**

```python
class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x


class LlamaSLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        cos, sin = build_rope_cache(cfg.head_dim, cfg.context_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, idx):
        B, T = idx.shape
        x = self.embed_tokens(idx)
        cos, sin = self.rope_cos[:T], self.rope_sin[:T]
        for layer in self.layers:
            x = layer(x, cos, sin)
        x = self.norm(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.context_len:]
            logits = self(idx_cond)[:, -1, :]
            if temperature != 1.0:
                logits = logits / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = torch.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_model.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Write Lab 02 (attention peek)** `labs/lab02_attention_peek.py`

```python
"""Lab 02 — print attention weights for one short sequence.
Run: python labs/lab02_attention_peek.py
"""
import torch
from slm.config import TOY
from slm.model import Attention, build_rope_cache

torch.manual_seed(0)
attn = Attention(TOY)
cos, sin = build_rope_cache(TOY.head_dim, TOY.context_len, TOY.rope_theta)
x = torch.randn(1, 5, TOY.d_model)

# Re-run attention internals to expose the weights (mirrors Attention.forward):
import torch.nn.functional as F
c = TOY
q = attn.q_proj(x).view(1, 5, c.n_heads, c.head_dim).transpose(1, 2)
k = attn.k_proj(x).view(1, 5, c.n_kv_heads, c.head_dim).transpose(1, 2)
from slm.model import apply_rope
q, k = apply_rope(q, cos[:5], sin[:5]), apply_rope(k, cos[:5], sin[:5])
scores = (q @ k.transpose(-2, -1)) * (c.head_dim ** -0.5)
scores = scores + torch.full((5, 5), float("-inf")).triu(1)
w = F.softmax(scores, dim=-1)[0, 0]  # head 0
print("Attention weights (row = query position, col = key position), head 0:")
for i, row in enumerate(w):
    print(f"  q{i}: " + " ".join(f"{p:.2f}" for p in row))
print("\nNote the upper triangle is 0.00 — that's the causal mask: a token")
print("can only attend to itself and earlier tokens.")
```

- [ ] **Step 6: Write Lab 03 (gibberish)** `labs/lab03_gibberish.py`

```python
"""Lab 03 — an UNTRAINED model already runs; it just talks nonsense.
This is the key insight: architecture (wiring) works instantly; only the
weights (the learned numbers) are missing. Run after training a tokenizer.

Run: python labs/lab03_gibberish.py
"""
import torch
from slm.config import TOY
from slm.model import LlamaSLM
from slm.tokenizer import train_tokenizer, decode

lines = open("tests/fixtures/tiny_stories.txt").read().splitlines()
tok = train_tokenizer(lines * 20, vocab_size=TOY.vocab_size, save_path="/tmp/lab03.json")

torch.manual_seed(0)
model = LlamaSLM(TOY)
start = torch.zeros((1, 1), dtype=torch.long)
out = model.generate(start, max_new_tokens=40)
print("UNTRAINED model says:\n")
print(decode(tok, out[0].tolist()))
print("\n^ Gibberish — but it generated! The pipeline works end-to-end.")
```

- [ ] **Step 7: Run both Labs**

Run: `python labs/lab02_attention_peek.py` then `python labs/lab03_gibberish.py`
Expected: Lab 02 shows a lower-triangular attention matrix; Lab 03 prints reconstructable-but-meaningless text.

- [ ] **Step 8: Update docs**

`CONCEPTS.md`: finalize **embedding**, **logits**, **pre-norm residual block**, **autoregressive generation**, **temperature / top-k sampling** (note Lab 05 will go deeper). `DEVLOG.md`: record the "untrained gibberish" milestone with a copied sample. `CHANGELOG.md` line. `REPRODUCE.md`: add the gibberish milestone (run Lab 03).

- [ ] **Step 9: Commit**

```bash
git add src/slm/model.py tests/test_model.py labs/lab02_attention_peek.py labs/lab03_gibberish.py CONCEPTS.md DEVLOG.md CHANGELOG.md REPRODUCE.md
git commit -m "feat: assemble LlamaSLM + generate; untrained-gibberish milestone (Labs 02-03)"
```

---

### Task 5: Training loop + overfit-one-batch — *Milestones: learning works, toy run*

**Files:**
- Create: `src/slm/train.py`
- Create: `src/slm/sample.py`
- Create: `tests/test_train.py`
- Create: `labs/lab05_sampling.py`

**Interfaces:**
- Consumes: `LlamaSLM`, `ModelConfig`, `get_batch`, tokenizer.
- Produces: `TrainConfig` dataclass (`lr, warmup_steps, max_steps, batch_size, grad_clip, weight_decay, log_every, sample_every, ckpt_path, seed`); `lr_at(step, cfg) -> float` (cosine schedule with linear warmup); `train(model, data, train_cfg, tok=None) -> list[tuple[int,float]]` (returns `(step, loss)` history, writes checkpoints + `loss_log.csv`, prints a sample every `sample_every`); `save_checkpoint(model, model_cfg, path)` / `load_checkpoint(path) -> (model, model_cfg)`; `plot_loss(csv_path, out_png)`. `sample.py` exposes `generate_text(ckpt_path, tok_path, prompt, max_new_tokens, temperature, top_k) -> str`.

- [ ] **Step 1: Write the failing test** in `tests/test_train.py`

```python
import torch
from slm.config import TOY
from slm.model import LlamaSLM
from slm.train import TrainConfig, lr_at, train


def test_lr_warmup_and_decay():
    cfg = TrainConfig(lr=1e-3, warmup_steps=10, max_steps=100)
    assert lr_at(0, cfg) < lr_at(5, cfg) < lr_at(10, cfg)   # warming up
    assert lr_at(10, cfg) == cfg.lr                          # peak at end of warmup
    assert lr_at(100, cfg) < lr_at(50, cfg)                  # decaying


def test_overfit_one_batch_drives_loss_down(tmp_path):
    torch.manual_seed(0)
    model = LlamaSLM(TOY)
    # one fixed tiny batch repeated => the model must memorize it
    data = list(range(0, 200))
    cfg = TrainConfig(lr=3e-3, warmup_steps=20, max_steps=400, batch_size=4,
                      log_every=100, sample_every=10_000,
                      ckpt_path=str(tmp_path / "ck.pt"), seed=0,
                      fixed_batch=True, context_len=16)
    history = train(model, data, cfg)
    first_loss = history[0][1]
    last_loss = history[-1][1]
    assert last_loss < first_loss * 0.3   # learning clearly happened
    assert last_loss < 1.0                 # nearly memorized the batch
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_train.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'slm.train'`.

- [ ] **Step 3: Implement `src/slm/train.py`**

```python
import csv
import math
from dataclasses import dataclass, asdict
import torch
import torch.nn.functional as F
from slm.config import ModelConfig
from slm.model import LlamaSLM
from slm.data import get_batch


@dataclass
class TrainConfig:
    lr: float = 3e-4
    warmup_steps: int = 100
    max_steps: int = 2000
    batch_size: int = 32
    context_len: int = 128
    grad_clip: float = 1.0
    weight_decay: float = 0.1
    log_every: int = 50
    sample_every: int = 500
    ckpt_path: str = "checkpoints/model.pt"
    seed: int = 0
    fixed_batch: bool = False  # True => always sample the same batch (overfit test)


def lr_at(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps if step + 1 < cfg.warmup_steps else cfg.lr * step / cfg.warmup_steps if False else cfg.lr * (step) / cfg.warmup_steps if step else cfg.lr / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    return 0.5 * cfg.lr * (1 + math.cos(math.pi * min(1.0, progress)))
```

> NOTE during implementation: replace the convoluted warmup line above with the clean version below — it is the intended logic (linear warmup to peak at `step == warmup_steps`, then cosine decay):
```python
def lr_at(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * step / cfg.warmup_steps
    if step == cfg.warmup_steps:
        return cfg.lr
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    return 0.5 * cfg.lr * (1 + math.cos(math.pi * min(1.0, progress)))
```
(The test asserts `lr_at(0) < lr_at(5) < lr_at(10)` and `lr_at(10) == cfg.lr`; the clean version satisfies it. Delete the convoluted variant.)

```python
def train(model: LlamaSLM, data: list[int], cfg: TrainConfig, tok=None):
    torch.manual_seed(cfg.seed)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay, betas=(0.9, 0.95))
    history: list[tuple[int, float]] = []
    rows: list[tuple[int, float]] = []
    model.train()
    for step in range(cfg.max_steps + 1):
        seed = cfg.seed if cfg.fixed_batch else cfg.seed + step
        x, y = get_batch(data, cfg.batch_size, cfg.context_len, seed=seed)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        for g in opt.param_groups:
            g["lr"] = lr_at(step, cfg)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        if step % cfg.log_every == 0:
            history.append((step, loss.item()))
            rows.append((step, loss.item()))
            print(f"step {step:>5} | loss {loss.item():.4f} | lr {lr_at(step, cfg):.2e}")
        if tok is not None and cfg.sample_every and step % cfg.sample_every == 0 and step > 0:
            _print_sample(model, tok)
    save_checkpoint(model, model.cfg, cfg.ckpt_path)
    _write_csv(rows, cfg.ckpt_path.replace(".pt", "_loss.csv"))
    return history


def _print_sample(model, tok, n=40):
    from slm.tokenizer import decode
    model.eval()
    start = torch.zeros((1, 1), dtype=torch.long)
    out = model.generate(start, max_new_tokens=n, temperature=0.8, top_k=40)
    print("  sample:", decode(tok, out[0].tolist()).replace("\n", " ")[:160])
    model.train()


def _write_csv(rows, path):
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "loss"])
        w.writerows(rows)


def save_checkpoint(model, model_cfg: ModelConfig, path: str):
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # Store config as a plain dict (not a pickled dataclass) so the checkpoint
    # can be loaded with weights_only=True — never unpickle arbitrary objects
    # from a downloaded model file (that is an arbitrary-code-execution vector).
    torch.save({"state_dict": model.state_dict(), "config": asdict(model_cfg)}, path)


def load_checkpoint(path: str):
    ckpt = torch.load(path, weights_only=True)  # tensors + plain containers only
    cfg = ModelConfig(**ckpt["config"])
    model = LlamaSLM(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, cfg


def plot_loss(csv_path: str, out_png: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    steps, losses = [], []
    with open(csv_path) as f:
        next(f)
        for line in f:
            s, l = line.strip().split(",")
            steps.append(int(s)); losses.append(float(l))
    plt.figure(); plt.plot(steps, losses); plt.xlabel("step"); plt.ylabel("loss")
    plt.title("Training loss"); plt.savefig(out_png, dpi=120)
    print("wrote", out_png)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_train.py -v`
Expected: PASS (2 passed). The overfit test is the proof that learning works.

- [ ] **Step 5: Implement `src/slm/sample.py`**

```python
import torch
from slm.train import load_checkpoint
from slm.tokenizer import load_tokenizer, encode, decode


def generate_text(ckpt_path, tok_path, prompt="", max_new_tokens=120,
                  temperature=0.8, top_k=40) -> str:
    model, _ = load_checkpoint(ckpt_path)
    tok = load_tokenizer(tok_path)
    ids = encode(tok, prompt) if prompt else [0]
    idx = torch.tensor([ids], dtype=torch.long)
    out = model.generate(idx, max_new_tokens, temperature=temperature, top_k=top_k)
    return decode(tok, out[0].tolist())


if __name__ == "__main__":
    import sys
    ck, tk = sys.argv[1], sys.argv[2]
    prompt = sys.argv[3] if len(sys.argv) > 3 else "Once upon a time"
    print(generate_text(ck, tk, prompt))
```

- [ ] **Step 6: Write Lab 05 (sampling)** `labs/lab05_sampling.py`

```python
"""Lab 05 — how temperature and top_k reshape the next-token distribution.
Run: python labs/lab05_sampling.py
"""
import torch
torch.manual_seed(0)
logits = torch.tensor([3.0, 2.0, 1.0, 0.5, 0.0, -1.0])
for temp in (0.2, 0.8, 1.5):
    p = torch.softmax(logits / temp, dim=-1)
    print(f"temp={temp}: " + " ".join(f"{x:.2f}" for x in p))
print("\nLow temp -> peaky (greedy, repetitive). High temp -> flat (creative,")
print("risky). top_k just zeroes everything outside the k most likely tokens.")
```

- [ ] **Step 7: Do a real toy training run on local CPU**

Run:
```bash
python - <<'PY'
import torch
from slm.config import TOY
from slm.model import LlamaSLM
from slm.tokenizer import train_tokenizer
from slm.data import tokenize_texts, load_tinystories
from slm.train import TrainConfig, train, plot_loss

texts = load_tinystories("train", limit=2000)
tok = train_tokenizer(texts, vocab_size=TOY.vocab_size, save_path="checkpoints/toy_tok.json")
data = tokenize_texts(tok, texts)
model = LlamaSLM(TOY)
cfg = TrainConfig(lr=3e-3, warmup_steps=50, max_steps=800, batch_size=32,
                  context_len=TOY.context_len, log_every=50, sample_every=200,
                  ckpt_path="checkpoints/toy.pt")
train(model, data, cfg, tok=tok)
plot_loss("checkpoints/toy_loss.csv", "checkpoints/toy_loss.png")
PY
```
Expected: loss decreases over steps; printed samples drift from gibberish toward word-like fragments; `checkpoints/toy_loss.png` shows a downward curve. (Toy quality stays rough — that's expected; coherence comes with the `small` Colab run.)

- [ ] **Step 8: Update docs**

`CONCEPTS.md`: **loss / cross-entropy**, **optimizer / AdamW**, **learning-rate schedule (warmup + cosine)**, **gradient clipping**, **checkpoint**, **overfitting (used here deliberately as a test)**. `DEVLOG.md`: record overfit-test result + the toy run's loss curve + a sample. `CHANGELOG.md` line. `REPRODUCE.md`: add the toy-training section (the heredoc above).

- [ ] **Step 9: Commit**

```bash
git add src/slm/train.py src/slm/sample.py tests/test_train.py labs/lab05_sampling.py CONCEPTS.md DEVLOG.md CHANGELOG.md REPRODUCE.md
git commit -m "feat: training loop + overfit test + sampling; toy CPU run produces a loss curve"
```

---

### Task 6: Colab training notebook for the `small` model — *Milestone: coherent stories*

**Files:**
- Create: `notebooks/colab_train.ipynb`
- Create: `notebooks/README.md`

**Interfaces:**
- Consumes: the entire `src/slm` package (cloned into Colab).
- Produces: a trained `small` checkpoint (`small.pt`) + tokenizer (`small_tok.json`) downloaded back locally into `checkpoints/`.

- [ ] **Step 1: Author the notebook** `notebooks/colab_train.ipynb` with these cells (use a notebook tool or write JSON; content of each cell below):

Cell 1 (setup):
```python
!git clone <YOUR_REPO_URL> model_learn || true
%cd model_learn
!pip -q install -r requirements.txt   # uv-exported list; Colab has no uv by default
import torch; print("CUDA:", torch.cuda.is_available())
```

Cell 2 (train small on GPU):
```python
import sys; sys.path.insert(0, "src")
import torch
from slm.config import SMALL
from slm.model import LlamaSLM
from slm.tokenizer import train_tokenizer
from slm.data import tokenize_texts, load_tinystories
from slm.train import TrainConfig, train, plot_loss

device = "cuda" if torch.cuda.is_available() else "cpu"
texts = load_tinystories("train", limit=None)         # full dataset
tok = train_tokenizer(texts, vocab_size=SMALL.vocab_size, save_path="checkpoints/small_tok.json")
data = tokenize_texts(tok, texts)
model = LlamaSLM(SMALL).to(device)
# Move batches to device: wrap get_batch results inside train() — for Colab,
# add `.to(device)` in a thin training cell, or set default device:
torch.set_default_device(device)
cfg = TrainConfig(lr=6e-4, warmup_steps=200, max_steps=20000, batch_size=64,
                  context_len=SMALL.context_len, log_every=100, sample_every=1000,
                  ckpt_path="checkpoints/small.pt")
train(model, data, cfg, tok=tok)
plot_loss("checkpoints/small_loss.csv", "checkpoints/small_loss.png")
```

Cell 3 (download artifacts):
```python
from google.colab import files
files.download("checkpoints/small.pt")
files.download("checkpoints/small_tok.json")
files.download("checkpoints/small_loss.png")
```

- [ ] **Step 2: Write `notebooks/README.md`** documenting: open in Colab → Runtime → change to GPU (T4) → run cells top to bottom → download the 3 files into local `checkpoints/`. Note the free-tier session limit and that `max_steps` can be lowered if the session is at risk of timing out; record the actual final `max_steps` and loss in `DEVLOG.md`.

- [ ] **Step 3: Run the notebook on Colab (manual)**

Expected: validation loss falls well below the toy run; `sample_every` printouts show increasingly coherent little stories by the end. Save `small_loss.png`.

- [ ] **Step 4: Verify the downloaded checkpoint locally on CPU**

Run: `python src/slm/sample.py checkpoints/small.pt checkpoints/small_tok.json "Once upon a time"`
Expected: a short, mostly-coherent children's story (the Phase-1 payoff).

- [ ] **Step 5: Update docs**

`DEVLOG.md`: record final `small` config, `max_steps`, training time, GPU type, final loss, and a verbatim generated story. `CHANGELOG.md` line. `REPRODUCE.md`: add the Colab section with the exact cells.

- [ ] **Step 6: Commit**

```bash
git add notebooks DEVLOG.md CHANGELOG.md REPRODUCE.md
git commit -m "feat: Colab notebook for small-model training; coherent-stories milestone"
```

---

### Task 7: Export to Hugging Face format + push to Hub — *Milestone: published*

**Files:**
- Create: `src/slm/export_hf.py`
- Create: `tests/test_export.py`

**Interfaces:**
- Consumes: `LlamaSLM`, `ModelConfig`, a trained checkpoint, the tokenizer JSON.
- Produces: `to_hf_config(cfg: ModelConfig) -> LlamaConfig`; `export_to_hf(ckpt_path, tok_path, out_dir) -> str` (writes a `LlamaForCausalLM` + `PreTrainedTokenizerFast` to `out_dir`, returns it); `push(out_dir, repo_id)`; and the key correctness guarantee: HF model logits equal hand-built logits within `atol=1e-4`.

- [ ] **Step 1: Write the failing test** in `tests/test_export.py`

```python
import torch
from slm.config import TOY
from slm.model import LlamaSLM
from slm.export_hf import to_hf_config, _copy_weights_into_hf


def test_hf_roundtrip_matches_handbuilt():
    torch.manual_seed(0)
    src = LlamaSLM(TOY).eval()
    from transformers import LlamaForCausalLM
    hf = LlamaForCausalLM(to_hf_config(TOY)).eval()
    _copy_weights_into_hf(src, hf)
    idx = torch.randint(0, TOY.vocab_size, (1, 12))
    with torch.no_grad():
        a = src(idx)
        b = hf(idx).logits
    assert torch.allclose(a, b, atol=1e-4), (a - b).abs().max().item()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_export.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'slm.export_hf'`.

- [ ] **Step 3: Implement `src/slm/export_hf.py`**

```python
import torch
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
from slm.config import ModelConfig
from slm.model import LlamaSLM
from slm.train import load_checkpoint


def to_hf_config(cfg: ModelConfig) -> LlamaConfig:
    return LlamaConfig(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.d_model,
        intermediate_size=cfg.ffn_hidden,
        num_hidden_layers=cfg.n_layers,
        num_attention_heads=cfg.n_heads,
        num_key_value_heads=cfg.n_kv_heads,
        head_dim=cfg.head_dim,
        max_position_embeddings=cfg.context_len,
        rms_norm_eps=cfg.rms_norm_eps,
        rope_theta=cfg.rope_theta,
        tie_word_embeddings=cfg.tie_embeddings,
        attention_bias=False,
        mlp_bias=False,
        hidden_act="silu",
    )


def _copy_weights_into_hf(src: LlamaSLM, hf: LlamaForCausalLM):
    sd = {}
    sd["model.embed_tokens.weight"] = src.embed_tokens.weight
    for i, blk in enumerate(src.layers):
        p = f"model.layers.{i}."
        sd[p + "input_layernorm.weight"] = blk.norm1.weight
        sd[p + "post_attention_layernorm.weight"] = blk.norm2.weight
        sd[p + "self_attn.q_proj.weight"] = blk.attn.q_proj.weight
        sd[p + "self_attn.k_proj.weight"] = blk.attn.k_proj.weight
        sd[p + "self_attn.v_proj.weight"] = blk.attn.v_proj.weight
        sd[p + "self_attn.o_proj.weight"] = blk.attn.o_proj.weight
        sd[p + "mlp.gate_proj.weight"] = blk.mlp.gate_proj.weight
        sd[p + "mlp.up_proj.weight"] = blk.mlp.up_proj.weight
        sd[p + "mlp.down_proj.weight"] = blk.mlp.down_proj.weight
    sd["model.norm.weight"] = src.norm.weight
    sd["lm_head.weight"] = src.lm_head.weight
    hf.load_state_dict(sd, strict=True)


def export_to_hf(ckpt_path: str, tok_path: str, out_dir: str) -> str:
    src, cfg = load_checkpoint(ckpt_path)
    src.eval()
    hf = LlamaForCausalLM(to_hf_config(cfg)).eval()
    _copy_weights_into_hf(src, hf)
    hf.save_pretrained(out_dir)
    fast = PreTrainedTokenizerFast(
        tokenizer_file=tok_path,
        eos_token="<|endoftext|>", bos_token="<|endoftext|>",
        unk_token="<|endoftext|>", pad_token="<|endoftext|>",
    )
    fast.save_pretrained(out_dir)
    return out_dir


def push(out_dir: str, repo_id: str):
    from huggingface_hub import HfApi
    HfApi().create_repo(repo_id, exist_ok=True)
    LlamaForCausalLM.from_pretrained(out_dir).push_to_hub(repo_id)
    PreTrainedTokenizerFast.from_pretrained(out_dir).push_to_hub(repo_id)
```

- [ ] **Step 4: Run the round-trip test**

Run: `pytest tests/test_export.py -v`
Expected: PASS. If it fails, the diff localizes the architecture mismatch (most likely RoPE convention or RMSNorm eps) — fix the hand-built model to match HF, since HF is the conversion target. This test is the linchpin of the whole "it converts to GGUF cleanly" promise.

- [ ] **Step 5: Export the trained small model and write a model card**

Run:
```bash
python -c "from slm.export_hf import export_to_hf; export_to_hf('checkpoints/small.pt','checkpoints/small_tok.json','export/tinystories-slm')"
```
Then create `export/tinystories-slm/README.md` (model card): what it is, training data (TinyStories), config, intended use (educational), limitations (tiny, stories-only), and the exact `sample.py` command to reproduce generation.

- [ ] **Step 6: Push to the Hub (requires `huggingface-cli login`)**

Run:
```bash
huggingface-cli login   # paste a token with write access
python -c "from slm.export_hf import push; push('export/tinystories-slm','<your-username>/tinystories-slm')"
```
Expected: repo appears on the Hub with weights, config, tokenizer, and model card. *(This is an outward-facing publish — confirm the repo name/visibility before running.)*

- [ ] **Step 7: Update docs**

`CONCEPTS.md`: **safetensors**, **config.json**, **model card**, **weight tying (input/output embeddings)**, **why HF format = portability**. `DEVLOG.md`: record the round-trip max-diff and the Hub URL. `CHANGELOG.md` line. `REPRODUCE.md`: add the export + push steps.

- [ ] **Step 8: Commit**

```bash
git add src/slm/export_hf.py tests/test_export.py export/tinystories-slm/README.md CONCEPTS.md DEVLOG.md CHANGELOG.md REPRODUCE.md
git commit -m "feat: export hand-built weights to LlamaForCausalLM (round-trip verified) + push to Hub"
```

---

### Task 8: GGUF conversion + Ollama run — *Milestone: runs in Ollama* 🎉

**Files:**
- Create: `scripts/convert_to_gguf.sh`
- Create: `scripts/Modelfile`
- Create: `labs/lab07_gguf_teardown.py`
- Create: `labs/lab08_quant_compare.py`

**Interfaces:**
- Consumes: the exported HF directory `export/tinystories-slm` from Task 7.
- Produces: `export/tinystories-slm-f16.gguf`, `...-Q8_0.gguf`, `...-Q4_K_M.gguf`; an Ollama model named `tinystories-slm`.

- [ ] **Step 1: Write `scripts/convert_to_gguf.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail
# Prereq (one-time):
#   git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp
#   pip install -r requirements.txt && cmake -B build && cmake --build build -j
LLAMA_CPP="${LLAMA_CPP:-$HOME/llama.cpp}"
SRC="export/tinystories-slm"
OUT="export"

python "$LLAMA_CPP/convert_hf_to_gguf.py" "$SRC" \
  --outfile "$OUT/tinystories-slm-f16.gguf" --outtype f16

"$LLAMA_CPP/build/bin/llama-quantize" \
  "$OUT/tinystories-slm-f16.gguf" "$OUT/tinystories-slm-Q8_0.gguf" Q8_0
"$LLAMA_CPP/build/bin/llama-quantize" \
  "$OUT/tinystories-slm-f16.gguf" "$OUT/tinystories-slm-Q4_K_M.gguf" Q4_K_M

echo "Wrote f16, Q8_0, Q4_K_M GGUFs to $OUT/"
```

- [ ] **Step 2: Run the conversion**

Run: `bash scripts/convert_to_gguf.sh`
Expected: three `.gguf` files; Q4_K_M is the smallest, f16 the largest. Sizes recorded in DEVLOG.

- [ ] **Step 3: Smoke-test the GGUF with llama.cpp directly**

Run: `~/llama.cpp/build/bin/llama-cli -m export/tinystories-slm-Q8_0.gguf -p "Once upon a time" -n 80`
Expected: coherent story text, close to the PyTorch sample — proving the weights survived conversion.

- [ ] **Step 4: Write `scripts/Modelfile`**

```
FROM ./export/tinystories-slm-Q4_K_M.gguf
PARAMETER temperature 0.8
PARAMETER top_k 40
PARAMETER stop "<|endoftext|>"
TEMPLATE """{{ .Prompt }}"""
```

- [ ] **Step 5: Create and run the Ollama model**

Run:
```bash
ollama create tinystories-slm -f scripts/Modelfile
ollama run tinystories-slm "Once upon a time"
```
Expected: Ollama streams a little story. **This is the Phase-1 finish line** — a model you built from scratch, running on your CPU through the same tool people use for Llama.

- [ ] **Step 6: Write Lab 07 (GGUF teardown)** `labs/lab07_gguf_teardown.py`

```python
"""Lab 07 — peek inside a .gguf: it's just a header + metadata + tensors.
Run: python labs/lab07_gguf_teardown.py export/tinystories-slm-Q8_0.gguf
"""
import struct, sys

path = sys.argv[1] if len(sys.argv) > 1 else "export/tinystories-slm-Q8_0.gguf"
with open(path, "rb") as f:
    magic = f.read(4)
    version, = struct.unpack("<I", f.read(4))
    n_tensors, = struct.unpack("<Q", f.read(8))
    n_kv, = struct.unpack("<Q", f.read(8))
print(f"magic        : {magic}  (should be b'GGUF')")
print(f"version      : {version}")
print(f"tensor count : {n_tensors}")
print(f"metadata kv  : {n_kv}")
print("\nThe header advertises how many tensors (the weights) and how many")
print("metadata entries (arch, hyperparams, tokenizer) follow. A .gguf is a")
print("self-describing container: the runtime reads this to know how to run it.")
```

- [ ] **Step 7: Write Lab 08 (quant compare)** `labs/lab08_quant_compare.py`

```python
"""Lab 08 — compare output + size across quantization levels.
Run: python labs/lab08_quant_compare.py
"""
import os, subprocess

LLAMA = os.path.expanduser("~/llama.cpp/build/bin/llama-cli")
for q in ("f16", "Q8_0", "Q4_K_M"):
    path = f"export/tinystories-slm-{q}.gguf"
    size_mb = os.path.getsize(path) / 1e6
    out = subprocess.run([LLAMA, "-m", path, "-p", "Once upon a time",
                          "-n", "40", "--seed", "0"],
                         capture_output=True, text=True).stdout
    print(f"=== {q}  ({size_mb:.1f} MB) ===\n{out.strip()[:200]}\n")
print("Smaller quant = smaller file + faster, with some quality loss.")
```

- [ ] **Step 8: Run both Labs**

Run: `python labs/lab07_gguf_teardown.py export/tinystories-slm-Q8_0.gguf` and `python labs/lab08_quant_compare.py`
Expected: Lab 07 prints `b'GGUF'` + counts; Lab 08 shows size shrinking f16 → Q8_0 → Q4_K_M with comparable story quality.

- [ ] **Step 9: Update docs**

`CONCEPTS.md`: **GGUF (container format)**, **quantization**, **Q8_0 vs Q4_K_M**, **inference engine (llama.cpp/Ollama)**, **Modelfile**, and **model compiler / deep-learning compiler** — frame this step explicitly as a *miniature model compiler*: `convert_hf_to_gguf.py` = export-to-IR, `llama-quantize` = an optimization pass, llama.cpp picking CPU kernels at load = codegen/runtime. Note the general landscape (ONNX Runtime, TVM, MLIR/IREE, TensorRT, OpenVINO, MLC-LLM) and that vendor stacks (incl. Tenstorrent's MLIR/Metalium toolchain) are the same idea for custom hardware — the transferable mental model for "download from HF → compile for hardware → run with confidence." `DEVLOG.md`: record GGUF sizes, the Ollama transcript, and quant comparison. `CHANGELOG.md`: mark Phase 1 complete. `REPRODUCE.md`: add the full convert → Ollama section. Update `README.md` quickstart to `ollama run tinystories-slm`.

- [ ] **Step 10: Commit**

```bash
git add scripts labs/lab07_gguf_teardown.py labs/lab08_quant_compare.py CONCEPTS.md DEVLOG.md CHANGELOG.md REPRODUCE.md README.md
git commit -m "feat: GGUF conversion + Ollama run; Phase 1 complete (Labs 07-08)"
```

---

## Self-Review (against the spec)

**Spec coverage:**
- §3 hand-build Llama-shaped → Tasks 4a/4b; HF-exactness enforced by Task 7 round-trip. ✓
- §3 reuse libs + Labs → tokenizer (T2/Lab01), attention (Lab02), gibberish (Lab03), sampling (Lab05), GGUF (Lab07/08). ✓
- §4 tech stack → `pyproject.toml` + `uv.lock` (uv-managed), `requirements.txt` exported for Colab (T1). ✓
- §5 dataset → `load_tinystories` (T3), used in T5/T6. ✓
- §6 model + two configs → T1 (configs), T4a/T4b (model). No GQA/MoE honored. ✓
- §7 milestone ladder → tokenizer (T2), batch (T3), gibberish (T4b), overfit (T5), toy run (T5), coherent (T6), published (T7), GGUF (T8), Ollama (T8). All nine rungs mapped. ✓
- §8 training strategy → T5 (loop, AdamW, cosine+warmup, clip, ckpt, sample callback, CSV/plot) + T6 (Colab). ✓
- §9 packaging→GGUF→run → T7 + T8. ✓
- §10 docs model → every task updates DEVLOG/CHANGELOG/CONCEPTS/REPRODUCE per Global Constraints. ✓
- §11 Labs → Labs 01,02,03,05,07,08 created. *(Lab04 RoPE-viz and Lab06 checkpoint-progression are listed in the spec as nice-to-haves; folded into CONCEPTS notes rather than standalone tasks to avoid bloat — noted here as an intentional trim, not a silent drop.)*
- §13 verification → shape tests (T4a/b), causal test (T4a), overfit test (T5), round-trip test (T7), GGUF smoke test (T8). ✓
- §14 compute → toy local (T5), small Colab (T6). ✓

**Placeholder scan:** The only non-literal item is `<YOUR_REPO_URL>` / `<your-username>` in Tasks 6–7, which are necessarily user-specific (the engineer fills their own repo); flagged inline. The `lr_at` task intentionally includes a "replace this" note with the clean implementation shown in full — not a placeholder, a deliberate cleanup instruction. No TBD/TODO logic gaps remain.

**Type consistency:** `ModelConfig` field names are used identically across config/model/export. `LlamaSLM` attribute names (`embed_tokens, layers, norm, lm_head`, block `norm1/attn/norm2/mlp`, attn `q_proj/k_proj/v_proj/o_proj`, mlp `gate_proj/up_proj/down_proj`) match exactly what `_copy_weights_into_hf` reads in Task 7. `get_batch`/`TrainConfig`/`train` signatures match their call sites. ✓
