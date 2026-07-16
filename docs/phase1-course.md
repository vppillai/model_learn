# Phase 1 — Build a Language Model From Scratch to Ollama: A Self-Study Course

## How this course works

This course retraces the build you already shipped, but this time *you* do the
understanding instead of watching the code get written. After a short math
Warm-Up, the course is organized into ten Modules that follow the exact order
you built things in — tokenizer, then data, then model, then training, then
export, then deployment. Every build Module (1 through 10) follows the same
six-section template, in this order: **Learning objectives** (what you'll be
able to explain afterward), **Frame** (the plain-language intuition and why
the piece exists), **Annotated code walkthrough** (the real code you wrote,
with commentary), **What we observed** (the actual output you got when you
ran it), **Gotchas & design decisions** (the real detours and why each choice
was made), and **Checkpoint** (a few quiz questions plus one "explain it back
in your own words" prompt). The Warm-Up that comes right after Module 0 is
the one exception — it's a lighter primer, not a full build Module, because
its job is just to hand you the handful of numeric building blocks (dot
product, matrix multiply, softmax, RMS-normalization, rotation) that Modules
4 and 6 lean on later.

## How to use this in NotebookLM

This document is meant to be uploaded to NotebookLM as its own source. Every
answer NotebookLM gives you should be grounded in the text below — nothing
here assumes you can go open the original repo, so you shouldn't need to
either. Treat NotebookLM as a tutor sitting on top of this course. Some
prompts that work well:

- *"Act as my tutor and quiz me on Module 4."*
- *"Check my understanding of RoPE — ask me to explain it, then correct me."*
- *"Walk me through the Warm-Up softmax example and then give me a new one."*
- *"I'll answer the Module 6 checkpoint; grade my answers."*

Use the Checkpoint questions at the end of each Module as your quiz bank, and
don't be shy about asking NotebookLM to invent variations on the worked
examples — the numbers in this document are real, so a good tutor session
should be able to check your arithmetic against them.

## Who this is for / level

This course is intuition-first: no heavy math, no proofs, just the concepts
you need to actually understand what you built and why. It assumes you're
comfortable reading Python, but nothing more specialized than that — anywhere
the underlying idea needs a piece of math (a dot product, a matrix multiply,
softmax), the Warm-Up below builds it up from a tiny worked example first.

## Table of contents

- [Module 0 — The Big Picture](#module-0--the-big-picture)
- [Warm-Up — The Math, on Tiny Arrays](#warm-up--the-math-on-tiny-arrays)
- [Module 1 — Environment & Reproducibility](#module-1--environment--reproducibility)
- [Module 2 — Tokenizer & BPE](#module-2--tokenizer--bpe)
- [Module 3 — Data Pipeline: Batching and the Shift-by-One Target](#module-3--data-pipeline-batching-and-the-shift-by-one-target)
- [Module 4 — Model Internals: RMSNorm, RoPE, Causal Attention, SwiGLU](#module-4--model-internals-rmsnorm-rope-causal-attention-swiglu)
- [Module 5 — Assembling the Model, Generation, and the Untrained "Echo Bias"](#module-5--assembling-the-model-generation-and-the-untrained-echo-bias)
- [Module 6 — Training](#module-6--training)
- [Module 7 — The Real Run (small on Colab)](#module-7--the-real-run-small-on-colab)
- [Module 8 — Packaging to Hugging Face Format](#module-8--packaging-to-hugging-face-format)
- [Module 9 — GGUF, Quantization, and Ollama](#module-9--gguf-quantization-and-ollama)
- [Module 10 — The Transferable Mental Model: Model Compilers](#module-10--the-transferable-mental-model-model-compilers)
- [Glossary](#glossary)

## Module 0 — The Big Picture

Every day, the machine-learning ecosystem hands people a simple three-step
loop: **download** a model someone else trained, **build** something with it
(fine-tune it, wrap it, prompt it), and **run** it somewhere — a server, a
laptop, a phone. Almost nobody outside a handful of labs ever does the middle
step for real: actually building a model's insides from nothing. That gap is
exactly what Phase 1 closes. Instead of downloading a model and treating its
internals as a black box, this course builds one, piece by piece, from a bare
tokenizer up to a trained transformer, and then hands the result to the
*exact same* runtimes everyone else uses to run a downloaded model — Hugging
Face `transformers`, `llama.cpp`, and Ollama. By the end, "download → build →
run" stops being someone else's pipeline and becomes one you've walked
end-to-end yourself, in both directions: you built the thing that would
normally arrive pre-built, and you shipped it into the same tools that run
pre-built things.

Concretely, the pipeline this course walks — in the order the Modules
below present it — looks like this:

**tokenizer → batches of shift-by-one training windows → a hand-built
Llama-shaped transformer → training → a Hugging Face–format checkpoint → a
quantized GGUF file → running locally in Ollama.**

Each arrow in that line is one or more Modules. The Warm-Up right after this
section gives you the tiny bits of math notation you need before Module 1
starts the walk.

## Warm-Up — The Math, on Tiny Arrays

**Learning objectives:** by the end of this Warm-Up you'll be able to read a
shape like `(batch, seq, d_model)` without flinching, compute a dot product
and a small matrix multiply by checking shapes instead of grinding through
arithmetic, explain in plain words what softmax and RMS-normalization each do
to a vector, and see why a 2D rotation is the trick RoPE uses to bake
position into attention. Every number below is real — it's the actual output
of a two-line PyTorch snippet, not something worked out by hand.

### Vectors & shapes

A token, once it's inside the model, is just a list of numbers — a vector of
length `d_model`. Stack a batch of sequences of these vectors and you get the
shape you'll see stamped on nearly every tensor in this course:
`(batch, seq, d_model)`.

```python
import torch
torch.manual_seed(0)
x = torch.randn(2, 3, 4)  # (batch=2, seq=3, d_model=4)
print("shape:", tuple(x.shape))
print("x[0,0] (one token vector):", [round(v, 3) for v in x[0,0].tolist()])
```
```
shape: (2, 3, 4)
x[0,0] (one token vector): [-1.126, -1.152, -0.251, -0.434]
```

That's 2 sequences of 3 tokens each, each token a 4-number vector — `x[0,0]`,
the first token of the first sequence, is that vector: `[-1.126, -1.152,
-0.251, -0.434]`. Nothing about these numbers is meaningful yet (they're
random), but the *shape* is the whole point: every module from here on — the
batcher in Module 3, the attention block in Module 4, the training loop in
Module 6 — is just doing arithmetic that keeps this `(batch, seq, d_model)`
shape intact (or deliberately reshapes it on purpose).

### Dot product

The dot product multiplies two vectors element-by-element and adds up the
results into a single number — a measure of how much two vectors "point the
same way."

```python
import torch
a = torch.tensor([1.0, 2.0])
b = torch.tensor([3.0, 4.0])
print("dot:", torch.dot(a, b).item())
```
```
dot: 11.0
```

`1*3 + 2*4 = 11`. That's it — one number out of two vectors in. This is the
entire arithmetic core of attention: in Module 4, "how much should this token
attend to that token" is computed by taking the dot product of a query
vector and a key vector.

### Matrix multiply & shape rules

A matrix multiply is just a lot of dot products at once: row `i` of the
first matrix dotted with column `j` of the second gives entry `(i, j)` of the
result. The rule that makes this legal: the *inner* dimensions have to match
— `(2,3) @ (3,4) → (2,4)`. The `3`s cancel out; what's left is the outer
shape.

```python
import torch
A = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # (2,3)
B = torch.tensor([[1.0, 0.0, 0.0, 1.0],
                   [0.0, 1.0, 0.0, 1.0],
                   [0.0, 0.0, 1.0, 1.0]])              # (3,4)
C = A @ B
print("A shape:", tuple(A.shape))
print("B shape:", tuple(B.shape))
print("C shape:", tuple(C.shape))
print(C.tolist())
```
```
A shape: (2, 3)
B shape: (3, 4)
C shape: (2, 4)
[[1.0, 2.0, 3.0, 6.0], [4.0, 5.0, 6.0, 15.0]]
```

Every "projection" in the model — turning a `d_model`-length vector into a
query, a key, a value, or a vocabulary of logits — is a matmul against a
weight matrix, and it only works because someone made sure the inner
dimension lines up. This is exactly the shape-bookkeeping behind Module 4's
`q @ k.transpose(-2, -1)` attention scores.

### Softmax

Softmax takes a list of raw scores and turns them into a probability
distribution: everything comes out positive and the whole thing sums to 1,
while preserving the *order* of the scores (bigger score in, bigger share of
the probability out).

```python
import torch
scores = torch.tensor([2.0, 1.0, 0.0])
probs = torch.softmax(scores, -1)
print("softmax:", probs.tolist())
print("sum:", probs.sum().item())
print("softmax T=0.5 (sharper):", torch.softmax(scores / 0.5, -1).tolist())
print("softmax T=2.0 (flatter):", torch.softmax(scores / 2.0, -1).tolist())
```
```
softmax: [0.6652409434318542, 0.2447284758090973, 0.09003057330846786]
sum: 1.0
softmax T=0.5 (sharper): [0.8668133020401001, 0.11731041967868805, 0.01587623916566372]
softmax T=2.0 (flatter): [0.5064803957939148, 0.30719590187072754, 0.18632373213768005]
```

`[2, 1, 0]` becomes roughly `[0.665, 0.245, 0.090]`, and yes, that sums to 1.
Dividing the scores by a *temperature* before the softmax changes how
confident the distribution looks without changing the ranking: dividing by
`0.5` (a low temperature) sharpens it to about `[0.867, 0.117, 0.016]` — much
more confident — while dividing by `2.0` (a high temperature) flattens it to
about `[0.506, 0.307, 0.186]` — much closer to uniform. This is the exact
function that turns raw attention scores into attention *weights* in Module
4, and it's also the last step before sampling a token in Module 5 and half
of the cross-entropy loss in Module 6.

### RMS-normalization

RMS-normalization rescales a vector by its own root-mean-square, so vectors
with wildly different magnitudes get pulled back onto a comparable scale
before they're used.

```python
import torch
v = torch.tensor([3.0, 4.0])
rms = v.pow(2).mean().sqrt().item()
print("rms:", rms)
normalized = (v / v.pow(2).mean().add(1e-5).sqrt()).tolist()
print("normalized:", normalized)
```
```
rms: 3.535533905029297
normalized: [0.8485277891159058, 1.1313704252243042]
```

The root-mean-square of `[3, 4]` is about `3.536` (mean of `9` and `16` is
`12.5`, and `sqrt(12.5) ≈ 3.536`), and dividing the original vector by that
gives `[0.849, 1.131]`. That tiny `1e-5` added before the square root is just
insurance against dividing by zero if a vector were ever all zeros — it
barely nudges these numbers. This is the *exact* kernel behind RMSNorm, the
normalization layer you'll build in Module 4.

### 2D rotation

Rotating a 2D vector by an angle spins it around the origin — and however
far you spin it, its length never changes. That length-preserving property
is the whole reason RoPE can use rotation to encode *position* without
distorting the *magnitude* information a query or key vector carries.

```python
import torch, math
theta = math.pi / 2  # 90 degrees
v = torch.tensor([1.0, 0.0])
R = torch.tensor([[math.cos(theta), -math.sin(theta)],
                  [math.sin(theta),  math.cos(theta)]])
rotated = R @ v
print("rotated:", rotated.tolist())
print("norm before:", v.norm().item())
print("norm after:", rotated.norm().item())
```
```
rotated: [6.123234262925839e-17, 1.0]
norm before: 1.0
norm after: 1.0
```

Rotating `[1, 0]` by 90° lands at `[0, 1]` — the `6.12e-17` on the x-axis is
just floating-point noise standing in for an exact zero (`cos(90°)` isn't
*quite* representable exactly). Notice the norm: `1.0` before, `1.0` after —
unchanged. RoPE (Module 4) applies rotations like this one, at different
angles per position, to pairs of numbers inside each query and key vector.
The rotation angle encodes *where* a token sits in the sequence; the
preserved length is exactly the property Module 4's RoPE test checks.

### Checkpoint

1. Work out `softmax([1, 1, 1])` by reasoning about what softmax does to
   equal scores, not by grinding through the formula. What comes out, and
   why?
2. You have a `(4, 8)` matrix and a `(3, 8)` matrix. Why can't you compute
   `A @ B` directly? What would have to change, and why do the inner
   dimensions of a matmul always have to match?
3. If you rotate a 2D vector, what property of it never changes — and why
   does that matter for a technique like RoPE, which rotates pieces of the
   query and key vectors?

**Explain it back:** in one sentence each — what does a dot product do, what
does softmax do, and what does a rotation do?

## Module 1 — Environment & Reproducibility

### Learning objectives

By the end of this Module you'll be able to explain what a virtual
environment buys a project that a shared, global Python install doesn't;
explain the difference between what `pyproject.toml` asks for and what
`uv.lock` actually pins; and explain why a dependency pin that's exactly
right for the machine you're developing on can be exactly wrong for a
different machine downstream (in this project's case, a Colab GPU box) —
and what you do about that.

### Frame

None of the model-building Modules that follow mean anything if you can't
reproduce the environment they ran in. `torch`, `tokenizers`, and
`transformers` are all fast-moving libraries with real behavioral
differences between versions — training that works on `torch==2.12` can
silently do something different on `torch==2.8`. So before touching any
model code, this project needed two separate things nailed down: an
**isolated** place for this project's dependencies to live (so they can't
collide with whatever else is installed on the machine, or with some other
project's dependencies), and a **reproducible** record of exactly which
versions got installed, not just a rough range.

Think of it like the difference between a recipe and a grocery receipt. A
recipe says "roughly a cup of flour, a couple of eggs" — that's
`pyproject.toml`, expressing intent with loose constraints like
`torch>=2.5`. A grocery receipt says "1.000 cup King Arthur all-purpose
flour, lot #4471" — that's `uv.lock`, the exact, fully-resolved set of
package versions (down to the specific build) that satisfied those loose
constraints the one time the resolver ran. Anyone who "re-shops from the
receipt" (runs `uv sync` against the checked-in lockfile) gets
byte-identical versions, not just versions that satisfy the same loose
range. And the virtual environment is the private kitchen the groceries go
into — a project-scoped Python install and package directory that keeps
this project's exact ingredient set from colliding with any other project's
kitchen on the same machine.

This project uses `uv` (a fast Python package manager and resolver) to
manage both the venv and the lockfile, but the concepts — isolate, then pin
exactly — apply no matter which tool does the job.

### Annotated code walkthrough

The dependency intent lives in `pyproject.toml`. Here's the real,
current dependencies block:

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

Every entry is a *loose* constraint — `torch>=2.5` means "any version 2.5 or
newer," not "exactly 2.5." That looseness is deliberate: it lets the
resolver pick compatible versions of everything together rather than
forcing an exact version this project doesn't actually require. `pytest` is
split into its own `[dependency-groups]` table because it's a development
tool, not something the shipped model code needs at runtime — keeping it
separate means a deployment that only needs `dependencies` doesn't have to
pull in the test framework too.

Right below that sits the one exception to "let the resolver pick freely" —
the CPU-torch pin:

```toml
[tool.uv.sources]
torch = { index = "pytorch-cpu" }

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true
```

`[tool.uv.sources]` tells `uv` "for the `torch` dependency specifically,
don't resolve it from the default index (PyPI) — resolve it from this named
index instead." `[[tool.uv.index]]` then defines that named index,
`pytorch-cpu`, pointing at PyTorch's own CPU-only wheel index, with
`explicit = true` meaning "only use this index for packages that
explicitly ask for it" (so it doesn't silently take over resolution for
every other dependency too). The Gotchas section below explains exactly why
this pin exists.

### What we observed

Running `uv sync` reads `pyproject.toml`, resolves every constraint against
the `pytorch-cpu` index for `torch` and the default index for everything
else, installs the result into `.venv/`, and writes the exact resolved
versions into `uv.lock` — the "grocery receipt" from the Frame section
above. `uv.lock` is a large, auto-generated file (not meant to be hand-read
line by line), but its header shows exactly what it's pinning against:

```toml
version = 1
revision = 3
requires-python = ">=3.13"
```

Outside this project's own `uv`-managed venv — anywhere a plain `pip
install -r requirements.txt` is the expected install path rather than a
`uv`-native one — this project also exports a portable `requirements.txt`
with:

```bash
uv export --no-hashes --format requirements-txt --no-emit-package torch -o requirements.txt
```

The resulting file opens with an autogenerated header recording the exact
command that produced it:

```
# This file was autogenerated by uv via the following command:
#    uv export --no-hashes --format requirements-txt --no-emit-package torch -o requirements.txt
aiohappyeyeballs==2.7.1
    # via aiohttp
aiohttp==3.14.1
    # via fsspec
...
```

Notice `torch` never appears in that list — that's `--no-emit-package
torch` doing its job, and it's the direct fix for Gotcha 2 below.

### Gotchas & design decisions

**Gotcha 1 — the default `torch` wheel pulled in a GPU stack this box can't
use.** A plain `uv sync`, before the CPU pin existed, resolved
`torch==2.12.1+cu130` plus roughly 20 `nvidia-*` CUDA-runtime packages,
ballooning the venv to about 4.8GB — on a machine with no NVIDIA GPU at all
(`torch.cuda.is_available()` was `False` regardless of which torch build
was installed). PyPI's default Linux `torch` wheel always bundles CUDA
runtime dependencies; there's no install-time hardware detection that skips
them. The fix was the `[tool.uv.sources]` pin shown above, forcing `torch`
to resolve from PyTorch's dedicated CPU-only index instead. Re-running `uv
sync` after adding the pin dropped the venv to about 1.1GB and installed
`torch==2.12.1+cpu` — roughly 3.7GB of unused CUDA machinery gone.

**Gotcha 2 — that same pin risked leaking a CPU-only `torch` into
`requirements.txt`.** A naive `uv export` propagates whatever `torch` is
actually pinned to, so the first `requirements.txt` said
`torch==2.12.1+cpu` for every platform — installing that on any GPU
machine would replace a working GPU `torch` with a CPU build, silently
killing GPU acceleration. The fix: exclude `torch` from the export
entirely with `--no-emit-package torch`, since a GPU host already ships
its own matched `torch`, and this project's own code (`import torch`)
doesn't care which `torch` provided it, as long as one is present. On
Colab specifically (Module 7 covers the full run), the notebook doesn't
install from `requirements.txt` at all — its setup cell runs a minimal
`pip install datasets tokenizers` instead, because even with `torch`
excluded, the rest of `requirements.txt` still pins a full transitive
dependency tree that conflicts with Colab's own co-tuned
`numpy`/`pandas`/`torch`. `requirements.txt` is the right artifact for a
fresh, empty environment; Colab's already-populated one calls for the
narrower install instead.

### Checkpoint

1. What does `uv.lock` give you that `pyproject.toml`'s `torch>=2.5` does
   not?
2. Why would the CPU torch pin have broken the Colab GPU run if left in
   `requirements.txt`?

**Explain it back:** in your own words, what is a virtual environment and
why use one here?

## Module 2 — Tokenizer & BPE

### Learning objectives

By the end of this Module you'll be able to explain why a tokenizer has to
exist at all before any model code runs; explain, step by step, what BPE
training actually does to arrive at a vocabulary; read the real
`train_tokenizer`/`load_tokenizer`/`encode`/`decode` functions and say what
each line is for; explain what the `Ġ` symbol means and why `Ġcat` and
`cat` are different tokens; and explain why `<|endoftext|>` ends up fixed
to token id `0`.

### Frame

A neural net never sees text. It only ever does arithmetic on numbers —
matrix multiplies, dot products, the same operations from the Warm-Up above
— so before a single character of a story can reach the model, something
has to translate that text into a list of integers, and translate the
model's output integers back into text. That translator is the
**tokenizer**, and it's the mandatory first (and last) step of the entire
pipeline: text in, integers out; integers in, text out.

The specific translation scheme this project uses is **BPE — byte-pair
encoding**. The idea is almost embarrassingly simple: start from the
smallest possible units (raw bytes), count which adjacent pair of units
shows up together most often across the training text, merge that single
most-frequent pair into one new unit, and repeat — over and over — until
the vocabulary reaches whatever target size you asked for. Common chunks of
text (like `"the "` or `"cat"`) get merged early and often, so they end up
as single tokens; rare or unusual byte sequences never get merged as much,
so they stay as smaller pieces. Nothing is ever left over unrepresented,
because the starting units are already the smallest possible thing text can
be broken into.

### Annotated code walkthrough

Here's the real, current `src/slm/tokenizer.py`, in full:

```python
import os
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
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    tok.save(save_path)
    return tok


def load_tokenizer(path: str) -> Tokenizer:
    return Tokenizer.from_file(path)


def encode(tok: Tokenizer, text: str) -> list[int]:
    return tok.encode(text).ids


def decode(tok: Tokenizer, ids: list[int]) -> str:
    return tok.decode(ids)
```

Walking through `train_tokenizer` line by line:

- `Tokenizer(BPE(unk_token=None))` builds a tokenizer whose *model*
  (the merge rules and vocabulary) is BPE, and explicitly says there's no
  "unknown token" fallback — more on exactly why that's safe below.
- `tok.pre_tokenizer = ByteLevel(add_prefix_space=False)` is the piece that
  makes this a *byte-level* BPE tokenizer rather than a character-level
  one: before any merging happens, the pre-tokenizer maps every raw UTF-8
  byte of the input to one of 256 printable stand-in characters (a
  reversible byte↔character mapping), and BPE then merges those. Operating
  on bytes instead of Unicode characters is what guarantees any input —
  any language, emoji, or stray binary garbage — is representable, since
  every possible byte value already has a defined mapping.
- `tok.decoder = ByteLevelDecoder()` is the exact inverse of the
  pre-tokenizer: it undoes that byte↔character remapping when turning
  token ids back into text, so `decode` gives you back real UTF-8 text
  rather than the visible stand-in characters.
- The `BpeTrainer` is configured with three things that matter: `vocab_size`
  (the target vocabulary size — training stops merging once it's reached),
  `special_tokens=[EOT]` (the list of tokens to insert before any of the
  byte alphabet or learned merges — `EOT` being `"<|endoftext|>"`, defined
  at the top of the file), and `initial_alphabet=ByteLevel.alphabet()`
  (seeds the starting vocabulary with all 256 single-byte tokens, before
  any merge has happened). Because special tokens are inserted first and
  `special_tokens` here contains exactly one entry, `<|endoftext|>` always
  lands at id `0` — verified directly: training a fresh tokenizer and
  calling `tok.token_to_id("<|endoftext|>")` returns `0` every time, and
  id `1` is the next token in the vocabulary (the first entry of the byte
  alphabet), never anything to do with `<|endoftext|>`.
- `tok.train_from_iterator(list(texts), trainer=trainer)` is where the
  actual merge-counting-and-merging loop runs, over whatever training texts
  were passed in.
- `os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)` creates
  the parent directory of `save_path` if it doesn't already exist, right
  before `tok.save(save_path)` writes the tokenizer out as a JSON file.
  This line wasn't there on day one — see the Gotchas section below for
  why it had to be added.

The other three functions are deliberately thin: `load_tokenizer` just
re-hydrates a `Tokenizer` from that saved JSON file; `encode` calls the
underlying tokenizer's `.encode(text)` and returns just the `.ids` list
(discarding the richer object `tokenizers` returns, since this project only
ever needs the plain integer ids); and `decode` calls the underlying
`.decode(ids)`, which routes through the `ByteLevelDecoder` configured
above to undo the byte-level remapping and hand back real text.

### What we observed

Training the same paragraph — `"the cat sat on the mat. the cat sat on the
hat. the rat sat on the cat."`, repeated 20 times — at increasing vocab
sizes and encoding `"the cat sat"` each time shows the merges forming live:

```
vocab= 260  'the cat sat' -> ['the', 'Ġ', 'c', 'at', 'Ġ', 's', 'at']
vocab= 270  'the cat sat' -> ['the', 'Ġcat', 'Ġsat']
vocab= 290  'the cat sat' -> ['the', 'Ġcat', 'Ġsat']
vocab= 320  'the cat sat' -> ['the', 'Ġcat', 'Ġsat']
```

At `vocab_size=260`, `"the "` has already merged into a single token
(`the`), but `cat` and `sat` haven't yet — they're still split into a
leading space token (`Ġ`), a first character, and a two-character remainder
(`at`). By `vocab_size=270`, just ten more merge slots later, `Ġcat` and
`Ġsat` have each collapsed into single tokens, and adding still more
vocabulary budget (290, 320) doesn't change this particular phrase any
further — the common chunks in this small paragraph have already found
their merges. This is exactly what "as the vocab grows, common chunks
become single tokens" looks like as real numbers, not just a description.

That `Ġ` deserves a direct explanation: it's the *visible* stand-in
character the byte-level pre-tokenizer uses for a literal space byte (a
GPT-2-era convention, carried into this tokenizer). `Ġcat` means "a token
that is a space followed by `cat`" — the version of the word that appears
after another word — while a bare `cat` (no `Ġ`) means the letters `c-a-t`
appearing with no preceding space, as inside a longer word or right at the
start of the text. They are different token ids representing different
byte sequences, even though a human reading the decoded text would just
see "cat" either way.

### Gotchas & design decisions

**Byte-level means nothing is ever "unrepresentable."** Because
`ByteLevel.alphabet()` seeds the vocabulary with all 256 possible byte
values before a single merge happens, and because the pre-tokenizer breaks
any input down into those bytes first, there is always a way to represent
whatever text comes in — worst case, one token per raw byte, with no
merges applying at all. That's exactly why `BPE(unk_token=None)` is safe:
traditional word-level tokenizers need an `<unk>` fallback for words they've
never seen, but a byte-level tokenizer never hits that situation, since the
byte alphabet is complete by construction.

**`train_tokenizer` had to learn to create its own save directory.** The
`os.makedirs(...)` line in the walkthrough above wasn't part of the
function on day one. It surfaced during the first real training run, when
`train_tokenizer` tried to write its output to a `checkpoints/` directory
that didn't exist yet — `tok.save(save_path)` failed outright, because the
function just called `tok.save()` on a path whose parent directory hadn't
been created. The project's other save functions (checkpoint-saving, CSV
loss logs) already made sure their target directory existed first;
`train_tokenizer` was the one place that didn't, and the fix was making it
consistent with the rest of the codebase — create the parent directory,
`exist_ok=True` so it's a no-op if the directory is already there, then
save.

### Checkpoint

1. Why is nothing ever "unrepresentable" with a byte-level BPE tokenizer?
2. What does `Ġcat` mean and how does it differ from `cat`?
3. Why is `<|endoftext|>` fixed to id 0?

**Explain it back:** walk through what BPE training actually does, step by
step.

## Module 3 — Data Pipeline: Batching and the Shift-by-One Target

### Learning objectives

By the end of this Module you'll be able to explain why language modeling's
training signal comes for free from ordinary text, with no separate labeling
step required; read the real `tokenize_texts` and `get_batch` functions and
say what each line does; state the exact relationship between a batch's
input `x` and its target `y`, in terms of both shapes and values; and
explain why `load_tinystories` — the function that pulls down the real
training corpus — is deliberately never called from the unit tests.

### Frame

Every supervised model needs labeled examples: an input, and the correct
answer for that input. Language modeling gets this label for free — it's
already sitting inside the raw text, with no separate annotation step
required. Take any window of tokens, and the "correct answer" for what
follows each token in that window is just whatever token actually comes
next in the real text. So if your input window `x` is 12 tokens long, the
target `y` for that exact same window is those same 12 tokens shifted one
position to the left: `y` at every position holds whatever token really
followed `x` at that position. That's the whole training signal for
language modeling — no labels file, no human annotation, just the text
itself, read one position ahead of where you started.

This is exactly the tiny-array indexing from the Warm-Up, just applied to a
much longer array: if `t` is the entire token stream and `s` is some
starting position, `x = t[s : s+context_len]` and `y = t[s+1 :
s+1+context_len]` — one slice, and that same slice again, shifted over by
one. Because every position in `x` gets its own real target sitting right
there in `y`, a single `context_len`-token input sequence doesn't hand you
one training example — it hands you `context_len` of them simultaneously,
one per position, all for the cost of a single forward pass.

Before any of that windowing can happen, though, the raw text has to become
one long, flat stream of token ids — that's what makes plain slicing
possible in the first place. `tokenize_texts` builds that stream: it runs
every document (story) through the tokenizer from Module 2, and glues all
of their token-id lists end to end into one continuous list, inserting the
reserved `<|endoftext|>` id (`0`, fixed there since Module 2) after each
document so there's a concrete marker for where one story ends and the next
begins. `get_batch` then does the windowing over that flat stream: given
`batch_size`, `context_len`, and a `seed`, it picks `batch_size` random
starting positions and, for each one, slices out both `x` (a
`context_len`-token window starting there) and `y` (the same length window,
starting exactly one token later).

### Annotated code walkthrough

Here's the real, current `src/slm/data.py`, in full:

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


def get_batch(data, batch_size: int, context_len: int, seed: int):
    # Accept a pre-built tensor (cheap) or a list (converted once). At scale,
    # re-tensorizing a multi-million-token list every call dominates runtime,
    # so train() passes a tensor built once before the loop.
    g = torch.Generator().manual_seed(seed)
    t = data if isinstance(data, torch.Tensor) else torch.tensor(data, dtype=torch.long)
    max_start = len(t) - context_len - 1
    assert max_start > 0, "not enough tokens for one context window"
    starts = torch.randint(0, max_start, (batch_size,), generator=g)
    x = torch.stack([t[s : s + context_len] for s in starts])
    y = torch.stack([t[s + 1 : s + 1 + context_len] for s in starts])
    return x, y


def load_tinystories(split: str = "train", limit: int | None = None) -> list[str]:
    """Real dataset loader for training runs (not used in unit tests).

    When `limit` is set (e.g. the local toy run), stream the dataset and take
    only the first `limit` stories, so we avoid downloading the full ~1.9GB
    corpus just to use a few thousand. When `limit` is None (the Colab `small`
    run), download the whole split.
    """
    from datasets import load_dataset
    if limit is not None:
        ds = load_dataset("roneneldan/TinyStories", split=split, streaming=True)
        out = []
        for row in ds:
            out.append(row["text"])
            if len(out) >= limit:
                break
        return out
    ds = load_dataset("roneneldan/TinyStories", split=split)
    return [row["text"] for row in ds]
```

`tokenize_texts` and `get_batch` are the two functions every real training
batch actually flows through; `load_tinystories` gets its own discussion in
Gotchas below. Walking through the first two line by line:

`tokenize_texts`:

- `stream: list[int] = []` starts an empty flat list that will hold every
  token id from every document, back to back, with nothing yet marking
  where one document's tokens stop and the next one's start.
- for each text `t`, `stream.extend(tok.encode(t).ids)` appends that one
  story's token ids onto the end of the running stream (using the same
  `encode` behavior built in Module 2).
- `stream.append(EOT_ID)` appends id `0` — `<|endoftext|>` — right after
  each story's tokens, so the finished stream reads
  `[story tokens][0][story tokens][0]...`. That single id is the only
  marker of document boundaries anywhere in this pipeline; nothing
  downstream needs to track story boundaries separately.
- the function returns that one flat list of ids. From here on, the data is
  just one long sequence of integers — the windowing that `get_batch` does
  next has no idea, and doesn't need to know, where the original document
  boundaries were, beyond the `0`s baked into the stream itself.

`get_batch`:

- `g = torch.Generator().manual_seed(seed)` creates a private random-number
  generator seeded with the given `seed`, instead of drawing from PyTorch's
  global RNG. That's what makes `get_batch` deterministic on its own terms:
  call it twice with the same `seed` and the same `data`, and you get back
  the exact same batch, every time (confirmed directly below in "What we
  observed").
- `t = data if isinstance(data, torch.Tensor) else torch.tensor(data,
  dtype=torch.long)` accepts either a plain Python list (like the one
  `tokenize_texts` returns) or an already-built tensor, and only pays the
  cost of converting to a tensor when it actually has to. The comment right
  above it explains why that matters: re-converting a multi-million-token
  list to a tensor on every single call would dominate training time, so
  the real training loop builds the tensor once, up front, and passes that
  same tensor into `get_batch` on every subsequent call.
- `max_start = len(t) - context_len - 1` is the last position in the stream
  from which you could still slice out a full `context_len`-token window
  *and* one more token past it for that window's shifted target — start any
  later than this and the slice would run off the end of the stream.
- `assert max_start > 0, "not enough tokens for one context window"` fails
  fast and explicitly if the data is too short to produce even one valid
  window — see Gotchas below for why that matters in practice.
- `starts = torch.randint(0, max_start, (batch_size,), generator=g)` draws
  `batch_size` random starting positions, each between `0` and `max_start`,
  using the private generator from above — so this draw, too, is fully
  determined by `seed`.
- `x = torch.stack([t[s : s + context_len] for s in starts])` slices out one
  `context_len`-token window per starting position and stacks them into a
  single tensor of shape `(batch_size, context_len)`.
- `y = torch.stack([t[s + 1 : s + 1 + context_len] for s in starts])` does
  the exact same slicing, from the exact same starting positions — just
  shifted one token later. This is the whole shift-by-one target from the
  Frame section above, written as two slices that differ only by that `+1`.
- the function returns `(x, y)`: the input batch and its target batch, both
  shaped `(batch_size, context_len)`.

### What we observed

`tests/test_data.py` confirms the two properties above directly against
real code: `test_tokenize_inserts_eot` checks that `0` really does show up
in `tokenize_texts`'s output stream, and `test_batch_shapes_and_shift`
checks both `x.shape == (4, 8)` and `y.shape == (4, 8)` for a `batch_size=4,
context_len=8` call, plus `torch.equal(x[:, 1:], y[:, :-1])` — i.e., every
`x` value except its first column lines up exactly with every `y` value
except its last column, which is the shift-by-one written as a tensor
equality instead of prose.

The project's DEVLOG makes that shift concrete by decoding an actual batch
back to text. From `DEVLOG.md` (2026-07-07):

> Printed a real batch and decoded it back to text to see the shift-by-one
> target directly: with `context_len=12`, `x[0]` decoded to `'kled above as
> the t'` and `y[0]` decoded to `'led above as the tw'` — the same window,
> slid forward by exactly one token.

Re-running that same idea fresh for this course, against the same
`tests/fixtures/tiny_stories.txt` fixture and the same `get_batch(...,
batch_size=4, context_len=12, seed=0)` call, gives:

```
x.shape torch.Size([4, 12])   y.shape torch.Size([4, 12])
x[0] ids: [70, 85, 76, 268, 288, 14, 0, 47, 78, 67, 69, 290]
y[0] ids: [85, 76, 268, 288, 14, 0, 47, 78, 67, 69, 290, 80]
x[0] decoded: 'ful day.Once u'
y[0] decoded: 'ul day.Once up'
```

Looking at the raw ids makes the shift unambiguous: `y[0]`'s first eleven
ids (`85, 76, 268, 288, 14, 0, 47, 78, 67, 69, 290`) are exactly `x[0]`'s
last eleven ids, and `y[0]` picks up one brand-new id (`80`) at the end that
`x[0]` never had. That's the shift-by-one at the level of actual token ids,
not just visually similar decoded strings. It's also worth noticing id `0`
sitting right in the middle of both lists, at `x[0]`'s 7th position — that's
an `<|endoftext|>` boundary from `tokenize_texts`, inserted between two
different stories in the fixture. `decode()` hides id `0` from the printed
text (it isn't `<|endoftext|>` in the output at all), which is exactly why
`'ful day.'` and `'Once u'` butt up against each other in the decoded string
with nothing visibly between them — invisibly, in the ids, there's a clean
document boundary sitting right there.

### Gotchas & design decisions

**`load_tinystories` is deliberately never used by the unit tests.** Real
training needs the actual TinyStories corpus (roughly 1.9GB, pulled via the
Hugging Face `datasets` library). `load_tinystories` supports two modes:
give it a `limit`, and it streams the dataset (`streaming=True`) and stops
after collecting the first `limit` stories, so a small toy run never has to
download the full corpus just to use a few thousand stories; leave `limit`
as `None`, and it downloads the entire split (the Colab `small` run in
Module 7 uses this path). But even the streaming path still touches a live
network and an external, upstream-hosted dataset — exactly the kind of
dependency a unit test should never have (network flakiness, CI slowness,
the dataset itself changing upstream). So `tests/test_data.py` never calls
`load_tinystories` at all. Instead, every test in that file builds its data
from `tests/fixtures/tiny_stories.txt`, a 30-line fixture checked directly
into the repo. The tests still exercise the exact same `tokenize_texts` and
`get_batch` code paths that real training uses — just against tiny, static,
offline text instead of the real dataset, so the whole test suite stays
fast, deterministic, and network-free.

**Determinism is scoped to the call, not global.** `get_batch`'s
`torch.Generator().manual_seed(seed)` is a private generator, not a call to
`torch.manual_seed(seed)` — this project's `test_batch_is_deterministic_with_seed`
test confirms two separate `get_batch` calls with `seed=0` against the same
data return the identical batch. Because that generator is private to the
one call, `get_batch`'s determinism doesn't depend on — and doesn't
disturb — whatever else in the program might be drawing from PyTorch's
global RNG elsewhere (weight initialization, for instance).

**The `assert max_start > 0` is a deliberate fail-fast, not defensive
noise.** If `context_len` is close to, or larger than, the length of the
actual token stream you hand `get_batch` — which happens immediately if you
try to batch from too little data, like accidentally pointing it at a
handful of tokens while asking for a long context window — the function
fails immediately with a clear message, rather than silently returning
malformed or wrapped-around windows that would quietly corrupt training.

### Checkpoint

1. Why does one input sequence give you a prediction target at *every*
   position, not just the end?
2. What exactly is the relationship between `x` and `y` in a batch?

**Explain it back:** why must the unit tests never download the real
TinyStories dataset?
