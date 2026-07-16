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

Mechanically, this is nothing fancier than plain Python slice indexing
applied to one long array: if `t` is the entire token stream and `s` is some
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
after collecting the first `limit` stories, so a run never has to download
the full corpus just to use a subset; leave `limit` as `None`, and it
downloads the entire split instead. The real Colab `small` run (Module 7
covers it in full) takes the first path, not the second — its data-loading
cell calls `load_tinystories("train", limit=200_000)`, a RAM-safe streaming
subset chosen deliberately: the notebook's own notes explain that 200,000
stories (~47M tokens) is already plenty for coherent little stories from
its 14M-parameter model, while holding the *entire* dataset in a Python
list would overflow Colab's roughly 12GB of RAM. But even the streaming
path still touches a live network and an external, upstream-hosted
dataset — exactly the kind of
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

## Module 4 — Model Internals: RMSNorm, RoPE, Causal Attention, SwiGLU

### Learning objectives

By the end of this Module you'll be able to explain what each of the four core
transformer pieces does and why it exists: read the real `RMSNorm` and say why
it rescales a token vector by its own root-mean-square (and why it does that
arithmetic in float32); read the real `build_rope_cache`/`rotate_half`/`apply_rope`
and explain how *rotating* a query and key bakes in word order while leaving
their lengths untouched — and why that encodes *relative* distance rather than
absolute position; read the real `Attention` and point to the exact lines where
tokens exchange information (Q·K scores) and where the causal mask forbids
looking at the future; and read the real `SwiGLU` and explain why it's called
per-token "private thinking." You'll also be able to hand-compute a tiny 3-token
causal-attention matrix and see why its first row is forced to be `[1, 0, 0]`.

### Frame

By the time a token reaches this Module's code, it's already a vector of length
`d_model` — exactly the `(batch, seq, d_model)` shape the Warm-Up drilled into
you (`TOY` uses `d_model=64`). Everything in a transformer block is arithmetic
that takes those vectors in and hands the same-shaped vectors back out. There
are four moving parts, and each has one job.

**RMSNorm keeps magnitudes healthy.** Stack dozens of layers and the numbers
flowing through them tend to drift — some token vectors grow huge, others shrink
toward nothing — and that drift destabilizes training. RMSNorm steps in *before*
each sub-layer and rescales every token vector by its own root-mean-square, so
whatever came in leaves at a consistent, comparable size. It's the exact RMS
kernel from the Warm-Up, wrapped in a learnable per-dimension `weight`.

**RoPE injects order by rotating q and k.** Attention, on its own, is
order-blind: shuffle the tokens and the raw dot products don't care. RoPE fixes
that by *rotating* each query and key vector by an angle proportional to the
token's position — token 0 rotates by nothing, token 5 rotates further, token 20
further still. Rotation is the perfect tool for this because, as the Warm-Up
showed, it never changes a vector's length — it only spins it. So position gets
baked in without corrupting the magnitude information the vector already
carries. The deeper payoff: when a rotated query later meets a rotated key,
their interaction depends only on the *difference* between their positions, not
the absolute positions themselves — RoPE encodes *relative* distance.

**Attention is the only place tokens exchange information.** Every token emits
three projected vectors: a Query ("what am I looking for?"), a Key ("what do I
offer?"), and a Value ("what do I carry?"). Each token scores its own Query
against every token's Key with a dot product (the Warm-Up's dot product, exactly
— big score means "these two point the same way"), turns those scores into
weights with softmax, and then blends everyone's Values by those weights. That
blend is the one and only moment in the whole architecture where information
moves *between* tokens. A **causal mask** constrains it: a token may attend to
itself and to earlier tokens, never to later ones — because at generation time
the later tokens don't exist yet.

**SwiGLU is per-token "private thinking."** Right after attention has let tokens
talk to each other, each token goes off and processes what it just gathered — on
its own, with no further cross-token interaction. SwiGLU expands the token vector
into a wider workspace (`ffn_hidden`), runs it through a gated nonlinearity, and
projects it back down to `d_model`. No token looks at any other token here; it's
purely private, per-position computation.

One detail worth stating up front, because it's a deliberate through-line of the
whole model: **every linear layer in this Module is bias-free** (`bias=False` on
every `nn.Linear`). That matches the Llama architecture this project is
rebuilding, and it's part of what lets the finished model line up bit-for-bit
with the stock implementation later (Module 8).

### Annotated code walkthrough

The four pieces below are the real, current contents of `src/slm/model.py`,
embedded verbatim and taken one at a time.

**RMSNorm** — rescale each token vector by its root-mean-square, in float32:

```python
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
```

- `self.weight = nn.Parameter(torch.ones(dim))` is a learnable per-dimension
  scale, one number per `d_model` slot, initialized to all-ones. At
  initialization it does nothing (multiplying by 1); training is free to nudge
  individual dimensions up or down later.
- `dtype = x.dtype` then `x = x.float()` — this is the **float32 trick**. The
  input might arrive in a lower precision (say bfloat16 on a GPU), but the
  normalization arithmetic is done in full float32 to avoid precision loss,
  and only the final result is cast back with `x.to(dtype)`. This is exactly
  what the stock Llama implementation does, and matching it is what keeps this
  hand-built model numerically identical to the official one — see Gotchas.
- `var = x.pow(2).mean(-1, keepdim=True)` is the mean of the squares along the
  last axis — the "MS" in RMS. `keepdim=True` preserves the trailing dimension
  so it broadcasts cleanly in the next step.
- `x = x * torch.rsqrt(var + self.eps)` multiplies by the *reciprocal* square
  root — i.e. divides by `sqrt(mean-of-squares)`, the root-mean-square. The
  `+ self.eps` (default `1e-5`) is the same divide-by-zero insurance you saw in
  the Warm-Up: harmless in normal cases, life-saving if a vector were ever all
  zeros.
- `return (self.weight * x.to(dtype))` casts back to the original precision and
  applies the learnable scale.

Notice what's *absent*: there is no subtracting of the mean anywhere. That
single omission is the whole difference between RMSNorm and the older LayerNorm
(Gotchas covers why that's a deliberate, cheaper choice).

**RoPE** — build a rotation cache once, then apply it to q and k:

```python
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
```

- `build_rope_cache` precomputes the rotation angles once, for every position,
  so attention never recomputes them. `inv_freq` is a set of frequencies — one
  per *pair* of dimensions (`torch.arange(0, head_dim, 2)` steps by two) —
  ranging from fast (angle changes a lot per position) to slow. Low dimensions
  spin quickly and encode fine-grained "am I one or two tokens apart"; high
  dimensions spin slowly and encode coarse long-range position. `theta`
  (`rope_theta`, `10000.0` here) sets how fast that frequency falloff is.
- `freqs = torch.outer(pos, inv_freq)` makes a `(seq, head_dim/2)` table where
  entry `(p, i)` is "position `p` times frequency `i`" — the rotation angle for
  that position and dimension-pair. `emb = torch.cat((freqs, freqs), dim=-1)`
  duplicates it to full `head_dim` width, and the function returns the cosine
  and sine of every angle. Precomputing `cos`/`sin` means the actual rotation
  is just multiplies and adds.
- `rotate_half` is the "spin" partner. It splits a vector into its first half
  and second half, then returns `[-second_half, first_half]`. Paired with the
  duplicated `cos`/`sin`, this is what makes `apply_rope` a genuine rotation:
  the standard 2D rule "new_x = x·cos − y·sin, new_y = x·sin + y·cos" falls out
  of `x * cos + rotate_half(x) * sin`.
- `apply_rope` broadcasts `cos`/`sin` across the batch and head axes
  (`unsqueeze(0).unsqueeze(0)`) and applies `x * cos + rotate_half(x) * sin` to
  the whole `(B, n_heads, T, head_dim)` tensor at once. This exact
  split-in-half convention (rather than rotating adjacent interleaved pairs) is
  the one the stock Llama implementation uses — picking it deliberately is part
  of the round-trip match in Module 8.

**Attention** — project to Q/K/V, apply RoPE, score, mask, softmax, blend:

```python
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
```

- The four `nn.Linear(..., bias=False)` projections are the matmuls-against-a-
  weight-matrix from the Warm-Up: `q_proj`, `k_proj`, `v_proj` each turn the
  incoming `d_model` vector into Query, Key, and Value vectors, and `o_proj`
  turns the blended result back into a `d_model` vector at the end. In `TOY`,
  `n_heads * head_dim` is `4 * 16 = 64`, exactly `d_model`.
- `.view(B, T, c.n_heads, c.head_dim).transpose(1, 2)` splits the projected
  vector into `n_heads` independent heads and moves the head axis forward, so
  each head attends in its own `head_dim`-sized subspace in parallel. Multiple
  heads let the model look for several different relationships at once.
- `q = apply_rope(q, cos, sin)` and the matching line for `k` inject position
  by rotating the queries and keys (but *not* the values — position matters for
  deciding who-attends-to-whom, not for what content gets carried).
- `scores = (q @ k.transpose(-2, -1)) * (c.head_dim ** -0.5)` is the heart of
  it: `q @ kᵀ` is the batch of dot products of every Query against every Key
  (the Warm-Up's matmul-as-many-dot-products), producing a `(T, T)` score grid
  per head. Multiplying by `head_dim ** -0.5` divides by `sqrt(head_dim)` — a
  scaling that keeps the scores from growing huge as `head_dim` grows, which
  would otherwise saturate the softmax.
- `mask = torch.full((T, T), float("-inf"), ...).triu(1)` builds a `(T, T)`
  matrix that is `-inf` strictly *above* the diagonal (`triu(1)` = keep the
  upper triangle, offset by 1) and `0` on and below it. Adding it to `scores`
  drives every "attend to a future token" entry to `-inf`.
- `attn = F.softmax(scores, dim=-1)` is the Warm-Up's softmax applied along each
  query's row of scores, turning them into weights that sum to 1 — and any score
  that's `-inf` becomes exactly `0`, which is precisely how the causal mask
  erases the future.
- `out = attn @ v` blends the Value vectors by those weights (matmul again),
  `.transpose(1, 2).reshape(...)` glues the heads back into one `d_model`-wide
  vector per token, and `self.o_proj(out)` is the final projection back into the
  model's working space.

**SwiGLU** — the per-token feed-forward, gate × up, then down:

```python
class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.up_proj = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.down_proj = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
```

- `gate_proj` and `up_proj` both expand the `d_model` vector up to the wider
  `ffn_hidden` workspace (`128` in `TOY`, versus `d_model=64`) — two separate
  projections of the same input.
- `F.silu(self.gate_proj(x))` runs the *gate* branch through SiLU (a smooth
  nonlinearity: `x * sigmoid(x)`), and `* self.up_proj(x)` multiplies it
  element-wise against the *up* branch. That multiplicative gating is the "GLU"
  (gated linear unit) idea — the gate branch learns to selectively let parts of
  the up branch through, which is more expressive than a single activation.
- `self.down_proj(...)` projects the wide result back down to `d_model`. Every
  operation here is per-token; nothing crosses between positions. All three
  layers are, again, bias-free.

### In-context worked example

Let's make attention concrete with the smallest possible case: **3 tokens, each
a `d=2` vector**, one head, no learned weights — just the arithmetic. This is
the same computation `Attention.forward` does, stripped to its core. Run:

```python
import torch
torch.manual_seed(0)
q = torch.randn(3, 2)          # 3 query vectors, 2 numbers each
k = torch.randn(3, 2)          # 3 key vectors
scores = (q @ k.T) / (2 ** 0.5)          # dot products, scaled by sqrt(head_dim=2)
mask = torch.triu(torch.full((3, 3), float("-inf")), 1)  # -inf above the diagonal
w = torch.softmax(scores + mask, -1)     # per-row softmax → attention weights
print(w)
```

The intermediate values, printed as they came out:

```
q = [[1.541, -0.2934], [-2.1788, 0.5684], [-1.0845, -1.3986]]
k = [[0.4033, 0.838], [-0.7193, -0.4033], [-0.5966, 0.182]]

raw scores (q @ kᵀ / sqrt2):
  [ 0.2656, -0.7001, -0.6879]
  [-0.2846,  0.9460,  0.9924]
  [-1.1381,  0.9505,  0.2775]

after adding the causal mask:
  [ 0.2656,  -inf,   -inf ]
  [-0.2846,  0.9460,  -inf ]
  [-1.1381,  0.9505,  0.2775]
```

Now the softmax of each row gives the `(3, 3)` attention-weight matrix, with its
upper triangle zeroed by the mask:

```
attention weights:
  q0:  1.0000   0.0000   0.0000
  q1:  0.2261   0.7739   0.0000
  q2:  0.0758   0.6120   0.3122
```

Read it row by row and every Warm-Up primitive shows up:

- **Row `q0` is `[1, 0, 0]`.** Token 0 is the very first token — there is nobody
  before it, so the mask sends both other entries to `-inf`, and softmax over a
  single surviving score always returns `1.0` (the Warm-Up's rule: softmax of a
  lone value, or equivalently one finite score against two `-inf`s, puts all the
  weight on that one). Token 0 can only attend to itself.
- **Row `q1` is `[0.226, 0.774, 0]`.** Token 1 sees tokens 0 and 1. Its raw
  scores were `-0.285` and `0.946` (dot products of its Query with each Key,
  scaled by `sqrt(2)`), and softmax of `[-0.285, 0.946]` is `[0.226, 0.774]` —
  more weight on the higher-scoring key, exactly as the Warm-Up's softmax
  preserves order. The future token 2 is zeroed out.
- **Row `q2` is `[0.076, 0.612, 0.312]`.** Token 2 sees all three; its three
  scaled dot products softmax to weights that sum to 1, with the largest share
  going to the key it pointed most toward (token 1).

Each entry started as a **dot product** (Warm-Up), the whole grid was one
**matmul** (`q @ kᵀ`, Warm-Up), and each row was turned into weights by
**softmax** (Warm-Up). The causal mask is nothing more than adding `-inf` before
that softmax.

The other two pieces reduce to their Warm-Up primitives just as cleanly. Here is
**RMSNorm** on a tiny vector, using the real class with its default all-ones
`weight`:

```python
from slm.model import RMSNorm
import torch
out = RMSNorm(2)(torch.tensor([3.0, 4.0]))
print([round(v, 4) for v in out.tolist()])   # [0.8485, 1.1314]
```

That's the identical result the Warm-Up's RMS example produced for `[3, 4]`
(`[0.849, 1.131]`) — because RMSNorm *is* that kernel: divide by the
root-mean-square (`sqrt((9+16)/2) ≈ 3.536`), scale by the learnable weight
(here `1`). The root-mean-square of the output is `1.0000`, which is the whole
point — whatever magnitude went in, a unit-RMS vector comes out.

And here is **RoPE** on a 2-element pair. First `rotate_half`, the spin partner:

```python
from slm.model import rotate_half
import torch
print(rotate_half(torch.tensor([0.8, 0.6])).tolist())   # [-0.6, 0.8]
```

It splits `[0.8, 0.6]` into halves and returns `[-0.6, 0.8]` — the second half
negated and swapped in front, the algebraic ingredient of a rotation. Now the
full `apply_rope`, rotating the pair `[1, 0]` by the angle for position 1:

```python
from slm.model import build_rope_cache, apply_rope
import torch
cos, sin = build_rope_cache(head_dim=2, seq_len=4, theta=10000.0)
x = torch.tensor([1.0, 0.0]).view(1, 1, 1, 2)       # (B, heads, T, head_dim)
r = apply_rope(x, cos[1:2], sin[1:2])
print([round(v, 4) for v in r.view(-1).tolist()])   # [0.5403, 0.8415]
```

`[1, 0]` rotated to `[0.5403, 0.8415]` — which is exactly `[cos(1), sin(1)]`, a
rotation by 1 radian, straight out of the Warm-Up's 2D-rotation section. The
norm before is `1.0` and the norm after is `1.0`: the rotation moved the vector
but did not stretch it.

Finally, the *relative*-distance property, made numeric. Rotate a fixed query
`q` and key `k` to various positions and take their dot product; what matters is
only the gap between the positions, never the absolute positions:

```python
from slm.model import build_rope_cache, apply_rope
import torch
cos, sin = build_rope_cache(head_dim=2, seq_len=8, theta=10000.0)
q = torch.tensor([0.7, 0.9]).view(1, 1, 1, 2)
k = torch.tensor([0.4, -0.3]).view(1, 1, 1, 2)
def rot(x, p): return apply_rope(x, cos[p:p+1], sin[p:p+1]).view(-1)
for m, n in [(1, 0), (4, 3), (6, 5)]:
    print(f"gap {m-n}:", round(torch.dot(rot(q, m), rot(k, n)).item(), 4))
```

```
gap 1: -0.4742
gap 1: -0.4742
gap 1: -0.4742
```

Positions `(1,0)`, `(4,3)`, `(6,5)` are three different places in the sequence,
but all are one token apart — and the rotated dot product is `-0.4742` in every
case. (Change the gap to 3 and every pair collapses to a *different* shared
value, `-0.0903`.) That identical-across-absolute-positions number is what
"RoPE encodes relative distance" means, as a real result.

### What we observed

Running the attention block on a real 5-token sequence (`TOY` config, one head)
and printing head 0's weight matrix shows the causal structure at slightly
larger scale than the hand example:

`labs/lab02_attention_peek.py`:

```
Attention weights (row = query position, col = key position), head 0:
  q0: 1.00 0.00 0.00 0.00 0.00
  q1: 0.41 0.59 0.00 0.00 0.00
  q2: 0.50 0.27 0.23 0.00 0.00
  q3: 0.20 0.25 0.33 0.22 0.00
  q4: 0.16 0.20 0.21 0.18 0.24
```

The entire upper triangle is `0.00` — every "attend to a later token" entry —
and each row sums to 1. Row `q0` is `[1.00, 0, 0, 0, 0]` for the same reason it
was in the hand example: the first token has only itself to attend to. This is
the causal mask working exactly as designed on live code.

Two unit tests (in `tests/test_model_components.py`) pin down the two most
important guarantees, and both pass:

- `test_rope_preserves_shape_and_norm` builds a random `(1, n_heads, 7, head_dim)`
  tensor, applies RoPE, and asserts the per-position vector norms are unchanged
  (`out.norm(dim=-1)` matches `x.norm(dim=-1)` to within `1e-4`). This is the
  length-preserving property of rotation, verified on the real `apply_rope` —
  RoPE spins the vector, it never resizes it.
- `test_attention_is_causal` runs attention on a 6-token sequence, then adds
  `10.0` to *only the last token* and runs it again. It asserts the outputs for
  positions 0 through 4 are identical (to within `1e-5`). Because the causal
  mask forbids earlier tokens from ever attending to the last one, changing the
  future genuinely cannot reach into the past — the earlier outputs come out
  bit-for-bit unchanged.

### Gotchas & design decisions

**Why RMSNorm instead of LayerNorm.** The classic LayerNorm does two things:
subtract the mean of each vector (re-centering it on zero), then divide by the
standard deviation. RMSNorm keeps only the second idea — divide by the
root-mean-square — and drops the mean-subtraction entirely. You can see that
directly in the code: there is no `x - x.mean(...)` anywhere. In practice the
re-centering turns out not to matter much for transformer quality, and skipping
it is cheaper (one fewer pass over the vector, no mean to compute and subtract).
That's why Llama and most modern LLMs use RMSNorm — and this project follows
suit to stay architecturally faithful.

**No GQA in Phase 1 (`n_kv_heads == n_heads`).** Some larger models save memory
with *grouped-query attention*, where several Query heads share a single Key/Value
head (`n_kv_heads < n_heads`). Notice `k_proj` and `v_proj` are already sized off
`n_kv_heads` while `q_proj` uses `n_heads` — but enabling real grouping
(`n_kv_heads < n_heads`) would additionally need a key/value-repeat step in the
forward pass (each shared K/V head expanded to line up with its group of Query
heads), which this code omits because the counts are equal in Phase 1. `TOY`
uses `n_heads=4, n_kv_heads=4`, so every Query head gets its own Key/Value head
and the `q @ k.transpose(-2, -1)` line broadcasts cleanly. Set `n_kv_heads=2`
without adding that repeat step and the forward pass raises a shape-mismatch
`RuntimeError` — so this isn't a dormant feature waiting to be flipped on, it's a
simplification Phase 1 makes on purpose.

**The float32 RMSNorm trick is required for the HF numerical match.** The
`x = x.float()` / `x.to(dtype)` dance in RMSNorm isn't cosmetic. The stock Llama
RMSNorm computes its normalization in float32 regardless of the input precision,
and this project's whole export story rests on this hand-built model producing
*identical* outputs to the official `LlamaForCausalLM` (a round-trip check of
max-abs-difference around `1e-5`, pure float noise — the correctness linchpin the
export in Module 8 depends on). If RMSNorm normalized in a lower precision
instead, the two implementations would diverge by more than float noise and that
match would break. Doing the RMS arithmetic in float32 and casting back is what
keeps them bit-for-bit aligned.

### Checkpoint

1. Using the Warm-Up's softmax, explain why the first row of the causal
   attention matrix is `[1, 0, 0]`. (Hint: what does the mask do to token 0's
   scores for tokens 1 and 2, and what does softmax return when only one score
   survives?)
2. Why does rotating a query/key vector (RoPE) encode *relative* position rather
   than absolute? Point to the worked example where three different
   position-pairs with the same gap produced the same dot product.
3. Which sub-layer lets tokens exchange information, and which processes each
   token privately? Name the exact operation in `Attention.forward` where the
   exchange happens.

**Explain it back:** trace one token vector through a block, in your own words —
`norm → attention → norm → SwiGLU`. What does each step do to the vector, which
step is the only one that mixes in information from *other* tokens, and why is
the vector normalized *before* each sub-layer rather than after? (You'll see
these four pieces wired together into a full residual block in Module 5.)

## Module 5 — Assembling the Model, Generation, and the Untrained "Echo Bias"

### Learning objectives

By the end of this module you'll be able to: assemble the four pieces from
Module 4 into a repeatable `Block` and stack those blocks into a whole model;
explain what "pre-norm residual" means and why the residual connection lets the
original token embedding survive all the way to the top of the stack; read the
`generate` loop and describe one full step of autoregressive generation
(predict → sample → append → repeat); and — the real payoff of this module —
explain the *echo bias*: why a brand-new, untrained model with tied embeddings
doesn't babble randomly but instead confidently repeats the token it just saw.
That last point is a genuine debugging finding from building this project, and
working through its mechanism will sharpen your intuition for dot products,
softmax, and residual streams more than any tidy example could.

### Frame

Module 4 built four self-contained pieces: RMSNorm (rescale a vector),
attention (the one place tokens exchange information), SwiGLU (per-token private
thinking), and RoPE (bake position into attention). This module wires them into
a working language model, and the wiring is almost embarrassingly simple.

The repeating unit is a **pre-norm residual block**. "Residual" means each
sub-layer *adds a correction* to its input rather than replacing it:

```
x = x + attn(norm(x))     # attention sub-layer
x = x + mlp(norm(x))      # feed-forward sub-layer
```

Two ideas are stacked in each line. **Pre-norm**: the normalization happens
*before* the sub-layer (inside the parentheses), which is more stable to train
at depth than the original "normalize after" Transformer design. **Residual**:
the `x + ...` means the sub-layer only has to learn a small nudge to add on top
of `x`, not reconstruct the whole representation from scratch. A quiet
side effect of that `x + ...` — one that becomes the star of this module — is
that the *input* to the block always survives in the output as a strong,
undiluted component. Stack several blocks and the very first thing that went in
(the raw token embedding) is still sitting there near the top.

The full model is then just: look up each token's embedding, run the sequence
through a stack of these blocks, apply one **final norm**, and project to
vocabulary-sized **logits** with a `lm_head`. The twist is that `lm_head` isn't
a fresh matrix — it *reuses the embedding table* (weight tying). "The vector
that represents token *t* going in" and "the row that scores token *t* coming
out" are literally the same numbers.

Finally, generating text is **autoregressive**: run the model to get a
distribution over the next token, sample one token from it, append it to the
sequence, and repeat with the now-longer sequence. Predict → sample → append →
repeat, one token at a time.

### Annotated code walkthrough

Here is the repeating block, straight from `src/slm/model.py`. It's the four
Module-4 pieces held together by two residual adds:

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
```

Two norms (one per sub-layer), one attention, one SwiGLU. In `forward`, notice
that `norm1(x)` and `norm2(x)` feed the *sub-layers*, but the thing being added
back is the un-normalized `x`. The normalization is a temporary "clean copy"
used only to compute the correction; the running representation `x` itself is
never overwritten, only added to. That is the residual stream, and it is why the
original embedding persists.

Now the whole model. The constructor lays out the parts and, crucially, ties the
output projection to the embedding table:

```python
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
```

Read the pieces in order:

- `embed_tokens` is the `(vocab_size, d_model)` lookup table from Module 4's
  embedding discussion — one row per possible token id.
- `layers` is a stack of `n_layers` identical `Block`s (`TOY` uses 2, `SMALL`
  uses 6).
- `self.norm` is the single final norm applied after the last block, before
  scoring.
- `lm_head` is a `bias=False` linear that turns each `d_model`-length vector
  into `vocab_size` logits.
- **The tying line**: `self.lm_head.weight = self.embed_tokens.weight`. This is
  not a copy — after it runs, the two names point at the *same* tensor in
  memory. Change one, you've changed the other. Remember this line; the entire
  deep dive below hangs on it.
- The last three lines precompute the RoPE cosine/sine tables once and stash
  them as **non-persistent buffers** (`persistent=False`). A buffer is model
  state that isn't a learnable parameter; "non-persistent" means it's kept out
  of the saved checkpoint. That's deliberate — the RoPE tables are fully
  determined by the config (`head_dim`, `context_len`, `rope_theta`) and can be
  rebuilt instantly, so there's no reason to bloat every checkpoint with numbers
  we can always recompute.

The forward pass is the Frame's four steps, literally:

```python
    def forward(self, idx):
        B, T = idx.shape
        x = self.embed_tokens(idx)
        cos, sin = self.rope_cos[:T], self.rope_sin[:T]
        for layer in self.layers:
            x = layer(x, cos, sin)
        x = self.norm(x)
        return self.lm_head(x)
```

Embed the token ids, slice the RoPE tables to the current sequence length `T`,
run every block in turn (each one reading and updating the same `x`), apply the
final norm, and project to logits. The output has shape
`(batch, T, vocab_size)`: at *every* position it emits a full score vector over
the vocabulary — a prediction of what comes next after that position.

Generation wraps `forward` in a loop:

```python
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

Walk one iteration — this *is* one full step of autoregressive generation:

1. **Crop the context.** `idx[:, -self.cfg.context_len:]` keeps only the last
   `context_len` tokens. The model was built to look at a fixed window; if the
   sequence has grown past it, we drop the oldest tokens so the forward pass and
   the RoPE tables stay in range.
2. **Predict.** `self(idx_cond)[:, -1, :]` runs the forward pass and keeps only
   the *last* position's logits — the prediction for the very next token. (The
   scores at earlier positions are real, but during generation we only care
   about what comes after the end.)
3. **Reshape the distribution (optional).** Dividing by `temperature` sharpens
   (low temperature) or flattens (high temperature) the scores, exactly the
   softmax-temperature trick from the Warm-Up; the `max(temperature, 1e-6)`
   guards against dividing by zero. Top-k keeps only the `k` highest logits and
   sets the rest to `-inf`, so unlikely tokens get zero probability and can
   never be drawn. With the defaults (`temperature=1.0`, `top_k=None`) both
   `if` blocks are skipped and we sample straight from the raw scores.
4. **Softmax → sample.** `softmax` turns logits into a probability distribution;
   `torch.multinomial` draws one token from it (so generation is random, not
   just "always the top token").
5. **Append.** `torch.cat([idx, nxt], dim=1)` glues the new token onto the
   sequence, and the loop repeats with the longer input.

The `@torch.no_grad()` decorator says "we're only running the model forward, not
training," which skips gradient bookkeeping and saves memory.

### What we observed

The striking thing about all of the above is that it runs *perfectly* on a model
that has learned nothing at all. `nn.Embedding` and `nn.Linear` are born filled
with small random numbers, so a freshly constructed `LlamaSLM(TOY)` already
produces finite logits, samples tokens, and extends a sequence end-to-end. The
architecture — the wiring — works the instant you build it. What's missing is
only the *weights*: the specific learned numbers that make the output mean
something.

Seeding an untrained model with a real prompt makes the split vivid.
`labs/lab03_gibberish.py` builds a tokenizer, constructs an untrained
`LlamaSLM(TOY)` under a fixed seed, and asks it to continue `"Once upon a
time"`:

```
=== seeded with a real prompt: 'Once upon a time' ===
Once upon a time time time time time time time time time time time time time time time time time time time time time time time time time time time time time time time
```

It generated thirty tokens without error — the pipeline is alive. But the output
isn't the random word-salad you might expect from random weights. It latches
onto one word and repeats it forever. That is not a coincidence and not a bug;
it's the echo bias, and it's the subject of the deep dive.

The lab also pokes the worst case — seeding from `<|endoftext|>` (token id 0):

```
=== seeded with <|endoftext|> (id 0) — the worst-case trigger ===
raw token ids: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
decoded: '' (empty — decode() hides <|endoftext|>, and the model just echoes it forever)
```

The model echoes token 0 twenty times over. Because the tokenizer hides the
special token when decoding, the *text* comes back empty — so the lab prints the
raw ids too, or the behavior would look like a mysterious blank instead of a
perfectly clear "it repeated the last token." Same phenomenon, most extreme
form.

### Deep dive — the echo bias

Here's the puzzle. Intuition says an untrained model, having learned nothing,
should be maximally *unsure* — its next-token distribution should be roughly
uniform, spreading probability thinly over the whole vocabulary. That intuition
is wrong for this architecture, and understanding exactly why is worth more than
the fix itself. An untrained tied-embedding model is not unsure at all. It is
*confidently wrong*: it puts almost all of its probability on one specific token
— the one it just saw. Prompt it with `"Once upon a time"` and it wants `time`,
then `time`, then `time`.

The mechanism is a two-step chain, and both steps lean on things you already
know from the Warm-Up.

**Step 1 — the residual stream keeps the last token's embedding on top.** Recall
the pre-norm block only ever *adds* to `x`. At initialization every sub-layer's
weights are small random numbers, so `attn(norm(x))` and `mlp(norm(x))` produce
only small corrections. The big, dominant thing in `x` after the whole stack is
therefore still the thing that went *in* at the bottom: the token's own
embedding vector. So at the final position, the hidden state the model is about
to score is, to a good approximation, just the embedding of the most recent
token (rescaled by the final norm, which changes its length but not its
direction).

**Step 2 — weight tying turns that into a self-dot-product.** The logit for
token *t* is the final hidden state dotted with row *t* of `lm_head`. But
`lm_head` *is* the embedding table (that tying line). So:

- The logit for the **last token** ≈ `embed(last) · embed(last)` — a vector
  dotted with *itself*. From the Warm-Up, a dot product measures how much two
  vectors point the same way; a vector points *perfectly* the same way as
  itself, so this self-dot-product is its squared length — a big positive
  number.
- The logit for **any other token** ≈ `embed(last) · embed(other)` — two
  *different*, essentially-random embedding rows. Random vectors in high
  dimensions point in unrelated directions, so their dot product hovers near
  zero.

One logit towering over a field of near-zeros. And the Warm-Up's softmax section
showed what softmax does with a gap: bigger score in, disproportionately bigger
probability out; a *large* gap saturates it. So the distribution collapses onto
that one token, `max_prob ≈ 1.0`, and generation — sampling, then appending —
locks into repeating it. Even with the defaults (no temperature sharpening, no
top-k), the raw distribution is *already* a spike, so sampling can't rescue it.

Running the untrained `TOY` model (fixed seed, a random 6-token input) gives the
real numbers behind that story:

- The argmax of the final logits is exactly the last input token — the echo,
  confirmed.
- The winning logit is about **64.8**; the next-highest is about **26.7** — a
  gap of roughly **38** logits. Softmax turns that gap into `max_prob` of
  `1.000000` (yes, `1.0` to six decimal places).
- That winning logit sits in the same ballpark as `|embed(last)|²`, which
  measures about **69.1** for this token — i.e. the top score really is
  approximately the last token's embedding dotted with itself, as the mechanism
  predicts. (It lands a little under the raw squared length because the final
  norm rescales the vector and the sub-layers do add their small corrections.)

The clincher is the counter-test. Rebuild the *same* model with one change —
`tie_embeddings=False`, so `lm_head` gets its own independent random matrix
instead of reusing the embedding table — and the bias evaporates. Now the last
token's embedding is dotted against unrelated `lm_head` rows, there's no
self-dot-product to win, and `max_prob` drops to about **0.0029**, right down
near the `1/2048 ≈ 0.00049` you'd get from a genuinely uniform guess. The echo
was never about the residuals *alone* or the tying *alone* — it needs both: the
residuals to carry the embedding to the top, and the tying to make the scorer
recognize it. The project's developer log records the same result holding across
both `TOY` and `SMALL`, across multiple seeds, and across varied (non-repeated)
inputs — a ~30–40 logit gap every time — so it's a property of the architecture,
not a fluke of one random init.

**Why the original test was wrong.** An early version of the model's test suite
included a check called `test_untrained_output_is_high_entropy` — it asserted
exactly the wrong-headed intuition above, that an untrained model's output would
be spread out (high entropy). When the model was assembled, that assertion
failed hard: `max_prob` came back a fully saturated `1.0`, the opposite of
high-entropy. The right response was *not* to force the model to be uniform —
the model was behaving correctly and faithfully to real Llama. The response was
to fix the *test's assumption*. The assertion was replaced with one that checks
what is actually, verifiably true:

```python
def test_untrained_model_echoes_last_token():
    # An untrained model does NOT produce a near-uniform distribution. Tied
    # embeddings + the pre-norm residual stream carry the last token's
    # embedding straight to the (tied) lm_head, so its self-dot-product
    # |embed(last)|^2 dominates every other token's logit — the model's
    # top prediction is the token it just saw. This is expected,
    # architecture-driven behavior, verified quantitatively in DEVLOG.md
    # (top logit ~= |embed(last)|^2). Training (Task 5) overwrites it.
    torch.manual_seed(0)
    m = LlamaSLM(TOY)
    idx = torch.randint(0, TOY.vocab_size, (1, 6))
    last = idx[0, -1].item()
    logits = m(idx)[0, -1]
    assert logits.argmax().item() == last
```

(The comment's "Task 5" is this project's internal name for the training work
Module 6 covers, and the "verified quantitatively in DEVLOG.md" claim is the
same real-number confirmation — winning logit, runner-up logit, `|embed(last)|²`
— walked through just above in this module's own deep dive.)

A companion test (`test_untrained_forward_is_finite_and_input_dependent`) still
guards the boring-but-vital failure modes — that the forward pass produces no
NaNs and genuinely depends on its input, so a dead or constant model can't sneak
through. The lesson is a good one to carry forward: when a test fails, the
assumption baked into the test is a suspect too, not just the code.

### Gotchas & design decisions

**Weight tying is a genuine trade-off.** On the plus side, the embedding table
is the single largest block of numbers in a small model like this — for `TOY`
it's the majority of all parameters — and tying means we store and train it
*once* instead of twice, roughly halving that block. (You can see the model's
own parameter count treats it as counted once.) It also encodes a reasonable
prior: the notion of "what token *t* means going in" and "how strongly to bet on
token *t* coming out" ought to be related. The minus side is this whole module's
deep dive: tying is precisely what makes the untrained echo bias so severe. But
that cost is temporary. It's an *initialization* artifact — training (Module 6)
reshapes the shared matrix so the self-dot-product no longer automatically wins,
and the model learns to bet on the token that should come *next* instead of the
one it just saw. In fact the bias leaves a measurable fingerprint on early
training: the starting loss comes in far *worse* than a uniform guesser's,
precisely because the model is confidently wrong, and the first chunk of
training is largely the optimizer un-learning the echo.

**RoPE tables as non-persistent buffers.** Registering `rope_cos`/`rope_sin`
with `persistent=False` keeps them out of the saved checkpoint. They're derived
data — fully recomputable from the config — so persisting them would only bloat
every checkpoint and risk a stale copy. This matters later: the export and
GGUF-packaging steps care about exactly which tensors count as real model state,
and position tables aren't it.

**Context cropping is a hard requirement, not politeness.** The
`idx[:, -context_len:]` slice in `generate` isn't just tidy — the RoPE tables
were only built out to `context_len` positions, and `forward` slices them to the
input length. Feed a longer sequence and you'd index past the end of those
tables. Cropping keeps generation inside the window the model was actually built
for.

### Checkpoint

1. An untrained model produces gibberish but still runs end-to-end. What does
   that tell you about the split between a model's *architecture* and its
   *weights* — which part does building the model give you for free, and which
   part does training have to supply?
2. Explain the mechanism of the echo bias in your own words. Why does it need
   *both* the pre-norm residual connections *and* weight tying — what does each
   one contribute, and what happens to the bias if you remove the tying
   (`tie_embeddings=False`)?
3. Why was the original `test_untrained_output_is_high_entropy` assertion
   *wrong*? What did it assume about an untrained model, and what does the model
   actually do instead (think about `max_prob` and the ~38-logit gap)?
4. Using the Warm-Up's dot product and softmax: why is the logit for the last
   token so much larger than the logit for any other token, and why does a gap
   of ~38 in the logits turn into a `max_prob` of essentially `1.0`?

**Explain it back:** describe one full step of autoregressive generation, from a
current sequence of token ids to a sequence that's one token longer. Name each
operation in order (crop → forward → take last position → optional
temperature/top-k → softmax → sample → append) and say in one line what each one
does.

## Module 6 — Training

### Learning objectives

By the end of this module you'll be able to: explain cross-entropy loss in
plain language and say why it penalizes a confident wrong answer far more
than an honestly uncertain one; read the real `lr_at` function and describe
the warmup-then-cosine-decay shape it produces, and explain why warmup
exists at all; explain what AdamW actually does with a gradient (adaptive
per-parameter step sizing, plus weight decay) and what gradient clipping
protects against; read the real `train` loop line by line and narrate one
full training step; read `save_checkpoint`/`load_checkpoint` and explain why
the config is stored as a plain dict and why loading needs
`map_location="cpu"`; and explain overfitting a single batch as the fastest
possible proof that learning works at all, before ever pointing the trainer
at a real dataset.

### Frame

Module 5 ended on a vivid demonstration of a model that runs but knows
nothing: an untrained `LlamaSLM`, seeded with `"Once upon a time"`, just
repeats `time` forever — confidently wrong, not merely random. Training is
the step that turns that blank, confidently-wrong architecture into
something that actually predicts real text. Four ideas do essentially all
of the work.

**Loss is "how wrong," and cross-entropy is the specific way this project
measures it.** At every position the model already emits a full probability
distribution over the vocabulary — that's the Warm-Up's softmax, one more
time. Cross-entropy loss looks up the one probability the model assigned to
whatever token *actually* came next, and turns that single number into a
penalty: `loss = -log(that probability)`. A probability near `1` (confident
and right) makes `-log(...)` land near `0` — almost no penalty. A
probability near `0` (confident and *wrong* — betting almost everything on
the wrong token) sends `-log(...)` shooting up toward a large number,
because the logarithm of a tiny number is a large negative number, and the
minus sign flips that into a large positive penalty. A middling probability
(honestly uncertain, spreading its bets) lands somewhere in between. That
asymmetry — confidently wrong costs far more than merely uncertain — is the
whole design point of cross-entropy, and the worked example below makes it
concrete with real numbers, reusing the Warm-Up's own softmax result.

**AdamW is the algorithm that turns a gradient into an actual weight
update.** Every parameter gets nudged by an amount that depends on its own
gradient, but not directly: AdamW keeps running averages of each
parameter's recent gradient and squared gradient, and uses those averages
to size that parameter's own step (a parameter with a history of small,
consistent gradients gets a confident, appropriately-sized step; a noisy
one gets dampened). The "W" adds weight decay — a small constant pull of
every weight toward zero at each step, independent of the gradient, which
discourages any one weight from growing needlessly large.

**The learning-rate schedule ramps up, peaks, then decays.** The learning
rate isn't one constant for the whole run. Early on, the freshly-random
weights (compounded by the echo bias from Module 5) produce large,
unreliable gradients, so taking a full-strength step immediately risks
blowing the weights somewhere useless. `lr_at` (below) ramps the learning
rate linearly from `0` up to a peak over `warmup_steps` steps, touches the
peak for exactly one step, then eases it back down along a cosine curve for
the rest of training — bigger, bolder steps while there's a lot of ground
to cover, smaller, more careful ones as the model settles.

**Gradient clipping is a safety cap, not a normal-operation mechanism.**
Every so often a single batch produces an unusually large gradient (an
outlier example, a numerical spike); clipping rescales the *whole* gradient
so its overall size never exceeds `grad_clip`, so one freak batch can't
yank every weight off a cliff in a single step.

**Overfitting one batch is the "does this even work" proof.** Before ever
pointing the trainer at real data, the fastest, cheapest test of whether
the whole pipeline is wired correctly is to feed it the exact same tiny
batch, over and over, and watch whether the loss collapses toward zero. If
a model can't even memorize one repeated batch, something upstream — data
flow, gradient flow, the loss computation itself — is broken, and no amount
of real training data will fix it. Only once this passes is it worth
spending real time (and, on Colab, real GPU minutes) on an actual dataset.

### Annotated code walkthrough

**The learning-rate schedule.** Here's the real, current `lr_at` from
`src/slm/train.py`:

```python
def lr_at(step: int, cfg: TrainConfig) -> float:
    """Linear warmup to the peak lr at step == warmup_steps, then cosine decay."""
    if step < cfg.warmup_steps:
        return cfg.lr * step / cfg.warmup_steps
    if step == cfg.warmup_steps:
        return cfg.lr
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    return 0.5 * cfg.lr * (1 + math.cos(math.pi * min(1.0, progress)))
```

- While `step < warmup_steps`: a straight linear ramp,
  `cfg.lr * step / warmup_steps` — `0.0` at step `0`, climbing in equal
  increments toward the peak as `step` grows.
- At `step == warmup_steps` exactly: return `cfg.lr` outright — the literal
  peak learning rate, the one step where warmup ends and decay begins.
- Past warmup: `progress` is how far through the *remaining* steps training
  is, clamped to at most `1.0` (`min(1.0, progress)`) so the curve can't
  overshoot into negative-cosine territory if a run ever ticks a few steps
  past `max_steps`. `0.5 * cfg.lr * (1 + cos(pi * progress))` is the
  standard cosine-decay curve: at `progress = 0` this evaluates to
  `0.5 * lr * (1 + 1) = lr` (matching the peak exactly), and at
  `progress = 1` it's `0.5 * lr * (1 + cos(pi)) = 0.5 * lr * (1 - 1) = 0` —
  the learning rate reaches exactly zero on the very last step.

**The training loop.** Here's the real, current `train`:

```python
def train(model: LlamaSLM, data: list[int], cfg: TrainConfig, tok=None):
    torch.manual_seed(cfg.seed)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay, betas=(0.9, 0.95))
    history: list[tuple[int, float]] = []
    rows: list[tuple[int, float]] = []
    model.train()
    # Batches are built on CPU (get_batch uses a CPU RNG for determinism);
    # move each to the model's device so the same code runs on CPU or GPU.
    device = next(model.parameters()).device
    # Tensorize the token stream ONCE (not per-step): at Colab scale the stream
    # is tens of millions of tokens and re-copying it every call would dominate.
    data_t = data if isinstance(data, torch.Tensor) else torch.tensor(data, dtype=torch.long)
    for step in range(cfg.max_steps + 1):
        seed = cfg.seed if cfg.fixed_batch else cfg.seed + step
        x, y = get_batch(data_t, cfg.batch_size, cfg.context_len, seed=seed)
        x, y = x.to(device), y.to(device)
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
```

Walking through one iteration of the loop — this *is* one full training step:

- `opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
  weight_decay=cfg.weight_decay, betas=(0.9, 0.95))` builds the optimizer
  once, before the loop, handing it every parameter in the model. `betas`
  are AdamW's two running-average decay rates (how much weight recent
  gradients get versus older ones) — `(0.9, 0.95)` is a standard choice for
  transformer training.
- `device = next(model.parameters()).device` reads whichever device the
  model's own parameters currently live on — `cpu`, or `cuda:0` if the
  model was moved there before `train()` was ever called. The comment right
  above it says why: `get_batch` builds its random starting positions with
  a private *CPU* generator (Module 3), so batches always arrive on CPU
  first and have to be moved onto wherever the model actually lives.
- `data_t = data if isinstance(...) else torch.tensor(...)` — the same
  tensorize-once pattern from `get_batch` (Module 3), applied at the
  `train()` level: the whole token stream is converted to a tensor exactly
  once before the loop starts, not on every step.
- Inside the loop, `seed = cfg.seed if cfg.fixed_batch else cfg.seed +
  step` picks the seed passed to `get_batch`. With `fixed_batch=False`
  (the normal case), every step gets a different seed and therefore a
  fresh random batch. With `fixed_batch=True`, every step reuses the exact
  same seed — and therefore the exact same batch — which is precisely the
  overfit-one-batch technique from the Frame above, wired in as a config
  flag rather than a separate code path.
- `x, y = get_batch(...)` then `x, y = x.to(device), y.to(device)` is the
  device-portability move itself: build the batch (CPU), then move both
  tensors onto the model's device. Module 7 walks through why this
  particular line matters so much once the model actually lives on a GPU.
- `logits = model(x)` runs the forward pass from Module 5 — shape
  `(batch, context_len, vocab_size)`, one full score vector per position.
- `loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
  y.reshape(-1))` is cross-entropy applied over *every position in the
  batch at once*: `logits.reshape(-1, vocab_size)` flattens the batch and
  sequence axes together into one long list of per-position score vectors,
  and `y.reshape(-1)` flattens the matching targets the same way, so every
  position's prediction is scored against its own true next token in a
  single call.
- `for g in opt.param_groups: g["lr"] = lr_at(step, cfg)` is how the
  schedule actually gets applied: AdamW doesn't know about warmup or
  cosine decay at all — the loop simply overwrites the optimizer's
  learning rate, fresh, before every single step, using whatever `lr_at`
  computes for the current `step`.
- `opt.zero_grad()`, `loss.backward()`, `clip_grad_norm_(...)`, `opt.step()`
  is the standard four-beat update: clear out any leftover gradients from
  the previous step, compute fresh gradients for every parameter via
  backpropagation, clip their combined size down to `cfg.grad_clip` if
  they're too large, then let AdamW actually apply the update.
- The periodic `if step % cfg.log_every == 0` block records `(step, loss)`
  pairs for the printed log and the CSV write at the end; the
  `if tok is not None and ...` block beneath it calls `_print_sample`,
  which generates a short sample from the model-in-progress every
  `sample_every` steps — that's exactly what produced the "coherence
  ladder" of samples in What We Observed below.
- After the loop finishes, `save_checkpoint(...)` writes the final weights
  and config to disk, and `_write_csv(...)` dumps the full loss history —
  the same CSV `plot_loss` (also in `train.py`) later turns into a PNG.

**Saving and loading a checkpoint.** Here's the real, current
`save_checkpoint`/`load_checkpoint`:

```python
def save_checkpoint(model, model_cfg: ModelConfig, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # Store config as a plain dict (not a pickled dataclass) so the checkpoint
    # can be loaded with weights_only=True — never unpickle arbitrary objects
    # from a downloaded model file (that is an arbitrary-code-execution vector).
    torch.save({"state_dict": model.state_dict(), "config": asdict(model_cfg)}, path)


def load_checkpoint(path: str):
    # map_location="cpu": a checkpoint trained on GPU (e.g. Colab) records CUDA
    # tensor locations; without remapping it fails to load on a CPU-only box.
    # Inference/packaging here is always CPU; callers can .to(device) after.
    ckpt = torch.load(path, weights_only=True, map_location="cpu")  # tensors + plain containers only
    cfg = ModelConfig(**ckpt["config"])
    model = LlamaSLM(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, cfg
```

- A checkpoint is two things bundled together: `model.state_dict()` (every
  learned weight tensor, by name) and `asdict(model_cfg)` — the
  `ModelConfig` dataclass turned into a *plain dictionary* of its fields
  (`vocab_size`, `d_model`, `n_layers`, and so on), rather than the
  dataclass object itself. That distinction is what makes `weights_only=True`
  possible on load (see Gotchas).
- `load_checkpoint` reverses the process: `torch.load(..., weights_only=True,
  map_location="cpu")` reads the file back as tensors and plain containers
  only, then `cfg = ModelConfig(**ckpt["config"])` rebuilds a real
  `ModelConfig` from that plain dict, `model = LlamaSLM(cfg)` constructs a
  fresh (untrained-shaped) model from it, and `model.load_state_dict(...)`
  copies the saved weights into that fresh model. `model.eval()` puts it in
  inference mode before handing it back.

**Temperature and top-k, reshaping a distribution.** `train()`'s
`_print_sample` helper calls `model.generate(..., temperature=0.8,
top_k=40)` every `sample_every` steps — the same two knobs Module 5's
`generate` loop exposed. `labs/lab05_sampling.py` isolates exactly what
those knobs do to a fixed set of scores, with no model attached at all:

```python
import torch
torch.manual_seed(0)
logits = torch.tensor([3.0, 2.0, 1.0, 0.5, 0.0, -1.0])
for temp in (0.2, 0.8, 1.5):
    p = torch.softmax(logits / temp, dim=-1)
    print(f"temp={temp}: " + " ".join(f"{x:.2f}" for x in p))
```

Real output:

```
temp=0.2: 0.99 0.01 0.00 0.00 0.00 0.00
temp=0.8: 0.69 0.20 0.06 0.03 0.02 0.00
temp=1.5: 0.46 0.24 0.12 0.09 0.06 0.03
```

The same six raw scores, softmaxed three different ways: dividing by a
*low* temperature (`0.2`) before the softmax sharpens the distribution to
`0.99` on the top score alone — nearly greedy, nearly deterministic;
dividing by a *high* temperature (`1.5`) flattens it to
`0.46/0.24/0.12/...` — much closer to uniform, more room for a
lower-scoring token to get sampled. This is the exact temperature mechanic
from the Warm-Up's softmax section, now shown across the specific values
`_print_sample` actually uses (`0.8`, moderately sharp) rather than the
defaults. Top-k works differently in kind: instead of reshaping every
probability, it zeroes out every score outside the `k` highest before the
softmax ever runs (Module 5's `generate` walkthrough has the exact line),
so a token ranked below the top `k` has *exactly* zero chance of being
sampled, however high temperature is set.

### In-context worked example

Take the Warm-Up's own softmax result and put a loss on top of it. Recall:
`softmax([2.0, 1.0, 0.0])` came out to `[0.665, 0.245, 0.090]` back in the
Warm-Up. Now suppose those are a model's predicted probabilities for three
possible next tokens — token 0, 1, 2 — and ask: what's the cross-entropy
loss if the *actual* next token turns out to be token 0 (the one the model
favored most)? What if it turns out to be token 2 (the one the model
favored least)?

```python
import torch, torch.nn.functional as F
logits = torch.tensor([2.0,1.0,0.0])
print('p:', torch.softmax(logits,-1).tolist())
print('loss true=0:', F.cross_entropy(logits.unsqueeze(0), torch.tensor([0])).item())
print('loss true=2 (confident-wrong):', F.cross_entropy(logits.unsqueeze(0), torch.tensor([2])).item())
```

Real output:

```
p: [0.6652409434318542, 0.2447284758090973, 0.09003057330846786]
loss true=0: 0.40760594606399536
loss true=2 (confident-wrong): 2.4076058864593506
```

`F.cross_entropy` takes raw logits directly (it applies the softmax
internally, then takes `-log` of the true class's probability), so these
two numbers are exactly `-log(0.665)` and `-log(0.090)`. When the true next
token really is the one the model put `66.5%` of its probability on, the
loss is a modest `0.41` — some penalty, because the model wasn't
*certain*, but not much. When the true next token is instead the one the
model gave only `9.0%` to — the model was confidently betting on something
else — the loss jumps to `2.41`, about six times larger for a probability
that's only about `7.4×` smaller. That's cross-entropy's whole design
showing up in two real numbers: confidently right costs almost nothing,
honestly uncertain costs a little, and confidently *wrong* costs
disproportionately more — exactly the "penalizes confident-and-wrong far
more than uncertain" line from the Frame above, now with an exact number
attached to both sides of it.

### What we observed

**Overfitting one batch, the proof that learning works.**
`tests/test_train.py`'s `test_overfit_one_batch_drives_loss_down` builds a
`TOY` model, hands it a fixed-batch `TrainConfig` (the same
`fixed_batch=True` mechanism from the Annotated Code section above) over
400 steps, and asserts `last_loss < first_loss * 0.3` and `last_loss <
1.0` — both pass. The project's own developer log recorded the sharpest
version of the same demonstration directly: with a single fixed batch
repeated over and over, loss collapsed from `62.2` to `0.0000` by around
step `50` — the model memorized that one batch essentially perfectly.
That's the "does learning work at all" question, answered with a real
number: yes, decisively.

**The toy training run (local CPU).** Training `TOY` (`213,312`
parameters) on 2,000 TinyStories, vocab size 2048, for 800 steps took
about 2 minutes on CPU. Loss: `62.4 → 12.1` (step 50) `→ 6.5` (100) `→
5.2` (200) → plateau around `4.3` (800), wiggling slightly there
afterward (per-step loss is noisy since each step draws a fresh random
batch, and 213K parameters is capacity-limited for real coherence — that's
exactly what the `SMALL` Colab run in Module 7 exists to fix). The
periodic samples climbed a visible "coherence ladder" as training
progressed:

```
step 200: "Once upon a time, there was a little girl named a saw a time.
She had a it was very in as and."
step 600: "Once upon a time, there was a little girl named Tim. Every
day excited one t adventure in the water and was so happy."
step 800: "...She was very happy and he thought, he wanted to play with
her. Once upon a time there were two friends with a"
```

Step 200 has real words and correct opening grammar but nonsense
semantics; step 600 has landed on a named character and largely coherent
clauses; step 800 reads as recognizable story structure. Nobody
hand-designed that progression — it's the direct, visible effect of loss
dropping from `62` toward `4.3` over those same 800 steps.

**Why the untrained loss starts *above* the uniform-guessing baseline.** A
model that knew absolutely nothing and guessed *uniformly* across all 2048
possible tokens would score a cross-entropy loss of exactly
`ln(2048) ≈ 7.6` — every position, however wrong, gets an equal, honest
`1/2048` chance on the true token. `TOY`'s untrained starting loss instead
measured around `62` — roughly *eight times worse* than that
honest-uncertainty baseline, not better and not merely equal to it. Module
5's deep dive explains exactly why: an untrained model with tied
embeddings doesn't guess uniformly at all, it confidently echoes the last
token it saw (a `max_prob` near `1.0`, driven by the self-dot-product
`|embed(last)|²`). Cross-entropy's asymmetry (from the Frame and the
worked example above) means that confidently betting on the *wrong* token
is punished far harder than spreading your bets honestly across all 2048
— which is precisely why the echo-biased, confidently-wrong starting point
scores worse than plain uniform guessing would. The first several dozen
steps of training are, in a very real sense, the optimizer unlearning that
echo before it can start learning anything about actual next-token
structure.

### Gotchas & design decisions

**The clean `lr_at` above is authoritative — a deliberately convoluted
draft in the project's own plan was not used.** The original Phase 1 plan
sketched a first version of the warmup/cosine schedule that the project's
developer log describes explicitly as "deliberately-convoluted," meant to
be replaced during implementation rather than copied verbatim. The version
embedded above — linear warmup, an exact peak at `step == warmup_steps`,
then a clamped cosine decay — is the real, current, shipped implementation
in `src/slm/train.py`, and it's the only version that matters going
forward.

**`weights_only=True` is a security choice, not an implementation
detail.** `torch.load` can, by default, unpickle arbitrary Python objects
— which means loading a checkpoint from an untrusted source (a downloaded
model file, say) can execute arbitrary code as a side effect of "just
loading a file." Storing the config as a plain dict (rather than pickling
the `ModelConfig` dataclass object itself) is what makes
`weights_only=True` possible on load: with that flag, `torch.load` refuses
to reconstruct anything beyond tensors and plain containers, closing off
that arbitrary-code-execution path entirely. This project's whole later
story — downloading a checkpoint someone else trained and running it
locally — is exactly the scenario where that protection matters.

**`load_checkpoint` needs `map_location="cpu"` to load a GPU-trained
checkpoint on this box.** `torch.save` records which device each tensor
lived on when it was saved. A checkpoint trained on a CUDA GPU (Module 7's
Colab run) carries CUDA location tags baked into the file; without
`map_location="cpu"`, loading that file on a machine with no GPU at all
fails outright, because there's nowhere to put a "CUDA tensor" on a
CPU-only box. Remapping every tensor to CPU at load time sidesteps the
problem entirely — inference and packaging in this project always happen
on CPU, and any caller that actually wants the model on a GPU can move it
there afterward with a plain `.to(device)`.

### Checkpoint

1. Using the worked example, why does confidently predicting the wrong
   token cost so much loss?
2. Why ramp the learning rate up (warmup) instead of starting at the peak?
3. What does overfitting a single batch prove, and why do it before a real
   dataset?

**Explain it back:** why does an untrained model score loss `~62` when
uniform guessing is only `~7.6`?

## Module 7 — The Real Run (small on Colab)

### Learning objectives

By the end of this module you'll be able to: explain why moving from the
`TOY` config to the `SMALL` config is a config change and not a code
change; read the real `SMALL` config values and connect them to its
13.8M-parameter count; read the exact device-portability handling inside
`train()` and explain what makes the identical function correct on both a
CPU and a Colab GPU; state the real, measured results of the Colab run
(eval loss and perplexity) and say what those numbers mean in plain terms;
and explain, from firsthand experience, why the project deliberately
stopped scaling up at roughly 14M parameters instead of going bigger.

### Frame

Every piece of Module 6 — `lr_at`, the `train` loop, cross-entropy,
`save_checkpoint`/`load_checkpoint` — is the *exact same code* this module
uses. Nothing about `train()` or `LlamaSLM` gets rewritten to go from the
small, fast, CPU-only toy run to what this project calls the "real" run:
the only two things that change are which `ModelConfig` gets passed in
(`SMALL` instead of `TOY`) and which physical device the model happens to
live on (a free-tier Colab T4 GPU instead of a laptop's CPU). That's the
entire payoff of writing device-portable code once in Module 6: scaling up
becomes a config swap, not a rewrite.

### Annotated code walkthrough

**The `SMALL` config.** Here's the real, current `SMALL`, alongside `TOY`
for comparison, from `src/slm/config.py`:

```python
TOY = ModelConfig(
    vocab_size=2048, d_model=64, n_layers=2, n_heads=4, n_kv_heads=4,
    head_dim=16, ffn_hidden=128, context_len=128,
)

SMALL = ModelConfig(
    vocab_size=8192, d_model=384, n_layers=6, n_heads=6, n_kv_heads=6,
    head_dim=64, ffn_hidden=1024, context_len=512,
)
```

Every field grows: vocabulary quadruples (`2048 → 8192`, room for a richer
byte-level BPE vocabulary trained on far more text), `d_model` grows
6-fold (`64 → 384`, the width of every token vector), depth triples
(`2 → 6` layers), each attention head carries 4× the working room
(`head_dim` `16 → 64`), the SwiGLU workspace grows 8-fold (`ffn_hidden`
`128 → 1024`), and the context window quadruples (`128 → 512` tokens of
history the model can look back on). Feeding `SMALL` into the exact same
`ModelConfig.n_params()` from Module 4's params-by-component discussion
gives **13,767,552** — the number this project rounds to "13.8M params,"
and notably the *same* number `n_params()` predicted before any Colab run
ever happened, since it's pure arithmetic over the config fields. `TOY`,
for comparison, comes out to `213,312` params — `SMALL` is roughly 65× the
capacity, still nowhere near a "large" language model, but enough of a
jump to move well past `TOY`'s capacity-limited plateau from Module 6.

**Device portability, in the exact lines that matter.** This is the same
`train()` from Module 6, but here's the piece worth re-reading with a real
GPU in mind:

```python
    device = next(model.parameters()).device
    ...
    for step in range(cfg.max_steps + 1):
        ...
        x, y = get_batch(data_t, cfg.batch_size, cfg.context_len, seed=seed)
        x, y = x.to(device), y.to(device)
```

`device = next(model.parameters()).device` never hardcodes a device
anywhere — it simply asks the model's own first parameter what device *it*
currently lives on. On Colab, the notebook cell builds the model, then
immediately moves it: `model = LlamaSLM(SMALL).to(device)` (with
`device = "cuda"`), so by the time `train()` runs, every one of the
model's parameters already lives on the GPU, and this line reads back
`cuda:0`. Locally, nobody ever calls `.to("cuda")`, so the same line reads
back `cpu`. Either way, `get_batch` itself always builds `x`/`y` on the
CPU — it uses a private CPU `torch.Generator` for determinism (Module 3),
which would break if that generator were ever asked to draw from a CUDA
default device. The fix is exactly the line above: build the batch on CPU
regardless, then explicitly move it, `x, y = x.to(device), y.to(device)`,
onto wherever the model actually lives, right before the forward pass. One
function, two devices, zero edits between the toy run and this one.

### What we observed

The Colab notebook (`notebooks/colab_train.ipynb`) streamed
`load_tinystories("train", limit=200_000)` — 200,000 stories, chosen
deliberately as a RAM-safe subset rather than the full ~1.9GB corpus,
since even that streamed subset tokenizes into a **44,290,410-token**
stream, and holding the *entire* dataset as a Python list risked
overflowing Colab's roughly 12GB of RAM. Training ran with `lr=6e-4,
warmup_steps=200, max_steps=20000, batch_size=64, context_len=512`, in
fp32, on a free-tier Colab T4 GPU. Step-0 loss came in at **361.7** —
another direct confirmation of Module 6's echo-bias math: just as `TOY`'s
untrained loss (~62) tracked its own `d_model` of 64, `SMALL`'s untrained
loss tracked its larger `d_model` of 384, because the dominant term in
that starting loss is still the self-dot-product of a
roughly-unit-variance embedding vector with itself. Loss fell into single
digits within a few hundred steps, then declined gradually across the
full 20,000 steps.

The three resulting artifacts — `small.pt`, `small_tok.json`,
`small_loss.png` — were downloaded from Colab into this project's local
`checkpoints/`. Evaluating the downloaded checkpoint back on this local
CPU box, against a fresh 500-story sample (30 batches), gave **loss
1.81, perplexity 6.1** — compared to the toy run's `~4.3` loss / `~74`
perplexity, and a uniform-guessing baseline of `ln(8192) ≈ 9.01` for
`SMALL`'s larger vocabulary. Perplexity is just `e^loss` — intuitively,
"on average, about how many tokens the model was genuinely torn between"
for each prediction; `6.1` means the model has usually narrowed the field
down to around six plausible next tokens, versus the toy run's `74` and a
uniform guesser's `8192`.

That drop in perplexity shows up directly in the generated text.
Prompting the Colab-trained model with `"Once upon a time"` (temperature
`0.8`, top_k `40`, seed `0`) produced:

> Once upon a time, there was a little boy named Tim. Tim loved to take
> pictures with his camera. One day, Tim went to the park with his mom. At
> the park, Tim found a nice spot under a big tree. Tim saw a bird's
> friend, Sarah. Tim asked, "Do you like my camera?" Sarah said, "Yes,
> it's very pretty." Tim took a photo of Tim and kept it in his pocket.
> They played and laughed and had fun. Soon

Named characters, dialogue, a beginning-middle-arc shape, correct grammar
throughout — real TinyStories prose, not the word-salad or half-formed
clauses of the toy run's samples. A couple of small 14M-on-a-subset
artifacts remain (Tim ends up taking "a photo of Tim"), but this is the
coherence milestone Phase 1 was built toward: a hand-written model,
trained by hand-written code, writing genuinely readable little stories.

### Gotchas & design decisions

**Colab's pip install had to stay minimal, or it broke the environment.**
Installing this project's full, pinned `requirements.txt` on Colab pulled
in a wall of resolver conflicts against Colab's own co-tuned
`numpy`/`pandas`/`torch`/`google-colab` stack, and then crashed a live
kernel with `ImportError: cannot import name '_center' from
'numpy._core.umath'` — the numpy upgrade happened underneath a kernel that
had already imported the *original* numpy, leaving its C extension and
Python files at mismatched versions. `requirements.txt` is the right
artifact for a fresh, empty environment (that's its whole reproducibility
job, from Module 1), but the wrong one for Colab's already-populated one.
The fix: the notebook's setup cell installs only this project's two
direct extra dependencies, `pip install datasets tokenizers`, and leaves
Colab's own torch/numpy/pandas alone entirely.

**The GPU-saved checkpoint wouldn't load on this CPU-only box —
`map_location="cpu"` again, this time for real.** Module 6 covered why
`load_checkpoint` needs `map_location="cpu"` as a code-level fact;
downloading `small.pt` straight off Colab is where that fact actually bit.
`torch.save` had recorded every tensor's CUDA location from training on
the T4, and loading the file locally without remapping failed with
"Attempting to deserialize object on a CUDA device but
torch.cuda.is_available() is False." Because `load_checkpoint` already
remaps to CPU unconditionally, the fix required no new code at all here —
it's the exact same line from Module 6, now earning its keep on a real,
downloaded, GPU-trained file.

**Scaling further was a deliberate choice to stop, not a limitation run
into by accident.** TinyStories is, by design, a small, simple-vocabulary
dataset (short children's stories) — it saturates in usable quality at
tens of millions of parameters, and pushing `SMALL`'s config meaningfully
larger would mostly slow down iteration and break the "runs instantly on
a CPU for a demo" property this project cares about, without buying
noticeably better stories from this particular dataset. 13.8M parameters
was the point where TinyStories' own ceiling and this project's own
"stays runnable everywhere" goal met in the middle.

### Checkpoint

1. What made scaling from `TOY` to `SMALL` a config change rather than a
   code change?
2. What does perplexity `6.1` mean in plain terms?

**Explain it back:** why did we finish Phase 1 at ~14M params instead of
going bigger?

## Module 8 — Packaging to Hugging Face Format

### Learning objectives

By the end of this Module you'll be able to: explain what the "Hugging Face
format" concretely consists of, and why that specific shape is what lets any
tool in the ecosystem open your model with zero custom code; read the real
`to_hf_config`, `_copy_weights_into_hf`, `export_to_hf`, and `push` functions
in `src/slm/export_hf.py`, in full, and say what each one is responsible for;
state the round-trip test's actual measured result and explain precisely
what that number proves (and what it would mean instead if it were large);
explain why `to_hf_config` overrides Llama's default special-token ids
rather than leaving them alone, and what a downstream user would experience
if that override were missing; and describe, from firsthand project
experience, the correct response the instant a write-scoped credential ends
up somewhere it can be seen.

### Frame

At the end of Module 7 you have exactly two files that matter:
`checkpoints/small.pt` (the trained weights plus the `ModelConfig` dict, in a
layout only this project's own `save_checkpoint`/`load_checkpoint` know how
to open) and `checkpoints/small_tok.json` (this project's own tokenizer).
Every tool used so far to load them — `load_checkpoint`, `LlamaSLM`,
`sample.py` — is code written earlier in this course. Nothing outside this
project has any idea what `small.pt`'s dictionary keys mean.

"Packaging to Hugging Face format" is the step that fixes that, and it's
worth being precise about what it actually is: not a rewrite, not a
re-implementation — a *repackaging*. The target shape is a plain directory
holding three kinds of files the whole open-source LLM ecosystem already
agrees on: `config.json` (a plain-text architecture spec — hidden size,
layer count, head count, vocabulary size, and, critically, the special-token
ids), one or more `model.safetensors` files (the weights themselves, in a
safe, memory-mappable container — unlike a pickled `.bin` file, safetensors
cannot execute arbitrary code on load, which is the modern reason it's the
default), and a handful of tokenizer files (`tokenizer.json`,
`tokenizer_config.json`, and friends) describing how to turn text into ids
and back. Any tool built against `transformers` — `from_pretrained`, the
GGUF converter Module 9 uses, the Hub's own model viewer — already knows how
to read exactly that shape, with nothing project-specific to teach it.

The reason this repackaging works with a plain weight *copy*, rather than
something lossier, traces back to a deliberate choice already flagged in
Module 4's Gotchas: this project matched stock Llama's architecture
faithfully on purpose — RMSNorm computed in float32 the way the official
implementation does, the same RoPE convention, the same SwiGLU shape —
specifically so that this moment would work. Think of the hand-built
`LlamaSLM` weights as a compiled binary, and Hugging Face format as
repackaging that same binary into the shipping container every dock on the
pipeline — `transformers`, the GGUF converter, the Hub — already knows how
to unload, no custom forklift required.

### Annotated code walkthrough

Here's the real, current `src/slm/export_hf.py`, in full:

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
        # Our tokenizer's <|endoftext|> is id 0 and serves as BOS/EOS/PAD.
        # Without this, LlamaConfig defaults (bos=1, eos=2) would tell runtimes
        # to stop on the wrong token, so generation never ends at a story break.
        bos_token_id=0,
        eos_token_id=0,
        pad_token_id=0,
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


def push(out_dir: str, repo_id: str, private: bool = True):
    from huggingface_hub import HfApi
    HfApi().create_repo(repo_id, exist_ok=True, private=private)
    LlamaForCausalLM.from_pretrained(out_dir).push_to_hub(repo_id, private=private)
    PreTrainedTokenizerFast.from_pretrained(out_dir).push_to_hub(repo_id, private=private)
    # The model card (README.md) isn't loaded by from_pretrained; upload it too.
    import os
    readme = os.path.join(out_dir, "README.md")
    if os.path.exists(readme):
        HfApi().upload_file(path_or_fileobj=readme, path_in_repo="README.md",
                            repo_id=repo_id, repo_type="model")
```

**`to_hf_config` — translating our config into theirs.** `ModelConfig`
(Module 1) and `LlamaConfig` (stock `transformers`) describe the same
architecture with different field names and different defaults, and this
function is a pure field-by-field translation between them: our `d_model`
becomes their `hidden_size`, our `ffn_hidden` becomes their
`intermediate_size`, our `n_layers`/`n_heads`/`n_kv_heads`/`head_dim` map
straight across, and `tie_word_embeddings=cfg.tie_embeddings` carries over
the exact weight-tying decision from Module 5 (where `tie_embeddings=True`
made `lm_head.weight` and `embed_tokens.weight` the literal same tensor) —
HF represents that identical fact as `tie_word_embeddings: true` in the
exported `config.json`. `attention_bias=False`, `mlp_bias=False`, and
`hidden_act="silu"` pin down the parts of stock Llama's config surface that
match this project's architecture: no bias terms anywhere, and the SwiGLU
activation from Module 4. The three lines at the bottom —
`bos_token_id=0`, `eos_token_id=0`, `pad_token_id=0` — are the one place
this function actively *overrides* a Llama default instead of translating a
field, and the comment explains why: Module 2 fixed this project's
`<|endoftext|>` at token id `0`, the *only* special token, doing
BOS/EOS/PAD duty all at once, but `LlamaConfig`'s own defaults are
`bos_token_id=1, eos_token_id=2` — ids that mean nothing in this project's
8,192-token vocabulary. More on why that override matters below.

**`_copy_weights_into_hf` — the same tensors, new names, zero
re-computation.** This function moves not one floating-point number: it
builds a dict, `sd`, mapping HF's expected state-dict keys directly onto
`LlamaSLM`'s live tensors — `src.embed_tokens.weight`, `blk.norm1.weight`,
`blk.attn.q_proj.weight`, and so on — then calls
`hf.load_state_dict(sd, strict=True)`. `strict=True` means: if a single key
is missing, or a single extra key is present, this line raises instead of
silently loading a partial model. Every name on the left
(`model.layers.{i}.input_layernorm.weight`, `self_attn.q_proj.weight`,
`mlp.gate_proj.weight`, …) is stock `LlamaForCausalLM`'s naming convention;
every value on the right is this project's own `Block`'s attribute
(`norm1`/`norm2`/`attn`/`mlp`, from Module 4/5). The loop over `src.layers`
handles all `n_layers` blocks identically, and the two lines above and
below it handle the embedding table and the tied `lm_head` at the top and
bottom of the stack. Because the *shapes* and *math* of `LlamaSLM` and
`LlamaForCausalLM` are already identical (that architectural-fidelity
choice from Module 4, again), this is purely a renaming exercise — nothing
here recomputes or approximates anything.

**`export_to_hf` — the whole pipeline, one call.** `load_checkpoint(ckpt_path)`
(Module 6/7's function, `map_location="cpu"` and all) rehydrates the trained
`LlamaSLM` and its `ModelConfig` from `small.pt`; a fresh, empty
`LlamaForCausalLM(to_hf_config(cfg))` is constructed; `_copy_weights_into_hf`
fills it; `hf.save_pretrained(out_dir)` writes `config.json` +
`model.safetensors` (plus a `generation_config.json`) to disk. The tokenizer
side is separate and deliberate: rather than re-deriving anything from this
project's own tokenizer file, `export_to_hf` wraps it directly in HF's
`PreTrainedTokenizerFast`, explicitly declaring `<|endoftext|>` as all four
of `eos_token`/`bos_token`/`unk_token`/`pad_token` — the HF-side mirror of
the same one-token-does-everything design from Module 2 — then calls
`fast.save_pretrained(out_dir)` to write the tokenizer files alongside the
model files.

**`push` — uploading, plus the one file `push_to_hub` forgets.**
`HfApi().create_repo(repo_id, exist_ok=True, private=private)` creates the
Hub repo (a no-op if it already exists); loading the just-exported model and
tokenizer back with `from_pretrained(out_dir)` and calling
`.push_to_hub(repo_id, private=private)` on each uploads the weights,
config, and tokenizer files. The comment on the next line flags something
worth noticing: `push_to_hub()` only uploads what `from_pretrained` itself
loads — model weights and tokenizer files — and a model card (`README.md`,
the human-facing page the Hub renders as the model's front page) isn't one
of those things, so the function checks for `out_dir`'s `README.md` and, if
present, uploads it explicitly via `HfApi().upload_file(...)`. Skip that
check and the model would technically be on the Hub, but its page would
render blank — no description, no usage example, no license.

### What we observed

`tests/test_export.py`'s `test_hf_roundtrip_matches_handbuilt` builds a
fresh, untrained `LlamaSLM(TOY)`, builds an empty
`LlamaForCausalLM(to_hf_config(TOY))`, copies the weights across with
`_copy_weights_into_hf`, feeds both the *same* random token ids, and
asserts `torch.allclose(a, b, atol=1e-4)` — a tolerance the actual run blew
past by more than an order of magnitude. That test run — comparing the
untrained `LlamaSLM(TOY)`'s forward pass against stock
`LlamaForCausalLM`'s forward pass on the same random input ids, immediately
after the weight copy — measured a maximum absolute logit difference of
**9.54 × 10⁻⁶** (mean difference **8.78 × 10⁻⁷**, this project's own
shorthand for it: "~9.5e-06") — a gap this small is floating-point rounding
noise, not an architectural difference. Module 4's Gotchas section promised this
exact number would show up here — "a round-trip check of max-abs-difference
around `1e-5` … the correctness linchpin the export in Module 8 depends
on" — and it did, on the first run, with no debugging required. That's the
actual proof, in numbers, that the from-scratch `LlamaSLM` this course
built starting in Module 4 isn't merely "Llama-like": it is, weight-for-
weight, the same architecture as the official `LlamaForCausalLM`, to within
float noise. Had that diff instead come out large — comparable to the
logits' own scale rather than a sliver of it — it would mean the two models
were computing genuinely different functions, and no amount of weight-
copying trickery would make the export trustworthy.

With that verified, the trained `small` checkpoint was exported to
`export/tinystories-slm/` (`config.json`, `model.safetensors`, and the
tokenizer files) and loaded back with plain, stock `transformers` — no
project code involved at all — confirming it still generates the same kind
of coherent TinyStories prose Module 7 already showed. It was then
published to a **private** Hub repository, `vysakhpillai/tinystories-slm`,
holding the weights, config, tokenizer files, and model card.

### Gotchas & design decisions

**Gotcha — the exported config first shipped Llama's default
`eos_token_id=2`, so generation never stopped.** Before `to_hf_config` set
`bos_token_id=eos_token_id=pad_token_id=0` explicitly, the exported
`config.json` carried `LlamaConfig`'s ordinary defaults, `bos_token_id=1,
eos_token_id=2` — perfectly sensible for a model whose vocabulary actually
assigns tokens 1 and 2 to BOS and EOS, meaningless for this project's
vocabulary, where those slots are just two more ordinary story tokens and
the *real* end-of-text marker is id `0`. `model.generate()` decides when to
stop by checking whether the model just emitted `eos_token_id` — so with
the default in place, it would keep generating indefinitely (up to
`max_new_tokens`), never recognizing this model's own `<|endoftext|>` as a
signal to stop. The fix was the three-line override already in the
walkthrough above; the observable effect, verified directly, was that five
of six sampled generations then stopped early exactly at a story break,
instead of all six running to the max length. This is the same id-0 fact
Module 2 established when it fixed `<|endoftext|>` as this tokenizer's only
special token — Module 8 is just the moment that fact has to be
re-declared in a second, HF-shaped config file, or the two disagree.

**Design decision — the Hub username doesn't have to match the GitHub
username, and here it doesn't.** This project's code lives on GitHub as
`vppillai`, but the Hugging Face account used to publish is a separate
identity, `vysakhpillai` — the two platforms have entirely independent
namespaces. The repo id passed to `push()` had to be the HF-namespaced
`vysakhpillai/tinystories-slm`, and the model card's own usage examples
needed the same correction, since a copy-pasted `vppillai/tinystories-slm`
would simply 404.

**Security — a write-scoped token ended up somewhere it could be seen, and
the only correct response is to treat it as already compromised.** While
authenticating to the Hub during this task, a write token (value redacted)
was pasted into the chat used to drive this session. A token with write
scope can push, delete, or overwrite repositories under that account — so
the instant a secret like that lands anywhere it might be logged,
screenshotted, or reviewed by anyone else, the right move isn't to hope
nobody looks: it's to revoke or rotate it immediately, at
huggingface.co/settings/tokens, and only re-authenticate afterward. This
project's actual response was exactly that — flag it and revoke it — and
the durable lesson generalizes well past Hugging Face: prefer short-lived,
read-only tokens for anything routine, reserve write scope for the one
moment it's actually needed, and treat "this credential was ever visible
somewhere it shouldn't have been" as equivalent to "this credential is
compromised," full stop, regardless of how the exposure happened.

### Checkpoint

1. Why does the round-trip logit diff of ~1e-5 matter — what does it prove?
2. What breaks downstream if the exported `config.json` keeps Llama's
   default `eos_token_id=2`?

**Explain it back:** why can our hand-built model load into `transformers`
with no custom code at all?

## Module 9 — GGUF, Quantization, and Ollama

### Learning objectives

By the end of this Module you'll be able to: explain what makes a `.gguf`
file "self-describing" and why that removes the need for any separate
config file; read the real `scripts/convert_to_gguf.sh` and
`scripts/Modelfile`, in full, and explain what each stage of the pipeline
does; read Lab 07's header-parsing code and explain exactly what a GGUF
header contains and why its two numbers (tensor count, metadata count)
matter; state the real file sizes this project produced at each
quantization level and explain what quantization trades away to get them;
and trace, from firsthand experience, the entire path a model takes from a
trained checkpoint file to a running `ollama run` session.

### Frame

Module 8 ended with the model sitting in Hugging Face format — a directory
`transformers`-based tools can open. That's already enough for a GPU server
running Python. But the actual finish line for this project is different:
run the hand-built model on an ordinary CPU, through the same lightweight
tooling millions of people already use to run downloaded models locally.
That tooling is `llama.cpp` (the inference engine) and Ollama (a friendly
wrapper around it), and neither one speaks `safetensors`/`config.json` —
they speak one specific container format, **GGUF**.

A GGUF file is a single, self-describing binary: a short header (a magic
number, a format version, how many tensors follow, how many metadata
entries follow), then all the tensors (the weights themselves), then the
metadata key-values (architecture name, hyperparameters, tokenizer details)
a runtime needs to know how to run the model at all. "Self-describing" is
the key word — unlike the HF directory from Module 8, where the weights
(`model.safetensors`) and the architecture spec (`config.json`) are two
separate files that have to be kept in sync, a GGUF file carries both in
one place, so `llama.cpp`/Ollama can load *just* the `.gguf` and know
everything they need.

Getting from Module 8's HF directory to a running Ollama model is a short
pipeline with two moves worth naming precisely, because the same two moves
show up in every deployment story for every model format: first,
**convert** the HF directory into GGUF's IR (a straight structural
translation, no precision loss — this project's `f16` GGUF); second,
**quantize** — an optimization pass that lowers the numeric precision of
the weights to shrink the file and speed up CPU inference, trading away a
small, controllable amount of quality. The inference engine, `llama.cpp`,
is the last piece: it loads a `.gguf`, picks CPU kernels suited to the
machine it's running on, and executes the forward pass token by token;
Ollama wraps that same engine with model management and the one-line
`ollama run <name>` experience. Module 10 steps back from these specific
file formats and names the general shape this whole pipeline is one
instance of.

### Annotated code walkthrough

Here's the real, current `scripts/convert_to_gguf.sh`, in full:

```bash
#!/usr/bin/env bash
set -euo pipefail
# Convert the exported HF model to GGUF, then quantize.
#
# Prereqs (one-time):
#   git clone --depth 1 https://github.com/ggerganov/llama.cpp ~/llama.cpp
#   uv pip install gguf sentencepiece protobuf          # converter deps
#   cd ~/llama.cpp && uv run cmake -B build -DLLAMA_CURL=OFF \
#     && uv run cmake --build build -j 4                # -j 4: avoid OOM on small boxes
#
# Run from the model_learn repo root:  bash scripts/convert_to_gguf.sh
LLAMA_CPP="${LLAMA_CPP:-$HOME/llama.cpp}"
SRC="export/tinystories-slm"
OUT="export"
QUANT="$LLAMA_CPP/build/bin/llama-quantize"

# 1. Export to GGUF IR (f16). The converter is pure Python; run it with the
#    project venv (which has gguf/torch/transformers/safetensors).
uv run python "$LLAMA_CPP/convert_hf_to_gguf.py" "$SRC" \
  --outfile "$OUT/tinystories-slm-f16.gguf" --outtype f16

# 2. Optimization pass: quantize f16 -> Q8_0 (near-lossless) and Q4_K_M (small).
"$QUANT" "$OUT/tinystories-slm-f16.gguf" "$OUT/tinystories-slm-Q8_0.gguf"   Q8_0
"$QUANT" "$OUT/tinystories-slm-f16.gguf" "$OUT/tinystories-slm-Q4_K_M.gguf" Q4_K_M

echo "Wrote f16, Q8_0, Q4_K_M GGUFs to $OUT/"
ls -lh "$OUT"/tinystories-slm-*.gguf
```

**Stage 1 — `convert_hf_to_gguf.py`, the export-to-IR step.** This is
llama.cpp's own converter, not project code, and it takes the Module 8
output directory (`export/tinystories-slm`, holding `config.json` +
`model.safetensors` + tokenizer files) and produces
`tinystories-slm-f16.gguf`: every weight re-encoded at 16-bit float
precision, plus the full metadata (architecture, hyperparameters,
tokenizer) the runtime will need — no quantization has happened yet at this
step, only a container-format change. It's run with `uv run`, this
project's own venv, specifically because that venv is the one place with
`gguf`/`torch`/`transformers`/`safetensors` all installed together; the
converter script itself lives inside the separately-cloned `~/llama.cpp`
checkout, not this repo.

**Stage 2 — `llama-quantize`, the optimization pass.** `llama-quantize` is
a compiled binary (built by the `cmake --build` step in the prerequisites
comment) that reads the f16 GGUF and writes a new GGUF with weights
re-encoded at lower precision — `Q8_0` and `Q4_K_M` here, run back-to-back
against the same f16 source so both quantized files derive from the
identical, unquantized starting point. Neither invocation touches the
original `f16` file; all three GGUFs — `f16`, `Q8_0`, `Q4_K_M` — end up
sitting side by side in `export/`, so any one of them can be handed to
`llama.cpp` or Ollama directly. The prerequisites comment at the top
documents `cmake --build build -j 4`, capping the build to 4 parallel
compiler jobs instead of letting it use every core on the box — why, is in
the Gotchas below.

Here's the real, current `scripts/Modelfile`, in full:

```
FROM ./export/tinystories-slm-Q4_K_M.gguf
PARAMETER temperature 0.8
PARAMETER top_k 40
PARAMETER stop "<|endoftext|>"
TEMPLATE """{{ .Prompt }}"""
```

**Line by line.** `FROM ./export/tinystories-slm-Q4_K_M.gguf` names the
single GGUF file this Ollama model is built from — the *smallest*, most
aggressively quantized one of the three, chosen deliberately for the
fastest CPU inference at acceptable quality. The two `PARAMETER` lines set
inference defaults baked into the Ollama model itself, so nobody running
`ollama run tinystories-slm` needs to remember to pass them: `temperature
0.8` and `top_k 40` are the exact same sampling settings Module 6/7 already
used from `sample.py`, and `stop "<|endoftext|>"` tells Ollama's own
generation loop to stop the moment this project's one special token
appears — the Ollama-side counterpart to Module 8's `eos_token_id=0` fix,
enforced here at the sampling-loop level instead of inside `config.json`.
`TEMPLATE """{{ .Prompt }}"""` is deliberately the simplest template Ollama
supports: no chat-formatting, no role markers, no system-prompt
scaffolding — just the raw prompt text handed straight to the model,
because this is a plain story-completion model, not an instruction-tuned
chat model, and any chat template would just inject tokens this
tokenizer's vocabulary and training never saw.

**Reading a GGUF's header, for real.** Lab 07
(`labs/lab07_gguf_teardown.py`) makes "self-describing" concrete by reading
nothing but the header, by hand, with no library:

```python
import struct, sys

path = sys.argv[1] if len(sys.argv) > 1 else "export/tinystories-slm-Q8_0.gguf"
with open(path, "rb") as f:
    magic = f.read(4)
    version, = struct.unpack("<I", f.read(4))
    n_tensors, = struct.unpack("<Q", f.read(8))
    n_kv, = struct.unpack("<Q", f.read(8))
```

Four fixed-size reads, in order, are the entire GGUF header: 4 raw bytes
for `magic` (expected to literally spell `GGUF`, a sanity check any reader
can perform before trusting the rest of the file), then three
little-endian integers unpacked with `struct.unpack` — a 4-byte unsigned
int for the format `version`, then two 8-byte unsigned ints for
`n_tensors` (how many weight tensors follow) and `n_kv` (how many metadata
key-value entries follow). Everything after those 24 bytes is exactly
`n_tensors` tensor descriptors followed by exactly `n_kv` metadata entries.
The header alone already tells a runtime precisely how much more there is
to read and of what kind — which is exactly what "self-describing" cashes
out to, in concrete bytes.

### What we observed

Running `scripts/convert_to_gguf.sh` produced three files in `export/`:
`tinystories-slm-f16.gguf` at **27MB**, `tinystories-slm-Q8_0.gguf` at
**15MB**, and `tinystories-slm-Q4_K_M.gguf` at **11MB** — corresponding to
**16.0**, **8.51**, and **6.21** bits per weight respectively. Q8_0's 8.51
bits (slightly above its nominal 8) comes from a small, fixed per-block
scale factor stored alongside the 8-bit weights; Q4_K_M's 6.21 bits (well
above its nominal ~4) is explained by the next paragraph.

Lab 07's header read against the Q8_0 file reported **56 tensors** and
**34 metadata entries** — the concrete, observed instance of "header +
tensors + metadata" from the Frame above. Running the same lab against the
Q4_K_M file makes a quieter but important point: **36 of those same 56
tensors fell back to a higher precision than Q4_K_M's nominal 4 bits**, not
because anything went wrong, but because their shapes don't fit Q4_K_M's
block structure — Q4_K_M's block-quantization scheme needs a tensor's
dimensions to divide evenly into its block size, and on a model this small
(`d_model=384`, `head_dim=64`, `ffn_hidden=1024` — all comparatively small
by production-model standards), several tensor dimensions likely fall
short of what that block structure expects, so `llama-quantize` stores
those specific tensors at higher precision rather than quantizing them
incorrectly. It's shape-driven and benign — a real, slightly-surprising
number this project's own tools produced, not a sign of a broken export.

Lab 08 (`labs/lab08_quant_compare.py`) then compares generation itself, not
just file size: it loops over all three quant levels, and for each one
that exists on disk, runs `llama-cli` once with a fixed prompt (`"Once
upon a time"`), a fixed seed (`0`), and a 40-token cap, using `-st`
(single-turn — exits after one generation instead of dropping into
`llama-cli`'s interactive REPL, which would otherwise hang the lab's
`subprocess.run` call) plus `--simple-io`/`--no-warmup`/`--no-display-prompt`
to keep the captured output to just the generated story. All three quant
levels — f16, Q8_0, Q4_K_M — produced fluent, coherent TinyStories-style
prose from the identical prompt and seed; the *text itself* differed
slightly between quant levels, because quantization perturbs the model's
logits by a small amount and sampling is sensitive to small logit changes —
but coherence held all the way down to the smallest, most aggressively
quantized file.

The actual finish line: with Ollama running and `scripts/Modelfile`
registered under the name `tinystories-slm`, `ollama run tinystories-slm
"Once upon a time"` produced, on ordinary CPU:

> Once upon a time, there was a little girl named Lily. She loved to play
> outside in the park. One day, she saw a big tree and wanted to climb it. But
> when she tried to climb the tree, she slipped and fell. Her knee hurt a lot!
> She started to cry because she couldn't reach the top of the tree. Her mom
> came running over and asked what happened. Lily told her that she was hurt and
> needed to rest. Her mom gave her some medicine and told her to rest when she
> was tired. After resting, ...

A hand-built tokenizer (Module 2), a hand-built transformer (Module 4/5),
trained by hand-written training code (Module 6/7), exported to a standard
format (Module 8), quantized down to 11MB, and run through the exact same
engine people use to run downloaded Llama models — generating a coherent,
punctuated, readable children's story, entirely on CPU.

### Gotchas & design decisions

**Gotcha — building llama.cpp at full parallelism ran this machine out of
memory.** The natural instinct, `cmake --build build` with no job cap,
spawns one compiler process per CPU core — on this box, 14 — and
llama.cpp's largest source files (model-family backends like `t5.cpp`,
`hunyuan-vl.cpp`) each need enough memory per compiler instance that 14 of
them running at once exceeded the box's roughly 11GB of RAM, killed with
`Killed signal terminated program cc1plus`. The fix already visible in the
prerequisites comment, `cmake --build build -j 4`, caps it to 4 concurrent
compiler processes; because `make`-style builds are incremental, re-running
with the lower job count resumed from the object files already compiled
rather than starting over.

**Gotcha — the GGUF converter didn't recognize this project's own
tokenizer.** `convert_hf_to_gguf.py` identifies *which* BPE tokenizer a
model uses by hashing a specific pre-tokenizer test string and checking
that hash against a hardcoded table of known tokenizers — it doesn't
inspect a fresh, custom-trained tokenizer's rules directly. This project's
tokenizer (Module 2's byte-level BPE, trained from scratch on TinyStories)
produces a hash the converter had never seen, `fe391dc4...`, so conversion
failed outright with `NotImplementedError: BPE pre-tokenizer was not
recognized`. The tokenizer genuinely *is* a standard GPT-2-style
byte-level BPE, just trained on this project's own data instead of
downloaded pre-trained — so the fix was registering that specific hash as
`"gpt-2"` in llama.cpp's own conversion code (`get_vocab_base_pre`), after
which the converter proceeded normally and the resulting GGUF records
`tokenizer.ggml.pre = gpt-2` in its metadata — exactly the kind of metadata
entry Lab 07's header count is pointing at.

**Design decision — Ollama needed a manually-started daemon, the `zstd`
package, and an absolute `FROM` path.** Three separate small frictions, all
environment-specific rather than code bugs: the official Ollama installer
failed the first time with "This version requires zstd for extraction"
until `zstd` was installed via the system package manager; on this VM,
systemd didn't auto-start the Ollama daemon after install, so `ollama
serve` had to be started by hand before `ollama create`/`ollama run` would
work at all; and Ollama 0.31.2 specifically rejected the Modelfile's clean,
repo-relative `FROM ./export/tinystories-slm-Q4_K_M.gguf` with `400 Bad
Request: invalid model name`, because that Ollama version treats a
relative-looking `FROM` value as a model-name reference rather than a
filesystem path — the fix was rewriting it to an absolute path for the
actual `ollama create` invocation, even though `scripts/Modelfile` itself is
kept in the cleaner relative form shown above for readability.

### Checkpoint

1. What does "self-describing" mean for a GGUF file, and why does it need
   no separate config?
2. Why does the same prompt and seed produce slightly different text
   across quant levels?
3. What is the Modelfile's job on top of the GGUF file itself?

**Explain it back:** trace the file, end to end, from `small.pt` all the
way to `ollama run`.

## Module 10 — The Transferable Mental Model: Model Compilers

### Learning objectives

By the end of this Module you'll be able to: name the three stages of Module
9's GGUF pipeline using compiler vocabulary — export to IR, optimization
pass, codegen + runtime — instead of GGUF-specific vocabulary; explain why
`llama-quantize` counts as an optimization pass rather than a mere format
change; recognize the same three-stage shape in other inference stacks
(ONNX Runtime, TVM, MLIR/IREE, TensorRT, OpenVINO, MLC-LLM, and vendor
toolchains like Tenstorrent's MLIR/Metalium) without needing to learn each
one from scratch; and state, from memory, the four-step mental model for
deploying a trained model onto a new piece of hardware.

### Frame

Every Module from 1 through 9 taught you a specific piece of *this*
project: a specific tokenizer, a specific architecture, a specific training
loop, a specific export path. This Module is different — it doesn't teach
you anything new to build. Instead, it renames what Module 9 already did,
and that renaming is the actual payoff of the whole course.

Look back at what Module 9's pipeline did, stripped of its GGUF-specific
words: it took a trained model living in one framework's native
representation (Hugging Face's `config.json` + `safetensors`), translated
it into a portable, self-describing intermediate format, then applied a
pass that changed *how* the model is represented internally (numeric
precision) without changing *what* it computes, and finally handed the
result to a runtime that picks concrete, hardware-suited kernels and
executes them. That is not a GGUF-specific story — it's the generic shape
every "run this trained model somewhere new" story takes. Once you can
name the shape, you can walk into an unfamiliar toolchain, built by a
different vendor for completely different silicon, and immediately know
which question to ask at each stage: what's the IR here, what optimization
passes does this stack run, and what does codegen and runtime look like for
this hardware?

That's the whole reframe: the GGUF path you just walked with your own
hands, end to end, *is* a miniature model compiler. Naming its stages once
is what lets the mental model travel to hardware this course never
touches.

### Annotated mapping

There's no new code in this Module — the "walkthrough" here is relabeling
the exact three moves Module 9 already made real, in the order you ran
them:

- **`convert_hf_to_gguf.py` = export to IR.** Module 9's first stage took
  the Module 8 HF directory (`config.json` + `model.safetensors` +
  tokenizer files — one framework's native format) and produced a `.gguf`
  file: a different container, the same computation, no precision change
  yet. That's exactly what "export to an intermediate representation"
  means in any compiler — leave the source framework's native format
  behind for a portable one that downstream tools agree on, without
  altering what the model computes.

- **`llama-quantize` (f16 → Q8_0 → Q4_K_M) = an optimization pass.** Module
  9's second stage read the f16 GGUF and wrote new GGUFs with weights
  re-encoded at lower numeric precision. Nothing about the model's
  *architecture* changed — same tensors, same shapes, same metadata — only
  the numeric representation each weight is stored and computed in. That's
  precisely what an optimization pass is in any compiler: a transformation
  applied to the IR that changes performance characteristics (here, file
  size and CPU inference speed) while preserving the computation it
  represents, at a small, controlled quality cost.

- **llama.cpp/Ollama picking CPU kernels at load = codegen + runtime.**
  When `llama.cpp` (or Ollama, wrapping it) loads a `.gguf` file, it
  doesn't just replay the file — it selects concrete, hardware-suited
  kernels for the machine it's actually running on and executes the
  forward pass token by token. That selection-and-execution step is
  codegen (choosing the concrete instructions for this hardware) plus
  runtime (actually running them), rolled into one load-and-go experience.

### What we observed / where it generalizes

Nothing new was run for this Module — "what we observed" here is a second
look at Module 9's own results, through the compiler lens instead of the
GGUF-specific one. The three real files this project produced
(`tinystories-slm-f16.gguf` at 27MB, `-Q8_0.gguf` at 15MB, `-Q4_K_M.gguf`
at 11MB) are three optimization-pass outputs derived from the same
exported IR; the coherent stories `ollama run` produced from all three are
proof the optimization pass preserved the computation while changing its
cost profile — exactly what "an optimization pass, not a rewrite" is
supposed to mean.

The reason this three-stage shape is worth memorizing is that it isn't
specific to `llama.cpp` at all. **ONNX Runtime** exports models to the ONNX
IR, runs graph-optimization passes (operator fusion, constant folding,
precision conversion) over that IR, then executes with hardware-specific
"execution providers." **TVM** and **MLIR/IREE** are, structurally,
compiler frameworks doing the same job more generally: import a model into
an IR, run a sequence of lowering/optimization passes, generate code for a
specific backend. **TensorRT** and **OpenVINO** are vendor-specific
versions of the same shape, tuned for NVIDIA and Intel hardware
respectively. **MLC-LLM** runs this exact shape specifically for large
language models across a wide range of target devices. And vendor
toolchains you'll meet in your own day job — Tenstorrent's MLIR/Metalium
stack included — are, at this level of abstraction, one more instance of
the same three moves: export to an IR, lower/optimize that IR, generate
and run hardware-specific code.

Once "convert a model for a target" clicks at this level of abstraction,
deploying a Hugging Face model onto custom hardware — any custom hardware
— stops looking like an unfamiliar, vendor-specific black box and starts
looking like a recipe you already know by heart: **export → IR →
lower/optimize → hardware engine.** Phase 4 of this project picks that
recipe back up and walks it against real hardware; this Module's only job
was to make sure you'd recognize the shape when you got there.

### Gotchas & design decisions

**This Module is the deliberate bridge, not a coincidence.** Every other
Module in this course ends by pointing forward to the next specific build
step. This one is different on purpose: it's the point in the course where
the specific tools — `llama.cpp`, Ollama, GGUF — step out of the way so
the *shape underneath them* can stand on its own. That shape is what's
meant to survive the trip from this project's CPU/Ollama finish line to a
completely different hardware stack — your day job's, MLIR/Metalium and
all — where none of Module 9's specific file formats or command-line tools
will be waiting for you, but the four-step recipe still applies.

**The mapping is a mental model, not a spec.** "Export to IR," "optimization
pass," and "codegen + runtime" are deliberately loose categories, not a
rigid three-stage standard every compiler literally implements with those
exact boundaries — some stacks fold quantization into export, some split
codegen and runtime into separate tools, some run dozens of optimization
passes instead of one. The value of the mental model isn't that every
toolchain matches it exactly; it's that it hands you a short list of
questions — what's the IR, what gets optimized, what actually executes —
that works as a first pass on *any* unfamiliar stack, including ones this
course never names.

**Resist the urge to over-claim what comes next.** It's tempting to read
"Tenstorrent's MLIR/Metalium toolchain" above and assume this course is
about to walk that toolchain the way Modules 1–9 walked `llama.cpp`. It
isn't — that's explicitly future work, not this course's job, and nothing
here should be read as a claim about specifically how that later work will
go. What this Module hands off is the concept, deliberately, ahead of the
concrete tools.

### Checkpoint

1. Map each of the three GGUF-path stages to its model-compiler role: which
   Module 9 tool is the "export to IR" step, which is the "optimization
   pass," and which is "codegen + runtime"?
2. Why is `llama-quantize` an "optimization pass" rather than a format
   change? What stays the same about the model, and what changes?
3. Name at least two other inference stacks (besides `llama.cpp`) that
   follow this same three-stage shape.

**Explain it back:** state the four-step mental model for deploying a
model onto new hardware, in your own words, without naming GGUF,
`llama.cpp`, or Ollama once.

## Glossary

**Attention / causal mask** — Each token produces a Query ("what am I
looking for"), a Key ("what I offer"), and a Value ("what I contain").
Attention scores every token's Query against every other token's Key,
turns those scores into weights with softmax, then blends Values by those
weights — the only place tokens exchange information. The causal mask
forces a token to attend only to itself and earlier tokens, never future
ones, since future tokens don't exist yet at generation time; Module 4
verifies this directly by showing that perturbing only the last token of a
sequence leaves every earlier position's output unchanged.

**Autoregressive generation** — Generating text one token at a time: run
the model, sample a next token from its output distribution, append it to
the sequence, and repeat with the now-longer sequence. Each new token
becomes part of the input used to predict the next one. Module 5 walks the
actual generation loop this idea compiles down to.

**Batch** — Multiple independent training examples processed together in
one forward pass, shaped `(batch_size, context_len)`, so many sequences get
processed in parallel per gradient step instead of one at a time. Module 3
builds the function that assembles a batch by drawing random starting
positions from the token stream.

**BPE (byte-pair encoding)** — The training algorithm behind this project's
tokenizer: start from raw bytes, then repeatedly merge the most frequent
adjacent pair into one new token, until the vocabulary reaches its target
size. Common chunks end up as single tokens; rare stuff stays as smaller
pieces, so nothing is ever "unrepresentable." Module 2 watches merges form
as the vocabulary size grows.

**Byte-level pre-tokenization / the `Ġ` symbol** — Operating on raw UTF-8
bytes instead of Unicode characters means any input is representable, with
no "unknown token" ever needed. A side effect: the space byte gets
remapped to the visible symbol `Ġ` in printed tokens (a GPT-2-era
convention), so `Ġcat` means "a space followed by `cat`" — a different
token from a bare `cat` appearing mid-word. Module 2 covers this in
detail.

**Checkpoint** — A saved snapshot of a model: its learned weights plus the
config needed to rebuild its architecture, so training can be resumed or
the model loaded for inference later. Storing the config as a plain dict,
rather than a pickled object, means the file can be loaded safely without
ever unpickling arbitrary code from a downloaded file. Module 6 introduces
saving and loading a checkpoint; Module 7 loads one that was trained on a
different device than it's read on.

**`config.json`** — The plain-text spec that tells any runtime how to
interpret a model's weights: hidden size, layer count, head count,
vocabulary size, normalization epsilon, RoPE's angle constant, and the
special-token ids. Getting the special-token ids right matters in
particular — declaring the right end-of-text token as the stop signal is
what makes generation actually stop. Module 8 covers writing this file as
part of packaging to Hugging Face format.

**Context window / sequence length (`context_len`)** — How many tokens the
model looks at, and predicts within, in one forward pass. A longer context
window lets the model use more preceding text to predict the next token,
but attention's cost grows with sequence length, so it isn't free to make
arbitrarily long. Module 3 introduces it as part of the batching function.

**`d_model` (hidden size)** — The length of the vector used to represent
one token as it flows through the network: every internal representation —
attention output, feed-forward output, everything — is a vector of exactly
this length. A bigger `d_model` means richer per-token representations, but
more parameters everywhere a token vector is touched. The Warm-Up
introduces the `(batch, seq, d_model)` shape convention this rests on;
Module 4 builds the pieces that operate on it.

**Device (CPU vs GPU) / device portability** — Tensors and a model live on
a specific device — the CPU or a CUDA GPU — and an operation needs all its
inputs on the same device. A GPU does the many small matrix multiplies of
training massively in parallel, so a real training run is
minutes-per-epoch on a GPU versus hours-plus on CPU. "Device-portable"
code runs unchanged on either, which is exactly what lets the same
training function train on a laptop's CPU and a Colab GPU with no edits —
Module 7 walks the exact lines that make that true.

**Embedding table** — The matrix bridging token ids and token vectors,
shaped `(vocab_size, d_model)`, with one row per possible token id. Row *i*
is the vector used to represent token id *i* — a plain lookup, always the
same row for the same id, with no idea what surrounds it at lookup time.
Module 5 assembles it into the full model and shows what happens when it
doubles as the output projection too.

**GGUF (container format)** — The single-file format `llama.cpp`/Ollama
consume: a self-describing binary holding a header, all the tensors
(weights), and metadata key-values (architecture, hyperparameters,
tokenizer). "Self-describing" means the runtime reads the file itself to
learn how to run the model, with no separate config file needed. Module 9
reads a real GGUF header byte by byte to make that concrete.

**Gradient clipping** — A safety cap on the total magnitude of the gradient
each training step. If one freak batch produces an enormous gradient,
clipping scales it down before the optimizer step, preventing a single
update from yanking the weights off a cliff. Module 6 covers it alongside
the rest of the training loop.

**HF format** — The standard container the whole open-source LLM ecosystem
understands: a directory holding `config.json` (the architecture spec),
`model.safetensors` (the weights), and tokenizer files. Repackaging a
hand-built model into this format is not a rewrite — it's a straight
weight copy into the shipping container every downstream tool already
knows how to open. Module 8 walks the actual repackaging step.

**How the embedding table learns despite being context-blind** — The
embedding table's *lookup* never looks at context, but the *values* in
each row are still shaped by it: during training, a token's row feeds into
attention and the feed-forward block, contributes to the loss, and
backpropagation sends a context-shaped gradient back into that exact row.
The same token id appears in countless different contexts across the
training data, so its row settles into a compromise that works reasonably
well everywhere it appears — disambiguating meaning *by* context isn't the
embedding table's job at all, that happens downstream in attention (Module
4), which combines this fixed starting vector with whatever tokens are
actually nearby.

**Inference engine (`llama.cpp` / Ollama)** — The runtime that actually
executes a model: it loads a GGUF file, picks CPU kernels suited to the
machine it's running on, and runs the forward pass token by token.
`llama.cpp` is the engine itself; Ollama wraps it with model management and
a simple one-line run command. Module 9 walks both.

**Learning-rate schedule (warmup + cosine decay)** — The learning rate
isn't held constant during training. It ramps up linearly from near-zero
over a fixed number of warmup steps, so early, wild gradients don't blow
up freshly random weights, then decays smoothly along a cosine curve back
toward zero, so late training takes small, careful steps to settle. Module
6 derives and walks the exact function that computes it.

**Lockfile (`uv.lock`)** — The exact, fully-resolved set of package
versions — down to the specific build — that satisfies a project's looser
dependency constraints. The dependency file expresses *intent* (a version
range); the lockfile is the *reproducible result*, so anyone syncing
against the same lockfile gets byte-identical dependency versions. Module
1 covers the distinction in full.

**Logits** — The raw, unnormalized scores a model outputs for each
possible next token: one number per vocabulary entry, before softmax turns
them into a probability distribution. Module 5 shows the model producing
logits of shape `(batch, context_len, vocab_size)`.

**Loss / cross-entropy** — The single number measuring how wrong a model's
predictions are: it compares the predicted probability distribution
against the actual next token, penalizing confident-and-wrong answers far
more than merely uncertain ones. Lower is better; a model guessing
uniformly over the vocabulary gives a known, computable baseline loss,
which is why an untrained model with the tied-embedding "echo" bias
actually starts *above* that baseline — confidently wrong, not just
uncertain, as Module 6 measures with real numbers.

**Model card (`README.md`)** — The human-facing documentation shipped
alongside a model on the Hugging Face Hub: what the model is, its training
data, its config, its intended use, and its limitations. Uploading a
model's weights doesn't upload this file automatically — it has to be
pushed explicitly, as Module 8 shows.

**Model compiler / deep-learning compiler** — The mental model that names
the general shape of "convert a trained model to run somewhere new": export
the model into a portable intermediate representation, run one or more
optimization passes over that representation, then generate and execute
hardware-specific code. Module 9's GGUF pipeline is one concrete instance
of this shape; Module 10 is the Module that names it explicitly and shows
where else the same shape appears.

**Modelfile** — Ollama's recipe for packaging a model: a `FROM` line
naming the GGUF file, plus inference defaults (temperature, top-k, a stop
token) and a prompt template. Registering one is what makes a one-line
`ollama run <name>` command work. Module 9 walks a real one line by line.

**Next-token target (the shift-by-one)** — Language modeling's core
training signal: for an input window of tokens, the target is the *same
window shifted one position left*, so at every position the target holds
whatever token actually came next. One input sequence therefore yields a
supervised training example at every position simultaneously. Module 3
confirms this directly by decoding a real input/target pair and showing
one is the other, slid forward by exactly one token.

**Optimizer / AdamW** — The algorithm that actually updates each parameter
using its gradient. AdamW adapts the step size per-parameter using running
averages of recent gradients, and applies weight decay to gently pull
weights toward zero as a form of regularization — the standard default
choice for training transformers. Module 6 wires it into the training
loop.

**Overfitting one batch** — A debugging technique: deliberately train on a
single fixed batch over and over until the loss drops to nearly zero. It's
the fastest proof that learning works at all — if a model can't even
memorize one batch, something in the gradient flow, loss computation, or
wiring is broken, and it's only worth spending time on a real dataset once
this passes. Module 6 walks the test that checks it.

**Parameter** — A single learnable number inside a model. "14 million
parameters" means 14 million such numbers, each nudged a little on every
training step; Module 7 states the real, computed parameter count for this
project's trained model.

**Pre-norm residual block** — The repeating pattern inside a transformer:
`x = x + attn(norm(x))`, then `x = x + mlp(norm(x))`. The residual (`x = x
+ ...`) means each sub-layer only has to learn a correction to add on top
of its input rather than reconstruct the whole representation from
scratch, which is also why the original input's direction persists
strongly through a deep stack. "Pre-norm" means normalization happens
before each sub-layer rather than after, which trains more stably at depth
than the original design; Module 5 assembles the real block and traces
exactly why that persistence matters.

**Q8_0 vs Q4_K_M** — Two quantization schemes with different trade-offs.
Q8_0 is simple 8-bit quantization, near-lossless; Q4_K_M is a roughly
4-bit "K-quant" that mixes precisions across a tensor — giving sensitive
weights more bits — to stay coherent at about half the file size. Module 9
measures the real file sizes and bits-per-weight each one produced.

**Quantization** — Storing a model's weights at lower numeric precision to
shrink its file size and speed up CPU inference, trading away a small,
controllable amount of quality. Because quantization perturbs the
resulting logits slightly, the same prompt and random seed can produce
slightly different generated text at each quantization level. Module 9
runs this project's model at three precision levels and compares the
output.

**RMSNorm** — A normalization layer that rescales each token's vector by
its own root-mean-square magnitude, keeping numbers in a healthy range as
they flow through many stacked layers (left unnormalized, magnitudes drift
and destabilize training). It's a cheaper cousin of LayerNorm — it skips
mean-subtraction and only rescales — which is why Llama and most modern
LLMs use it. Module 4 builds it, and the Warm-Up supplies the underlying
arithmetic.

**RoPE (rotary positional encoding)** — Injects word-order information
into attention by *rotating* each token's query and key vector by an angle
proportional to its position, with later tokens rotating further. Rotation
preserves a vector's length — it never stretches or shrinks it, only spins
it — and the payoff is that the relationship between a rotated query and
key ends up depending only on their *relative* distance, not their
absolute positions. Module 4 builds it and works a numeric example that
confirms the relative-distance property directly.

**Round-trip equivalence** — The test that a hand-built model and the
official reference implementation produce *identical* logits for the same
input, down to ordinary floating-point noise. It's the proof that a
from-scratch architecture is genuinely the real thing, not merely "close"
— and everything downstream (publishing, quantized export, running in
other tools) depends on it holding. Module 8 covers the actual test and
its measured result.

**Safetensors** — A safe, fast weights-file format: just tensors plus
metadata, memory-mappable, and — unlike a pickled file — incapable of
executing arbitrary code on load. It's the modern default for
distributing model weights; Module 8 uses it as part of packaging to
Hugging Face format.

**Special token (`<|endoftext|>`)** — A token that isn't natural text —
it's inserted deliberately, here between documents, so the model can learn
"this text ended, a new one begins." Module 2 covers why this token is
fixed to id `0` by convention, and Modules 8 and 9 both rely on that fixed
id downstream to make generation stop in the right place.

**SwiGLU / feed-forward** — The per-token "private thinking" step, applied
right after attention lets tokens exchange information. It expands each
token's vector into a wider workspace, applies a gated nonlinearity, then
projects back down to the model's normal width — with no cross-token
interaction happening here at all, purely per-token processing. Module 4
builds it.

**Temperature / top-k sampling** — Two knobs that reshape a next-token
probability distribution before sampling. Temperature divides logits
before softmax: a low temperature sharpens the distribution (more
confident, more repetitive), a high temperature flattens it (more random);
top-k zeroes out every token outside the `k` most likely ones, so very
unlikely tokens can never be picked regardless of temperature. Module 6
isolates both knobs on a fixed set of logits to show the effect
numerically.

**Tied-embedding "echo" bias at initialization** — A surprising, real
property of tied-embedding, pre-norm-residual transformers: *before any
training at all*, the model strongly favors predicting the most recently
seen token again. The cause is a two-step chain — residual connections
keep the input token's embedding direction dominant in the final hidden
state, and because the output projection reuses that same embedding matrix
(weight tying), that token's self-dot-product logit vastly outscores every
other token's cross-dot-product logit, saturating softmax to a probability
near 1.0. It disappears entirely once the embeddings are untied; Module
5's deep dive walks the full mechanism with real, measured numbers.

**Token / token id** — A token is one "chunk" a tokenizer produces — a
whole word, part of a word, or a single character or byte; the token id is
the integer that represents it in the model's vocabulary. Module 2 covers
how these are produced and shows real examples.

**Tokenizer** — The function that converts text into a list of integers,
and back again. Neural networks only operate on numbers, so this is the
mandatory first translation step in any language-model pipeline. Module 2
builds one from scratch.

**Virtual environment (venv)** — An isolated Python installation and
package directory scoped to one project, so its dependencies can't
collide with other projects or the system Python. Module 1 covers
creating and using one.

**Vocabulary** — The full set of tokens a tokenizer knows, each mapped to
a unique id. Its size is a real architecture cost: vocabulary size times
hidden size is the size of the model's embedding table, so a bigger
vocabulary means a bigger model at the same hidden size. Module 2 covers
training a vocabulary to a specific target size.

**Weight tying (input/output embeddings)** — Using the *same* matrix for a
model's input embedding table and its output projection. This halves the
model's single largest parameter block and ties "the meaning of a token
going in" to "the score for that token coming out" — it's also the root
cause of the untrained echo bias above. Module 5 introduces the design
choice; Module 8 shows how Hugging Face format records the same fact in
`config.json`.
