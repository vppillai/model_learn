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
