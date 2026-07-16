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
