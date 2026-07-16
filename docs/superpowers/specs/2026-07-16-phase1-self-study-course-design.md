# Phase 1 Self-Study Course — Design Spec

**Date:** 2026-07-16
**Status:** Approved (design), pending implementation plan
**Topic:** A single, self-contained tutoring document for NotebookLM that lets vpillai
re-learn everything Phase 1 covered — this time actively, not by watching code get written.

## Motivation

Phase 1 (hand-build an SLM → train → export to HF → GGUF → Ollama) is complete and
proven. But the *learning* was front-loaded into Claude writing the code while vpillai
executed the steps that needed a human (Colab runs, tool installs). The concepts were
framed in the plan and captured in `CONCEPTS.md`/`DEVLOG.md`, but never delivered as the
guided, predict→observe→explain-back loop the plan's Execution Protocol designed.

This project reclaims that learning by producing **one master document** engineered as a
**structured self-study course with annotated code**, to be fed into **NotebookLM** as a
single source. vpillai then uses NotebookLM as a tutor (chat, quizzes, audio overview),
returns with findings, and we update the docs. This spec covers only the authoring of that
document; the NotebookLM session and any follow-up doc revisions are downstream.

## The governing constraint: NotebookLM grounds answers only in its sources

NotebookLM cannot see the git repo, cannot run the labs, and cannot resolve a path
reference like "see Lab 03" or "DEVLOG 2026-07-07." It answers strictly from the text of
uploaded sources. Three rules follow and shape every decision below:

1. **No dangling references.** Every cross-reference in the existing docs is *inlined* — the
   concept, the code it refers to, and the output it produced all live together in the same
   section. The course must read correctly to someone who has never seen the repo.
2. **Heading hierarchy is the navigation.** NotebookLM builds its internal structure from
   headings. A clean table of contents and consistent nesting is what lets it answer
   "tutor me on the training module" accurately.
3. **Observed reality is embedded.** Actual outputs — the untrained gibberish sample, loss
   62→~4.3 on the toy run, eval perplexity 6.1 on `small`, round-trip max logit diff
   9.5e-06, GGUF sizes f16 27MB / Q8_0 15MB / Q4_K_M 11MB — appear in the prose so the
   tutor can ground "what actually happened," not just theory.

## Decisions (from brainstorming)

- **Packaging:** ONE combined master document (not multiple purpose-built docs, not raw repo
  files). Simplest to upload; NotebookLM handles long single sources well.
- **Tutoring style:** Structured self-study course — learning objectives, an ordered path
  following the build, and per-module checkpoints. (Lightweight quizzes are folded into each
  module as course checkpoints; there is no separate Q&A bank.)
- **Code:** Embedded, annotated walkthroughs of the real `src/slm/*.py` — line/section-level
  commentary, not a raw appendix and not concepts-only.
- **Structural approach:** Hybrid (chosen over pure pipeline-order and pure conceptual-layers)
  — pipeline/build order as the spine, one rigorous template per module, "deep dive" callouts
  for the genuinely hard bits (e.g. the echo-bias investigation, RoPE rotation intuition).
- **Level:** Intuition-first, no heavy math — consistent with the project's learning profile
  and the existing `CONCEPTS.md` register.

## Deliverable

A single Markdown file at **`docs/phase1-course.md`** (versioned in-repo). Expected length
~10–15k words — well within NotebookLM's 500k-word/source limit. Upload path: the `.md`
directly (supported), or paste-as-text / export to PDF or a Google Doc if preferred.

### Front matter
- **Course overview:** the big-picture map ("the ecosystem lets you download→build→run a
  model; here we *build one* end to end"), and the reverse pipeline we traverse.
- **How to use this in NotebookLM:** suggested prompts (e.g. *"quiz me on Module 4,"* *"act as
  a tutor and check my understanding of RoPE,"* *"give me the audio overview of the training
  module"*), and a note that answers are grounded in this document.
- **Level & prerequisites:** intuition-first, Python literacy assumed, no heavy math.
- **Table of contents.**

### Module template (the reusable unit)
Every module contains, in this order:
1. **Learning objectives** — what the reader will be able to explain afterward.
2. **Frame** — plain-language intuition + analogy; the "why it exists." Sourced from the
   plan's per-task Frame text and `CONCEPTS.md`.
3. **Annotated code walkthrough** — the real source embedded, with inline commentary.
4. **What we observed** — the actual milestone output, inlined.
5. **Gotchas & design decisions** — the real detours from `DEVLOG.md`, and *why* each choice
   was made.
6. **Checkpoint** — 2–4 mini-quiz questions plus one "explain it back in your own words"
   prompt (delivering the Execution Protocol's step 5).

### Module list (maps 1:1 to Phase 1 tasks)
- **0 — Overview** (front matter above).
- **1 — Environment & reproducibility:** uv, venv, lockfile; the CPU-torch-pin gotcha and the
  Colab `requirements.txt` leak.
- **2 — Tokenizer & BPE:** train/load/encode/decode, special tokens, byte-level/`Ġ` (Lab 01).
- **3 — Data pipeline:** the token stream, batching, and the shift-by-one next-token target.
- **4 — Model internals:** RMSNorm, RoPE, causal attention, SwiGLU (Lab 02).
- **5 — Assembly + generation** + **deep dive: the untrained "echo bias"** (Lab 03).
- **6 — Training:** loss/cross-entropy, AdamW, warmup+cosine LR, grad clip, overfit-one-batch,
  the toy run (Lab 05).
- **7 — The real run:** `small` (13.8M) on Colab, device portability, eval ppl 6.1.
- **8 — Packaging to HF format** + **round-trip equivalence** (the correctness linchpin).
- **9 — GGUF, quantization, llama.cpp, Ollama** (Labs 07/08).
- **10 — The transferable mental model:** the GGUF path as a miniature model compiler
  (export→IR→lower/optimize→hardware engine) and the Tenstorrent MLIR/Metalium tie-in.
- **Glossary** — the ~40 `CONCEPTS.md` entries, deduplicated, for quick lookup.

## Source material (all already in-repo)

- `CONCEPTS.md` — ~40 plain-language concept entries (primary source for Frame + Glossary).
- `DEVLOG.md` — dated narrative + real gotchas (primary source for "What we observed" and
  "Gotchas & decisions").
- `docs/superpowers/plans/2026-06-22-slm-phase1-base-model.md` — the per-task Frame text and
  the canonical code (primary source for the annotated walkthroughs).
- `src/slm/*.py` — the actual current implementation (authoritative for embedded code; the
  plan's snippets must be reconciled against the real files before embedding).
- `labs/*.py` and their expected outputs — the observable demonstrations.
- `REPRODUCE.md` — the follow-along commands.

## Success criteria

1. A reader with no repo access can follow the course top-to-bottom and understand the full
   Phase 1 arc — every concept, its code, and its observed result are self-contained.
2. Every module has all six template sections; no section references an external file by path
   without inlining what it points to.
3. Embedded code matches the *current* `src/slm/*.py` (reconciled, not copied from the plan's
   pre-implementation snippets — e.g. the `lr_at` clean version, not the convoluted draft).
4. NotebookLM can answer module-scoped tutoring prompts and run the checkpoint quizzes using
   only this document.
5. Numbers cited (params, loss, ppl, diffs, file sizes) match `DEVLOG.md`/`CHANGELOG.md`.

## Out of scope

- The NotebookLM session itself and any doc revisions that come out of it (a later cycle).
- Phase 2/3/4 content — this course covers Phase 1 only.
- Changing any `src/`, tests, or existing docs — this task only *adds* `docs/phase1-course.md`.
  (If reconciliation surfaces a genuine error in an existing doc, note it; don't silently edit.)
- Audio/PDF generation — output is Markdown; conversion is the reader's choice.
