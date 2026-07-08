# Concepts (plain-language, concept-on-contact)

Each entry: what it is, why it exists, and the Lab that shows it.

## parameter
A single learnable number in the model. "14M parameters" = 14 million such
numbers, adjusted during training. Lab: see `n_params()` in `src/slm/config.py`.

## virtual environment (venv)
An isolated Python installation + package directory scoped to one project,
so its dependencies (exact `torch`/`transformers` versions) can't collide
with other projects or the system Python. `uv venv` creates one at `.venv/`.

## lockfile (`uv.lock`)
The exact, fully-resolved set of package versions (down to the build, e.g.
`torch==2.12.1+cpu`) that satisfies `pyproject.toml`'s looser constraints
(e.g. `torch>=2.5`). `pyproject.toml` is the *intent*; `uv.lock` is the
*reproducible result* — anyone who runs `uv sync` against the same lockfile
gets byte-identical dependency versions.

## tokenizer
The function that converts text to a list of integers (and back). Neural
nets only operate on numbers, so this is the mandatory first translation
step. Lab: `labs/lab01_bpe_by_hand.py`.

## token / token id
A token is one "chunk" the tokenizer produces (could be a whole word, part
of a word, or a single character/byte); the token id is the integer that
represents it in the model's vocabulary. Lab: `labs/lab01_bpe_by_hand.py`.

## BPE (byte-pair encoding)
The training algorithm behind our tokenizer: start from raw bytes, then
repeatedly merge the most frequent adjacent pair into one new token, until
the vocabulary reaches the target size. Common chunks end up as single
tokens; rare stuff stays as smaller pieces — nothing is ever
"unrepresentable." Lab: `labs/lab01_bpe_by_hand.py` shows merges forming as
vocab size grows.

## vocabulary
The full set of tokens a tokenizer knows, each mapped to a unique id. Size
is a real architecture cost: `vocab_size × d_model` is the size of the
model's embedding table (see `n_params()`), so a bigger vocabulary means a
bigger model for the same `d_model`.

## special token (`<|endoftext|>`)
A token that isn't natural text — it's inserted by us as a marker, here
between documents/stories, so the model can learn "this text ended, a new
one begins." Fixed to id `0` by convention (see `EOT` in
`src/slm/tokenizer.py`).

## byte-level pre-tokenization / the `Ġ` symbol
Our tokenizer operates on raw UTF-8 bytes, not Unicode characters, so any
input is representable (no "unknown token" needed). A side effect: the
space byte gets remapped to the visible symbol `Ġ` in printed
tokens/vocabularies (a GPT-2-era convention) — `Ġcat` means "a token that is
a space followed by `cat`," a different token from a bare `cat` that
appears mid-word. Lab: `labs/lab01_bpe_by_hand.py`.

## d_model (hidden size)
The length of the vector used to represent one token as it flows through
the network — every internal representation (attention output,
feed-forward output, everything) is a vector of exactly this length. `TOY`
uses 64; `SMALL` uses 384. Bigger `d_model` means richer per-token
representations, but more parameters everywhere a token vector is touched
(embedding table, attention projections, feed-forward).

## embedding table
The matrix bridging token ids and token vectors: shape
`(vocab_size, d_model)`, one row per possible token id. Row *i* is the
vector used to represent token id *i* — a plain lookup, always the same row
for the same id. At inference time it has no idea what surrounds it.

## how the embedding table learns despite being context-blind
The *lookup* never looks at context, but the *values* in each row are
shaped by it. During training, a token's row feeds into attention/FFN,
contributes to a loss, and backpropagation sends a context-shaped gradient
back into that exact row. The same token id appears in countless different
contexts across the training data, so its row settles into a compromise
that works reasonably well everywhere it appears. Disambiguating meaning
*by* context is not the embedding table's job — that happens downstream in
attention, which combines this fixed starting vector with the tokens
actually around it. See `DEVLOG.md` (2026-07-07) for a live before/after
demo of one row changing after two gradient steps with conflicting targets.

## context window / sequence length (`context_len`)
How many tokens the model looks at (and predicts within) in one forward
pass. `TOY` uses 128; `SMALL` uses 512. Longer context lets the model use
more preceding text to predict the next token, but attention cost grows
with sequence length, so it's not free to make arbitrarily long.

## batch
Multiple independent training examples processed together in one forward
pass (shape `(batch_size, context_len)`), so the GPU/CPU processes many
sequences in parallel per gradient step instead of one at a time. Lab: see
`get_batch()` in `src/slm/data.py`.

## next-token target (the shift-by-one)
Language modeling's training signal: for input `x` (a window of tokens),
the target `y` is the *same window shifted one position left* — at every
position, `y` holds whatever token actually came next. One sequence yields
a supervised example at every position simultaneously. Confirmed directly
in `DEVLOG.md` (2026-07-07): decoding `x[0]` and `y[0]` from a real batch
gives the same text slid forward by exactly one token.

## RMSNorm
A normalization layer that rescales each token's vector by its
root-mean-square magnitude, keeping numbers in a healthy range as they flow
through many stacked layers (unnormalized, magnitudes can drift and
destabilize training). Cheaper cousin of LayerNorm — it skips
mean-subtraction and only rescales — which is why Llama and most modern
LLMs use it. Lab: `RMSNorm` in `src/slm/model.py`.

## RoPE (rotary positional encoding)
Injects word-order information into attention by *rotating* each token's
query/key vector by an angle proportional to its position — later tokens
rotate further. Rotation preserves a vector's length (never stretches or
shrinks it, only spins it), which is exactly what `test_rope_preserves_shape_and_norm`
checks. The payoff: the relationship between a rotated query and key
depends only on their *relative* distance, not their absolute positions.

## attention / causal mask
Each token produces a Query ("what am I looking for"), Key ("what I
offer"), and Value ("what I contain"). Attention scores every token's Query
against every other token's Key, then blends Values weighted by those
scores — the only place tokens exchange information. The causal mask
forces a token to only attend to itself and earlier tokens, never future
ones, since future tokens don't exist yet at generation time. Verified
directly: perturbing only the last token of a 6-token sequence left
positions 0-4's output bit-for-bit identical (see `DEVLOG.md` 2026-07-07).

## SwiGLU / feed-forward
The per-token "private thinking" step, applied right after attention lets
tokens exchange information. Expands each token's vector into a wider
workspace (`ffn_hidden`), applies a gated nonlinearity, then projects back
down to `d_model`. No cross-token interaction here — purely per-token
processing. This block is also the future Mixture-of-Experts "expert"
(Phase 3).

## tied-embedding "echo" bias at initialization
A surprising, real property of tied-embedding + pre-norm-residual
transformers: **before any training**, the model strongly favors predicting
the *most recently seen token again*. Cause: residual connections keep the
input token's embedding direction dominant in the final hidden state, and
because `lm_head` reuses that same embedding matrix (weight tying), that
token's self-dot-product logit vastly outscores every other token's
cross-dot-product logit (a ~30-40 logit gap, saturating softmax to
`max_prob ≈ 1.0`). Confirmed empirically across `TOY` and `SMALL`, multiple
seeds, and varied (non-repeated) inputs; disappears entirely when
`tie_embeddings=False`. Not a bug — a real architectural property that
training overwrites. See `DEVLOG.md` (2026-07-07) for the full
investigation. Lab: `labs/lab03_gibberish.py` shows it live — prompting
with `"Once upon a time"` degenerates into repeating `"time"` forever.

## logits
The raw, unnormalized scores the model outputs for each possible next
token — one number per vocabulary entry, before `softmax` turns them into a
probability distribution. `LlamaSLM.forward()` returns logits of shape
`(batch, context_len, vocab_size)`.

## pre-norm residual block
The `Block` pattern: `x = x + attn(norm(x)); x = x + mlp(norm(x))`. The
*residual* (`x = x + ...`) means each sub-layer only has to learn a
*correction* to add on top of its input, rather than reconstruct the whole
representation from scratch — this is also why the original input's
direction persists strongly through the stack (see the tied-embedding echo
bias above). "Pre-norm" means normalization happens *before* each sub-layer
rather than after, which is more stable to train at depth than the
original post-norm Transformer design.

## autoregressive generation
Generating text one token at a time: run the model, sample a next token
from its output distribution, append it to the sequence, repeat. Each new
token becomes part of the input for predicting the next one. Lab:
`generate()` in `src/slm/model.py`.

## temperature / top-k sampling
Two knobs that reshape the next-token probability distribution before
sampling. Temperature divides logits before `softmax`: low temperature
sharpens the distribution (more confident/repetitive), high temperature
flattens it (more random). Top-k zeroes out every token outside the `k`
most likely, preventing very unlikely tokens from ever being picked. Lab
05 (Task 5) explores both in more depth.
