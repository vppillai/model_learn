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
