# Reproduce This From Scratch

Follow-along guide (distilled from DEVLOG). Build the whole thing yourself.

## 0. Environment (uv)
```bash
# install uv if needed: curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv && uv sync          # creates .venv, installs deps from uv.lock
source .venv/bin/activate   # or prefix commands with `uv run`
```
Note: `pyproject.toml` pins `torch` to the CPU wheel index
(`download.pytorch.org/whl/cpu`) since local dev here has no GPU. If you
have a CUDA GPU locally, remove the `[tool.uv.sources]`/`[[tool.uv.index]]`
block so `torch` resolves to the default (CUDA-enabled) wheel instead.

`requirements.txt` (for Colab / non-uv environments) intentionally omits
`torch` — Colab already provides a GPU-matched torch build; installing the
CPU pin from this project would overwrite it.

## Milestone: configs
```bash
uv run pytest tests/test_config.py -v
PYTHONPATH=src uv run python -c "from slm.config import TOY, SMALL; print(TOY.n_params(), SMALL.n_params())"
```

## Milestone: tokenizer
```bash
uv run pytest tests/test_tokenizer.py -v
PYTHONPATH=src uv run python labs/lab01_bpe_by_hand.py
```
Watch the `labs/lab01_bpe_by_hand.py` output: as `vocab_size` grows, common
chunks like `"the "`/`"cat"` collapse from multiple tokens into one.

## Milestone: real batches
```bash
uv run pytest tests/test_data.py -v
PYTHONPATH=src uv run python -c "
from slm.tokenizer import train_tokenizer
from slm.data import tokenize_texts, get_batch
lines = open('tests/fixtures/tiny_stories.txt').read().splitlines()
tok = train_tokenizer(lines * 20, vocab_size=300, save_path='/tmp/task3_tok.json')
stream = tokenize_texts(tok, lines)
x, y = get_batch(stream, batch_size=2, context_len=12, seed=0)
print(tok.decode(x[0].tolist()))
print(tok.decode(y[0].tolist()))
"
```
The two printed lines are the same text, `y` slid forward by one token.
