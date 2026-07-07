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
