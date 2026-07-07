# Developer Log

Dated, chronological record of what we did and why — including dead-ends.

## 2026-07-07 — Task 1: scaffold + configs

- Created package skeleton (`src/slm/`), `ModelConfig` dataclass with `TOY`
  (213,312 params) and `SMALL` (13,767,552 params) presets, pytest wiring
  (`pytest.ini` with `pythonpath = src`).
- `uv` 0.11.21 was already installed on this box; no install step needed.
- **Gotcha 1 — default `torch` wheel pulls in CUDA on a CPU-only box.**
  Plain `uv sync` resolved `torch==2.12.1+cu130` plus ~20 `nvidia-*` packages
  (~4.8GB venv total), even though this machine has no NVIDIA GPU
  (`torch.cuda.is_available()` was `False` regardless). PyPI's default Linux
  torch wheel always bundles CUDA runtime deps; there's no install-time
  auto-detection. Fixed by pinning torch to the CPU wheel index in
  `pyproject.toml`:
  ```toml
  [tool.uv.sources]
  torch = { index = "pytorch-cpu" }

  [[tool.uv.index]]
  name = "pytorch-cpu"
  url = "https://download.pytorch.org/whl/cpu"
  explicit = true
  ```
  Re-running `uv sync` after this dropped the venv to 1.1GB and installed
  `torch==2.12.1+cpu`.
- **Gotcha 2 — that same pin leaks into `requirements.txt` and would break
  Colab.** `uv export` propagates the CPU-only pin, so the naive
  `requirements.txt` said `torch==2.12.1+cpu` for all non-macOS platforms —
  installing it on Colab would replace Colab's preinstalled GPU-matched
  torch with a CPU build, silently killing GPU acceleration for the future
  `small` training run (Task 6). Fixed by excluding torch from the export:
  `uv export --no-hashes --format requirements-txt --no-emit-package torch -o requirements.txt`.
  Colab already ships a working GPU torch; the Colab notebook (Task 6) will
  note explicitly that `requirements.txt` intentionally omits torch.
- Working install commands (for reference / REPRODUCE.md):
  ```bash
  uv venv
  uv sync
  uv export --no-hashes --format requirements-txt --no-emit-package torch -o requirements.txt
  ```
