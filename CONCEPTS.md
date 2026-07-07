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
