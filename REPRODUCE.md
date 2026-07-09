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

## Milestone: learning works (overfit one batch)
```bash
uv run pytest tests/test_train.py -v
```
`test_overfit_one_batch_drives_loss_down` trains on one fixed batch and
asserts loss collapses toward 0 — the proof that gradient flow works.

## Milestone: toy training run (local CPU, ~2 min)
```bash
PYTHONPATH=src uv run python - <<'PY'
import torch
from slm.config import TOY
from slm.model import LlamaSLM
from slm.tokenizer import train_tokenizer
from slm.data import tokenize_texts, load_tinystories
from slm.train import TrainConfig, train, plot_loss

texts = load_tinystories("train", limit=2000)          # streams; no full download
tok = train_tokenizer(texts, vocab_size=TOY.vocab_size, save_path="checkpoints/toy_tok.json")
data = tokenize_texts(tok, texts)
model = LlamaSLM(TOY)
cfg = TrainConfig(lr=3e-3, warmup_steps=50, max_steps=800, batch_size=32,
                  context_len=TOY.context_len, log_every=50, sample_every=200,
                  ckpt_path="checkpoints/toy.pt")
train(model, data, cfg, tok=tok)
plot_loss("checkpoints/toy_loss.csv", "checkpoints/toy_loss.png")
PY
```
Expected: loss falls ~62 → ~4.3; printed samples drift from gibberish toward
word-like, then story-shaped fragments. Generate from the checkpoint:
```bash
PYTHONPATH=src uv run python -m slm.sample checkpoints/toy.pt checkpoints/toy_tok.json "Once upon a time"
```
(Toy quality stays rough — coherence comes with the `small` Colab run.)

## Milestone: coherent stories (`small` model on Colab GPU)
The `small` (~14M param) model trains on a GPU, not local CPU. Full
instructions: `notebooks/README.md`. In short:
1. Push this repo to a remote (GitHub) so Colab can clone it.
2. Open `notebooks/colab_train.ipynb` in Colab, set a T4 GPU, edit `REPO_URL`,
   run all cells (~30–60 min).
3. Download `small.pt`, `small_tok.json`, `small_loss.png` into `checkpoints/`.
4. Verify locally on CPU:
```bash
PYTHONPATH=src uv run python -m slm.sample checkpoints/small.pt checkpoints/small_tok.json "Once upon a time"
```
Expected: a short, mostly-coherent children's story. (Final loss and an
example story are recorded in `DEVLOG.md` after the run.)

## Milestone: published (export to HF format + push to Hub)
```bash
# 1. Verify the architecture is bit-for-bit Llama (round-trip test):
uv run pytest tests/test_export.py -v

# 2. Export the trained small model to standard HF format:
PYTHONPATH=src uv run python -c "from slm.export_hf import export_to_hf; export_to_hf('checkpoints/small.pt','checkpoints/small_tok.json','export/tinystories-slm')"

# 3. Confirm it loads with stock transformers (no custom code):
PYTHONPATH=src uv run python -c "
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast
m = LlamaForCausalLM.from_pretrained('export/tinystories-slm').eval()
t = PreTrainedTokenizerFast.from_pretrained('export/tinystories-slm')
ids = t('Once upon a time', return_tensors='pt').input_ids
print(t.decode(m.generate(ids, max_new_tokens=80, do_sample=True, top_k=40)[0], skip_special_tokens=True))
"

# 4. Push to the Hub (needs `uv run hf auth login` with a write token first):
PYTHONPATH=src uv run python -c "from slm.export_hf import push; push('export/tinystories-slm','<your-hf-username>/tinystories-slm', private=True)"
```
