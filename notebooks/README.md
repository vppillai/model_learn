# Training the `small` model on Google Colab

The local `toy` run (Task 5) proved the pipeline works on CPU in ~2 minutes.
The `small` model (~14M params) needs a GPU to train in reasonable time — this
is what produces genuinely coherent little stories. We use Colab's free T4.

The notebook runs the **exact same** `src/slm` code as the local run; only the
config (`SMALL` vs `TOY`) and the compute (GPU vs CPU) differ.

## Prerequisite: the repo must be reachable by Colab

Colab's first cell does `git clone`, so the repo needs to live on a remote.
This is already set up: the repo is public at
**https://github.com/vppillai/model_learn**, and the notebook's `REPO_URL` is
pre-filled to it. Nothing to do here for the first run.

If you make more local commits before running, push them so Colab clones the
latest:

```bash
git push
```

**No-git fallback:** if you'd rather not use the remote, upload the `src/`
folder and `requirements.txt` to the Colab session (Files panel → upload),
skip the clone cell, and `pip install -r requirements.txt`.

## Steps

1. Open `notebooks/colab_train.ipynb` in Colab
   (colab.research.google.com → File → Open notebook → GitHub, or upload it).
2. **Runtime → Change runtime type → T4 GPU.** (The train cell asserts a GPU
   is present and stops early if not.)
3. Edit `REPO_URL` in cell 1.
4. Run cells top to bottom. The training cell prints a loss line every 100
   steps and a generated sample every 1000 — watch the stories get coherent.
5. Run the download cell to save `small.pt`, `small_tok.json`, and
   `small_loss.png` into your local `checkpoints/` directory.

## Troubleshooting

- **`ImportError: cannot import name '_center' from 'numpy._core.umath'`** (or
  similar numpy/pandas binary errors): something upgraded numpy/pandas inside
  the running kernel. Cell 1 installs only `datasets tokenizers` precisely to
  avoid this — but if you hit it, do **Runtime → Restart session** and run from
  the top. Do *not* `pip install -r requirements.txt` on Colab: that pins the
  full transitive tree and clobbers Colab's co-tuned numpy/pandas/torch.
- **`CUDA available: False`**: Runtime → Change runtime type → T4 GPU, then
  restart and rerun.

## Config notes (finalized here per spec §16)

- `limit=200_000` stories (~47M tokens). The current data pipeline holds the
  token stream in memory; the **full** dataset as a Python list would exceed
  Colab's ~12GB RAM. 200k stories is ample for coherent output from a 14M
  model and keeps the run comfortably within a free session.
- `max_steps=20000, batch_size=64, lr=6e-4, warmup_steps=200`. On a T4 this is
  roughly 30–60 min. **If your session risks timing out, lower `max_steps`**
  (10000 still gives decent results).
- fp32 (no mixed precision) — simple and fast enough at 14M params. AMP could
  speed it up further but adds complexity we don't need here.

## After the run

Record in `DEVLOG.md`: the actual `max_steps` used, GPU type, wall-clock time,
final loss, and a verbatim generated story. Then verify the downloaded
checkpoint locally on CPU:

```bash
PYTHONPATH=src uv run python -m slm.sample checkpoints/small.pt checkpoints/small_tok.json "Once upon a time"
```
