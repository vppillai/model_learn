import csv
import math
import os
from dataclasses import dataclass, asdict
import torch
import torch.nn.functional as F
from slm.config import ModelConfig
from slm.model import LlamaSLM
from slm.data import get_batch


@dataclass
class TrainConfig:
    lr: float = 3e-4
    warmup_steps: int = 100
    max_steps: int = 2000
    batch_size: int = 32
    context_len: int = 128
    grad_clip: float = 1.0
    weight_decay: float = 0.1
    log_every: int = 50
    sample_every: int = 500
    ckpt_path: str = "checkpoints/model.pt"
    seed: int = 0
    fixed_batch: bool = False  # True => always sample the same batch (overfit test)


def lr_at(step: int, cfg: TrainConfig) -> float:
    """Linear warmup to the peak lr at step == warmup_steps, then cosine decay."""
    if step < cfg.warmup_steps:
        return cfg.lr * step / cfg.warmup_steps
    if step == cfg.warmup_steps:
        return cfg.lr
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    return 0.5 * cfg.lr * (1 + math.cos(math.pi * min(1.0, progress)))


def train(model: LlamaSLM, data: list[int], cfg: TrainConfig, tok=None):
    torch.manual_seed(cfg.seed)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay, betas=(0.9, 0.95))
    history: list[tuple[int, float]] = []
    rows: list[tuple[int, float]] = []
    model.train()
    # Batches are built on CPU (get_batch uses a CPU RNG for determinism);
    # move each to the model's device so the same code runs on CPU or GPU.
    device = next(model.parameters()).device
    # Tensorize the token stream ONCE (not per-step): at Colab scale the stream
    # is tens of millions of tokens and re-copying it every call would dominate.
    data_t = data if isinstance(data, torch.Tensor) else torch.tensor(data, dtype=torch.long)
    for step in range(cfg.max_steps + 1):
        seed = cfg.seed if cfg.fixed_batch else cfg.seed + step
        x, y = get_batch(data_t, cfg.batch_size, cfg.context_len, seed=seed)
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        for g in opt.param_groups:
            g["lr"] = lr_at(step, cfg)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        if step % cfg.log_every == 0:
            history.append((step, loss.item()))
            rows.append((step, loss.item()))
            print(f"step {step:>5} | loss {loss.item():.4f} | lr {lr_at(step, cfg):.2e}")
        if tok is not None and cfg.sample_every and step % cfg.sample_every == 0 and step > 0:
            _print_sample(model, tok)
    save_checkpoint(model, model.cfg, cfg.ckpt_path)
    _write_csv(rows, cfg.ckpt_path.replace(".pt", "_loss.csv"))
    return history


def _print_sample(model, tok, n=40):
    from slm.tokenizer import decode
    model.eval()
    device = next(model.parameters()).device
    start = torch.zeros((1, 1), dtype=torch.long, device=device)
    out = model.generate(start, max_new_tokens=n, temperature=0.8, top_k=40)
    print("  sample:", decode(tok, out[0].tolist()).replace("\n", " ")[:160])
    model.train()


def _write_csv(rows, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "loss"])
        w.writerows(rows)


def save_checkpoint(model, model_cfg: ModelConfig, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # Store config as a plain dict (not a pickled dataclass) so the checkpoint
    # can be loaded with weights_only=True — never unpickle arbitrary objects
    # from a downloaded model file (that is an arbitrary-code-execution vector).
    torch.save({"state_dict": model.state_dict(), "config": asdict(model_cfg)}, path)


def load_checkpoint(path: str):
    # map_location="cpu": a checkpoint trained on GPU (e.g. Colab) records CUDA
    # tensor locations; without remapping it fails to load on a CPU-only box.
    # Inference/packaging here is always CPU; callers can .to(device) after.
    ckpt = torch.load(path, weights_only=True, map_location="cpu")  # tensors + plain containers only
    cfg = ModelConfig(**ckpt["config"])
    model = LlamaSLM(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, cfg


def plot_loss(csv_path: str, out_png: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    steps, losses = [], []
    with open(csv_path) as f:
        next(f)
        for line in f:
            s, l = line.strip().split(",")
            steps.append(int(s)); losses.append(float(l))
    plt.figure(); plt.plot(steps, losses); plt.xlabel("step"); plt.ylabel("loss")
    plt.title("Training loss"); plt.savefig(out_png, dpi=120)
    print("wrote", out_png)
