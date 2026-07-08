import torch
from slm.config import TOY
from slm.model import LlamaSLM
from slm.train import TrainConfig, lr_at, train


def test_lr_warmup_and_decay():
    cfg = TrainConfig(lr=1e-3, warmup_steps=10, max_steps=100)
    assert lr_at(0, cfg) < lr_at(5, cfg) < lr_at(10, cfg)   # warming up
    assert lr_at(10, cfg) == cfg.lr                          # peak at end of warmup
    assert lr_at(100, cfg) < lr_at(50, cfg)                  # decaying


def test_overfit_one_batch_drives_loss_down(tmp_path):
    torch.manual_seed(0)
    model = LlamaSLM(TOY)
    # one fixed tiny batch repeated => the model must memorize it
    data = list(range(0, 200))
    cfg = TrainConfig(lr=3e-3, warmup_steps=20, max_steps=400, batch_size=4,
                      log_every=100, sample_every=10_000,
                      ckpt_path=str(tmp_path / "ck.pt"), seed=0,
                      fixed_batch=True, context_len=16)
    history = train(model, data, cfg)
    first_loss = history[0][1]
    last_loss = history[-1][1]
    assert last_loss < first_loss * 0.3   # learning clearly happened
    assert last_loss < 1.0                 # nearly memorized the batch
