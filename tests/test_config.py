from slm.config import ModelConfig, TOY, SMALL


def test_toy_is_tiny_and_consistent():
    assert TOY.d_model == TOY.n_heads * TOY.head_dim
    assert TOY.n_kv_heads == TOY.n_heads  # no GQA in Phase 1
    assert TOY.n_params() < 1_000_000


def test_small_targets_about_14M_params():
    assert SMALL.d_model == SMALL.n_heads * SMALL.head_dim
    assert SMALL.n_kv_heads == SMALL.n_heads
    assert 10_000_000 < SMALL.n_params() < 16_000_000
