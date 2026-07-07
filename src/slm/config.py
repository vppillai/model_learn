from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    ffn_hidden: int
    context_len: int
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    tie_embeddings: bool = True

    def n_params(self) -> int:
        """Approximate trainable parameter count (bias-free linears)."""
        embed = self.vocab_size * self.d_model  # tied: counted once
        attn = self.n_layers * (
            self.d_model * self.n_heads * self.head_dim          # q
            + 2 * self.d_model * self.n_kv_heads * self.head_dim  # k, v
            + self.n_heads * self.head_dim * self.d_model          # o
        )
        ffn = self.n_layers * (3 * self.d_model * self.ffn_hidden)
        norms = self.n_layers * 2 * self.d_model + self.d_model  # RMSNorm weights
        return embed + attn + ffn + norms


TOY = ModelConfig(
    vocab_size=2048, d_model=64, n_layers=2, n_heads=4, n_kv_heads=4,
    head_dim=16, ffn_hidden=128, context_len=128,
)

SMALL = ModelConfig(
    vocab_size=8192, d_model=384, n_layers=6, n_heads=6, n_kv_heads=6,
    head_dim=64, ffn_hidden=1024, context_len=512,
)
