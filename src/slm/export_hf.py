import torch
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
from slm.config import ModelConfig
from slm.model import LlamaSLM
from slm.train import load_checkpoint


def to_hf_config(cfg: ModelConfig) -> LlamaConfig:
    return LlamaConfig(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.d_model,
        intermediate_size=cfg.ffn_hidden,
        num_hidden_layers=cfg.n_layers,
        num_attention_heads=cfg.n_heads,
        num_key_value_heads=cfg.n_kv_heads,
        head_dim=cfg.head_dim,
        max_position_embeddings=cfg.context_len,
        rms_norm_eps=cfg.rms_norm_eps,
        rope_theta=cfg.rope_theta,
        tie_word_embeddings=cfg.tie_embeddings,
        attention_bias=False,
        mlp_bias=False,
        hidden_act="silu",
        # Our tokenizer's <|endoftext|> is id 0 and serves as BOS/EOS/PAD.
        # Without this, LlamaConfig defaults (bos=1, eos=2) would tell runtimes
        # to stop on the wrong token, so generation never ends at a story break.
        bos_token_id=0,
        eos_token_id=0,
        pad_token_id=0,
    )


def _copy_weights_into_hf(src: LlamaSLM, hf: LlamaForCausalLM):
    sd = {}
    sd["model.embed_tokens.weight"] = src.embed_tokens.weight
    for i, blk in enumerate(src.layers):
        p = f"model.layers.{i}."
        sd[p + "input_layernorm.weight"] = blk.norm1.weight
        sd[p + "post_attention_layernorm.weight"] = blk.norm2.weight
        sd[p + "self_attn.q_proj.weight"] = blk.attn.q_proj.weight
        sd[p + "self_attn.k_proj.weight"] = blk.attn.k_proj.weight
        sd[p + "self_attn.v_proj.weight"] = blk.attn.v_proj.weight
        sd[p + "self_attn.o_proj.weight"] = blk.attn.o_proj.weight
        sd[p + "mlp.gate_proj.weight"] = blk.mlp.gate_proj.weight
        sd[p + "mlp.up_proj.weight"] = blk.mlp.up_proj.weight
        sd[p + "mlp.down_proj.weight"] = blk.mlp.down_proj.weight
    sd["model.norm.weight"] = src.norm.weight
    sd["lm_head.weight"] = src.lm_head.weight
    hf.load_state_dict(sd, strict=True)


def export_to_hf(ckpt_path: str, tok_path: str, out_dir: str) -> str:
    src, cfg = load_checkpoint(ckpt_path)
    src.eval()
    hf = LlamaForCausalLM(to_hf_config(cfg)).eval()
    _copy_weights_into_hf(src, hf)
    hf.save_pretrained(out_dir)
    fast = PreTrainedTokenizerFast(
        tokenizer_file=tok_path,
        eos_token="<|endoftext|>", bos_token="<|endoftext|>",
        unk_token="<|endoftext|>", pad_token="<|endoftext|>",
    )
    fast.save_pretrained(out_dir)
    return out_dir


def push(out_dir: str, repo_id: str, private: bool = True):
    from huggingface_hub import HfApi
    HfApi().create_repo(repo_id, exist_ok=True, private=private)
    LlamaForCausalLM.from_pretrained(out_dir).push_to_hub(repo_id, private=private)
    PreTrainedTokenizerFast.from_pretrained(out_dir).push_to_hub(repo_id, private=private)
    # The model card (README.md) isn't loaded by from_pretrained; upload it too.
    import os
    readme = os.path.join(out_dir, "README.md")
    if os.path.exists(readme):
        HfApi().upload_file(path_or_fileobj=readme, path_in_repo="README.md",
                            repo_id=repo_id, repo_type="model")
