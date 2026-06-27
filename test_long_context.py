"""
test_long_context.py
--------------------
Fast smoke tests for the long-context architecture path.
"""

from __future__ import annotations

import torch

from generate import build_qa_prompt
from model import (
    DEFAULT_MODEL_PRESET,
    MODEL_PRESETS,
    MiniGPT,
    RotaryPositionalEncoding,
    estimate_parameter_count,
)
from tokenizer import CharTokenizer
from train import hparams_from_checkpoint_config


def test_rope_scaling_changes_base() -> None:
    plain = RotaryPositionalEncoding(head_dim=16, max_seq_len=32, base=10000.0)
    scaled = RotaryPositionalEncoding(
        head_dim=16,
        max_seq_len=32,
        base=10000.0,
        scaling={"type": "ntk", "factor": 4.0},
    )
    assert scaled.base > plain.base


def test_long_context_forward_shape() -> None:
    model = MiniGPT(
        vocab_size=64,
        d_model=32,
        n_heads=4,
        n_layers=2,
        max_seq_len=128,
        n_kv_heads=2,
        norm_type="rmsnorm",
        qkv_fused=False,
        rope_scaling={"type": "ntk", "factor": 4.0},
    )
    x = torch.randint(0, 64, (2, 96))
    logits, caches = model(x)
    assert logits.shape == (2, 96, 64)
    assert len(caches) == 2


def test_prompt_history_respects_token_budget() -> None:
    tokenizer = CharTokenizer()
    tokenizer.build_vocab("Q:A:<eos> old answer current question")
    history = [(f"old{i}", f"answer{i}") for i in range(12)]
    prompt = build_qa_prompt(
        history,
        "current question",
        tokenizer=tokenizer,
        max_prompt_tokens=80,
    )
    assert prompt.endswith("Q: current question\nA:")
    assert len(tokenizer.encode(prompt)) <= 80
    assert "old11" in prompt


def test_default_preset_is_larger_dense_transformer() -> None:
    vocab_size = 100_278
    daily = MODEL_PRESETS[DEFAULT_MODEL_PRESET]
    old = MODEL_PRESETS["gravity-120m"]

    assert daily.n_kv_heads == daily.n_heads
    assert daily.norm_type == "rmsnorm"
    assert not daily.qkv_fused
    assert estimate_parameter_count(
        vocab_size,
        daily.d_model,
        daily.n_heads,
        daily.n_layers,
        daily.ffn_mult,
        daily.n_kv_heads,
        daily.norm_type,
        daily.qkv_fused,
    ) > estimate_parameter_count(
        vocab_size,
        old.d_model,
        old.n_heads,
        old.n_layers,
        old.ffn_mult,
        old.n_kv_heads,
        old.norm_type,
        old.qkv_fused,
    )


def test_resume_can_enable_requested_long_rope_scaling() -> None:
    requested = MODEL_PRESETS["gravity-daily-long"]
    checkpoint = MODEL_PRESETS["gravity-daily"]
    fallback = {
        "d_model": requested.d_model,
        "n_heads": requested.n_heads,
        "n_layers": requested.n_layers,
        "dropout": requested.dropout,
        "ffn_mult": requested.ffn_mult,
        "n_kv_heads": requested.n_kv_heads,
        "norm_type": requested.norm_type,
        "qkv_fused": requested.qkv_fused,
        "rope_base": requested.rope_base,
        "rope_scaling": requested.rope_scaling,
    }
    cfg = {
        "d_model": checkpoint.d_model,
        "n_heads": checkpoint.n_heads,
        "n_layers": checkpoint.n_layers,
        "dropout": checkpoint.dropout,
        "ffn_mult": checkpoint.ffn_mult,
        "n_kv_heads": checkpoint.n_kv_heads,
        "norm_type": checkpoint.norm_type,
        "qkv_fused": checkpoint.qkv_fused,
        "rope_base": checkpoint.rope_base,
        "rope_scaling": checkpoint.rope_scaling,
    }

    resolved = hparams_from_checkpoint_config(cfg, fallback)

    assert resolved["rope_scaling"] == requested.rope_scaling


if __name__ == "__main__":
    test_rope_scaling_changes_base()
    test_long_context_forward_shape()
    test_prompt_history_respects_token_budget()
    test_default_preset_is_larger_dense_transformer()
    test_resume_can_enable_requested_long_rope_scaling()
    print("long-context smoke tests passed")
