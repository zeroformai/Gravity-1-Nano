"""
model.py
--------
Decoder-only Transformer for Gravity.

The default architecture follows the proven small-LLM pattern used by LLaMA,
Qwen, Mistral-style models:
  - token embeddings with tied LM head
  - pre-normalization blocks
  - RMSNorm by default
  - rotary position embeddings (RoPE)
  - optional NTK-style RoPE scaling for longer context windows
  - dense multi-head self-attention for the daily preset
  - optional grouped-query attention (GQA) for smaller/legacy presets
  - SwiGLU feed-forward blocks
  - KV cache for generation

Legacy fused MHA + LayerNorm is still supported for old checkpoints.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint


@dataclass(frozen=True)
class ModelPreset:
    """Architecture preset without tokenizer-dependent vocabulary size."""

    d_model: int
    n_heads: int
    n_layers: int
    dropout: float = 0.1
    ffn_mult: float = 4.0
    n_kv_heads: Optional[int] = None
    norm_type: str = "rmsnorm"
    qkv_fused: bool = False
    rope_base: float = 10000.0
    rope_scaling: Optional[dict[str, float | str]] = None


DEFAULT_MODEL_PRESET = "gravity-daily"

MODEL_PRESETS: dict[str, ModelPreset] = {
    "tiny": ModelPreset(d_model=128, n_heads=4, n_layers=4, n_kv_heads=2),
    "small": ModelPreset(d_model=256, n_heads=8, n_layers=6, n_kv_heads=4),
    "gravity-30m": ModelPreset(d_model=256, n_heads=8, n_layers=8, n_kv_heads=4),
    "gravity-60m": ModelPreset(d_model=416, n_heads=8, n_layers=8, n_kv_heads=4),
    # Compatibility preset: cl100k_base + local <eos> gives about 122M params.
    "gravity-120m": ModelPreset(d_model=720, n_heads=12, n_layers=9, n_kv_heads=4),
    "gravity-120m-long": ModelPreset(
        d_model=720,
        n_heads=12,
        n_layers=9,
        n_kv_heads=4,
        rope_scaling={"type": "ntk", "factor": 4.0},
    ),
    # Daily-use dense Transformer: full KV heads, about 304M params with cl100k.
    "gravity-daily": ModelPreset(
        d_model=1024,
        n_heads=16,
        n_layers=16,
        dropout=0.08,
        n_kv_heads=16,
    ),
    "gravity-daily-long": ModelPreset(
        d_model=1024,
        n_heads=16,
        n_layers=16,
        dropout=0.08,
        n_kv_heads=16,
        rope_scaling={"type": "ntk", "factor": 4.0},
    ),
    # Larger dense target for stronger generality when hardware allows it.
    "gravity-700m-dense": ModelPreset(
        d_model=1536,
        n_heads=24,
        n_layers=20,
        dropout=0.08,
        n_kv_heads=24,
    ),
}


def _ffn_hidden_dim(d_model: int, ffn_mult: float) -> int:
    hidden_dim = int(2 * ffn_mult * d_model / 3)
    return ((hidden_dim + 7) // 8) * 8


def estimate_parameter_count(
    vocab_size: int,
    d_model: int,
    n_heads: int,
    n_layers: int,
    ffn_mult: float = 4.0,
    n_kv_heads: Optional[int] = None,
    norm_type: str = "rmsnorm",
    qkv_fused: bool = False,
) -> int:
    """Analytically estimate trainable params for this tied-embedding model."""
    if d_model % n_heads != 0:
        raise ValueError("d_model must be divisible by n_heads")

    n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
    if n_heads % n_kv_heads != 0:
        raise ValueError("n_heads must be divisible by n_kv_heads")
    if qkv_fused and n_kv_heads != n_heads:
        raise ValueError("qkv_fused requires n_kv_heads == n_heads")

    head_dim = d_model // n_heads
    kv_dim = n_kv_heads * head_dim
    hidden_dim = _ffn_hidden_dim(d_model, ffn_mult)

    embedding = vocab_size * d_model
    if qkv_fused:
        attention_per_layer = 4 * d_model * d_model
    else:
        attention_per_layer = (
            d_model * d_model
            + 2 * d_model * kv_dim
            + d_model * d_model
        )
    ffn_per_layer = 3 * d_model * hidden_dim
    norm_per_block = 4 * d_model if norm_type == "layernorm" else 2 * d_model
    final_norm = 2 * d_model if norm_type == "layernorm" else d_model
    return embedding + n_layers * (
        attention_per_layer + ffn_per_layer + norm_per_block
    ) + final_norm


def attention_variant(n_heads: int, n_kv_heads: Optional[int]) -> str:
    kv_heads = n_heads if n_kv_heads is None else n_kv_heads
    if kv_heads == n_heads:
        return "dense_mha"
    return f"gqa_{n_heads}q_{kv_heads}kv"


class RMSNorm(nn.Module):
    """Root-mean-square normalization used by modern decoder LLMs."""

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_float = x.float()
        normed = x_float * torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (normed.to(dtype) * self.weight)


def create_norm(norm_type: str, d_model: int) -> nn.Module:
    normalized = norm_type.lower()
    if normalized == "rmsnorm":
        return RMSNorm(d_model)
    if normalized == "layernorm":
        return nn.LayerNorm(d_model)
    raise ValueError(f"Unknown norm_type: {norm_type}")


class RotaryPositionalEncoding(nn.Module):
    """Rotary Position Embedding applied to per-head Q/K vectors."""

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int = 2048,
        base: float = 10000.0,
        scaling: Optional[dict[str, float | str]] = None,
    ):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        self.head_dim = head_dim
        self.base = self._scaled_base(base, head_dim, scaling)
        inv_freq = 1.0 / (self.base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    @staticmethod
    def _scaled_base(
        base: float,
        head_dim: int,
        scaling: Optional[dict[str, float | str]],
    ) -> float:
        if not scaling:
            return base
        scaling_type = str(scaling.get("type", "none")).lower()
        factor = float(scaling.get("factor", 1.0))
        if factor <= 1.0 or scaling_type in {"none", "off"}:
            return base
        if scaling_type in {"ntk", "dynamic_ntk"}:
            exponent = head_dim / max(1, head_dim - 2)
            return base * (factor ** exponent)
        raise ValueError(f"Unknown rope_scaling type: {scaling_type}")

    def _build_cache(self, seq_len: int) -> None:
        positions = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(positions, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        seq_len = x.shape[2]
        seq_end = offset + seq_len
        if seq_end > self.cos_cached.shape[0]:
            self._build_cache(seq_end)

        cos = self.cos_cached[offset:seq_end].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[offset:seq_end].unsqueeze(0).unsqueeze(0)
        return (x * cos) + (_rotate_half(x) * sin)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def _repeat_kv(x: torch.Tensor, repeats: int) -> torch.Tensor:
    if repeats == 1:
        return x
    return x.repeat_interleave(repeats, dim=1)


class CausalSelfAttention(nn.Module):
    """Causal self-attention with optional GQA and KV cache."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        residual_scale: float = 1.0,
        n_kv_heads: Optional[int] = None,
        qkv_fused: bool = False,
        rope_base: float = 10000.0,
        rope_scaling: Optional[dict[str, float | str]] = None,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        if n_heads % n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if qkv_fused and n_kv_heads != n_heads:
            raise ValueError("qkv_fused requires n_kv_heads == n_heads")

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.kv_dim = n_kv_heads * self.head_dim
        self.n_kv_repeats = n_heads // n_kv_heads
        self.dropout = dropout
        self.residual_scale = residual_scale
        self.qkv_fused = qkv_fused

        if qkv_fused:
            self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        else:
            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.k_proj = nn.Linear(d_model, self.kv_dim, bias=False)
            self.v_proj = nn.Linear(d_model, self.kv_dim, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.resid_drop = nn.Dropout(dropout)
        self.rope = RotaryPositionalEncoding(
            self.head_dim,
            max_seq_len,
            base=rope_base,
            scaling=rope_scaling,
        )

    def _shape_q(self, t: torch.Tensor, batch: int, seq_len: int) -> torch.Tensor:
        return t.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

    def _shape_kv(self, t: torch.Tensor, batch: int, seq_len: int) -> torch.Tensor:
        return t.view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        batch, seq_len, channels = x.shape
        offset = 0 if kv_cache is None else kv_cache[0].shape[2]

        if self.qkv_fused:
            qkv = self.qkv_proj(x)
            q_raw, k_raw, v_raw = qkv.split(channels, dim=-1)
        else:
            q_raw = self.q_proj(x)
            k_raw = self.k_proj(x)
            v_raw = self.v_proj(x)

        q = self._shape_q(q_raw, batch, seq_len)
        k = self._shape_kv(k_raw, batch, seq_len)
        v = self._shape_kv(v_raw, batch, seq_len)

        q = self.rope(q, offset=offset)
        k = self.rope(k, offset=offset)

        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            k = torch.cat([cached_k, k], dim=2)
            v = torch.cat([cached_v, v], dim=2)

        new_kv_cache = (k, v)
        attn_k = _repeat_kv(k, self.n_kv_repeats)
        attn_v = _repeat_kv(v, self.n_kv_repeats)

        attn_mask = None
        if padding_mask is not None:
            total_len = attn_k.shape[2]
            attn_mask = padding_mask[:, :total_len].unsqueeze(1).unsqueeze(2)
            attn_mask = attn_mask.to(dtype=q.dtype)
            attn_mask = (1.0 - attn_mask) * torch.finfo(q.dtype).min

        if attn_mask is not None:
            scale = 1.0 / math.sqrt(self.head_dim)
            attn_weights = torch.matmul(q, attn_k.transpose(-2, -1)) * scale
            q_len = q.shape[2]
            k_len = attn_k.shape[2]
            causal_mask = torch.triu(
                torch.ones(q_len, k_len, device=q.device, dtype=torch.bool),
                diagonal=k_len - q_len + 1,
            )
            attn_weights = attn_weights.masked_fill(
                causal_mask.unsqueeze(0).unsqueeze(0),
                float("-inf"),
            )
            attn_weights = attn_weights + attn_mask
            attn_weights = F.softmax(attn_weights, dim=-1)
            if self.training:
                attn_weights = F.dropout(attn_weights, p=self.dropout)
            out = torch.matmul(attn_weights, attn_v)
        else:
            out = F.scaled_dot_product_attention(
                q,
                attn_k,
                attn_v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=(kv_cache is None),
            )

        out = out.transpose(1, 2).contiguous().view(batch, seq_len, channels)
        out = self.resid_drop(self.out_proj(out)) * self.residual_scale
        return out, new_kv_cache


class SwiGLUFeedForward(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(self, d_model: int, dropout: float = 0.1, ffn_mult: float = 4.0):
        super().__init__()
        hidden_dim = _ffn_hidden_dim(d_model, ffn_mult)
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)
        self.w3 = nn.Linear(d_model, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    """Pre-norm decoder block."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        residual_scale: float = 1.0,
        ffn_mult: float = 4.0,
        n_kv_heads: Optional[int] = None,
        norm_type: str = "rmsnorm",
        qkv_fused: bool = False,
        rope_base: float = 10000.0,
        rope_scaling: Optional[dict[str, float | str]] = None,
    ):
        super().__init__()
        self.ln1 = create_norm(norm_type, d_model)
        self.attn = CausalSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            max_seq_len=max_seq_len,
            dropout=dropout,
            residual_scale=residual_scale,
            n_kv_heads=n_kv_heads,
            qkv_fused=qkv_fused,
            rope_base=rope_base,
            rope_scaling=rope_scaling,
        )
        self.ln2 = create_norm(norm_type, d_model)
        self.ffn = SwiGLUFeedForward(d_model, dropout, ffn_mult=ffn_mult)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        attn_out, new_kv_cache = self.attn(self.ln1(x), kv_cache, padding_mask)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x, new_kv_cache


class MiniGPT(nn.Module):
    """Decoder-only language model."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        ffn_mult: float = 4.0,
        n_kv_heads: Optional[int] = None,
        norm_type: str = "rmsnorm",
        qkv_fused: bool = False,
        rope_base: float = 10000.0,
        rope_scaling: Optional[dict[str, float | str]] = None,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        if n_heads % n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")

        self.max_seq_len = max_seq_len
        self.n_layers = n_layers
        self.gradient_checkpointing = gradient_checkpointing
        self.config = {
            "architecture": "decoder_only_transformer",
            "attention": attention_variant(n_heads, n_kv_heads),
            "vocab_size": vocab_size,
            "d_model": d_model,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "max_seq_len": max_seq_len,
            "dropout": dropout,
            "ffn_mult": ffn_mult,
            "n_kv_heads": n_kv_heads,
            "norm_type": norm_type,
            "qkv_fused": qkv_fused,
            "rope_base": rope_base,
            "rope_scaling": rope_scaling,
        }

        residual_scale = 1.0 / math.sqrt(2 * n_layers)
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.emb_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                max_seq_len=max_seq_len,
                dropout=dropout,
                residual_scale=residual_scale,
                ffn_mult=ffn_mult,
                n_kv_heads=n_kv_heads,
                norm_type=norm_type,
                qkv_fused=qkv_fused,
                rope_base=rope_base,
                rope_scaling=rope_scaling,
            )
            for _ in range(n_layers)
        ])
        self.ln_final = create_norm(norm_type, d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)

    def set_gradient_checkpointing(self, enabled: bool = True) -> None:
        self.gradient_checkpointing = enabled

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def forward(
        self,
        idx: torch.Tensor,
        kv_caches: Optional[list[Optional[tuple[torch.Tensor, torch.Tensor]]]] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, list[Optional[tuple[torch.Tensor, torch.Tensor]]]]:
        batch, seq_len = idx.shape
        if seq_len == 0:
            raise ValueError("Input sequence must not be empty.")

        past_len = 0
        if kv_caches is not None and kv_caches and kv_caches[0] is not None:
            past_len = kv_caches[0][0].shape[2]
        if past_len + seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {past_len + seq_len} exceeds max_seq_len={self.max_seq_len}."
            )

        x = self.emb_drop(self.token_emb(idx))
        if kv_caches is None:
            kv_caches = [None] * self.n_layers

        new_kv_caches: list[Optional[tuple[torch.Tensor, torch.Tensor]]] = []
        for i, block in enumerate(self.blocks):
            use_checkpoint = (
                self.gradient_checkpointing
                and self.training
                and kv_caches[i] is None
            )
            if use_checkpoint:
                def checkpointed_block(hidden: torch.Tensor) -> torch.Tensor:
                    out, _ = block(hidden, None, padding_mask)
                    return out

                x = activation_checkpoint(checkpointed_block, x, use_reentrant=False)
                new_cache = None
            else:
                x, new_cache = block(x, kv_caches[i], padding_mask)
            new_kv_caches.append(new_cache)

        logits = self.lm_head(self.ln_final(x))
        return logits, new_kv_caches

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
