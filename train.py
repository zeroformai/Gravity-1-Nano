"""
train.py
--------
End-to-end training script for MiniGPT.

Usage:
    python train.py
    python train.py --max_iters 5000 --seq_len 512
"""

import argparse
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

from model import (
    DEFAULT_MODEL_PRESET,
    MODEL_PRESETS,
    MiniGPT,
    attention_variant,
    estimate_parameter_count,
)
from tokenizer import create_tokenizer
from utils import (
    load_text_files,
    parse_qa_examples,
    parse_qa_conversations,
    build_history_examples,
    encode_qa_examples,
    split_examples,
    prepare_qa_batch_tensors,
    get_qa_batch,
    estimate_qa_loss,
    set_seed,
    get_autocast,
    get_grad_scaler,
)


class TrainingProgress:
    """Render a compact single-line training progress indicator."""

    def __init__(self, total_steps: int, width: int = 28, enabled: bool = True) -> None:
        self.total_steps = total_steps
        self.width = max(10, width)
        self.enabled = enabled
        self.started_at = time.time()
        self.last_render_len = 0
        self.last_train_loss = None
        self.last_val_loss = None

    @staticmethod
    def _format_duration(seconds: float) -> str:
        seconds = max(0, int(seconds))
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            return f"{hours:d}h{minutes:02d}m{seconds:02d}s"
        if minutes:
            return f"{minutes:d}m{seconds:02d}s"
        return f"{seconds:d}s"

    def _clear_line(self) -> None:
        if not self.enabled:
            return
        sys.stdout.write("\r" + (" " * self.last_render_len) + "\r")
        sys.stdout.flush()

    def render(self, step: int, loss: float, avg_loss: float | None = None, lr: float | None = None) -> None:
        if not self.enabled:
            return

        elapsed = time.time() - self.started_at
        steps_per_sec = step / elapsed if elapsed > 0 else 0.0
        remaining_steps = max(0, self.total_steps - step)
        eta = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0.0
        progress = step / self.total_steps if self.total_steps else 1.0
        filled = min(self.width, int(self.width * progress))
        bar = "#" * filled + "-" * (self.width - filled)

        avg = f"{avg_loss:.4f}" if avg_loss is not None else "n/a"
        train = f"{self.last_train_loss:.4f}" if self.last_train_loss is not None else "n/a"
        val = f"{self.last_val_loss:.4f}" if self.last_val_loss is not None else "n/a"
        lr_str = f" lr {lr:.2e}" if lr is not None else ""
        line = (
            f"[{bar}] {progress * 100:6.2f}% "
            f"step {step:>{len(str(self.total_steps))}d}/{self.total_steps} "
            f"loss {loss:.4f} avg {avg} "
            f"eval train {train} val {val} "
            f"{steps_per_sec:.2f} it/s elapsed {self._format_duration(elapsed)} "
            f"eta {self._format_duration(eta)}{lr_str}"
        )
        sys.stdout.write("\r" + line)
        if len(line) < self.last_render_len:
            sys.stdout.write(" " * (self.last_render_len - len(line)))
        sys.stdout.flush()
        self.last_render_len = len(line)

    def set_evaluating(self, step: int) -> None:
        if not self.enabled:
            return
        self._clear_line()
        line = f"[Train] Evaluating at step {step}/{self.total_steps}..."
        sys.stdout.write("\r" + line)
        sys.stdout.flush()
        self.last_render_len = len(line)

    def update_metrics(self, train_loss: float, val_loss: float) -> None:
        self.last_train_loss = train_loss
        self.last_val_loss = val_loss

    def finish(self) -> None:
        if not self.enabled:
            return
        sys.stdout.write("\n")
        sys.stdout.flush()


# ---------------------------------------------------------------------------
# Learning rate schedule: linear warmup + cosine decay
# ---------------------------------------------------------------------------

def get_lr(
    step: int,
    max_lr: float,
    min_lr: float,
    warmup_steps: int,
    total_steps: int,
) -> float:
    """Compute learning rate with linear warmup and cosine decay."""
    if step < warmup_steps:
        # Linear warmup
        return max_lr * (step + 1) / warmup_steps
    if step >= total_steps:
        return min_lr
    # Cosine decay
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MiniGPT")
    parser.add_argument("--data", type=str, default="corpus/qa_10k.txt",
                        help="Comma/semicolon separated corpus files or glob patterns.")
    parser.add_argument(
        "--dialogues",
        type=str,
        default="corpus/dialogues_clean.txt",
        help="Optional comma/semicolon separated multi-turn dialogue files.",
    )
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument(
        "--val_frac",
        type=float,
        default=0.05,
        help="Fraction reserved for validation.",
    )
    parser.add_argument(
        "--model_preset",
        choices=sorted(MODEL_PRESETS),
        default=DEFAULT_MODEL_PRESET,
        help="Architecture preset. Individual size flags override this.",
    )
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--n_heads", type=int, default=None)
    parser.add_argument("--n_layers", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--ffn_mult", type=float, default=None)
    parser.add_argument("--n_kv_heads", type=int, default=None)
    parser.add_argument("--norm_type", choices=["rmsnorm", "layernorm"], default=None)
    parser.add_argument("--qkv_fused", action="store_true",
                        help="Use legacy fused QKV multi-head attention.")
    parser.add_argument("--rope_base", type=float, default=None)
    parser.add_argument(
        "--rope_scaling_type",
        choices=["none", "ntk"],
        default=None,
        help="RoPE scaling strategy for longer context windows.",
    )
    parser.add_argument(
        "--rope_scaling_factor",
        type=float,
        default=None,
        help="RoPE scaling factor. Values above 1 stretch usable context.",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_iters", type=int, default=5000)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--eval_iters", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=3e-5,
                        help="Minimum learning rate after cosine decay.")
    parser.add_argument("--warmup_steps", type=int, default=200,
                        help="Number of linear warmup steps.")
    parser.add_argument("--weight_decay", type=float, default=0.1,
                        help="Weight decay for AdamW (applied to non-bias, non-LN params).")
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.03,
        help="Cross-entropy label smoothing to reduce brittle memorization and overconfident hallucinations.",
    )
    parser.add_argument("--grad_accum_steps", type=int, default=1,
                        help="Gradient accumulation steps for larger effective batch size.")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Resume model weights and architecture config from a checkpoint .pt file.",
    )
    parser.add_argument(
        "--history_examples",
        type=int,
        default=2,
        help="Number of prior turns to include when training dialogue examples.",
    )
    parser.add_argument(
        "--history_weight",
        type=int,
        default=3,
        help="Repeat history-aware dialogue examples to improve multi-turn chat.",
    )
    parser.add_argument(
        "--amp",
        choices=["auto", "on", "off"],
        default="auto",
        help="Use CUDA mixed precision to speed up training",
    )
    parser.add_argument(
        "--progress_width",
        type=int,
        default=28,
        help="Width of the training progress bar",
    )
    parser.add_argument(
        "--no_progress",
        action="store_true",
        help="Disable the single-line progress bar and print periodic logs only",
    )
    parser.add_argument(
        "--eval_first",
        action="store_true",
        help="Run validation after the first training step",
    )
    parser.add_argument(
        "--tokenizer_kind",
        choices=["tiktoken", "char"],
        default="tiktoken",
        help="Tokenizer implementation to use",
    )
    parser.add_argument(
        "--use_padding_mask",
        action="store_true",
        help="Use attention padding masks. Slower, but preserves strict masking for non-right-padded data.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Training device selection.",
    )
    parser.add_argument(
        "--grad_checkpointing",
        action="store_true",
        help="Trade compute for lower activation memory on larger models.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Use torch.compile for the training model when available.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Load data, build tokenizer/model, report sizes, then exit.",
    )
    parser.add_argument(
        "--auto_batch_size",
        action="store_true",
        help="Automatically scale physical batch size to maximize VRAM usage on CUDA.",
    )
    parser.add_argument(
        "--max_search_batch_size",
        type=int,
        default=1024,
        help="Maximum batch size to test during VRAM auto-fitting.",
    )
    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_model_hparams(args: argparse.Namespace) -> dict:
    preset = MODEL_PRESETS[args.model_preset]
    hparams = {
        "d_model": preset.d_model,
        "n_heads": preset.n_heads,
        "n_layers": preset.n_layers,
        "dropout": preset.dropout,
        "ffn_mult": preset.ffn_mult,
        "n_kv_heads": preset.n_kv_heads or preset.n_heads,
        "norm_type": preset.norm_type,
        "qkv_fused": preset.qkv_fused,
        "rope_base": preset.rope_base,
        "rope_scaling": preset.rope_scaling,
    }
    for name in ("d_model", "n_heads", "n_layers", "dropout", "ffn_mult", "n_kv_heads", "norm_type", "rope_base"):
        value = getattr(args, name)
        if value is not None:
            hparams[name] = value
    if args.qkv_fused:
        hparams["qkv_fused"] = True
    if args.rope_scaling_type and args.rope_scaling_type != "none":
        hparams["rope_scaling"] = {
            "type": args.rope_scaling_type,
            "factor": float(args.rope_scaling_factor or 1.0),
        }
    elif args.rope_scaling_type == "none":
        hparams["rope_scaling"] = None

    if hparams["d_model"] % hparams["n_heads"] != 0:
        raise ValueError("--d_model must be divisible by --n_heads.")
    if hparams["n_heads"] % hparams["n_kv_heads"] != 0:
        raise ValueError("--n_heads must be divisible by --n_kv_heads.")
    if hparams["qkv_fused"] and hparams["n_kv_heads"] != hparams["n_heads"]:
        raise ValueError("--qkv_fused requires --n_kv_heads == --n_heads.")
    return hparams


def hparams_from_checkpoint_config(cfg: dict, fallback: dict) -> dict:
    """Use checkpoint architecture when resuming older or custom checkpoints."""
    hparams = fallback.copy()
    for key in ("d_model", "n_heads", "n_layers", "dropout", "ffn_mult"):
        if key in cfg:
            hparams[key] = cfg[key]
    hparams["n_kv_heads"] = cfg.get("n_kv_heads", cfg.get("n_heads", hparams["n_heads"]))
    hparams["norm_type"] = cfg.get("norm_type", "layernorm")
    hparams["qkv_fused"] = cfg.get("qkv_fused", True)
    hparams["rope_base"] = cfg.get("rope_base", 10000.0)
    checkpoint_rope_scaling = cfg.get("rope_scaling")
    requested_rope_scaling = fallback.get("rope_scaling")
    hparams["rope_scaling"] = requested_rope_scaling or checkpoint_rope_scaling
    return hparams


# ---------------------------------------------------------------------------
# Parameter group helpers
# ---------------------------------------------------------------------------

def get_param_groups(model: torch.nn.Module, weight_decay: float) -> list[dict]:
    """
    Separate parameters into two groups:
      - Decayed: weights of Linear layers (not biases, not LayerNorm)
      - Non-decayed: biases, LayerNorm weights/biases, embedding weights
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Don't decay biases, LayerNorm params, or embeddings
        if param.ndim == 1 or "ln" in name or "emb" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


# ---------------------------------------------------------------------------
# VRAM optimization and memory telemetry
# ---------------------------------------------------------------------------

def report_cuda_memory(msg: str = "") -> None:
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
        prefix = f" [{msg}]" if msg else ""
        print(f"[VRAM]{prefix} Allocated: {allocated:.1f} MB | Reserved: {reserved:.1f} MB | Max Allocated: {max_allocated:.1f} MB")


def find_max_batch_size(
    model: torch.nn.Module,
    seq_len: int,
    device: torch.device,
    use_amp: bool = True,
    start_batch_size: int = 16,
    max_batch_size: int = 1024,
) -> int:
    """
    Search for the maximum batch size that fits in VRAM by running dummy forward,
    backward, and optimizer steps.
    """
    import gc
    import torch
    import torch.nn.functional as F

    print("\n[Train] Starting VRAM auto-fitting batch size search...")
    report_cuda_memory("Before search")
    
    # Save the initial model weights (cpu-cloned to save GPU memory)
    initial_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    # Create a temporary optimizer to accurately simulate training memory footprint (optimizer states)
    param_groups = get_param_groups(model, 0.1)
    
    import inspect
    use_fused = False
    sig = inspect.signature(torch.optim.AdamW)
    if "fused" in sig.parameters and device.type == "cuda":
        use_fused = True
        
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=3e-4,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=use_fused,
    )
    
    # We will search in powers of 2 first
    current_bs = start_batch_size
    last_working_bs = None
    
    model.train()
    
    while current_bs <= max_batch_size:
        gc.collect()
        torch.cuda.empty_cache()
        
        try:
            # Run 3 dummy steps to fully allocate gradients and optimizer states in VRAM
            for _ in range(3):
                vocab_size = model.token_emb.num_embeddings
                x = torch.randint(0, vocab_size, (current_bs, seq_len), device=device)
                y = torch.randint(0, vocab_size, (current_bs, seq_len), device=device)
                mask = torch.ones((current_bs, seq_len), dtype=torch.bool, device=device)
                
                optimizer.zero_grad(set_to_none=True)
                with get_autocast(device.type, enabled=use_amp):
                    logits, _ = model(x, padding_mask=mask)
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        y.view(-1),
                        ignore_index=-100,
                    )
                loss.backward()
                optimizer.step()
                
            last_working_bs = current_bs
            print(f"  Batch size {current_bs:<4d} fits successfully.")
            current_bs *= 2
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  Batch size {current_bs:<4d} exceeded VRAM limits (OOM).")
                break
            else:
                raise e

    if last_working_bs is None:
        print("[Train] Auto batch size search failed. Defaulting to batch size 16.")
        model.load_state_dict(initial_state)
        gc.collect()
        torch.cuda.empty_cache()
        return 16

    # Binary search between last_working_bs and current_bs - 1
    low = last_working_bs
    high = min(current_bs - 1, max_batch_size)
    best_bs = last_working_bs

    while low <= high:
        mid = (low + high) // 2
        if mid <= best_bs:
            low = mid + 1
            continue
            
        gc.collect()
        torch.cuda.empty_cache()
        
        try:
            for _ in range(3):
                vocab_size = model.token_emb.num_embeddings
                x = torch.randint(0, vocab_size, (mid, seq_len), device=device)
                y = torch.randint(0, vocab_size, (mid, seq_len), device=device)
                mask = torch.ones((mid, seq_len), dtype=torch.bool, device=device)
                
                optimizer.zero_grad(set_to_none=True)
                with get_autocast(device.type, enabled=use_amp):
                    logits, _ = model(x, padding_mask=mask)
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        y.view(-1),
                        ignore_index=-100,
                    )
                loss.backward()
                optimizer.step()
                
            best_bs = mid
            print(f"  Binary search: Batch size {mid:<4d} fits successfully.")
            low = mid + 1
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  Binary search: Batch size {mid:<4d} exceeded VRAM limits (OOM).")
                high = mid - 1
            else:
                raise e

    # Round down best_bs to the nearest multiple of 8 (or 4/2 if very small) for optimal hardware alignment
    if best_bs >= 8:
        best_bs = (best_bs // 8) * 8
    elif best_bs >= 4:
        best_bs = (best_bs // 4) * 4

    print(f"[Train] Auto-fit batch size selected: {best_bs}")
    
    # Restore model state
    model.load_state_dict(initial_state)
    
    # Clean up search optimizer and cache
    del optimizer
    gc.collect()
    torch.cuda.empty_cache()
    report_cuda_memory("After search cleanup")
    print()
    
    return best_bs


def main() -> None:
    args = parse_args()
    set_seed(42)

    device = select_device(args.device)
    print(f"[Train] Using device: {device}")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    raw_texts = load_text_files(args.data)
    raw_text = "\n".join(text for _, text in raw_texts)

    dialogue_text = ""
    dialogue_examples = []
    if args.dialogues:
        dialogue_texts = load_text_files(args.dialogues)
        dialogue_text = "\n\n".join(text for _, text in dialogue_texts)
        conversations = parse_qa_conversations(dialogue_text)
        dialogue_examples = build_history_examples(
            conversations,
            max_history=args.history_examples,
        )
        dialogue_examples = dialogue_examples * max(1, args.history_weight)

    tokenizer = create_tokenizer(args.tokenizer_kind)
    tokenizer.build_vocab(raw_text + "\n" + dialogue_text)

    resume_checkpoint = None
    resume_config = None
    if args.resume_from:
        resume_checkpoint = torch.load(args.resume_from, map_location=device, weights_only=True)
        resume_config = resume_checkpoint["config"]
        if tokenizer.vocab_size != resume_config["vocab_size"]:
            raise ValueError(
                f"Tokenizer vocab size {tokenizer.vocab_size} does not match "
                f"checkpoint vocab size {resume_config['vocab_size']}."
            )
        args.seq_len = max(args.seq_len, int(resume_config.get("max_seq_len", args.seq_len)))
        print(f"[Train] Resuming from checkpoint: {args.resume_from}")

    model_hparams = resolve_model_hparams(args)
    if resume_config is not None:
        model_hparams = hparams_from_checkpoint_config(resume_config, model_hparams)

    qa_examples = parse_qa_examples(raw_text)
    qa_examples.extend(dialogue_examples)
    encoded_examples = encode_qa_examples(qa_examples, tokenizer, args.seq_len)
    train_examples, val_examples = split_examples(
        encoded_examples,
        val_fraction=args.val_frac,
    )
    train_batches = prepare_qa_batch_tensors(train_examples, args.seq_len, device)
    val_batches = prepare_qa_batch_tensors(val_examples, args.seq_len, device)

    model = MiniGPT(
        vocab_size=tokenizer.vocab_size,
        d_model=model_hparams["d_model"],
        n_heads=model_hparams["n_heads"],
        n_layers=model_hparams["n_layers"],
        max_seq_len=args.seq_len,
        dropout=model_hparams["dropout"],
        ffn_mult=model_hparams["ffn_mult"],
        n_kv_heads=model_hparams["n_kv_heads"],
        norm_type=model_hparams["norm_type"],
        qkv_fused=model_hparams["qkv_fused"],
        rope_base=model_hparams["rope_base"],
        rope_scaling=model_hparams["rope_scaling"],
        gradient_checkpointing=args.grad_checkpointing,
    ).to(device)
    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model_state"])
        print("[Train] Loaded model weights from resume checkpoint.")

    estimated_params = estimate_parameter_count(
        vocab_size=tokenizer.vocab_size,
        d_model=model_hparams["d_model"],
        n_heads=model_hparams["n_heads"],
        n_layers=model_hparams["n_layers"],
        ffn_mult=model_hparams["ffn_mult"],
        n_kv_heads=model_hparams["n_kv_heads"],
        norm_type=model_hparams["norm_type"],
        qkv_fused=model_hparams["qkv_fused"],
    )
    actual_params = model.count_parameters()
    print(
        f"[Train] Preset: {args.model_preset} | "
        f"d_model={model_hparams['d_model']} n_heads={model_hparams['n_heads']} "
        f"n_kv_heads={model_hparams['n_kv_heads']} n_layers={model_hparams['n_layers']} "
        f"ffn_mult={model_hparams['ffn_mult']} norm={model_hparams['norm_type']} "
        f"attention={attention_variant(model_hparams['n_heads'], model_hparams['n_kv_heads'])} "
        f"qkv_fused={model_hparams['qkv_fused']} "
        f"rope_scaling={model_hparams['rope_scaling']}"
    )
    print(
        f"[Train] Model has {actual_params:,} trainable parameters "
        f"(estimate {estimated_params:,})."
    )

    if args.dry_run:
        print("[Train] Dry run complete. No optimizer step or checkpoint written.")
        return

    use_amp = args.amp == "on" or (args.amp == "auto" and device.type == "cuda")

    if args.auto_batch_size and device.type == "cuda":
        target_eff_batch = args.batch_size * args.grad_accum_steps
        max_search = min(args.max_search_batch_size, len(train_examples))
        opt_batch_size = find_max_batch_size(
            model=model,
            seq_len=args.seq_len,
            device=device,
            use_amp=use_amp,
            start_batch_size=16,
            max_batch_size=max_search,
        )
        
        if target_eff_batch <= opt_batch_size:
            args.batch_size = opt_batch_size
            args.grad_accum_steps = 1
        else:
            args.batch_size = opt_batch_size
            args.grad_accum_steps = max(1, math.ceil(target_eff_batch / opt_batch_size))
            
        print(f"[Train] VRAM auto-fit configuration:")
        print(f"  -> Physical micro-batch size: {args.batch_size}")
        print(f"  -> Gradient accumulation steps: {args.grad_accum_steps}")
        print(f"  -> Effective batch size: {args.batch_size * args.grad_accum_steps}")

    # AdamW with separated parameter groups for weight decay
    param_groups = get_param_groups(model, args.weight_decay)
    
    # Check if fused AdamW is supported and on CUDA
    import inspect
    use_fused = False
    sig = inspect.signature(torch.optim.AdamW)
    if "fused" in sig.parameters and device.type == "cuda":
        use_fused = True
        
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=use_fused,
    )

    scaler = get_grad_scaler(device.type, enabled=use_amp)
    if use_amp:
        print("[Train] AMP mixed precision: enabled")
    if use_fused:
        print("[Train] Fused AdamW optimizer: enabled")
    if args.grad_checkpointing:
        print("[Train] Gradient checkpointing: enabled")
    if args.compile:
        if hasattr(torch, "compile"):
            print("[Train] torch.compile: enabled")
            model = torch.compile(model)
        else:
            print("[Train] torch.compile requested but unavailable; continuing uncompiled.")

    print(f"\n[Train] Starting training for {args.max_iters} steps...")
    print(f"[Train] Warmup: {args.warmup_steps} steps | "
          f"LR: {args.learning_rate} -> {args.min_lr} (cosine) | "
          f"Weight decay: {args.weight_decay} | "
          f"Label smoothing: {args.label_smoothing}")
    if args.grad_accum_steps > 1:
        print(f"[Train] Gradient accumulation: {args.grad_accum_steps} steps "
              f"(effective batch size: {args.batch_size * args.grad_accum_steps})")
    if not args.use_padding_mask:
        print("[Train] Padding attention mask: disabled for faster causal attention")
    print()

    model.train()
    report_cuda_memory("Training Init")
    progress = TrainingProgress(
        args.max_iters,
        width=args.progress_width,
        enabled=not args.no_progress and sys.stdout.isatty(),
    )
    recent_losses = []

    for step in range(1, args.max_iters + 1):
        if step == 2:  # Step 2 because step 1 optimizer update has completed and states are fully allocated
            report_cuda_memory("Running training (after Step 1)")
        # Update learning rate
        lr = get_lr(step - 1, args.learning_rate, args.min_lr, args.warmup_steps, args.max_iters)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Gradient accumulation loop
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for micro_step in range(args.grad_accum_steps):
            x, y, mask = get_qa_batch(train_batches, args.batch_size, args.seq_len, device)
            with get_autocast(device.type, enabled=use_amp):
                logits, _ = model(x, padding_mask=mask if args.use_padding_mask else None)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1),
                    ignore_index=-100,
                    label_smoothing=args.label_smoothing,
                )
                loss = loss / args.grad_accum_steps
            scaler.scale(loss).backward()
            accum_loss += loss.item()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        loss_value = accum_loss
        recent_losses.append(loss_value)
        if len(recent_losses) > 50:
            recent_losses.pop(0)
        avg_loss = sum(recent_losses) / len(recent_losses)
        progress.render(step, loss_value, avg_loss, lr)

        should_eval = step % args.eval_every == 0 or (args.eval_first and step == 1)
        if should_eval:
            progress.set_evaluating(step)
            metrics = estimate_qa_loss(
                model,
                train_batches,
                val_batches,
                args.batch_size,
                args.seq_len,
                device,
                args.eval_iters,
                use_amp,
                use_padding_mask=args.use_padding_mask,
            )
            progress.update_metrics(metrics["train"], metrics["val"])
            if not progress.enabled:
                print(
                    f"  Step {step:>5d}/{args.max_iters} | "
                    f"train loss: {metrics['train']:.4f} | "
                    f"val loss: {metrics['val']:.4f} | "
                    f"batch loss: {loss_value:.4f} | "
                    f"recent avg: {avg_loss:.4f} | "
                    f"lr: {lr:.2e}"
                )
            else:
                progress.render(step, loss_value, avg_loss, lr)

    progress.finish()
    print("[Train] Training complete.")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    model_path = os.path.join(args.checkpoint_dir, "model.pt")
    tokenizer_path = os.path.join(args.checkpoint_dir, "tokenizer.json")
    vocab_path = os.path.join(args.checkpoint_dir, "vocab.json")

    torch.save(
        {
            "model_state": (
                model._orig_mod.state_dict()
                if hasattr(model, "_orig_mod")
                else model.state_dict()
            ),
            "config": {
                "architecture": "decoder_only_transformer",
                "attention": attention_variant(model_hparams["n_heads"], model_hparams["n_kv_heads"]),
                "vocab_size": tokenizer.vocab_size,
                "model_preset": (
                    resume_config.get("model_preset", args.model_preset)
                    if resume_config is not None
                    else args.model_preset
                ),
                "d_model": model_hparams["d_model"],
                "n_heads": model_hparams["n_heads"],
                "n_layers": model_hparams["n_layers"],
                "max_seq_len": args.seq_len,
                "dropout": model_hparams["dropout"],
                "ffn_mult": model_hparams["ffn_mult"],
                "n_kv_heads": model_hparams["n_kv_heads"],
                "norm_type": model_hparams["norm_type"],
                "qkv_fused": model_hparams["qkv_fused"],
                "rope_base": model_hparams["rope_base"],
                "rope_scaling": model_hparams["rope_scaling"],
                "tokenizer_kind": args.tokenizer_kind,
                "label_smoothing": args.label_smoothing,
            },
        },
        model_path,
    )
    tokenizer.save(tokenizer_path)
    tokenizer.save(vocab_path)

    print(f"[Train] Model saved -> {model_path}")
    print(f"[Train] Tokenizer saved -> {tokenizer_path}")


if __name__ == "__main__":
    main()
