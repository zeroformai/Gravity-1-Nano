"""
train_dpo.py
------------
Direct Preference Optimization for Gravity checkpoints.

Input JSONL format:
    {"prompt": "...", "chosen": "...", "rejected": "..."}
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from generate import build_qa_prompt, load_model
from model import MiniGPT
from tokenizer import EOS_TOKEN
from train import get_param_groups, get_lr
from utils import get_autocast, get_grad_scaler, set_seed


@dataclass(frozen=True)
class PreferenceExample:
    prompt: str
    chosen: str
    rejected: str


@dataclass(frozen=True)
class EncodedPreference:
    chosen_ids: list[int]
    chosen_mask: list[int]
    rejected_ids: list[int]
    rejected_mask: list[int]


@dataclass(frozen=True)
class PreferenceBatchTensors:
    """Pre-padded preference pairs for fast indexed DPO batching."""

    chosen_x: torch.Tensor
    chosen_y: torch.Tensor
    chosen_mask: torch.Tensor
    rejected_x: torch.Tensor
    rejected_y: torch.Tensor
    rejected_mask: torch.Tensor

    def __len__(self) -> int:
        return self.chosen_x.size(0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DPO tune a Gravity checkpoint")
    parser.add_argument("--preferences", type=str, default="preferences.jsonl")
    parser.add_argument("--checkpoint", type=str, default="checkpoints_gravity120/model.pt")
    parser.add_argument("--tokenizer", type=str, default="checkpoints_gravity120/tokenizer.json")
    parser.add_argument("--vocab", type=str, default=None, help="Alias for --tokenizer")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_dpo")
    parser.add_argument("--seq_len", type=int, default=None,
                        help="Override checkpoint context length for filtering only.")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_iters", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--min_lr", type=float, default=5e-7)
    parser.add_argument("--warmup_steps", type=int, default=20)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--log_every", type=int, default=1,
                        help="Print training metrics every N optimizer steps.")
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--amp", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--reference_free", action="store_true",
                        help="Use zero reference log-ratio to save memory.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_preferences(path: str) -> list[PreferenceExample]:
    examples: list[PreferenceExample] = []
    skipped_invalid = 0
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                skipped_invalid += 1
                print(f"[DPO] Skipping invalid JSON at {path}:{line_no}: {exc}")
                continue
            try:
                prompt = str(record["prompt"]).strip()
                chosen = str(record["chosen"]).strip()
                rejected = str(record["rejected"]).strip()
            except KeyError as exc:
                skipped_invalid += 1
                print(f"[DPO] Skipping incomplete preference at {path}:{line_no}: {exc}")
                continue
            if prompt and chosen and rejected and chosen != rejected:
                examples.append(PreferenceExample(prompt, chosen, rejected))
    if not examples:
        raise ValueError(f"No preference examples found in {path}")
    if skipped_invalid:
        print(f"[DPO] Skipped {skipped_invalid} invalid preference lines.")
    return examples


def normalize_prompt(prompt: str) -> str:
    stripped = prompt.strip()
    if stripped.startswith("Q:") and "\nA:" in stripped:
        return stripped if stripped.endswith("A:") else stripped.split("\nA:", 1)[0] + "\nA:"
    return build_qa_prompt([], stripped)


def encode_answer(tokenizer, prompt: str, answer: str, seq_len: int) -> tuple[list[int], list[int]] | None:
    prompt_text = normalize_prompt(prompt)
    full_text = f"{prompt_text} {answer.strip()}{EOS_TOKEN}"
    prompt_ids = tokenizer.encode(prompt_text)
    full_ids = tokenizer.encode(full_text)
    if len(full_ids) < 2 or len(full_ids) > seq_len:
        return None

    target_count = len(full_ids) - 1
    answer_start = max(0, len(prompt_ids) - 1)
    mask = [0] * target_count
    for index in range(answer_start, target_count):
        mask[index] = 1
    if not any(mask):
        return None
    return full_ids, mask


def encode_preferences(
    examples: list[PreferenceExample],
    tokenizer,
    seq_len: int,
) -> list[EncodedPreference]:
    encoded: list[EncodedPreference] = []
    skipped = 0
    for example in examples:
        chosen = encode_answer(tokenizer, example.prompt, example.chosen, seq_len)
        rejected = encode_answer(tokenizer, example.prompt, example.rejected, seq_len)
        if chosen is None or rejected is None:
            skipped += 1
            continue
        encoded.append(EncodedPreference(
            chosen_ids=chosen[0],
            chosen_mask=chosen[1],
            rejected_ids=rejected[0],
            rejected_mask=rejected[1],
        ))
    if not encoded:
        raise ValueError("No preference examples fit the sequence length.")
    if skipped:
        print(f"[DPO] Skipped {skipped} long/invalid preference pairs.")
    print(f"[DPO] Encoded {len(encoded)} preference pairs.")
    return encoded


def pad_sequences(
    sequences: list[list[int]],
    loss_masks: list[list[int]],
    pad_id: int,
    device: torch.device,
    max_sequence_len: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_len = max_sequence_len or max(len(ids) for ids in sequences)
    input_len = max_len - 1
    batch = len(sequences)
    x = torch.full((batch, input_len), pad_id, dtype=torch.long)
    y = torch.full((batch, input_len), pad_id, dtype=torch.long)
    mask = torch.zeros((batch, input_len), dtype=torch.bool)

    for row, ids in enumerate(sequences):
        source = ids[:-1]
        target = ids[1:]
        length = len(source)
        x[row, :length] = torch.tensor(source, dtype=torch.long)
        y[row, :length] = torch.tensor(target, dtype=torch.long)
        mask[row, :len(loss_masks[row])] = torch.tensor(loss_masks[row], dtype=torch.bool)
    return x.to(device), y.to(device), mask.to(device)


def prepare_preference_tensors(
    examples: list[EncodedPreference],
    pad_id: int,
    device: torch.device,
) -> PreferenceBatchTensors:
    """Pre-pad encoded preference pairs once to avoid per-step tensor construction."""
    if not examples:
        raise ValueError("No encoded preference examples are available.")
    max_sequence_len = max(
        max(len(item.chosen_ids), len(item.rejected_ids))
        for item in examples
    )
    chosen_x, chosen_y, chosen_mask = pad_sequences(
        [item.chosen_ids for item in examples],
        [item.chosen_mask for item in examples],
        pad_id,
        device,
        max_sequence_len,
    )
    rejected_x, rejected_y, rejected_mask = pad_sequences(
        [item.rejected_ids for item in examples],
        [item.rejected_mask for item in examples],
        pad_id,
        device,
        max_sequence_len,
    )
    return PreferenceBatchTensors(
        chosen_x=chosen_x,
        chosen_y=chosen_y,
        chosen_mask=chosen_mask,
        rejected_x=rejected_x,
        rejected_y=rejected_y,
        rejected_mask=rejected_mask,
    )


def get_preference_batch(
    examples: list[EncodedPreference] | PreferenceBatchTensors,
    batch_size: int,
    pad_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if isinstance(examples, PreferenceBatchTensors):
        indices = torch.randint(0, len(examples), (batch_size,), device=examples.chosen_x.device)
        return (
            examples.chosen_x.index_select(0, indices),
            examples.chosen_y.index_select(0, indices),
            examples.chosen_mask.index_select(0, indices),
            examples.rejected_x.index_select(0, indices),
            examples.rejected_y.index_select(0, indices),
            examples.rejected_mask.index_select(0, indices),
        )

    sampled = random.choices(examples, k=batch_size)
    max_sequence_len = max(
        max(len(item.chosen_ids), len(item.rejected_ids))
        for item in sampled
    )
    chosen_x, chosen_y, chosen_mask = pad_sequences(
        [item.chosen_ids for item in sampled],
        [item.chosen_mask for item in sampled],
        pad_id,
        device,
        max_sequence_len,
    )
    rejected_x, rejected_y, rejected_mask = pad_sequences(
        [item.rejected_ids for item in sampled],
        [item.rejected_mask for item in sampled],
        pad_id,
        device,
        max_sequence_len,
    )
    return chosen_x, chosen_y, chosen_mask, rejected_x, rejected_y, rejected_mask


def sequence_logps(
    model: MiniGPT,
    x: torch.Tensor,
    y: torch.Tensor,
    loss_mask: torch.Tensor,
    use_amp: bool,
) -> torch.Tensor:
    logits, _ = model(x)
    log_probs = F.log_softmax(logits, dim=-1)
    token_logps = log_probs.gather(-1, y.unsqueeze(-1)).squeeze(-1)
    token_logps = token_logps * loss_mask
    lengths = loss_mask.sum(dim=-1).clamp_min(1)
    return token_logps.sum(dim=-1) / lengths


def paired_sequence_logps(
    model: MiniGPT,
    chosen_x: torch.Tensor,
    chosen_y: torch.Tensor,
    chosen_mask: torch.Tensor,
    rejected_x: torch.Tensor,
    rejected_y: torch.Tensor,
    rejected_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score chosen/rejected responses in one model forward pass."""
    x = torch.cat([chosen_x, rejected_x], dim=0)
    y = torch.cat([chosen_y, rejected_y], dim=0)
    loss_mask = torch.cat([chosen_mask, rejected_mask], dim=0)
    logits, _ = model(x)
    log_probs = F.log_softmax(logits, dim=-1)
    token_logps = log_probs.gather(-1, y.unsqueeze(-1)).squeeze(-1)
    token_logps = token_logps * loss_mask
    lengths = loss_mask.sum(dim=-1).clamp_min(1)
    logps = token_logps.sum(dim=-1) / lengths
    return logps.chunk(2, dim=0)


def dpo_loss(
    policy_model: MiniGPT,
    reference_model: MiniGPT | None,
    batch,
    beta: float,
    use_amp: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    chosen_x, chosen_y, chosen_mask, rejected_x, rejected_y, rejected_mask = batch
    with get_autocast(chosen_x.device.type, enabled=use_amp):
        pi_chosen, pi_rejected = paired_sequence_logps(
            policy_model,
            chosen_x,
            chosen_y,
            chosen_mask,
            rejected_x,
            rejected_y,
            rejected_mask,
        )
        pi_logratio = pi_chosen - pi_rejected

        if reference_model is None:
            ref_logratio = torch.zeros_like(pi_logratio)
        else:
            with torch.no_grad():
                ref_chosen, ref_rejected = paired_sequence_logps(
                    reference_model,
                    chosen_x,
                    chosen_y,
                    chosen_mask,
                    rejected_x,
                    rejected_y,
                    rejected_mask,
                )
                ref_logratio = ref_chosen - ref_rejected

        logits = beta * (pi_logratio - ref_logratio)
        loss = -F.logsigmoid(logits).mean()
        accuracy = (logits > 0).float().mean()

    metrics = {
        "loss": float(loss.detach().item()),
        "margin": float((pi_logratio - ref_logratio).detach().mean().item()),
        "accuracy": float(accuracy.detach().item()),
        "chosen_logp": float(pi_chosen.detach().mean().item()),
        "rejected_logp": float(pi_rejected.detach().mean().item()),
    }
    return loss, metrics


def average_metrics(metrics: list[dict[str, float]]) -> dict[str, float]:
    return {
        key: sum(item[key] for item in metrics) / len(metrics)
        for key in metrics[0]
    }


@torch.no_grad()
def estimate_dpo(
    policy_model: MiniGPT,
    reference_model: MiniGPT | None,
    examples: list[EncodedPreference] | PreferenceBatchTensors,
    batch_size: int,
    pad_id: int,
    device: torch.device,
    beta: float,
    use_amp: bool,
    eval_iters: int = 5,
) -> dict[str, float]:
    policy_model.eval()
    if reference_model is not None:
        reference_model.eval()
    metrics: list[dict[str, float]] = []
    for _ in range(eval_iters):
        batch = get_preference_batch(examples, batch_size, pad_id, device)
        _, batch_metrics = dpo_loss(policy_model, reference_model, batch, beta, use_amp)
        metrics.append(batch_metrics)
    policy_model.train()
    return {
        key: sum(item[key] for item in metrics) / len(metrics)
        for key in metrics[0]
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    tokenizer_path = args.vocab or args.tokenizer
    device = select_device(args.device)
    use_amp = args.amp == "on" or (args.amp == "auto" and device.type == "cuda")

    print(f"[DPO] Using device: {device}")
    policy_model, tokenizer = load_model(args.checkpoint, tokenizer_path, device)
    seq_len = args.seq_len or policy_model.max_seq_len
    raw_examples = load_preferences(args.preferences)
    encoded_examples = encode_preferences(raw_examples, tokenizer, seq_len)
    preference_batches = prepare_preference_tensors(encoded_examples, 0, device)

    if args.dry_run:
        print("[DPO] Dry run complete. No optimizer step or checkpoint written.")
        return

    reference_model = None
    if not args.reference_free:
        print("[DPO] Loading frozen reference model.")
        reference_model, _ = load_model(args.checkpoint, tokenizer_path, device)
        reference_model.eval()
        for param in reference_model.parameters():
            param.requires_grad_(False)

    param_groups = get_param_groups(policy_model, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.learning_rate, betas=(0.9, 0.95))
    scaler = get_grad_scaler(device.type, enabled=use_amp)
    if use_amp:
        print("[DPO] AMP mixed precision: enabled")

    print(f"[DPO] Starting DPO for {args.max_iters} steps | beta={args.beta}")
    policy_model.train()
    for step in range(1, args.max_iters + 1):
        lr = get_lr(step - 1, args.learning_rate, args.min_lr, args.warmup_steps, args.max_iters)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accum_metrics = []
        for _ in range(args.grad_accum_steps):
            batch = get_preference_batch(preference_batches, args.batch_size, 0, device)
            loss, metrics = dpo_loss(policy_model, reference_model, batch, args.beta, use_amp)
            loss = loss / args.grad_accum_steps
            scaler.scale(loss).backward()
            accum_metrics.append(metrics)

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        if step == 1 or step % args.log_every == 0 or step == args.max_iters:
            train_metrics = average_metrics(accum_metrics)
            print(
                f"[DPO][train] step {step:>5d}/{args.max_iters} "
                f"loss {train_metrics['loss']:.4f} margin {train_metrics['margin']:.4f} "
                f"acc {train_metrics['accuracy']:.3f} "
                f"chosen_logp {train_metrics['chosen_logp']:.4f} "
                f"rejected_logp {train_metrics['rejected_logp']:.4f} "
                f"lr {lr:.2e}",
                flush=True,
            )

        if step == 1 or step % args.eval_every == 0 or step == args.max_iters:
            eval_metrics = estimate_dpo(
                policy_model,
                reference_model,
                preference_batches,
                args.batch_size,
                0,
                device,
                args.beta,
                use_amp,
                eval_iters=min(5, max(1, math.ceil(len(encoded_examples) / args.batch_size))),
            )
            print(
                f"[DPO][eval]  step {step:>5d}/{args.max_iters} "
                f"eval_loss {eval_metrics['loss']:.4f} "
                f"margin {eval_metrics['margin']:.4f} acc {eval_metrics['accuracy']:.3f} "
                f"chosen_logp {eval_metrics['chosen_logp']:.4f} "
                f"rejected_logp {eval_metrics['rejected_logp']:.4f}",
                flush=True,
            )

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    model_path = os.path.join(args.checkpoint_dir, "model.pt")
    out_tokenizer_path = os.path.join(args.checkpoint_dir, "tokenizer.json")
    vocab_path = os.path.join(args.checkpoint_dir, "vocab.json")

    torch.save(
        {
            "model_state": policy_model.state_dict(),
            "config": {
                **policy_model.config,
                "dpo_beta": args.beta,
                "dpo_base_checkpoint": args.checkpoint,
            },
        },
        model_path,
    )
    tokenizer.save(out_tokenizer_path)
    if Path(out_tokenizer_path) != Path(vocab_path):
        shutil.copyfile(out_tokenizer_path, vocab_path)

    print(f"[DPO] Model saved -> {model_path}")
    print(f"[DPO] Tokenizer saved -> {out_tokenizer_path}")


if __name__ == "__main__":
    main()
