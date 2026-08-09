"""
train_dpo_lora.py
-----------------
Preference optimization for the LoRA fine-tuned Qwen model, continuing from
the SFT adapter produced by train_lora.py.

Algorithm upgrades over the legacy train_dpo.py:
  1. Reference log-probs are precomputed once from the frozen SFT state —
     exact reference, no second model in memory, no per-step reference pass.
  2. cDPO label smoothing absorbs judge noise in synthetic preference pairs.
  3. A small NLL term on chosen answers anchors absolute likelihood so the
     margin cannot grow by sinking both responses.
  4. Real shuffled epochs over the full dataset plus a held-out validation
     split with best-adapter selection (the old loop sampled with
     replacement and covered a few percent of the data).
  5. Kept from the old design: chosen and rejected scored in one forward.

Usage:
    python build_dpo_dataset.py
    python train_dpo_lora.py
    python train_dpo_lora.py --save_merged
    python train_dpo_lora.py --device cpu
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import math
import os
import random
import time

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from train_lora import encode_conversation, get_lr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DPO on top of the SFT LoRA adapter")
    parser.add_argument("--model_dir", type=str,
                        default="hf_models/qwen3-0.6b")
    parser.add_argument("--sft_adapter", type=str,
                        default="checkpoints_lora/qwen3-06b-gravity/adapter_best")
    parser.add_argument("--data_dir", type=str, default="dpo_data")
    parser.add_argument("--out_dir", type=str,
                        default="checkpoints_lora/qwen3-06b-gravity-dpo")
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--micro_batch", type=int, default=4,
                        help="Preference pairs per micro-batch (2x sequences).")
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--label_smoothing", type=float, default=0.1,
                        help="cDPO smoothing for noisy preference labels.")
    parser.add_argument("--nll_coef", type=float, default=0.05,
                        help="Weight of the per-token NLL anchor on chosen.")
    parser.add_argument("--length_normalized", action="store_true",
                        help="Use per-token average log-probs in the margin.")
    parser.add_argument("--eval_every", type=int, default=20)
    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_merged", action="store_true")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "rocm", "cpu"],
                        help="auto 优先 CUDA，其次 ROCm (AMD)，最后 CPU (fp32)。")
    return parser.parse_args()


def load_pairs(path: str) -> list[dict]:
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs


def encode_pair(tokenizer, record: dict, seq_len: int):
    """Encode chosen/rejected continuations; score only the final response."""
    prompt_ids, _ = encode_conversation(tokenizer, record["prompt_messages"], seq_len) or (None, None)
    if prompt_ids is None:
        return None

    sides = []
    for key in ("chosen", "rejected"):
        messages = record["prompt_messages"] + [
            {"role": "assistant", "content": record[key]}
        ]
        encoded = encode_conversation(tokenizer, messages, seq_len)
        if encoded is None:
            return None
        ids, labels = encoded
        # Mask any earlier assistant turns: only the final response is scored.
        labels = [-100] * len(prompt_ids) + labels[len(prompt_ids):]
        if not any(l != -100 for l in labels):
            return None
        sides.append((ids, labels))
    weight = float(record.get("weight", 1.0) or 1.0)
    return sides[0], sides[1], max(0.01, weight)


class PairBatcher:
    """Yield padded (2B, T) tensors: chosen block stacked over rejected block."""

    def __init__(self, examples, micro_batch: int, pad_id: int, device, seed: int):
        self.examples = examples
        self.micro_batch = micro_batch
        self.pad_id = pad_id
        self.device = device
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return math.ceil(len(self.examples) / self.micro_batch)

    def epoch(self, shuffle: bool = True):
        order = list(range(len(self.examples)))
        if shuffle:
            self.rng.shuffle(order)
        for start in range(0, len(order), self.micro_batch):
            indices = order[start:start + self.micro_batch]
            chunk = [self.examples[i] for i in indices]
            sequences = [pair[0] for pair in chunk] + [pair[1] for pair in chunk]
            weights = [pair[2] for pair in chunk]
            max_len = max(len(ids) for ids, _ in sequences)
            batch_ids, batch_labels, batch_mask = [], [], []
            for ids, labels in sequences:
                pad = max_len - len(ids)
                batch_ids.append(ids + [self.pad_id] * pad)
                batch_labels.append(labels + [-100] * pad)
                batch_mask.append([1] * len(ids) + [0] * pad)
            yield (
                indices,
                torch.tensor(batch_ids, dtype=torch.long, device=self.device),
                torch.tensor(batch_labels, dtype=torch.long, device=self.device),
                torch.tensor(batch_mask, dtype=torch.long, device=self.device),
                torch.tensor(weights, dtype=torch.float32, device=self.device),
            )


def autocast_forward(device_type: str):
    """bf16 autocast on CUDA/ROCm; plain fp32 execution on CPU."""
    if device_type in ("cuda", "rocm"):
        return torch.autocast(device_type, dtype=torch.bfloat16)
    return contextlib.nullcontext()


def sequence_logps(model, ids, labels, mask):
    """Sum of response-token log-probs and response token counts, per row."""
    with autocast_forward(ids.device.type):
        logits = model(input_ids=ids, attention_mask=mask).logits
    logprobs = F.log_softmax(logits[:, :-1].float(), dim=-1)
    targets = ids[:, 1:]
    score_mask = (labels[:, 1:] != -100)
    token_logps = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    seq_logp = (token_logps * score_mask).sum(dim=-1)
    token_counts = score_mask.sum(dim=-1).clamp_min(1)
    return seq_logp, token_counts


def disable_dropout(model: torch.nn.Module) -> None:
    """Policy and reference should differ only by weights, not by noise."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0


@torch.no_grad()
def precompute_reference(model, batcher: PairBatcher, name: str):
    model.eval()
    count = len(batcher.examples)
    ref_logp = torch.zeros(2, count)
    ref_tokens = torch.zeros(2, count)
    for indices, ids, labels, mask, _weights in batcher.epoch(shuffle=False):
        seq_logp, token_counts = sequence_logps(model, ids, labels, mask)
        half = len(indices)
        for row, example_index in enumerate(indices):
            ref_logp[0, example_index] = seq_logp[row].item()
            ref_logp[1, example_index] = seq_logp[half + row].item()
            ref_tokens[0, example_index] = token_counts[row].item()
            ref_tokens[1, example_index] = token_counts[half + row].item()
    print(f"[DPO] Precomputed reference log-probs for {count} {name} pairs.")
    return ref_logp, ref_tokens


def dpo_metrics(
    pi_chosen, pi_rejected, ref_chosen, ref_rejected,
    chosen_tokens, weights, beta: float, label_smoothing: float, nll_coef: float,
):
    z = beta * ((pi_chosen - ref_chosen) - (pi_rejected - ref_rejected))
    weights = weights.float().clamp_min(0.01)
    normalizer = weights.sum().clamp_min(1e-6)
    pref_loss_per_pair = -(
        (1.0 - label_smoothing) * F.logsigmoid(z)
        + label_smoothing * F.logsigmoid(-z)
    )
    nll_per_pair = -(pi_chosen / chosen_tokens)
    pref_loss = (pref_loss_per_pair * weights).sum() / normalizer
    nll = (nll_per_pair * weights).sum() / normalizer
    loss = pref_loss + nll_coef * nll
    weighted_acc = (((z > 0).float() * weights).sum() / normalizer)
    return loss, {
        "loss": float(loss.detach().item()),
        "pref_loss": float(pref_loss.detach().item()),
        "nll": float(nll.detach().item()),
        "acc": float(weighted_acc.detach().item()),
        "margin": float(((z / beta) * weights).sum().detach().item() / float(normalizer.detach().item())),
        "chosen_drift": float(((pi_chosen - ref_chosen) * weights).sum().detach().item() / float(normalizer.detach().item())),
        "mean_weight": float(weights.detach().mean().item()),
    }


def run_batch(model, batch, ref_logp, ref_tokens, args, device):
    indices, ids, labels, mask, weights = batch
    half = len(indices)
    seq_logp, token_counts = sequence_logps(model, ids, labels, mask)
    index_tensor = torch.tensor(indices, dtype=torch.long)
    ref_c = ref_logp[0, index_tensor].to(device)
    ref_r = ref_logp[1, index_tensor].to(device)
    pi_c, pi_r = seq_logp[:half], seq_logp[half:]
    tok_c = token_counts[:half].float()

    if args.length_normalized:
        tok_r = token_counts[half:].float()
        ref_tc = ref_tokens[0, index_tensor].to(device).clamp_min(1)
        ref_tr = ref_tokens[1, index_tensor].to(device).clamp_min(1)
        pi_c, pi_r = pi_c / tok_c, pi_r / tok_r
        ref_c, ref_r = ref_c / ref_tc, ref_r / ref_tr
        anchor_tokens = torch.ones_like(tok_c)
    else:
        anchor_tokens = tok_c

    return dpo_metrics(
        pi_c, pi_r, ref_c, ref_r, anchor_tokens,
        weights,
        args.beta, args.label_smoothing, args.nll_coef,
    )


@torch.no_grad()
def evaluate(model, batcher, ref_logp, ref_tokens, args, device):
    model.eval()
    totals: dict[str, float] = {}
    batches = 0
    for batch in batcher.epoch(shuffle=False):
        _, metrics = run_batch(model, batch, ref_logp, ref_tokens, args, device)
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value
        batches += 1
    model.train()
    return {key: value / max(1, batches) for key, value in totals.items()}


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if args.device == "auto":
        if torch.cuda.is_available():
            args.device = "cuda"
        elif getattr(torch, "hip", None) is not None and torch.hip.is_available():
            args.device = "rocm"
        else:
            args.device = "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is not available.")
    device = torch.device(args.device)
    if device.type == "cpu":
        print("[DPO] Training on CPU with the checkpoint dtype where possible.")

    base = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base, args.model_dir)
    data_dir = os.path.join(base, args.data_dir)
    out_dir = os.path.join(base, args.out_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    def encode_split(name: str):
        records = load_pairs(os.path.join(data_dir, f"{name}.jsonl"))
        encoded, skipped = [], 0
        for record in records:
            item = encode_pair(tokenizer, record, args.seq_len)
            if item is None:
                skipped += 1
            else:
                encoded.append(item)
        print(f"[DPO] {name}: {len(encoded)} pairs encoded"
              + (f", {skipped} skipped" if skipped else "") + ".")
        return encoded

    train_pairs = encode_split("train")
    val_pairs = encode_split("val")
    if not train_pairs or not val_pairs:
        raise ValueError("DPO needs at least one encoded train pair and one encoded val pair.")

    model_kwargs = {
        "dtype": torch.bfloat16 if device.type == "cuda" else "auto",
        "attn_implementation": "sdpa",
        "local_files_only": True,
    }
    if (
        os.path.exists(os.path.join(model_dir, "model.safetensors"))
        or os.path.exists(os.path.join(model_dir, "model.safetensors.index.json"))
    ):
        model_kwargs["use_safetensors"] = True
    if importlib.util.find_spec("accelerate") is not None:
        model_kwargs["low_cpu_mem_usage"] = True
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        **model_kwargs,
    ).to(device)
    model.config.use_cache = False
    model = PeftModel.from_pretrained(
        model, os.path.join(base, args.sft_adapter), is_trainable=True,
    )
    disable_dropout(model)
    print(f"[DPO] Continuing from SFT adapter: {args.sft_adapter}")

    trainable_params = []
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.float()
            trainable_params.append(param)
    print(f"[DPO] Trainable params: {sum(p.numel() for p in trainable_params):,}")

    train_batcher = PairBatcher(train_pairs, args.micro_batch, pad_id, device, args.seed)
    val_batcher = PairBatcher(val_pairs, args.micro_batch, pad_id, device, args.seed)

    # The adapter still holds its SFT weights here, so this one pass captures
    # the exact reference policy for the whole run.
    train_ref = precompute_reference(model, train_batcher, "train")
    val_ref = precompute_reference(model, val_batcher, "val")
    model.train()

    steps_per_epoch = math.ceil(len(train_batcher) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))
    min_lr = args.learning_rate * args.min_lr_ratio
    print(f"[DPO] {steps_per_epoch} steps/epoch x {args.epochs} epochs = {total_steps} steps "
          f"(warmup {warmup_steps}) | beta={args.beta} eps={args.label_smoothing} "
          f"nll_coef={args.nll_coef} length_normalized={args.length_normalized}")

    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.learning_rate, betas=(0.9, 0.95), eps=1e-8,
    )

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "training_meta.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    baseline = evaluate(model, val_batcher, *val_ref, args=args, device=device)
    print(f"[DPO] baseline val: loss {baseline['loss']:.4f} acc {baseline['acc']:.3f} "
          f"margin {baseline['margin']:.3f}")

    global_step = 0
    best_val = float("inf")
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        micro_iter = train_batcher.epoch(shuffle=True)
        for _ in range(steps_per_epoch):
            lr = get_lr(global_step, args.learning_rate, min_lr, warmup_steps, total_steps)
            for group in optimizer.param_groups:
                group["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            step_metrics, micro_count = {}, 0
            for _ in range(args.grad_accum):
                try:
                    batch = next(micro_iter)
                except StopIteration:
                    break
                loss, metrics = run_batch(model, batch, *train_ref, args=args, device=device)
                (loss / args.grad_accum).backward()
                for key, value in metrics.items():
                    step_metrics[key] = step_metrics.get(key, 0.0) + value
                micro_count += 1
            if micro_count == 0:
                break
            step_metrics = {k: v / micro_count for k, v in step_metrics.items()}

            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            global_step += 1

            if global_step % args.log_every == 0 or global_step == total_steps:
                elapsed = time.time() - started
                rate = global_step / elapsed
                print(f"[DPO] epoch {epoch} step {global_step}/{total_steps} "
                      f"loss {step_metrics['loss']:.4f} acc {step_metrics['acc']:.3f} "
                      f"margin {step_metrics['margin']:.2f} "
                      f"chosen_drift {step_metrics['chosen_drift']:.2f} "
                      f"lr {lr:.2e} ({rate:.2f} it/s)", flush=True)

            if global_step % args.eval_every == 0 or global_step == total_steps:
                val_metrics = evaluate(model, val_batcher, *val_ref, args=args, device=device)
                marker = ""
                if val_metrics["loss"] < best_val:
                    best_val = val_metrics["loss"]
                    model.save_pretrained(os.path.join(out_dir, "adapter_best"))
                    marker = " (best, saved)"
                print(f"[DPO] eval step {global_step}: loss {val_metrics['loss']:.4f} "
                      f"acc {val_metrics['acc']:.3f} margin {val_metrics['margin']:.2f} "
                      f"chosen_drift {val_metrics['chosen_drift']:.2f}{marker}", flush=True)

    model.save_pretrained(os.path.join(out_dir, "adapter_last"))
    tokenizer.save_pretrained(os.path.join(out_dir, "adapter_last"))
    print(f"[DPO] Final adapter -> {os.path.join(out_dir, 'adapter_last')}")
    print(f"[DPO] Best adapter (val loss {best_val:.4f}) -> {os.path.join(out_dir, 'adapter_best')}")

    if args.save_merged:
        print("[DPO] Merging adapter into base weights...")
        # 先移回 CPU 再 merge：ROCm/AMD 上直接保存 GPU 张量可能出错
        model = model.to("cpu")
        merged = model.merge_and_unload()
        merged_dir = os.path.join(out_dir, "merged")
        merged.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
        print(f"[DPO] Merged model -> {merged_dir}")

    print(f"[DPO] Done in {(time.time() - started) / 60:.1f} min.")


if __name__ == "__main__":
    main()
