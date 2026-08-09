"""
train_lora.py
-------------
LoRA supervised fine-tuning of a local Qwen3-0.6B model on the
chat-format dataset produced by build_sft_dataset.py.

Plain PyTorch training loop (no Trainer/accelerate), answer-only loss:
labels are -100 everywhere except assistant message content + <|im_end|>.

Usage:
    python build_sft_dataset.py
    python train_lora.py
    python train_lora.py --epochs 3 --micro_batch 4 --grad_accum 4 --save_merged
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
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA SFT for local Qwen model")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "rocm", "cpu"],
                        help="auto 优先 CUDA，其次 ROCm (AMD)，最后 CPU (fp32)。")
    parser.add_argument("--model_dir", type=str,
                        default="hf_models/qwen35-2b")
    parser.add_argument("--data_dir", type=str, default="sft_data")
    parser.add_argument("--out_dir", type=str,
                        default="checkpoints_lora/qwen35-2b-gravity")
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--micro_batch", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", type=str,
                        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
                        help="Comma-separated module names adapted by LoRA.")
    parser.add_argument("--eval_every", type=int, default=50)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad_checkpointing", action="store_true")
    parser.add_argument("--save_merged", action="store_true",
                        help="Also export base+adapter merged weights.")
    return parser.parse_args()


def load_conversations(path: str) -> list[list[dict]]:
    conversations = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                conversations.append(json.loads(line)["messages"])
    return conversations


def encode_conversation(
    tokenizer,
    messages: list[dict],
    seq_len: int,
) -> tuple[list[int], list[int]] | None:
    """Tokenize in Qwen chat format, training only assistant content + im_end."""
    input_ids: list[int] = []
    labels: list[int] = []

    def add(text: str, train: bool = False) -> None:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        input_ids.extend(ids)
        labels.extend(ids if train else [-100] * len(ids))

    for message in messages:
        role, content = message["role"], message["content"]
        if role == "assistant":
            add(f"{IM_START}assistant\n")
            add(content, train=True)
            add(IM_END, train=True)
            add("\n")
        else:
            add(f"{IM_START}{role}\n{content}{IM_END}\n")

    if len(input_ids) > seq_len:
        return None
    return input_ids, labels


class PaddedBatches:
    """Shuffle, batch, and right-pad encoded examples each epoch."""

    def __init__(
        self,
        examples: list[tuple[list[int], list[int]]],
        micro_batch: int,
        pad_id: int,
        device: torch.device,
        seed: int,
    ):
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
            chunk = [self.examples[i] for i in order[start:start + self.micro_batch]]
            max_len = max(len(ids) for ids, _ in chunk)
            batch_ids, batch_labels, batch_mask = [], [], []
            for ids, labels in chunk:
                pad = max_len - len(ids)
                batch_ids.append(ids + [self.pad_id] * pad)
                batch_labels.append(labels + [-100] * pad)
                batch_mask.append([1] * len(ids) + [0] * pad)
            yield (
                torch.tensor(batch_ids, dtype=torch.long, device=self.device),
                torch.tensor(batch_labels, dtype=torch.long, device=self.device),
                torch.tensor(batch_mask, dtype=torch.long, device=self.device),
            )


def get_lr(step: int, max_lr: float, min_lr: float, warmup: int, total: int) -> float:
    if step < warmup:
        return max_lr * (step + 1) / max(1, warmup)
    if step >= total:
        return min_lr
    progress = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, batches: PaddedBatches, device: torch.device) -> float:
    model.eval()
    total_loss, count = 0.0, 0
    for ids, labels, mask in batches.epoch(shuffle=False):
        with autocast_context(device):
            out = model(input_ids=ids, attention_mask=mask, labels=labels)
        total_loss += out.loss.item()
        count += 1
    model.train()
    return total_loss / max(1, count)


def resolve_device(choice: str) -> torch.device:
    """解析 --device：auto 优先 CUDA，其次 ROCm (AMD)，最后 CPU。"""
    if choice == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch, "hip", None) is not None and torch.hip.is_available():
            return torch.device("rocm")
        return torch.device("cpu")
    if choice == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is required for --device cuda (bf16 training).")
    return torch.device(choice)


def autocast_context(device: torch.device):
    """CUDA/ROCm 上 bf16 autocast；CPU 上纯 fp32。"""
    if device.type in ("cuda", "rocm"):
        return torch.autocast(device.type, dtype=torch.bfloat16)
    return contextlib.nullcontext()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = resolve_device(args.device)
    if device.type == "cpu":
        print("[train] 使用 CPU (fp32)，训练会明显变慢，建议使用 --device cuda/rocm")
    else:
        print(f"[train] 使用 {device.type.upper()} (bf16)")

    base = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base, args.model_dir)
    data_dir = os.path.join(base, args.data_dir)
    out_dir = os.path.join(base, args.out_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    train_convs = load_conversations(os.path.join(data_dir, "train.jsonl"))
    val_convs = load_conversations(os.path.join(data_dir, "val.jsonl"))

    def encode_all(convs: list[list[dict]], name: str):
        encoded, skipped = [], 0
        for messages in convs:
            item = encode_conversation(tokenizer, messages, args.seq_len)
            if item is None:
                skipped += 1
            else:
                encoded.append(item)
        if skipped:
            print(f"[SFT] {name}: skipped {skipped} conversations over {args.seq_len} tokens.")
        lengths = [len(ids) for ids, _ in encoded]
        if not lengths:
            raise ValueError(f"No {name} conversations fit within {args.seq_len} tokens.")
        print(f"[SFT] {name}: {len(encoded)} examples, "
              f"avg {sum(lengths) / len(lengths):.0f} / max {max(lengths)} tokens.")
        return encoded

    train_examples = encode_all(train_convs, "train")
    val_examples = encode_all(val_convs, "val")

    model_kwargs = {
        "dtype": torch.bfloat16,
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
    model = AutoModelForCausalLM.from_pretrained(model_dir, **model_kwargs).to(device)
    model.config.use_cache = False
    if args.grad_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[name.strip() for name in args.target_modules.split(",") if name.strip()],
    )
    model = get_peft_model(model, lora_config)

    # Keep LoRA master weights in fp32 for stable AdamW updates; matmuls
    # still run in bf16 under autocast.
    trainable_params = []
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.float()
            trainable_params.append(param)
    model.print_trainable_parameters()

    train_batches = PaddedBatches(train_examples, args.micro_batch, pad_id, device, args.seed)
    val_batches = PaddedBatches(val_examples, args.micro_batch, pad_id, device, args.seed)

    steps_per_epoch = math.ceil(len(train_batches) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))
    min_lr = args.learning_rate * args.min_lr_ratio
    print(f"[SFT] {steps_per_epoch} optimizer steps/epoch x {args.epochs} epochs "
          f"= {total_steps} steps (warmup {warmup_steps}).")
    print(f"[SFT] Effective batch size: {args.micro_batch * args.grad_accum}.")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "training_meta.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    model.train()
    global_step = 0
    best_val = float("inf")
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        micro_iter = train_batches.epoch(shuffle=True)
        for _ in range(steps_per_epoch):
            lr = get_lr(global_step, args.learning_rate, min_lr, warmup_steps, total_steps)
            for group in optimizer.param_groups:
                group["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            accum_loss, micro_count = 0.0, 0
            for _ in range(args.grad_accum):
                try:
                    ids, labels, mask = next(micro_iter)
                except StopIteration:
                    break
                with autocast_context(device):
                    out = model(input_ids=ids, attention_mask=mask, labels=labels)
                loss = out.loss / args.grad_accum
                loss.backward()
                accum_loss += loss.item()
                micro_count += 1
            if micro_count == 0:
                break

            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            global_step += 1

            if global_step % args.log_every == 0 or global_step == total_steps:
                elapsed = time.time() - started
                rate = global_step / elapsed
                eta = (total_steps - global_step) / rate if rate > 0 else 0.0
                print(f"[SFT] epoch {epoch} step {global_step}/{total_steps} "
                      f"loss {accum_loss:.4f} lr {lr:.2e} "
                      f"({rate:.2f} it/s, eta {eta / 60:.1f} min)", flush=True)

            if global_step % args.eval_every == 0 or global_step == total_steps:
                val_loss = evaluate(model, val_batches, device)
                marker = ""
                if val_loss < best_val:
                    best_val = val_loss
                    model.save_pretrained(os.path.join(out_dir, "adapter_best"))
                    marker = " (best, saved)"
                print(f"[SFT] eval step {global_step}: val loss {val_loss:.4f}{marker}",
                      flush=True)

    model.save_pretrained(os.path.join(out_dir, "adapter_last"))
    tokenizer.save_pretrained(os.path.join(out_dir, "adapter_last"))
    print(f"[SFT] Final adapter -> {os.path.join(out_dir, 'adapter_last')}")
    print(f"[SFT] Best adapter (val {best_val:.4f}) -> {os.path.join(out_dir, 'adapter_best')}")

    if args.save_merged:
        print("[SFT] Merging adapter into base weights...")
        # 先移回 CPU 再 merge：ROCm/AMD 上直接保存 GPU 张量可能出错
        model = model.to("cpu")
        merged = model.merge_and_unload()
        merged_dir = os.path.join(out_dir, "merged")
        merged.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
        print(f"[SFT] Merged model -> {merged_dir}")

    print(f"[SFT] Done in {(time.time() - started) / 60:.1f} min.")


if __name__ == "__main__":
    main()
