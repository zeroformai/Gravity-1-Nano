"""
train_rl.py
-----------
Lightweight online reinforcement learning for Gravity checkpoints.

The loop samples answers from the current policy, scores them against Q/A
targets with a problem-solving reward, and updates the model with a clipped
REINFORCE-style objective plus a small reference KL penalty.
"""

from __future__ import annotations

import argparse
import os
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from generate import (
    build_qa_prompt,
    extract_answer,
    get_bad_token_ids,
    has_bad_text,
    is_low_quality_answer,
    load_model,
)
from model import MiniGPT
from tokenizer import EOS_TOKEN
from train import get_lr, get_param_groups
from utils import get_autocast, get_grad_scaler, load_text_files, parse_qa_examples, set_seed


@dataclass(frozen=True)
class RLExample:
    question: str
    target: str


@dataclass(frozen=True)
class Rollout:
    prompt: str
    question: str
    target: str
    answer: str
    ids: list[int]
    mask: list[int]
    reward: float
    entropy: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RL tune a Gravity checkpoint")
    parser.add_argument("--data", type=str, default="corpus/priority_qa.txt;corpus/qa_10k.txt")
    parser.add_argument("--checkpoint", type=str, default="checkpoints_dpo_casual_step8_final/model.pt")
    parser.add_argument("--tokenizer", type=str, default="checkpoints_dpo_casual_step8_final/tokenizer.json")
    parser.add_argument("--vocab", type=str, default=None, help="Alias for --tokenizer")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_rl")
    parser.add_argument("--max_examples", type=int, default=2000)
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_iters", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--learning_rate", type=float, default=2e-6)
    parser.add_argument("--min_lr", type=float, default=2e-7)
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--kl_coef", type=float, default=0.03)
    parser.add_argument("--reward_baseline", type=float, default=0.35)
    parser.add_argument("--advantage_clip", type=float, default=1.5)
    parser.add_argument(
        "--allow_negative_only_updates",
        action="store_true",
        help="Allow optimizer steps when every rollout scores at or below the reward baseline.",
    )
    parser.add_argument("--temperature", type=float, default=0.55)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--amp", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
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


def load_rl_examples(data_spec: str, limit: int) -> list[RLExample]:
    examples: list[RLExample] = []
    for _, text in load_text_files(data_spec):
        for item in parse_qa_examples(text):
            if item.question.strip() and item.answer.strip():
                examples.append(RLExample(item.question.strip(), item.answer.strip()))
    if not examples:
        raise ValueError(f"No Q/A examples found in {data_spec}")
    random.shuffle(examples)
    if limit > 0:
        examples = examples[:limit]
    print(f"[RL] Loaded {len(examples):,} reward prompts.")
    return examples


def _features(text: str) -> set[str]:
    normalized = re.sub(r"\s+", " ", text.lower()).strip()
    words = {
        word
        for word in re.findall(r"[a-z0-9']+", normalized)
        if len(word) > 1
    }
    for segment in re.findall(r"[\u4e00-\u9fff]+", normalized):
        words.update(segment[i:i + 2] for i in range(max(0, len(segment) - 1)))
        words.update(segment[i:i + 3] for i in range(max(0, len(segment) - 2)))
    return words


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(needle.lower() in lowered for needle in needles)


def reward_answer(question: str, answer: str, target: str) -> float:
    stripped = re.sub(r"\s+", " ", answer).strip()
    if not stripped or has_bad_text(stripped) or is_low_quality_answer(stripped, question):
        return -1.0

    answer_feats = _features(stripped)
    target_feats = _features(target)
    question_feats = _features(question)
    if not answer_feats:
        return -0.6

    target_recall = len(answer_feats & target_feats) / max(1, len(target_feats))
    target_jaccard = len(answer_feats & target_feats) / max(1, len(answer_feats | target_feats))
    question_use = len(answer_feats & question_feats) / max(1, len(question_feats))
    reward = 0.15 + 0.65 * target_recall + 0.35 * target_jaccard + 0.15 * question_use

    length = len(stripped)
    if 24 <= length <= 420:
        reward += 0.15
    elif length < 8:
        reward -= 0.35
    elif length > 700:
        reward -= 0.25

    if _contains_any(
        stripped,
        (
            "first", "then", "because", "step", "check", "verify", "try",
            "goal", "constraint", "example", "next",
            "先", "再", "因为", "步骤", "目标", "约束", "例子", "检查", "验证",
        ),
    ):
        reward += 0.15

    if stripped in question or question in stripped:
        reward -= 0.3
    if EOS_TOKEN in stripped:
        reward -= 0.2
    return max(-1.0, min(1.5, reward))


def _apply_top_k_top_p(logits: torch.Tensor, top_k: int | None, top_p: float | None) -> torch.Tensor:
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        values, _ = torch.topk(logits, k)
        logits = logits.masked_fill(logits < values[:, -1].unsqueeze(-1), float("-inf"))

    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = False
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, float("-inf"))
    return logits


@torch.no_grad()
def sample_answer(
    model: MiniGPT,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    device: torch.device,
    suppress_bad_tokens: bool = True,
) -> tuple[str, list[int], list[int], float]:
    was_training = model.training
    model.eval()
    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        raise ValueError("Prompt encodes to an empty sequence.")
    if len(prompt_ids) >= model.max_seq_len:
        prompt_ids = prompt_ids[-(model.max_seq_len - 1):]

    ctx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    logits, kv_caches = model(ctx)
    generated_ids = list(prompt_ids)
    entropy_total = 0.0
    entropy_count = 0
    bad_token_ids = None
    if suppress_bad_tokens:
        bad_token_ids = get_bad_token_ids(tokenizer, model.token_emb.num_embeddings, device)
        bad_token_ids = bad_token_ids[bad_token_ids < model.token_emb.num_embeddings]

    for _ in range(max_new_tokens):
        next_logits = logits[:, -1, :].clone()
        if bad_token_ids is not None:
            if bad_token_ids.numel() > 0:
                next_logits.index_fill_(1, bad_token_ids, float("-inf"))
        next_logits = _apply_top_k_top_p(next_logits / max(temperature, 1e-6), top_k, top_p)
        probs = torch.softmax(next_logits, dim=-1)
        entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)
        entropy_total += float(entropy.item())
        entropy_count += 1

        next_id = torch.multinomial(probs, num_samples=1)
        token_id = int(next_id.item())
        generated_ids.append(token_id)
        if token_id == tokenizer.eos_id:
            break
        if kv_caches and kv_caches[0] is not None and kv_caches[0][0].shape[2] + 1 >= model.max_seq_len:
            break
        logits, kv_caches = model(next_id, kv_caches=kv_caches)

    decoded = tokenizer.decode(generated_ids, skip_special=False)
    answer = extract_answer(prompt, decoded)
    target_count = len(generated_ids) - 1
    answer_start = max(0, len(prompt_ids) - 1)
    mask = [0] * target_count
    for index in range(answer_start, target_count):
        mask[index] = 1
    if was_training:
        model.train()
    return answer, generated_ids, mask, entropy_total / max(1, entropy_count)


def pad_rollouts(
    rollouts: list[Rollout],
    pad_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    max_len = max(len(item.ids) for item in rollouts)
    input_len = max_len - 1
    batch = len(rollouts)
    x = torch.full((batch, input_len), pad_id, dtype=torch.long)
    y = torch.full((batch, input_len), pad_id, dtype=torch.long)
    mask = torch.zeros((batch, input_len), dtype=torch.bool)
    rewards = torch.tensor([item.reward for item in rollouts], dtype=torch.float32)

    for row, item in enumerate(rollouts):
        ids = item.ids
        length = len(ids) - 1
        x[row, :length] = torch.tensor(ids[:-1], dtype=torch.long)
        y[row, :length] = torch.tensor(ids[1:], dtype=torch.long)
        mask[row, :len(item.mask)] = torch.tensor(item.mask, dtype=torch.bool)
    return x.to(device), y.to(device), mask.to(device), rewards.to(device)


def sequence_logps(
    model: MiniGPT,
    x: torch.Tensor,
    y: torch.Tensor,
    loss_mask: torch.Tensor,
    use_amp: bool,
) -> torch.Tensor:
    with get_autocast(x.device.type, enabled=use_amp):
        logits, _ = model(x)
        log_probs = F.log_softmax(logits, dim=-1)
        token_logps = log_probs.gather(-1, y.unsqueeze(-1)).squeeze(-1)
        token_logps = token_logps * loss_mask
        lengths = loss_mask.sum(dim=-1).clamp_min(1)
        return token_logps.sum(dim=-1) / lengths


def rl_loss(
    policy_model: MiniGPT,
    reference_model: MiniGPT,
    rollouts: list[Rollout],
    pad_id: int,
    device: torch.device,
    reward_baseline: float,
    advantage_clip: float,
    kl_coef: float,
    use_amp: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    x, y, mask, rewards = pad_rollouts(rollouts, pad_id, device)
    pi_logp = sequence_logps(policy_model, x, y, mask, use_amp)
    with torch.no_grad():
        ref_logp = sequence_logps(reference_model, x, y, mask, use_amp)

    advantages = (rewards - reward_baseline).clamp(-advantage_clip, advantage_clip)
    kl_delta = pi_logp - ref_logp
    loss = -(advantages.detach() * pi_logp).mean() + kl_coef * kl_delta.pow(2).mean()
    metrics = {
        "loss": float(loss.detach().item()),
        "reward": float(rewards.mean().item()),
        "advantage": float(advantages.mean().item()),
        "kl_delta": float(kl_delta.detach().mean().item()),
        "policy_logp": float(pi_logp.detach().mean().item()),
        "ref_logp": float(ref_logp.detach().mean().item()),
        "entropy": sum(item.entropy for item in rollouts) / len(rollouts),
    }
    return loss, metrics


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    tokenizer_path = args.vocab or args.tokenizer
    device = select_device(args.device)
    use_amp = args.amp == "on" or (args.amp == "auto" and device.type == "cuda")

    print(f"[RL] Using device: {device}")
    examples = load_rl_examples(args.data, args.max_examples)
    policy_model, tokenizer = load_model(args.checkpoint, tokenizer_path, device)
    seq_len = args.seq_len or policy_model.max_seq_len
    policy_model.max_seq_len = seq_len

    if args.dry_run:
        sample = random.choice(examples)
        prompt = build_qa_prompt([], sample.question)
        answer, _, _, entropy = sample_answer(
            policy_model,
            tokenizer,
            prompt,
            args.max_new_tokens,
            args.temperature,
            args.top_k,
            args.top_p,
            device,
        )
        print(f"[RL] Dry run sample reward: {reward_answer(sample.question, answer, sample.target):.3f}")
        print(f"[RL] Dry run entropy: {entropy:.3f}")
        print("[RL] Dry run complete. No optimizer step or checkpoint written.")
        return

    print("[RL] Loading frozen reference model.")
    reference_model, _ = load_model(args.checkpoint, tokenizer_path, device)
    reference_model.eval()
    for param in reference_model.parameters():
        param.requires_grad_(False)

    param_groups = get_param_groups(policy_model, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.learning_rate, betas=(0.9, 0.95))
    scaler = get_grad_scaler(device.type, enabled=use_amp)
    if use_amp:
        print("[RL] AMP mixed precision: enabled")

    print(
        f"[RL] Starting online RL for {args.max_iters} steps | "
        f"batch={args.batch_size} kl_coef={args.kl_coef}"
    )
    policy_model.train()
    skipped_updates = 0
    for step in range(1, args.max_iters + 1):
        lr = get_lr(step - 1, args.learning_rate, args.min_lr, args.warmup_steps, args.max_iters)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        sampled = random.choices(examples, k=args.batch_size)
        rollouts: list[Rollout] = []
        for example in sampled:
            prompt = build_qa_prompt([], example.question)
            answer, ids, mask, entropy = sample_answer(
                policy_model,
                tokenizer,
                prompt,
                args.max_new_tokens,
                args.temperature,
                args.top_k,
                args.top_p,
                device,
            )
            if len(ids) > seq_len:
                ids = ids[:seq_len]
                mask = mask[:max(0, seq_len - 1)]
            reward = reward_answer(example.question, answer, example.target)
            rollouts.append(Rollout(prompt, example.question, example.target, answer, ids, mask, reward, entropy))

        best_reward = max(item.reward for item in rollouts)
        skipped_update = False
        if not args.allow_negative_only_updates and best_reward <= args.reward_baseline:
            skipped_update = True
            skipped_updates += 1
            rewards = torch.tensor([item.reward for item in rollouts], dtype=torch.float32)
            metrics = {
                "loss": 0.0,
                "reward": float(rewards.mean().item()),
                "advantage": float((rewards - args.reward_baseline).mean().item()),
                "kl_delta": 0.0,
                "policy_logp": 0.0,
                "ref_logp": 0.0,
                "entropy": sum(item.entropy for item in rollouts) / len(rollouts),
            }
        else:
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = rl_loss(
                policy_model,
                reference_model,
                rollouts,
                pad_id=0,
                device=device,
                reward_baseline=args.reward_baseline,
                advantage_clip=args.advantage_clip,
                kl_coef=args.kl_coef,
                use_amp=use_amp,
            )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

        if step == 1 or step % args.log_every == 0 or step == args.max_iters:
            print(
                f"[RL][train] step {step:>5d}/{args.max_iters} "
                f"loss {metrics['loss']:.4f} reward {metrics['reward']:.3f} "
                f"adv {metrics['advantage']:.3f} kl_delta {metrics['kl_delta']:.4f} "
                f"pi_logp {metrics['policy_logp']:.4f} entropy {metrics['entropy']:.3f} "
                f"lr {lr:.2e}" + (" skipped_update" if skipped_update else ""),
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
                "rl_base_checkpoint": args.checkpoint,
                "rl_kl_coef": args.kl_coef,
                "rl_reward_baseline": args.reward_baseline,
                "rl_skipped_updates": skipped_updates,
            },
        },
        model_path,
    )
    tokenizer.save(out_tokenizer_path)
    if Path(out_tokenizer_path) != Path(vocab_path):
        shutil.copyfile(out_tokenizer_path, vocab_path)

    print(f"[RL] Model saved -> {model_path}")
    print(f"[RL] Tokenizer saved -> {out_tokenizer_path}")


if __name__ == "__main__":
    main()
