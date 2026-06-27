"""
generate_candidates.py
----------------------
Generate multiple candidate answers per prompt for preference collection.

The output is JSONL:
    {"prompt": "...", "candidates": [{"answer": "...", "tokens": 42, ...}, ...]}
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from generate import DEFAULT_FALLBACK_CORPUS, answer_question, load_model
from utils import load_text_files, parse_qa_examples, set_seed


DEFAULT_PROMPTS = [
    "hi, are you there?",
    "can you chat with me for a bit?",
    "I do not know what to say, but I want some company.",
    "today felt a little heavy.",
    "ask me something easy.",
    "say something gentle.",
    "I am bored. Talk with me.",
    "I feel a little lonely tonight.",
    "tell me something small and warm.",
    "stay with me for a moment.",
    "我有点无聊，陪我聊聊。",
    "在吗？",
    "今天有点累。",
    "不知道说什么，但想有人陪。",
    "问我一个轻松的问题。",
    "说点温柔的话。",
    "陪我待一会儿。",
    "我有点孤单。",
    "随便聊点日常吧。",
    "今天过得一般。",
]

CASUAL_VARIANTS = [
    "please answer casually.",
    "keep it short and warm.",
    "ask me one easy question.",
    "just stay with me for a moment.",
    "make it feel natural.",
    "不要讲大道理。",
    "简单温柔一点。",
    "像朋友一样回我。",
    "问我一个轻松的问题。",
    "陪我随便聊聊。",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Gravity preference candidates")
    parser.add_argument("--checkpoint", type=str, default="checkpoints_gravity120/model.pt")
    parser.add_argument("--tokenizer", type=str, default="checkpoints_gravity120/tokenizer.json")
    parser.add_argument("--vocab", type=str, default=None, help="Alias for --tokenizer")
    parser.add_argument("--prompts", type=str, default=None,
                        help="Text file with one prompt per line, or a Q/A corpus to reuse questions from.")
    parser.add_argument("--output", type=str, default="preference_candidates.jsonl")
    parser.add_argument("--num_candidates", type=int, default=1)
    parser.add_argument("--max_prompts", type=int, default=500)
    parser.add_argument("--max_new_tokens", type=int, default=40)
    parser.add_argument("--attempt_multiplier", type=int, default=1,
                        help="Generation attempts per requested candidate.")
    parser.add_argument("--resume", action="store_true",
                        help="Append missing prompts to an existing output JSONL instead of starting over.")
    parser.add_argument("--temperature", type=float, default=0.55)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--allow_retrieval_fallback", action="store_true",
                        help="Allow exact/retrieval fallback. Default disables it to expose model behavior.")
    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_prompts(spec: str | None, limit: int) -> list[str]:
    if not spec:
        prompts = DEFAULT_PROMPTS[:]
        variant_index = 0
        while len(prompts) < limit:
            base = DEFAULT_PROMPTS[len(prompts) % len(DEFAULT_PROMPTS)]
            variant = CASUAL_VARIANTS[variant_index % len(CASUAL_VARIANTS)]
            prompts.append(f"{base} {variant}")
            variant_index += 1
        return prompts[:limit]

    prompts: list[str] = []
    for path, text in load_text_files(spec):
        try:
            examples = parse_qa_examples(text)
            prompts.extend(example.question for example in examples)
        except ValueError:
            _ = path
            prompts.extend(line.strip() for line in text.splitlines() if line.strip())

    seen: set[str] = set()
    unique = []
    for prompt in prompts:
        if prompt not in seen:
            seen.add(prompt)
            unique.append(prompt)
    random.shuffle(unique)
    return unique[:limit]


def load_completed_prompts(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            prompt = str(record.get("prompt", "")).strip()
            if prompt:
                completed.add(prompt)
    return completed


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    tokenizer_path = args.vocab or args.tokenizer
    device = select_device(args.device)

    print(f"[Candidates] Using device: {device}")
    model, tokenizer = load_model(args.checkpoint, tokenizer_path, device)
    prompts = load_prompts(args.prompts, args.max_prompts)
    print(f"[Candidates] Loaded {len(prompts)} prompts.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed_prompts = load_completed_prompts(output_path) if args.resume else set()
    if completed_prompts:
        print(f"[Candidates] Resuming: {len(completed_prompts)} prompts already completed.")

    open_mode = "a" if args.resume else "w"
    with output_path.open(open_mode, encoding="utf-8") as f:
        for index, prompt in enumerate(prompts, start=1):
            if prompt in completed_prompts:
                print(f"[Candidates] {index}/{len(prompts)} skipped existing")
                continue
            candidates = []
            seen_answers: set[str] = set()
            attempts = 0
            max_attempts = max(args.num_candidates, args.num_candidates * max(1, args.attempt_multiplier))
            while len(candidates) < args.num_candidates and attempts < max_attempts:
                attempts += 1
                temperature = max(0.05, args.temperature * random.uniform(0.75, 1.25))
                top_p = min(0.98, max(0.5, args.top_p * random.uniform(0.9, 1.05)))
                answer, tokens, elapsed = answer_question(
                    model=model,
                    tokenizer=tokenizer,
                    question=prompt,
                    history=[],
                    max_new_tokens=args.max_new_tokens,
                    temperature=temperature,
                    top_k=args.top_k,
                    top_p=top_p,
                    device=device,
                    suppress_bad_tokens=True,
                    fallback_corpus=DEFAULT_FALLBACK_CORPUS if args.allow_retrieval_fallback else None,
                    use_retrieval_fallback=args.allow_retrieval_fallback,
                    hallucination_entropy=100.0,
                )
                normalized = " ".join(answer.split())
                if normalized and normalized not in seen_answers:
                    seen_answers.add(normalized)
                    candidates.append({
                        "answer": answer,
                        "tokens": tokens,
                        "elapsed": elapsed,
                        "temperature": temperature,
                        "top_p": top_p,
                    })

            record = {"prompt": prompt, "candidates": candidates}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[Candidates] {index}/{len(prompts)} {len(candidates)} candidates")

    print(f"[Candidates] Wrote {output_path}")


if __name__ == "__main__":
    main()
