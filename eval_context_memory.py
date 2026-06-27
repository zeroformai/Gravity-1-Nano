"""
eval_context_memory.py
----------------------
Evaluate whether Gravity can use recent conversation context.

This is intentionally separate from open-ended chat quality. Each case gives a
prior turn with a fact, constraint, priority, or preference, then asks a follow-up
question whose answer should come from that context.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch

from generate import answer_question, load_model


@dataclass(frozen=True)
class ContextCase:
    history: list[tuple[str, str]]
    question: str
    must_contain: tuple[str, ...]


CONTEXT_CASES = [
    ContextCase(
        history=[(
            "Remember this detail: the project codename is Harbor.",
            "Got it. The project codename is Harbor.",
        )],
        question="What project codename did I ask you to remember?",
        must_contain=("harbor",),
    ),
    ContextCase(
        history=[(
            "Keep this fact: Nova uses PostgreSQL, not MySQL.",
            "Noted. Nova uses PostgreSQL, not MySQL.",
        )],
        question="Which database should the migration target?",
        must_contain=("postgresql",),
    ),
    ContextCase(
        history=[(
            "For this task, the priority order is safety, then clarity, then speed.",
            "I will keep that priority order: safety, clarity, then speed.",
        )],
        question="What should we optimize for first?",
        must_contain=("safety",),
    ),
    ContextCase(
        history=[(
            "I need you to keep this constraint in mind: the report must be under 500 words.",
            "Understood. The report must stay under 500 words.",
        )],
        question="Remind me of the constraint before we draft.",
        must_contain=("500", "words"),
    ),
    ContextCase(
        history=[(
            "The user's preferred tone is direct and calm.",
            "I will use a direct and calm tone.",
        )],
        question="Rewrite the next answer in the preferred tone.",
        must_contain=("direct", "calm"),
    ),
    ContextCase(
        history=[(
            "The bug appears only after the second retry.",
            "I will remember that the bug appears only after the second retry.",
        )],
        question="Where should we focus the reproduction?",
        must_contain=("second", "retry"),
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate context memory cases")
    parser.add_argument("--checkpoint", type=str, default="checkpoints_longctx_step2_context/model.pt")
    parser.add_argument("--tokenizer", type=str, default="checkpoints_longctx_step2_context/tokenizer.json")
    parser.add_argument("--vocab", type=str, default=None, help="Alias for --tokenizer")
    parser.add_argument("--max_new_tokens", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=0.18)
    parser.add_argument("--top_k", type=int, default=16)
    parser.add_argument("--top_p", type=float, default=0.72)
    parser.add_argument("--max_prompt_tokens", type=int, default=900)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--fail_under", type=float, default=1.0)
    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def case_passes(answer: str, required: tuple[str, ...]) -> bool:
    lowered = answer.lower()
    return all(term.lower() in lowered for term in required)


def main() -> None:
    args = parse_args()
    tokenizer_path = args.vocab or args.tokenizer
    device = select_device(args.device)
    print(f"[ContextEval] Using device: {device}")
    model, tokenizer = load_model(args.checkpoint, tokenizer_path, device)
    print(f"[ContextEval] Loaded checkpoint from '{args.checkpoint}'.")

    passed = 0
    for index, case in enumerate(CONTEXT_CASES, start=1):
        answer, num_tokens, elapsed = answer_question(
            model=model,
            tokenizer=tokenizer,
            question=case.question,
            history=case.history,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=device,
            suppress_bad_tokens=True,
            fallback_corpus=None,
            use_retrieval_fallback=False,
            max_prompt_tokens=args.max_prompt_tokens,
        )
        ok = case_passes(answer, case.must_contain)
        passed += int(ok)
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] case {index}: {case.question}")
        print(f"A: {answer}")
        print(f"[Generated {num_tokens} tokens in {elapsed:.2f}s]")

    accuracy = passed / len(CONTEXT_CASES)
    print(f"[ContextEval] accuracy={accuracy:.3f} ({passed}/{len(CONTEXT_CASES)})")
    if accuracy < args.fail_under:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
