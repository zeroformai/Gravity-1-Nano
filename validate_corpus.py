"""
Validate Gravity corpus files before training.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tokenizer import EOS_TOKEN
from utils import load_text_files, parse_qa_conversations, parse_qa_examples


MOJIBAKE_MARKERS = ("�", "鎴", "浣", "鐨", "鍦", "涓", "绱", "銆", "锛", "紵", "€")


def count_lines(path: str) -> int:
    return len(Path(path).read_text(encoding="utf-8").splitlines())


def suspect_mojibake(text: str) -> int:
    return sum(text.count(marker) for marker in MOJIBAKE_MARKERS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Gravity training corpus")
    parser.add_argument("--data", default="corpus/qa_10k.txt")
    parser.add_argument("--dialogues", default="corpus/dialogues_clean.txt")
    parser.add_argument("--target_lines", type=int, default=10000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    total_lines = 0
    total_examples = 0
    seen_questions: set[str] = set()
    duplicate_questions = 0
    marker_count = 0

    for path, text in load_text_files(args.data):
        examples = parse_qa_examples(text)
        lines = count_lines(path)
        total_lines += lines
        total_examples += len(examples)
        marker_count += suspect_mojibake(text)
        for example in examples:
            if example.question in seen_questions:
                duplicate_questions += 1
            seen_questions.add(example.question)
        if text.count(EOS_TOKEN) != len(examples):
            raise ValueError(f"EOS count mismatch in {path}")
        print(f"[Corpus] {path}: {lines:,} lines | {len(examples):,} Q/A records")

    if args.dialogues:
        for path, text in load_text_files(args.dialogues):
            conversations = parse_qa_conversations(text)
            marker_count += suspect_mojibake(text)
            turns = sum(len(conversation) for conversation in conversations)
            print(
                f"[Dialogues] {path}: {count_lines(path):,} lines | "
                f"{len(conversations):,} conversations | {turns:,} turns"
            )

    if total_lines < args.target_lines:
        raise ValueError(
            f"Corpus has {total_lines:,} lines, below target {args.target_lines:,}."
        )
    if marker_count:
        raise ValueError(f"Detected {marker_count:,} mojibake marker occurrences.")

    print(
        f"[OK] {total_lines:,} data lines | {total_examples:,} Q/A records | "
        f"{duplicate_questions:,} duplicate questions"
    )


if __name__ == "__main__":
    main()
