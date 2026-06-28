"""
Validate Gravity corpus files before training.

This script intentionally avoids importing torch-backed training utilities so
corpus checks can run in a lightweight text-only environment.
"""

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

from tokenizer import EOS_TOKEN


MOJIBAKE_MARKERS = (
    "\ufffd",
    "锟",
    "閹",
    "娴",
    "閻",
    "閸",
    "娑",
    "缁",
    "閵",
    "閿",
    "鈧",
)


def load_text_files(spec: str) -> list[tuple[str, str]]:
    paths: list[str] = []
    for part in re.split(r"[,;]", spec or ""):
        pattern = part.strip()
        if not pattern:
            continue
        matches = sorted(glob.glob(pattern))
        paths.extend(matches or [pattern])

    if not paths:
        raise ValueError("No input files were specified.")

    loaded: list[tuple[str, str]] = []
    for path in paths:
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(path)
        text = file_path.read_text(encoding="utf-8")
        print(f"[Corpus] Loaded {len(text):,} characters from '{path}'.")
        loaded.append((str(file_path), text))
    return loaded


def parse_qa_examples(text: str) -> list[tuple[str, str]]:
    if not text.strip():
        raise ValueError("Dataset is empty.")

    pattern = re.compile(r"^Q: (.*?)\nA: (.*?)(?:<eos>)(?:\r?\n|$)", re.MULTILINE | re.DOTALL)
    examples = [(q.strip(), a.strip()) for q, a in pattern.findall(text)]
    if not examples:
        raise ValueError("No Q/A examples found. Expected records ending with <eos>.")

    q_count = len(re.findall(r"^Q: ", text, flags=re.MULTILINE))
    eos_count = text.count(EOS_TOKEN)
    if len(examples) != q_count or len(examples) != eos_count:
        raise ValueError(
            "Malformed Q/A dataset: every 'Q:' record must have an 'A:' answer "
            f"ending with {EOS_TOKEN}. Parsed {len(examples)} examples, found "
            f"{q_count} questions and {eos_count} EOS markers."
        )
    return examples


def parse_qa_conversations(text: str) -> list[list[tuple[str, str]]]:
    conversations: list[list[tuple[str, str]]] = []
    for chunk in re.split(r"(?:\r?\n){2,}", text.strip()):
        if chunk.strip():
            conversations.append(parse_qa_examples(chunk + "\n"))
    return conversations


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
        for question, _ in examples:
            if question in seen_questions:
                duplicate_questions += 1
            seen_questions.add(question)
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
