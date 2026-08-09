"""
build_sft_dataset.py
--------------------
Convert the project's Q/A corpora into chat-format JSONL for fine-tuning a
pretrained model (Qwen3-0.6B).

Sources and how they are treated:
  - corpus/qa_10k.txt       template-generated; deduplicated by answer, keeping
                            at most --max_variants question phrasings per answer
  - corpus/priority_qa.txt  hand-written single-turn pairs, kept as-is
  - data.txt                hand-written single-turn pairs, kept as-is
  - dialogues.txt           blank-line separated multi-turn conversations
  - corpus/dialogues_clean.txt   same
  - corpus/context_memory.txt    same (teaches in-context memory)
  - corpus/roleplay_companion.jsonl  persona records; persona becomes the
                            system prompt, "ideal" becomes the assistant reply
  - capability_curriculum.py compact anchors for instruction following,
                            reasoning, coding, and problem understanding

Output:
  sft_data/train.jsonl and sft_data/val.jsonl, one record per line:
    {"messages": [{"role": "system", ...}, {"role": "user", ...}, ...]}

Usage:
    python build_sft_dataset.py
    python build_sft_dataset.py --val_frac 0.05 --max_variants 2
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import OrderedDict

from capability_curriculum import (
    build_capability_sft_conversations,
    repeat_records,
)

QA_PATTERN = re.compile(r"^Q: (.*?)\nA: (.*?)<eos>", re.MULTILINE | re.DOTALL)

DEFAULT_SYSTEM_PROMPT = (
    "你是 Gravity，一个温柔、克制的本地陪伴助手。"
    "你先倾听和共情，再在用户需要时给出小而具体的下一步。"
    "回复简短自然，不说教，不堆砌建议。"
    "You are Gravity, a gentle local companion. Reply briefly and warmly, "
    "in the same language the user uses."
)

QA_SOURCES = [
    ("corpus/qa_10k.txt", True),
    ("corpus/priority_qa.txt", False),
    ("data.txt", False),
]

DIALOGUE_SOURCES = [
    "dialogues.txt",
    "corpus/dialogues_clean.txt",
    "corpus/context_memory.txt",
]

ROLEPLAY_SOURCE = "corpus/roleplay_companion.jsonl"


def parse_qa_pairs(text: str) -> list[tuple[str, str]]:
    return [(q.strip(), a.strip()) for q, a in QA_PATTERN.findall(text)]


def dedup_by_answer(
    pairs: list[tuple[str, str]],
    max_variants: int,
) -> list[tuple[str, str]]:
    """Group template pairs by identical answer, keep few diverse questions."""
    grouped: "OrderedDict[str, list[str]]" = OrderedDict()
    for question, answer in pairs:
        grouped.setdefault(answer, []).append(question)

    kept: list[tuple[str, str]] = []
    for answer, questions in grouped.items():
        picks = [questions[0]]
        if max_variants > 1 and len(questions) > 1:
            first_chars = set(questions[0])
            rest = sorted(
                questions[1:],
                key=lambda q: len(set(q) ^ first_chars),
                reverse=True,
            )
            picks.extend(rest[: max_variants - 1])
        kept.extend((q, answer) for q in picks)
    return kept


def cap_frame_families(
    pairs: list[tuple[str, str]],
    frame_prefix: int,
    frame_cap: int,
    rng: random.Random,
) -> list[tuple[str, str]]:
    """
    Limit how many distinct answers may share one sentence frame.

    Template generation left ~90% of unique answers opening with one of three
    frames ("我在听。", "For ...", "I am ..."); unchecked, fine-tuning collapses
    onto them. Answers are grouped by their first `frame_prefix` characters and
    each family keeps at most `frame_cap` sampled answers (with their question
    variants).
    """
    families: "OrderedDict[str, list[str]]" = OrderedDict()
    for _, answer in pairs:
        if answer not in families.setdefault(answer[:frame_prefix], []):
            families[answer[:frame_prefix]].append(answer)

    keep_answers: set[str] = set()
    for members in families.values():
        if len(members) <= frame_cap:
            keep_answers.update(members)
        else:
            keep_answers.update(rng.sample(members, frame_cap))
    return [(q, a) for q, a in pairs if a in keep_answers]


def qa_to_conversation(question: str, answer: str) -> list[dict]:
    return [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]


def parse_dialogue_blocks(text: str) -> list[list[dict]]:
    """Blank-line separated blocks of Q/A records become multi-turn chats."""
    conversations = []
    for block in re.split(r"\n\s*\n", text):
        pairs = parse_qa_pairs(block)
        if not pairs:
            continue
        messages = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]
        for question, answer in pairs:
            messages.append({"role": "user", "content": question})
            messages.append({"role": "assistant", "content": answer})
        conversations.append(messages)
    return conversations


def roleplay_system_prompt(role: dict) -> str:
    parts = [f"你现在扮演 {role.get('name', '一位陪伴者')}。"]
    if role.get("persona"):
        parts.append(f"人设：{role['persona']}")
    if role.get("relationship"):
        parts.append(f"与用户的关系：{role['relationship']}")
    if role.get("style"):
        parts.append(f"说话风格：{role['style']}")
    if role.get("boundaries"):
        parts.append(f"边界：{role['boundaries']}")
    facts = role.get("facts") or []
    if facts:
        parts.append("需要记住的事实：" + "；".join(facts))
    return "\n".join(parts)


def parse_roleplay_records(path: str) -> list[list[dict]]:
    conversations = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            system = roleplay_system_prompt(record.get("role", {}))
            memories = record.get("memories") or []
            if memories:
                system += "\n对话背景：" + "；".join(memories)

            messages = [{"role": "system", "content": system}]
            for turn in record.get("history") or []:
                if isinstance(turn, dict):
                    user_text, assistant_text = turn.get("user"), turn.get("assistant")
                else:
                    user_text, assistant_text = turn[0], turn[1]
                if user_text:
                    messages.append({"role": "user", "content": user_text})
                if assistant_text:
                    messages.append({"role": "assistant", "content": assistant_text})

            messages.append({"role": "user", "content": record["user"]})
            messages.append({"role": "assistant", "content": record["ideal"]})
            conversations.append(messages)
    return conversations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build chat-format SFT dataset")
    parser.add_argument("--out_dir", type=str, default="sft_data")
    parser.add_argument("--val_frac", type=float, default=0.05)
    parser.add_argument("--max_variants", type=int, default=2,
                        help="Question phrasings kept per unique templated answer.")
    parser.add_argument("--frame_cap", type=int, default=25,
                        help="Max distinct answers sharing one sentence frame.")
    parser.add_argument("--frame_prefix", type=int, default=4,
                        help="Answer prefix length used to identify a frame.")
    parser.add_argument("--include_capability_curriculum",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="Mix in compact SFT anchors for instruction/reason/code/problem understanding.")
    parser.add_argument("--capability_copies", type=int, default=3,
                        help="How many copies of the capability curriculum to mix into SFT.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = os.path.dirname(os.path.abspath(__file__))

    conversations: list[list[dict]] = []
    seen_pairs: set[tuple[str, str]] = set()
    stats: list[tuple[str, int]] = []
    frame_rng = random.Random(args.seed)

    for rel_path, is_templated in QA_SOURCES:
        path = os.path.join(base, rel_path)
        pairs = parse_qa_pairs(open(path, encoding="utf-8").read())
        raw_count = len(pairs)
        if is_templated:
            pairs = dedup_by_answer(pairs, args.max_variants)
            pairs = cap_frame_families(
                pairs, args.frame_prefix, args.frame_cap, frame_rng
            )
        fresh = [p for p in pairs if p not in seen_pairs]
        seen_pairs.update(fresh)
        conversations.extend(qa_to_conversation(q, a) for q, a in fresh)
        stats.append((rel_path, len(fresh)))
        note = f" (deduped from {raw_count})" if is_templated else ""
        print(f"[Data] {rel_path}: {len(fresh)} single-turn conversations{note}")

    for rel_path in DIALOGUE_SOURCES:
        path = os.path.join(base, rel_path)
        blocks = parse_dialogue_blocks(open(path, encoding="utf-8").read())
        conversations.extend(blocks)
        stats.append((rel_path, len(blocks)))
        print(f"[Data] {rel_path}: {len(blocks)} multi-turn conversations")

    roleplay = parse_roleplay_records(os.path.join(base, ROLEPLAY_SOURCE))
    conversations.extend(roleplay)
    stats.append((ROLEPLAY_SOURCE, len(roleplay)))
    print(f"[Data] {ROLEPLAY_SOURCE}: {len(roleplay)} roleplay conversations")

    if args.include_capability_curriculum:
        capability = build_capability_sft_conversations(DEFAULT_SYSTEM_PROMPT)
        expanded = repeat_records(capability, args.capability_copies)
        conversations.extend(expanded)
        stats.append(("capability_curriculum", len(expanded)))
        print(
            f"[Data] capability_curriculum: {len(expanded)} conversations "
            f"({len(capability)} unique x {max(0, args.capability_copies)})"
        )

    rng = random.Random(args.seed)
    rng.shuffle(conversations)
    val_count = max(1, int(len(conversations) * args.val_frac))
    val_set = conversations[:val_count]
    train_set = conversations[val_count:]

    out_dir = os.path.join(base, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    for name, subset in (("train.jsonl", train_set), ("val.jsonl", val_set)):
        out_path = os.path.join(out_dir, name)
        with open(out_path, "w", encoding="utf-8") as f:
            for messages in subset:
                f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
        print(f"[Data] Wrote {len(subset)} conversations -> {out_path}")

    turn_counts = [sum(1 for m in c if m["role"] == "assistant") for c in conversations]
    print(
        f"[Data] Total {len(conversations)} conversations, "
        f"{sum(turn_counts)} assistant turns, "
        f"max {max(turn_counts)} turns per conversation."
    )


if __name__ == "__main__":
    main()
