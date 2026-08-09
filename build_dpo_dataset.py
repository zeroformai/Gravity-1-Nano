"""
build_dpo_dataset.py
--------------------
Convert preferences.jsonl (+ roleplay ideal/rejected pairs + capability
curriculum pairs) into chat-format preference records for train_dpo_lora.py.

Record layout, one JSON per line:
    {"prompt_messages": [{"role": "system"|"user"|"assistant", ...}, ...],
     "chosen": "...", "rejected": "..."}

preferences.jsonl provenance note: records with all_candidates_rejected=true
hold a judge-written chosen against a real sampled model failure — the most
useful kind of pair here, so they are kept.

Usage:
    python build_dpo_dataset.py
"""

from __future__ import annotations

import argparse
import json
import os
import random

from build_sft_dataset import (
    DEFAULT_SYSTEM_PROMPT,
    ROLEPLAY_SOURCE,
    roleplay_system_prompt,
)
from capability_curriculum import (
    build_capability_dpo_pairs,
    repeat_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build chat-format DPO dataset")
    parser.add_argument("--preferences", type=str, default="preferences.jsonl")
    parser.add_argument("--out_dir", type=str, default="dpo_data")
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--include_capability_curriculum",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="Mix in DPO preference anchors for instruction/reason/code/problem understanding.")
    parser.add_argument("--capability_copies", type=int, default=4,
                        help="How many copies of the capability preference curriculum to mix into DPO.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_preference_records(path: str) -> list[dict]:
    records = []
    skipped = 0
    if not os.path.exists(path):
        print(f"[DPO-Data] {path}: not found, skipping external preferences.")
        return records
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            prompt = str(item.get("prompt", "")).strip()
            chosen = str(item.get("chosen", "")).strip()
            rejected = str(item.get("rejected", "")).strip()
            if not prompt or not chosen or not rejected or chosen == rejected:
                skipped += 1
                continue
            records.append({
                "prompt_messages": [
                    {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "chosen": chosen,
                "rejected": rejected,
                "category": "external",
                "weight": float(item.get("weight", 1.0) or 1.0),
                "source": item.get("source", "preferences"),
            })
    if skipped:
        print(f"[DPO-Data] Skipped {skipped} empty/degenerate records.")
    print(f"[DPO-Data] {path}: {len(records)} preference pairs.")
    return records


def load_roleplay_pairs(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            ideal = str(item.get("ideal", "")).strip()
            rejected = str(item.get("rejected", "")).strip()
            if not ideal or not rejected or ideal == rejected:
                continue
            system = roleplay_system_prompt(item.get("role", {}))
            memories = item.get("memories") or []
            if memories:
                system += "\n对话背景：" + "；".join(memories)
            prompt_messages = [{"role": "system", "content": system}]
            for turn in item.get("history") or []:
                if isinstance(turn, dict):
                    user_text, assistant_text = turn.get("user"), turn.get("assistant")
                else:
                    user_text, assistant_text = turn[0], turn[1]
                if user_text:
                    prompt_messages.append({"role": "user", "content": user_text})
                if assistant_text:
                    prompt_messages.append({"role": "assistant", "content": assistant_text})
            prompt_messages.append({"role": "user", "content": item["user"]})
            records.append({
                "prompt_messages": prompt_messages,
                "chosen": ideal,
                "rejected": rejected,
                "category": "roleplay",
                "weight": 1.0,
                "source": "roleplay",
            })
    print(f"[DPO-Data] {path}: {len(records)} roleplay preference pairs.")
    return records


def main() -> None:
    args = parse_args()
    base = os.path.dirname(os.path.abspath(__file__))

    records = load_preference_records(os.path.join(base, args.preferences))
    records.extend(load_roleplay_pairs(os.path.join(base, ROLEPLAY_SOURCE)))
    if args.include_capability_curriculum:
        capability = build_capability_dpo_pairs(DEFAULT_SYSTEM_PROMPT)
        expanded = repeat_records(capability, args.capability_copies)
        records.extend(expanded)
        print(
            f"[DPO-Data] capability_curriculum: {len(expanded)} preference pairs "
            f"({len(capability)} unique x {max(0, args.capability_copies)})"
        )

    if not records:
        raise ValueError("No DPO preference records were collected.")

    rng = random.Random(args.seed)
    rng.shuffle(records)
    val_count = max(1, int(len(records) * args.val_frac))
    splits = (("val.jsonl", records[:val_count]), ("train.jsonl", records[val_count:]))

    out_dir = os.path.join(base, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    for name, subset in splits:
        out_path = os.path.join(out_dir, name)
        with open(out_path, "w", encoding="utf-8") as f:
            for record in subset:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"[DPO-Data] Wrote {len(subset)} pairs -> {out_path}")


if __name__ == "__main__":
    main()
