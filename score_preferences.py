"""
score_preferences.py
--------------------
Convert generated candidates into DPO preference pairs.

Modes:
  - heuristic: rank candidates with simple local quality checks.
  - manual: show candidates and ask a human to pick best/worst.
  - deepseek: ask DeepSeek to pick best/worst through its OpenAI-compatible API.

The output is JSONL:
    {"prompt": "...", "chosen": "...", "rejected": "...", "source": "..."}
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from generate import has_bad_text, is_low_quality_answer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Gravity preference pairs")
    parser.add_argument("--input", type=str, default="preference_candidates.jsonl")
    parser.add_argument("--output", type=str, default="preferences.jsonl")
    parser.add_argument("--append", action="store_true",
                        help="Append to the output JSONL instead of replacing it.")
    parser.add_argument("--dedupe", action="store_true",
                        help="Skip prompt/chosen/rejected triples already present in output.")
    parser.add_argument("--mode", choices=["heuristic", "manual", "deepseek"], default="heuristic")
    parser.add_argument("--domain", choices=["casual", "general"], default="casual",
                        help="Judging domain. casual keeps DeepSeek focused on everyday chat.")
    parser.add_argument("--min_score_gap", type=float, default=1.0)
    parser.add_argument("--deepseek_model", type=str, default="deepseek-v4-pro")
    parser.add_argument("--deepseek_base_url", type=str, default="https://api.deepseek.com")
    parser.add_argument("--deepseek_temperature", type=float, default=0.0)
    parser.add_argument("--deepseek_timeout", type=float, default=60.0)
    parser.add_argument("--deepseek_retries", type=int, default=2)
    parser.add_argument("--deepseek_parallel", type=int, default=4,
                        help="Number of concurrent DeepSeek judge requests.")
    parser.add_argument("--no_synthetic_chosen", action="store_true",
                        help="Disable DeepSeek-generated ideal answers when every candidate is bad.")
    return parser.parse_args()


def length_score(answer: str) -> float:
    words = re.findall(r"\w+|[\u4e00-\u9fff]", answer)
    count = len(words)
    if 12 <= count <= 90:
        return 2.0
    if 6 <= count < 12 or 90 < count <= 140:
        return 1.0
    return 0.0


def repetition_penalty(answer: str) -> float:
    compact = re.sub(r"\s+", "", answer)
    if not compact:
        return 3.0
    unique_ratio = len(set(compact)) / max(1, len(compact))
    penalty = 0.0
    if unique_ratio < 0.25 and len(compact) > 20:
        penalty += 2.0
    if re.search(r"(.{1,5})\1{3,}", compact):
        penalty += 2.0
    return penalty


def heuristic_score(prompt: str, answer: str) -> float:
    score = 0.0
    if not answer.strip() or has_bad_text(answer):
        return -10.0
    if is_low_quality_answer(answer, prompt):
        score -= 4.0
    score += length_score(answer)
    score -= repetition_penalty(answer)
    lowered = answer.lower()
    if any(marker in lowered for marker in ("i am not confident", "fallback")):
        score -= 1.0
    if any(marker in lowered for marker in ("first", "start", "next step", "small", "plan")):
        score += 0.7
    if "?" in answer or "?" in prompt:
        score += 0.3
    return score


def load_candidate_records(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if "prompt" not in record or "candidates" not in record:
                raise ValueError(f"Invalid candidate record at line {line_no}")
            records.append(record)
    return records


def preference_key(record: dict) -> str:
    payload = {
        "prompt": " ".join(str(record.get("prompt", "")).split()),
        "chosen": " ".join(str(record.get("chosen", "")).split()),
        "rejected": " ".join(str(record.get("rejected", "")).split()),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_existing_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    keys: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                keys.add(preference_key(json.loads(line)))
            except json.JSONDecodeError:
                continue
    return keys


def extract_candidates(record: dict) -> list[str]:
    return [
        item.get("answer", "").strip()
        for item in record.get("candidates", [])
        if isinstance(item, dict) and item.get("answer", "").strip()
    ]


def choose_heuristic(record: dict, min_gap: float) -> dict | None:
    prompt = record["prompt"]
    candidates = extract_candidates(record)
    if len(candidates) < 2:
        return None

    scored = [(heuristic_score(prompt, answer), answer) for answer in candidates]
    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, chosen = scored[0]
    worst_score, rejected = scored[-1]
    if chosen == rejected or best_score - worst_score < min_gap:
        return None
    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "chosen_score": best_score,
        "rejected_score": worst_score,
        "source": "heuristic",
    }


def choose_manual(record: dict) -> dict | None:
    prompt = record["prompt"]
    candidates = extract_candidates(record)
    if len(candidates) < 2:
        return None

    print("\nPrompt:")
    print(prompt)
    for index, answer in enumerate(candidates, start=1):
        print(f"\n[{index}] {answer}")

    raw_best = input("\nBest candidate number (blank to skip): ").strip()
    if not raw_best:
        return None
    raw_worst = input("Worst candidate number: ").strip()
    try:
        best = int(raw_best) - 1
        worst = int(raw_worst) - 1
    except ValueError:
        return None
    if best == worst or not (0 <= best < len(candidates)) or not (0 <= worst < len(candidates)):
        return None
    return {
        "prompt": prompt,
        "chosen": candidates[best],
        "rejected": candidates[worst],
        "source": "human",
    }


def build_deepseek_judge_prompt(prompt: str, candidates: list[str], domain: str = "casual") -> str:
    numbered = "\n\n".join(
        f"Candidate {index}:\n{answer}"
        for index, answer in enumerate(candidates, start=1)
    )
    if domain == "casual":
        domain_rules = (
            "This dataset is only for everyday casual chat and gentle companionship.\n"
            "Do not prefer professional, technical, academic, legal, medical, financial, coding, or workplace answers.\n"
            "Prefer natural small talk, emotional presence, light questions, warmth, brevity, and conversational flow.\n"
            "If the prompt seems professional, convert it into a casual, non-expert response or a gentle boundary.\n"
        )
    else:
        domain_rules = ""

    return (
        "You are judging candidate answers for a small local assistant named Gravity.\n"
        f"{domain_rules}"
        "Pick the best and worst answer for preference training.\n"
        "Prefer answers that are helpful, safe, grounded, concise, and directly responsive.\n"
        "Penalize hallucination, repetition, invalid text, generic dismissal, unsafe advice, and ignoring constraints.\n"
            "If all candidates are bad, set all_rejected=true and write a correct, valuable ideal_answer.\n"
            "If there is only one candidate, set all_rejected=true and write an ideal_answer unless the candidate is already perfect.\n"
        "The ideal_answer must answer the prompt directly and should be suitable as the chosen answer for DPO.\n"
        "Return only valid JSON with this shape: "
        '{"best": 1, "worst": 2, "all_rejected": false, "ideal_answer": "", "reason": "short reason"}\n'
        "When all_rejected=true, best may be null but worst must be the worst candidate number.\n\n"
        f"Prompt:\n{prompt}\n\n{numbered}"
    )


def call_deepseek_judge(
    prompt: str,
    candidates: list[str],
    model: str,
    base_url: str,
    temperature: float,
    timeout: float,
    retries: int,
    domain: str,
) -> dict | None:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is required for --mode deepseek.")

    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a strict preference-data judge. Return JSON only."},
            {"role": "user", "content": build_deepseek_judge_prompt(prompt, candidates, domain)},
        ],
        "temperature": temperature,
        "stream": False,
        "response_format": {"type": "json_object"},
    }
    payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    url = base_url.rstrip("/") + "/chat/completions"

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        request = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
            content = data["choices"][0]["message"]["content"]
            return json.loads(content)
        except (urllib.error.URLError, urllib.error.HTTPError, KeyError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
                continue

    print(f"[DeepSeek] Judge failed: {last_error}")
    return None


def choose_deepseek(record: dict, args: argparse.Namespace) -> dict | None:
    prompt = record["prompt"]
    candidates = extract_candidates(record)
    if len(candidates) < 1:
        return None

    judgment = call_deepseek_judge(
        prompt=prompt,
        candidates=candidates,
        model=args.deepseek_model,
        base_url=args.deepseek_base_url,
        temperature=args.deepseek_temperature,
        timeout=args.deepseek_timeout,
        retries=args.deepseek_retries,
        domain=args.domain,
    )
    if not judgment:
        return None
    try:
        worst = int(judgment["worst"]) - 1
    except (KeyError, TypeError, ValueError):
        return None
    if not (0 <= worst < len(candidates)):
        return None

    all_rejected = bool(judgment.get("all_rejected", False))
    ideal_answer = str(judgment.get("ideal_answer", "")).strip()
    if all_rejected and not args.no_synthetic_chosen and ideal_answer:
        return {
            "prompt": prompt,
            "chosen": ideal_answer,
            "rejected": candidates[worst],
            "judge_reason": str(judgment.get("reason", "")),
            "source": f"deepseek:{args.deepseek_model}:synthetic_chosen",
            "all_candidates_rejected": True,
        }

    try:
        best = int(judgment["best"]) - 1
    except (KeyError, TypeError, ValueError):
        return None
    if len(candidates) < 2 or best == worst or not (0 <= best < len(candidates)):
        return None
    return {
        "prompt": prompt,
        "chosen": candidates[best],
        "rejected": candidates[worst],
        "judge_reason": str(judgment.get("reason", "")),
        "source": f"deepseek:{args.deepseek_model}",
        "all_candidates_rejected": False,
    }


def write_preference(
    f,
    preference: dict | None,
    index: int,
    total: int,
    existing_keys: set[str],
) -> tuple[int, int]:
    if preference is None:
        print(f"[Preferences] {index}/{total} skipped")
        return 0, 0
    key = preference_key(preference)
    if key in existing_keys:
        print(f"[Preferences] {index}/{total} duplicate skipped")
        return 0, 1
    f.write(json.dumps(preference, ensure_ascii=False) + "\n")
    f.flush()
    existing_keys.add(key)
    source = preference.get("source", "unknown")
    print(f"[Preferences] {index}/{total} accepted source={source}")
    return 1, 0


def score_records_parallel_deepseek(
    records: list[dict],
    args: argparse.Namespace,
    output_file,
    existing_keys: set[str],
) -> tuple[int, int]:
    max_workers = max(1, args.deepseek_parallel)
    written = 0
    skipped_duplicates = 0
    total = len(records)
    print(f"[DeepSeek] Parallel judge requests: {max_workers}")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(choose_deepseek, record, args): index
            for index, record in enumerate(records, start=1)
        }
        for future in as_completed(futures):
            index = futures[future]
            try:
                preference = future.result()
            except Exception as exc:
                print(f"[Preferences] {index}/{total} failed: {exc}")
                continue
            added, duplicate = write_preference(
                output_file,
                preference,
                index,
                total,
                existing_keys,
            )
            written += added
            skipped_duplicates += duplicate
    return written, skipped_duplicates


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    records = load_candidate_records(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing_keys = load_existing_keys(output_path) if args.dedupe else set()

    written = 0
    skipped_duplicates = 0
    open_mode = "a" if args.append else "w"
    with output_path.open(open_mode, encoding="utf-8") as f:
        if args.mode == "deepseek" and args.deepseek_parallel > 1:
            written, skipped_duplicates = score_records_parallel_deepseek(
                records,
                args,
                f,
                existing_keys,
            )
        else:
            for index, record in enumerate(records, start=1):
                if args.mode == "manual":
                    preference = choose_manual(record)
                elif args.mode == "deepseek":
                    preference = choose_deepseek(record, args)
                else:
                    preference = choose_heuristic(record, args.min_score_gap)
                added, duplicate = write_preference(
                    f,
                    preference,
                    index,
                    len(records),
                    existing_keys,
                )
                written += added
                skipped_duplicates += duplicate

    action = "Appended" if args.append else "Wrote"
    print(
        f"[Preferences] {action} {written} preference pairs -> {output_path} "
        f"(duplicates skipped: {skipped_duplicates})"
    )


if __name__ == "__main__":
    main()
