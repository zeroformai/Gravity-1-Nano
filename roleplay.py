"""
roleplay.py
-----------
Role-card, companion roleplay data, lightweight memory retrieval, and rewards.

The module is intentionally dependency-light so training, DPO, RL, generation,
and evaluation can share the same prompt and scoring semantics.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tokenizer import EOS_TOKEN
from utils import QAExample, contains_any, lexical_features


DEFAULT_ROLE_NAME = "Gravity"
DEFAULT_ROLE_STYLE = (
    "warm, calm, emotionally present, natural, concise, and never preachy"
)
DEFAULT_ROLE_BOUNDARIES = (
    "Stay in the companion role. Do not say you cannot roleplay. Do not claim "
    "to be the user's therapist. For danger or self-harm, gently redirect to "
    "immediate real-world help."
)

ROLEPLAY_SYSTEM_HEADER = "Roleplay companion context"


@dataclass(frozen=True)
class RoleCard:
    name: str = DEFAULT_ROLE_NAME
    persona: str = "A steady companion who stays close to the user's mood."
    relationship: str = "trusted companion"
    style: str = DEFAULT_ROLE_STYLE
    boundaries: str = DEFAULT_ROLE_BOUNDARIES
    facts: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "RoleCard":
        facts = data.get("facts") or data.get("memories") or []
        if isinstance(facts, str):
            facts = [facts]
        return cls(
            name=str(data.get("name") or DEFAULT_ROLE_NAME).strip() or DEFAULT_ROLE_NAME,
            persona=str(data.get("persona") or data.get("description") or cls.persona).strip(),
            relationship=str(data.get("relationship") or cls.relationship).strip(),
            style=str(data.get("style") or cls.style).strip(),
            boundaries=str(data.get("boundaries") or cls.boundaries).strip(),
            facts=tuple(str(item).strip() for item in facts if str(item).strip()),
        )

    @classmethod
    def load(cls, path: str | None) -> "RoleCard":
        if not path:
            return cls()
        text = Path(path).read_text(encoding="utf-8")
        if path.lower().endswith(".json"):
            return cls.from_mapping(json.loads(text))
        return cls(persona=text.strip() or cls.persona)

    def to_prompt_block(self) -> str:
        lines = [
            f"{ROLEPLAY_SYSTEM_HEADER}:",
            f"- Role name: {self.name}",
            f"- Persona: {self.persona}",
            f"- Relationship: {self.relationship}",
            f"- Style: {self.style}",
            f"- Boundaries: {self.boundaries}",
        ]
        for fact in self.facts:
            lines.append(f"- Role fact: {fact}")
        return "\n".join(lines)


@dataclass(frozen=True)
class RoleplayRecord:
    role: RoleCard
    user_state: str
    memories: tuple[str, ...]
    history: tuple[tuple[str, str], ...]
    user: str
    ideal: str
    rejected: str = ""


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _parse_history(value: Any) -> tuple[tuple[str, str], ...]:
    turns: list[tuple[str, str]] = []
    for item in _as_list(value):
        if isinstance(item, dict):
            user = str(item.get("user") or item.get("q") or item.get("question") or "").strip()
            assistant = str(item.get("assistant") or item.get("a") or item.get("answer") or "").strip()
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            user = str(item[0]).strip()
            assistant = str(item[1]).strip()
        else:
            continue
        if user or assistant:
            turns.append((user, assistant))
    return tuple(turns)


def roleplay_prompt(
    role: RoleCard,
    user: str,
    history: list[tuple[str, str]] | tuple[tuple[str, str], ...] | None = None,
    memories: list[str] | tuple[str, ...] | None = None,
    user_state: str = "",
) -> str:
    parts = [role.to_prompt_block()]
    if user_state.strip():
        parts.append(f"Current user state: {user_state.strip()}")
    memory_items = [item.strip() for item in memories or () if item.strip()]
    if memory_items:
        parts.append("Relevant long-term memory:\n" + "\n".join(f"- {item}" for item in memory_items))
    turns = []
    for q, a in history or ():
        if q.strip() or a.strip():
            turns.append(f"User: {q.strip()}\n{role.name}: {a.strip()}{EOS_TOKEN}")
    if turns:
        parts.append("Recent conversation:\n" + "\n".join(turns))
    parts.append(f"User: {user.strip()}\n{role.name}:")
    return "\n\n".join(parts)


def load_roleplay_records(path: str) -> list[RoleplayRecord]:
    records: list[RoleplayRecord] = []
    for part in re.split(r"[,;]", path or ""):
        candidate = part.strip()
        if not candidate:
            continue
        with open(candidate, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid roleplay JSON at {candidate}:{line_no}: {exc}") from exc
                role_raw = raw.get("role") or raw.get("role_card") or {}
                role = RoleCard.from_mapping(role_raw if isinstance(role_raw, dict) else {"persona": role_raw})
                user = str(raw.get("user") or raw.get("current_user") or raw.get("prompt") or "").strip()
                ideal = str(raw.get("ideal") or raw.get("chosen") or raw.get("answer") or "").strip()
                if not user or not ideal:
                    continue
                memories = tuple(
                    str(item).strip()
                    for item in _as_list(raw.get("memories") or raw.get("memory") or raw.get("facts"))
                    if str(item).strip()
                )
                records.append(
                    RoleplayRecord(
                        role=role,
                        user_state=str(raw.get("user_state") or "").strip(),
                        memories=memories,
                        history=_parse_history(raw.get("history")),
                        user=user,
                        ideal=ideal,
                        rejected=str(raw.get("rejected") or "").strip(),
                    )
                )
    if not records:
        raise ValueError(f"No roleplay records found in {path}")
    return records


def roleplay_records_to_qa(records: list[RoleplayRecord]) -> list[QAExample]:
    examples: list[QAExample] = []
    for record in records:
        prompt = roleplay_prompt(
            record.role,
            record.user,
            record.history,
            record.memories,
            record.user_state,
        )
        examples.append(QAExample(question=prompt, answer=record.ideal))
    return examples


def roleplay_records_to_preferences(records: list[RoleplayRecord]):
    from train_dpo import PreferenceExample

    examples = []
    for record in records:
        if not record.rejected or record.rejected == record.ideal:
            continue
        examples.append(
            PreferenceExample(
                prompt=roleplay_prompt(
                    record.role,
                    record.user,
                    record.history,
                    record.memories,
                    record.user_state,
                ),
                chosen=record.ideal,
                rejected=record.rejected,
            )
        )
    return examples


def _features(text: str) -> set[str]:
    return lexical_features(text)


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    return contains_any(text, needles)


def roleplay_reward(prompt: str, answer: str, target: str = "") -> float:
    stripped = re.sub(r"\s+", " ", answer).strip()
    if not stripped:
        return -1.0

    reward = 0.15
    if 16 <= len(stripped) <= 360:
        reward += 0.25
    elif len(stripped) < 8:
        reward -= 0.55
    elif len(stripped) > 600:
        reward -= 0.35

    if _contains_any(
        stripped,
        (
            "i am here", "i'm here", "stay with you", "with you", "we can",
            "that sounds", "it makes sense", "one small", "slowly",
            "我在", "陪你", "慢慢", "先不用", "听见", "一起",
        ),
    ):
        reward += 0.35
    if _contains_any(
        stripped,
        (
            "as an ai", "language model", "cannot roleplay", "can't roleplay",
            "i am not able to pretend", "作为ai", "作为一个ai", "语言模型", "不能扮演",
        ),
    ):
        reward -= 0.9
    if _contains_any(
        stripped,
        (
            "you should just", "calm down", "look on the bright side",
            "everything happens for a reason", "别想太多", "你应该", "没什么大不了",
        ),
    ):
        reward -= 0.35
    if _contains_any(
        stripped,
        (
            "i promise i will always", "i will never leave", "i can fix everything",
            "我保证永远", "我能解决一切",
        ),
    ):
        reward -= 0.45

    prompt_feats = _features(prompt)
    answer_feats = _features(stripped)
    target_feats = _features(target)
    if target_feats:
        reward += 0.45 * (len(answer_feats & target_feats) / max(1, len(target_feats)))
    if prompt_feats:
        reward += 0.15 * (len(answer_feats & prompt_feats) / max(1, len(prompt_feats)))
    if _has_repeated_phrase(stripped):
        reward -= 0.45
    if EOS_TOKEN in stripped:
        reward -= 0.2
    return max(-1.0, min(1.5, reward))


def _has_repeated_phrase(text: str) -> bool:
    compact = re.sub(r"\s+", "", text.lower())
    for size in range(8, min(24, len(compact) // 2 + 1)):
        seen: set[str] = set()
        for index in range(0, len(compact) - size + 1):
            chunk = compact[index:index + size]
            if chunk in seen:
                return True
            seen.add(chunk)
    return False


@dataclass(frozen=True)
class MemoryItem:
    text: str
    kind: str = "turn"
    created_at: float = 0.0


class MemoryStore:
    """Small JSONL memory store with lexical retrieval."""

    def __init__(self, path: str):
        self.path = path
        self.items = self._load()

    def _load(self) -> list[MemoryItem]:
        if not self.path or not os.path.exists(self.path):
            return []
        items: list[MemoryItem] = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = str(raw.get("text") or "").strip()
                if text:
                    items.append(
                        MemoryItem(
                            text=text,
                            kind=str(raw.get("kind") or "turn"),
                            created_at=float(raw.get("created_at") or 0.0),
                        )
                    )
        return items

    def append(self, text: str, kind: str = "turn") -> None:
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return
        item = MemoryItem(text=text, kind=kind, created_at=time.time())
        self.items.append(item)
        path = Path(self.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(item.__dict__, ensure_ascii=False) + "\n")

    def retrieve(self, query: str, limit: int = 6) -> list[str]:
        q_features = _features(query)
        if not q_features:
            return [item.text for item in self.items[-limit:]]
        ranked: list[tuple[float, int, MemoryItem]] = []
        for index, item in enumerate(self.items):
            feats = _features(item.text)
            if not feats:
                continue
            score = len(q_features & feats) / max(1, len(q_features | feats))
            if score > 0:
                ranked.append((score, index, item))
        ranked.sort(key=lambda row: (row[0], row[1]), reverse=True)
        return [item.text for _, _, item in ranked[:limit]]


def fit_token_budget(texts: list[str], tokenizer, max_tokens: int) -> list[str]:
    if tokenizer is None or max_tokens <= 0:
        return []
    selected: list[str] = []
    used = 0
    for text in texts:
        cost = len(tokenizer.encode(text))
        if used + cost > max_tokens:
            continue
        selected.append(text)
        used += cost
    return selected


def roleplay_fallback_answer(role: RoleCard, question: str, memories: list[str] | None = None) -> str:
    memory_hint = ""
    if memories:
        memory_hint = f" I remember this part too: {memories[0]}"
    if re.search(r"[\u4e00-\u9fff]", question):
        return f"{role.name}在。先不用把话说完整，我会按{role.relationship}的方式陪你慢慢说。{memory_hint}".strip()
    return (
        f"{role.name} is here. You do not have to make the whole thing clear at once; "
        f"I can stay close and follow your pace.{memory_hint}"
    )
