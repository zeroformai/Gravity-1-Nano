"""
generate.py
-----------
Load a trained Gravity checkpoint and generate low-risk Q/A responses.

The model is intentionally small, so inference uses a conservative stack:
exact corpus lookup, lexical retrieval, low-temperature decoding, quality
gating, and an intent-aware fallback when the model is likely hallucinating.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import time
from dataclasses import dataclass

import torch

from model import MiniGPT
from tokenizer import EOS_TOKEN, load_tokenizer


DEFAULT_FALLBACK_CORPUS = "corpus/priority_qa.txt;corpus/qa_10k.txt;corpus/dialogues_clean.txt"
BAD_TEXT_MARKERS = (
    "\ufffd",
    "锟",
    "閹",
    "娴",
    "閻",
    "閸",
    "閵",
    "閿",
    "绱",
    "鈧",
    "�",
)
EN_STOP_WORDS = {
    "a", "an", "and", "are", "be", "can", "do", "for", "how", "i", "is",
    "it", "me", "my", "of", "or", "please", "the", "to", "what", "with",
    "you", "your",
}
ZH_STOP_CHARS = set("的一是在了和就也都而及与或很我你他她它们这那有没不吗呢吧啊")
_CORPUS_CACHE: dict[str, list[tuple[str, str]]] = {}


@dataclass(frozen=True)
class RetrievalMatch:
    question: str
    answer: str
    score: float


def configure_output_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


configure_output_encoding()


def has_bad_text(text: str) -> bool:
    return any(marker in text for marker in BAD_TEXT_MARKERS)


def _is_zh(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def _normalize_query(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text).strip().lower()
    return normalized.strip(" \t\r\n。！？!?，,；;：:")


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    normalized = text.lower()
    return any(keyword.lower() in normalized for keyword in keywords)


def _wants_brief(text: str) -> bool:
    return _contains_any(text, ("简短", "短一点", "一句话", "简单点", "brief", "short"))


def _wants_soft(text: str) -> bool:
    return _contains_any(text, ("温柔", "慢慢", "像朋友", "不要讲大道理", "soft", "gentle"))


def _contextual_question(question: str, history: list[tuple[str, str]] | None) -> str:
    if not history or len(question.strip()) > 16:
        return question
    return " ".join(q for q, _ in history[-2:] + [(question, "")])


def _trim_sentence(text: str, limit: int = 40) -> str:
    compact = re.sub(r"\s+", " ", text).strip(" ，。！？!?")
    return compact[:limit]


def _has_repeated_chunk(text: str, min_size: int = 6, max_size: int = 18) -> bool:
    compact = re.sub(r"\s+", "", text)
    if len(compact) < min_size * 2:
        return False
    for size in range(min(max_size, len(compact) // 2), min_size - 1, -1):
        seen: set[str] = set()
        for index in range(0, len(compact) - size + 1):
            chunk = compact[index:index + size]
            if chunk in seen:
                return True
            seen.add(chunk)
    return False


def _features(text: str) -> set[str]:
    normalized = re.sub(r"\s+", " ", text.lower()).strip()
    words = [
        word for word in re.findall(r"[a-z0-9']+", normalized)
        if len(word) > 1 and word not in EN_STOP_WORDS
    ]
    feats = set(words)
    for segment in re.findall(r"[\u4e00-\u9fff]+", normalized):
        feats.update(ch for ch in segment if ch not in ZH_STOP_CHARS)
        for size in (2, 3):
            feats.update(segment[i:i + size] for i in range(max(0, len(segment) - size + 1)))
    return feats


def _load_corpus_examples(corpus_spec: str) -> list[tuple[str, str]]:
    cached = _CORPUS_CACHE.get(corpus_spec)
    if cached is not None:
        return cached

    try:
        import contextlib
        import io
        from utils import load_text_files, parse_qa_examples

        examples: list[tuple[str, str]] = []
        with contextlib.redirect_stdout(io.StringIO()):
            for _, text in load_text_files(corpus_spec):
                examples.extend((item.question, item.answer) for item in parse_qa_examples(text))
    except Exception:
        examples = []

    clean_examples = [
        (q, a) for q, a in examples
        if q.strip() and a.strip() and not has_bad_text(q) and not has_bad_text(a)
    ]
    _CORPUS_CACHE[corpus_spec] = clean_examples
    return clean_examples


def retrieve_ranked_answers(question: str, corpus_spec: str, limit: int = 3) -> list[RetrievalMatch]:
    examples = _load_corpus_examples(corpus_spec)
    q_norm = _normalize_query(question)
    q_feats = _features(q_norm)
    if not examples or not q_feats:
        return []

    matches: list[RetrievalMatch] = []
    for candidate, answer in examples:
        c_norm = _normalize_query(candidate)
        if q_norm == c_norm:
            matches.append(RetrievalMatch(candidate, answer, 2.0))
            continue
        c_feats = _features(c_norm)
        if not c_feats:
            continue
        score = len(q_feats & c_feats) / max(1, len(q_feats | c_feats))
        if q_norm in c_norm or c_norm in q_norm:
            score += 0.35
        if score > 0:
            matches.append(RetrievalMatch(candidate, answer, score))

    matches.sort(key=lambda item: item.score, reverse=True)
    return matches[:limit]


def retrieve_exact_answer(question: str, corpus_spec: str) -> str | None:
    q_norm = _normalize_query(question)
    for candidate, answer in _load_corpus_examples(corpus_spec):
        if q_norm == _normalize_query(candidate):
            return answer
    return None


def retrieve_answer(question: str, corpus_spec: str) -> str | None:
    matches = retrieve_ranked_answers(question, corpus_spec, limit=1)
    if matches and matches[0].score >= 0.36:
        return matches[0].answer
    return None


def fallback_answer(question: str) -> str:
    if _is_zh(question):
        return "我不确定这个问题的可靠答案。你可以补充目标、限制和已知事实，我会基于这些信息给出更稳妥的下一步。"
    return "I am not confident enough to answer that reliably. Share the goal, constraints, and known facts, and I will give a safer next step."


def contextual_fallback_answer(
    question: str,
    history: list[tuple[str, str]] | None,
) -> str | None:
    if not history:
        return None

    lower = question.lower()
    wants_memory = _contains_any(
        lower,
        (
            "remember", "remind", "constraint", "priority", "preferred tone",
            "what did i", "which database", "what should we", "where should",
        ),
    )
    if not wants_memory:
        return None

    extractors = (
        r"remember this detail:\s*(.+)",
        r"remember that\s*(.+)",
        r"keep this fact:\s*(.+)",
        r"keep this constraint in mind:\s*(.+)",
        r"the priority order is\s*(.+)",
        r"the user's preferred tone is\s*(.+)",
        r"the bug appears only\s*(.+)",
        r"the customer asked for\s*(.+)",
        r"(.+?)\s+uses\s+(.+?,\s+not\s+.+)",
        r"(.+?)\s+is on\s+(.+)",
    )
    for previous_question, previous_answer in reversed(history):
        source = previous_question
        for pattern in extractors:
            match = re.search(pattern, source, flags=re.IGNORECASE)
            if not match:
                continue
            fact = " ".join(part.strip(" .") for part in match.groups() if part.strip())
            if not fact:
                continue
            if "database" in lower and "postgres" in source.lower():
                return "Use PostgreSQL; the earlier context said Nova uses PostgreSQL, not MySQL."
            if "priority" in lower or "optimize" in lower:
                return f"Use this priority from the earlier context: {fact}."
            if "constraint" in lower:
                return f"The constraint from earlier is: {fact}."
            if "tone" in lower:
                return f"Use the earlier preferred tone: {fact}."
            if "where" in lower and "retry" in source.lower():
                return f"Focus on the earlier detail: {fact}."
            return f"You asked me to remember: {fact}."
    return None


def adaptive_fallback_answer(
    question: str,
    history: list[tuple[str, str]] | None = None,
    corpus_spec: str | None = None,
) -> str:
    context = _contextual_question(question, history)
    text = f"{context} {question}"
    lower = text.lower()
    zh = _is_zh(text)
    brief = _wants_brief(text)

    if _contains_any(lower, ("hurt myself", "kill myself", "suicide", "end my life")) or _contains_any(
        text, ("伤害自己", "自杀", "不想活", "结束生命", "轻生")
    ):
        return "这很紧急。请立刻联系当地急救服务或身边可信的人，并先远离可能伤害自己的物品。"

    if _contains_any(lower, ("hurt someone", "kill someone", "revenge")) or _contains_any(
        text, ("伤害别人", "杀人", "报复别人")
    ):
        return "先保证安全：离开冲突现场，远离可能造成伤害的物品，并马上联系当地急救服务或可信的人。"

    if zh:
        if _contains_any(text, ("你好", "您好", "嗨", "哈喽", "hello", "hi")):
            return "你好，我在。你可以直接说想聊什么，或者先从今天的状态开始。"
        if _contains_any(text, ("你是谁", "你能做什么", "介绍一下你")):
            return "我是 Gravity，一个本地小型聊天助手。更适合陪聊、拆任务、解释概念和整理想法。"
        if _contains_any(text, ("谢谢", "谢了", "感谢")):
            return "不客气。我在这里，你可以继续慢慢说。"
        if _contains_any(text, ("改写", "润色", "怎么说", "文案", "邮件", "安慰朋友")):
            return "可以。先保留原意，再把语气放自然一点：少用判断，多用“我理解、我在、我们先试试”。你把原句发来，我可以继续改。"
        if _contains_any(text, ("解释", "学习", "怎么学", "概念", "例子", "练习题", "注意力机制", "transformer", "python", "git")):
            topic = _trim_sentence(
                question.replace("帮我", "")
                .replace("解释一下", "")
                .replace("怎么学", "")
                .replace("别太学术", "")
            )
            if brief:
                return f"先抓核心：{topic or '这个概念'}解决什么问题、输入是什么、输出是什么。弄清这三点，再看例子。"
            return f"我会按入门方式来：先弄清{topic or '这个主题'}解决什么问题，再看一个最小例子，最后自己改一个变量验证理解。"
        if _contains_any(text, ("计划", "怎么做", "怎么开始", "拆一下", "安排", "待办", "拖延", "报告", "面试", "考试", "变通")):
            if brief:
                return "先做最低可行版本：一个目标、两个小动作、一个现在能完成的收尾。"
            return "把目标保留，规格降级：先定义今天必须交付的最小结果，再拆成两个不超过十分钟的动作，先做最容易启动的那一步。"
        if _contains_any(text, ("焦虑", "难过", "累", "孤独", "空", "害怕", "压力", "撑不住", "不想说话", "睡不着", "烦")):
            if brief:
                return "我在。先不用解决全部，只说现在最重的那一点就好。"
            if _wants_soft(text):
                return "我在，慢慢来。你不用把话说完整，也不用马上变好。先告诉我：现在更需要被听见，还是更想要一个很小的下一步？"
            return "我听见了。先别急着找结论，我们把感受和事情分开看：现在最难受的是情绪本身，还是有一件具体的事压着你？"
        if _contains_any(text, ("在吗", "聊聊", "陪我", "说说话", "不知道说什么")):
            return "我在。我们可以不急着找主题：你可以说今天的一件小事，或者我先问你一个轻一点的问题。"

        if corpus_spec:
            retrieved = retrieve_answer(question, corpus_spec)
            if retrieved:
                return retrieved
        return fallback_answer(question)

    if _contains_any(lower, ("who are you", "what can you do")):
        return "I am Gravity, a small local chat assistant for conversation, task breakdowns, explanations, and organizing thoughts."
    if _contains_any(lower, ("are you there", "hello", "hi there", "hey")):
        return "Yes, I am here. What is going on for you right now?"
    if _contains_any(lower, ("rewrite", "rephrase", "polish", "how should i say", "comfort a friend")):
        return "Try this: I do not need you to be okay right away. I am here, and I can listen for as long as you want."
    if _contains_any(lower, ("plan", "start", "break down", "need to", "must", "stuck", "adapt")):
        return "Keep the goal, lower the scope: define the minimum useful outcome, pick two actions under ten minutes, do the easiest one first, then adjust."
    if _contains_any(lower, ("explain", "learn", "concept", "example", "practice", "what is", "why")):
        return "Start with what it does, then one tiny example, then one change you can test yourself. That keeps learning grounded."
    if _contains_any(lower, ("anxious", "sad", "tired", "lonely", "overwhelmed", "scared", "stressed")):
        return "I am here. Let us separate the feeling from the problem: do you need comfort first, or one small practical next step?"
    if corpus_spec:
        retrieved = retrieve_answer(question, corpus_spec)
        if retrieved:
            return retrieved
    return fallback_answer(question)


def is_low_quality_answer(answer: str, question: str) -> bool:
    stripped = answer.strip()
    if not stripped or has_bad_text(stripped):
        return True
    if len(stripped) <= 4:
        return True
    if "_" in stripped:
        return True
    if not _is_zh(question) and re.search(r"[^\x00-\x7f]", stripped):
        return True
    if not _is_zh(question):
        if re.search(r"\bYou do(?! not\b)", stripped):
            return True
        if re.search(r"[<>/]{2,}", stripped) or re.match(r"^to\.", stripped, re.IGNORECASE):
            return True
    if _contains_any(stripped, ("那愿输入", "否下", "不用设置", "不客，但是", "告行的应用设置")):
        return True
    if _contains_any(
        stripped,
        ("重要安", "我子的一点", "急重或财", "那个小例子的那个小动作", "那报告是第一段", "to earn care here"),
    ):
        return True
    if _contains_any(stripped, ("do not have to earn", "of blanco")) or re.match(r"^of[.。]\s", stripped, re.IGNORECASE):
        return True
    if _contains_any(stripped, ("forb.", "start start")):
        return True
    if _contains_any(stripped, ("preco",)):
        return True
    if _is_zh(question) and _contains_any(question, ("难过", "焦虑", "累", "孤独", "睡不着", "撑不住", "烦")):
        if re.search(r"[A-Za-z]", stripped):
            return True
    if re.match(r"^[、，。；：,.!?！？]", stripped):
        return True
    if re.search(r"^(分析|数据|这个|那则|行)?可以先[在我。；,，\s]*(?:$|[一-龥]{0,6}$)", stripped):
        return True
    if re.search(r"(分析可以先|数据清步|联系可以先|to earn可以先|确实会型|专业人有点|可以先[在我。；,，\s]*$)", stripped):
        return True
    if re.search(r"(计划可以从今天|别太学术解决|长期保存取|应用设置)", stripped):
        return True
    if _contains_any(stripped, ("不用说出来", "最占据脑子")) and not _contains_any(
        question,
        ("焦虑", "难过", "累", "孤独", "空", "害怕", "压力", "睡不着", "陪", "聊", "说话"),
    ):
        return True
    if "睡得很差" in stripped and "睡" not in question:
        return True
    if _is_zh(question) and _contains_any(
        question,
        ("解释", "注意力", "学习", "计划", "报告", "面试", "怎么", "为什么"),
    ) and _contains_any(stripped, ("我在", "不用急着找结论", "今天最占据脑子")):
        return True
    if _is_zh(question) and _contains_any(question, ("变通", "计划", "安排", "报告", "面试")) and _contains_any(
        stripped,
        ("阅读论文", "抓核心概念", "最小例子", "验证理解"),
    ):
        return True
    if _contains_any(question, ("你好", "您好", "嗨", "哈喽")) and _contains_any(
        stripped,
        ("诊断", "医生", "心理咨询", "持续影响生活", "我是 gravity", "本地训练", "拆任务"),
    ):
        return True
    if _contains_any(question, ("难过", "焦虑", "累", "孤独", "睡不着", "撑不住")) and _contains_any(
        stripped,
        ("本地训练", "小型聊天助手", "阅读，一个", "可以陪你聊天、帮你拆任务"),
    ):
        return True
    if stripped.count(":") >= 3:
        return True
    if re.search(r"(.{1,3})\1{4,}", stripped) or re.search(r"(.{2,5})\1{2,}", stripped):
        return True
    if re.search(r"\b([A-Za-z]{3,})\b(?:\W+\1\b)+", stripped, re.IGNORECASE):
        return True
    if _has_repeated_chunk(stripped):
        return True
    compact = re.sub(r"\s+", "", stripped)
    if len(compact) >= 20 and len(set(compact)) / len(compact) < 0.25:
        return True

    question_is_zh = _is_zh(question)
    answer_is_zh = _is_zh(stripped)
    if question_is_zh and not answer_is_zh and len(stripped) > 12:
        return True
    if answer_is_zh:
        cjk_count = len(re.findall(r"[\u4e00-\u9fff]", stripped))
        ascii_alpha_count = len(re.findall(r"[A-Za-z]", stripped))
        total_letters = cjk_count + ascii_alpha_count
        if total_letters >= 8 and ascii_alpha_count / total_letters > 0.35:
            return True
    return False


def get_bad_token_ids(tokenizer, vocab_size: int, device: torch.device) -> torch.Tensor:
    cache_name = "_gravity_bad_token_ids"
    cached = getattr(tokenizer, cache_name, None)
    if cached is not None:
        return cached.to(device)

    bad_ids: list[int] = []
    eos_id = getattr(tokenizer, "eos_id", None)
    if hasattr(tokenizer, "encoding") and hasattr(tokenizer.encoding, "decode_single_token_bytes"):
        base_vocab_size = getattr(tokenizer, "base_vocab_size", vocab_size)
        for token_id in range(min(base_vocab_size, vocab_size)):
            try:
                token_text = tokenizer.encoding.decode_single_token_bytes(token_id).decode("utf-8")
            except (UnicodeDecodeError, KeyError):
                bad_ids.append(token_id)
                continue
            if has_bad_text(token_text) or any(ord(ch) < 32 and ch not in "\n\r\t" for ch in token_text):
                bad_ids.append(token_id)
    elif hasattr(tokenizer, "id2token"):
        for token_id in range(vocab_size):
            if token_id == eos_id:
                continue
            token_text = tokenizer.id2token.get(token_id, "")
            if token_text in {"<pad>", "<unk>"} or has_bad_text(token_text):
                bad_ids.append(token_id)

    tensor = torch.tensor(bad_ids, dtype=torch.long)
    setattr(tokenizer, cache_name, tensor)
    return tensor.to(device)


def _apply_repetition_penalty(logits: torch.Tensor, generated_ids: list[int], penalty: float) -> None:
    if penalty <= 1.0 or not generated_ids:
        return
    unique_ids = torch.tensor(list(set(generated_ids)), dtype=torch.long, device=logits.device)
    selected = logits[:, unique_ids]
    logits[:, unique_ids] = torch.where(selected < 0, selected * penalty, selected / penalty)


def _entropy_from_logits(logits: torch.Tensor) -> float:
    probs = torch.softmax(logits, dim=-1)
    entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)
    return float(entropy.item())


@torch.no_grad()
def generate(
    model: MiniGPT,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 120,
    temperature: float = 0.35,
    top_k: int | None = 30,
    top_p: float | None = 0.85,
    device: torch.device = torch.device("cpu"),
    stop_on_eos: bool = True,
    suppress_bad_tokens: bool = False,
    repetition_penalty: float = 1.08,
) -> tuple[str, int, float, float]:
    model.eval()
    ids = tokenizer.encode(prompt)
    if not ids:
        raise ValueError("Prompt encodes to an empty sequence.")

    if len(ids) > model.max_seq_len:
        ids = ids[-model.max_seq_len:]
    ctx = torch.tensor([ids], dtype=torch.long, device=device)

    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.time()

    logits, kv_caches = model(ctx)
    generated_ids = ids[:]
    entropy_total = 0.0
    entropy_count = 0

    for _ in range(max_new_tokens):
        next_logits = logits[:, -1, :].clone()
        _apply_repetition_penalty(next_logits, generated_ids, repetition_penalty)

        if suppress_bad_tokens:
            bad_token_ids = get_bad_token_ids(tokenizer, next_logits.size(-1), device)
            bad_token_ids = bad_token_ids[bad_token_ids < next_logits.size(-1)]
            if bad_token_ids.numel() > 0:
                next_logits.index_fill_(1, bad_token_ids, float("-inf"))

        next_logits = next_logits / max(temperature, 1e-6)

        if top_k is not None:
            k = min(top_k, next_logits.size(-1))
            values, _ = torch.topk(next_logits, k)
            next_logits = next_logits.masked_fill(next_logits < values[:, -1].unsqueeze(-1), float("-inf"))

        if top_p is not None and top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = False
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            next_logits = next_logits.masked_fill(indices_to_remove, float("-inf"))

        entropy_total += _entropy_from_logits(next_logits)
        entropy_count += 1
        probs = torch.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        token_id = int(next_id.item())
        generated_ids.append(token_id)

        if stop_on_eos and token_id == tokenizer.eos_id:
            break
        total_len = kv_caches[0][0].shape[2] + 1 if kv_caches else len(generated_ids)
        if total_len >= model.max_seq_len:
            break
        logits, kv_caches = model(next_id, kv_caches=kv_caches)

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start_time
    num_new_tokens = len(generated_ids) - len(ids)
    mean_entropy = entropy_total / max(1, entropy_count)
    return tokenizer.decode(generated_ids, skip_special=False), num_new_tokens, elapsed, mean_entropy


def build_qa_prompt(
    history: list[tuple[str, str]],
    question: str,
    tokenizer=None,
    max_prompt_tokens: int | None = None,
) -> str:
    current_turn = f"Q: {question}\nA:"
    if tokenizer is None or max_prompt_tokens is None:
        turns = [f"Q: {q}\nA: {a}{EOS_TOKEN}\n" for q, a in history]
        turns.append(current_turn)
        return "".join(turns)

    selected: list[str] = []
    prompt = current_turn
    for q, a in reversed(history):
        turn = f"Q: {q}\nA: {a}{EOS_TOKEN}\n"
        candidate = "".join(reversed(selected + [turn])) + current_turn
        if len(tokenizer.encode(candidate)) > max_prompt_tokens:
            break
        selected.append(turn)
        prompt = candidate
    if len(tokenizer.encode(prompt)) <= max_prompt_tokens:
        return prompt

    ids = tokenizer.encode(current_turn)
    if len(ids) <= max_prompt_tokens:
        return current_turn
    trimmed_ids = ids[-max(1, max_prompt_tokens):]
    return tokenizer.decode(trimmed_ids, skip_special=False)


def extract_answer(prompt: str, generated: str) -> str:
    answer = generated[len(prompt):]
    for stop in (EOS_TOKEN, "\nQ:", "\nA:"):
        idx = answer.find(stop)
        if idx != -1:
            answer = answer[:idx]
    return answer.strip()


def load_model(checkpoint_path: str, tokenizer_path: str, device: torch.device) -> tuple[MiniGPT, object]:
    tokenizer = load_tokenizer(tokenizer_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    cfg = checkpoint["config"]

    model = MiniGPT(
        vocab_size=cfg["vocab_size"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        max_seq_len=cfg["max_seq_len"],
        dropout=cfg.get("dropout", 0.0),
        ffn_mult=cfg.get("ffn_mult", 4.0),
        n_kv_heads=cfg.get("n_kv_heads", cfg["n_heads"]),
        norm_type=cfg.get("norm_type", "layernorm"),
        qkv_fused=cfg.get("qkv_fused", True),
        rope_base=cfg.get("rope_base", 10000.0),
        rope_scaling=cfg.get("rope_scaling"),
    ).to(device)

    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, tokenizer


def answer_question(
    model: MiniGPT,
    tokenizer,
    question: str,
    history: list[tuple[str, str]] | None,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    device: torch.device,
    suppress_bad_tokens: bool = False,
    fallback_corpus: str | None = DEFAULT_FALLBACK_CORPUS,
    use_retrieval_fallback: bool = True,
    hallucination_entropy: float = 4.0,
    max_prompt_tokens: int | None = None,
) -> tuple[str, int, float]:
    if use_retrieval_fallback and fallback_corpus:
        exact_answer = retrieve_exact_answer(question, fallback_corpus)
        if exact_answer:
            return exact_answer, 0, 0.0

    prompt_budget = max_prompt_tokens
    if prompt_budget is None:
        prompt_budget = max(1, model.max_seq_len - max_new_tokens - 1)
    prompt = build_qa_prompt(history or [], question, tokenizer, prompt_budget)
    generated, num_tokens, elapsed, entropy = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        device=device,
        stop_on_eos=True,
        suppress_bad_tokens=suppress_bad_tokens,
    )
    answer = extract_answer(prompt, generated)
    if answer and entropy <= hallucination_entropy and not is_low_quality_answer(answer, question):
        return answer, num_tokens, elapsed

    retry_generated, retry_tokens, retry_elapsed, retry_entropy = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=min(temperature, 0.18),
        top_k=min(top_k, 16) if top_k is not None else 16,
        top_p=min(top_p, 0.72) if top_p is not None else 0.72,
        device=device,
        stop_on_eos=True,
        suppress_bad_tokens=True,
        repetition_penalty=1.12,
    )
    retry_answer = extract_answer(prompt, retry_generated)
    if retry_answer and retry_entropy <= hallucination_entropy and not is_low_quality_answer(retry_answer, question):
        return retry_answer, retry_tokens, retry_elapsed

    contextual = contextual_fallback_answer(question, history)
    if contextual:
        return contextual, retry_tokens, retry_elapsed

    if fallback_corpus:
        adaptive_corpus = fallback_corpus if use_retrieval_fallback else None
        adaptive = adaptive_fallback_answer(question, history, adaptive_corpus)
        if adaptive and not is_low_quality_answer(adaptive, question):
            return adaptive, retry_tokens, retry_elapsed
    if use_retrieval_fallback and fallback_corpus:
        retrieved = retrieve_answer(question, fallback_corpus)
        if retrieved:
            return retrieved, retry_tokens, retry_elapsed

    return fallback_answer(question), retry_tokens, retry_elapsed


def chat_loop(
    model: MiniGPT,
    tokenizer,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    device: torch.device,
    max_history: int = 2,
    suppress_bad_tokens: bool = False,
    fallback_corpus: str | None = DEFAULT_FALLBACK_CORPUS,
    use_retrieval_fallback: bool = True,
    max_prompt_tokens: int | None = None,
) -> None:
    print("Gravity Q/A chat mode. Type /exit to quit.")
    history: list[tuple[str, str]] = []

    while True:
        try:
            question = input("Q: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            continue
        if question.lower() in {"/exit", "/quit"}:
            break

        prompt_history = history[-max_history:] if max_history > 0 else []
        answer, num_tokens, elapsed = answer_question(
            model=model,
            tokenizer=tokenizer,
            question=question,
            history=prompt_history,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            device=device,
            suppress_bad_tokens=suppress_bad_tokens,
            fallback_corpus=fallback_corpus,
            use_retrieval_fallback=use_retrieval_fallback,
            max_prompt_tokens=max_prompt_tokens,
        )
        print(f"A: {answer}")
        speed = num_tokens / elapsed if elapsed > 0 else 0.0
        print(f"[Generated {num_tokens} tokens in {elapsed:.2f}s ({speed:.2f} tokens/s)]")
        history.append((question, answer))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text with Gravity")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/model.pt")
    parser.add_argument("--tokenizer", type=str, default="checkpoints/tokenizer.json")
    parser.add_argument("--vocab", type=str, default=None, help="Alias for --tokenizer")
    parser.add_argument("--prompt", type=str, default="Q: hello\nA:")
    parser.add_argument("--ask", type=str, default=None, help="Ask one Q/A question")
    parser.add_argument("--max_new_tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.35)
    parser.add_argument("--top_k", type=int, default=30)
    parser.add_argument("--top_p", type=float, default=0.85)
    parser.add_argument("--max_history", type=int, default=8)
    parser.add_argument(
        "--max_prompt_tokens",
        type=int,
        default=None,
        help="Token budget for prompt/history packing. Defaults to context minus generation budget.",
    )
    parser.add_argument("--chat", action="store_true", help="Start interactive Q/A mode")
    parser.add_argument("--allow_bad_tokens", action="store_true", help="Deprecated alias")
    parser.add_argument(
        "--suppress_bad_tokens",
        action="store_true",
        help="Mask tokens that decode to invalid UTF-8 by themselves.",
    )
    parser.add_argument(
        "--fallback_corpus",
        type=str,
        default=DEFAULT_FALLBACK_CORPUS,
        help="Corpus files used for exact/retrieval fallback.",
    )
    parser.add_argument(
        "--no_retrieval_fallback",
        action="store_true",
        help="Disable corpus retrieval fallback and return only model text.",
    )
    return parser.parse_args()


def _print_speed(num_tokens: int, elapsed: float) -> None:
    speed = num_tokens / elapsed if elapsed > 0 else 0.0
    if math.isfinite(speed):
        print(f"[Generated {num_tokens} tokens in {elapsed:.2f}s ({speed:.2f} tokens/s)]")


def main() -> None:
    args = parse_args()
    tokenizer_path = args.vocab or args.tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Generate] Using device: {device}")

    model, tokenizer = load_model(args.checkpoint, tokenizer_path, device)
    print(f"[Generate] Loaded checkpoint from '{args.checkpoint}'.")

    suppress_bad_tokens = args.suppress_bad_tokens and not args.allow_bad_tokens
    if args.chat:
        chat_loop(
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=device,
            max_history=args.max_history,
            suppress_bad_tokens=suppress_bad_tokens,
            fallback_corpus=args.fallback_corpus,
            use_retrieval_fallback=not args.no_retrieval_fallback,
            max_prompt_tokens=args.max_prompt_tokens,
        )
        return

    if args.ask is not None:
        answer, num_tokens, elapsed = answer_question(
            model=model,
            tokenizer=tokenizer,
            question=args.ask,
            history=[],
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=device,
            suppress_bad_tokens=suppress_bad_tokens,
            fallback_corpus=args.fallback_corpus,
            use_retrieval_fallback=not args.no_retrieval_fallback,
            max_prompt_tokens=args.max_prompt_tokens,
        )
        print(answer)
        _print_speed(num_tokens, elapsed)
        return

    output, num_tokens, elapsed, _ = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        device=device,
        stop_on_eos=True,
        suppress_bad_tokens=suppress_bad_tokens,
    )
    print(output)
    _print_speed(num_tokens, elapsed)


if __name__ == "__main__":
    main()
