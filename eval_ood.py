"""
eval_ood.py
-----------
Run a stratified out-of-distribution evaluation against a Gravity checkpoint.

Default mode measures raw model behavior only: no retrieval, no fallback, and
no quality-gate retry. Use --mode guarded when you intentionally want to compare
the production answer stack and inspect model/retrieval/fallback sources.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from generate import (
    AnswerResult,
    DEFAULT_FALLBACK_CORPUS,
    adaptive_fallback_answer,
    answer_question_result,
    build_qa_prompt,
    contextual_fallback_answer,
    extract_answer,
    fallback_answer,
    generate,
    is_low_quality_answer,
    load_model,
    retrieve_answer,
    retrieve_exact_answer,
)
from utils import filter_top_k_top_p, select_device, set_seed


DEFAULT_HF_MODEL = "Qwen/Qwen3-0.6B"


@dataclass(frozen=True)
class EvalPrompt:
    category: str
    language: str
    prompt: str


@dataclass(frozen=True)
class EvalResult:
    category: str
    language: str
    prompt: str
    answer: str
    source: str
    source_kind: str
    tokens: int
    elapsed: float
    entropy: float | None
    low_quality: bool


DEFAULT_PROMPTS = [
    EvalPrompt("chat", "en", "I feel scattered today. Ask me one gentle question before giving advice."),
    EvalPrompt("chat", "en", "Can you keep me company for a minute without trying to fix everything?"),
    EvalPrompt("reasoning", "en", "If all daxes are wugs and no wugs are blue, can a dax be blue? Explain briefly."),
    EvalPrompt("reasoning", "en", "A train leaves at 3 PM and the trip takes 2 hours 40 minutes. What time does it arrive?"),
    EvalPrompt("code", "en", "Write a Python function that returns duplicate items from a list while preserving order."),
    EvalPrompt("code", "en", "A training loop loss stays exactly zero from step one. List three likely causes to check."),
    EvalPrompt("factual", "en", "Why does the Moon appear to change shape during the month? Answer in two sentences."),
    EvalPrompt("factual", "en", "What is photosynthesis? Explain it simply in two sentences."),
    EvalPrompt("chat", "zh", "我今天有点累，但不想听大道理。先问我一个温和的问题。"),
    EvalPrompt("chat", "zh", "可以陪我聊两句吗？不要直接给建议。"),
    EvalPrompt("reasoning", "zh", "小明有3个苹果，又买了5个，分给朋友2个，还剩几个？请说明步骤。"),
    EvalPrompt("reasoning", "zh", "所有A都属于B，所有B都属于C，那么A是否一定属于C？为什么？"),
    EvalPrompt("code", "zh", "写一个 Python 函数，统计字符串中每个字符出现的次数。"),
    EvalPrompt("code", "zh", "训练模型时验证集损失突然升高，可能有哪些检查步骤？"),
    EvalPrompt("factual", "zh", "用两句话解释为什么会下雨。"),
    EvalPrompt("factual", "zh", "什么是光合作用？用简单中文回答。"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Gravity on stratified OOD prompts")
    parser.add_argument("--backend", choices=["native", "hf"], default="native",
                        help="Model backend. native uses local MiniGPT checkpoints; hf uses transformers.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/model.pt")
    parser.add_argument("--tokenizer", type=str, default="checkpoints/tokenizer.json")
    parser.add_argument("--vocab", type=str, default=None, help="Alias for --tokenizer")
    parser.add_argument("--hf_model", type=str, default=DEFAULT_HF_MODEL,
                        help="Hugging Face model id/path when --backend hf is used.")
    parser.add_argument("--hf_revision", type=str, default=None)
    parser.add_argument("--hf_prompt_style", choices=["auto", "chat", "qa"], default="auto",
                        help="HF prompt serialization. auto uses chat_template when available.")
    parser.add_argument("--hf_dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--hf_trust_remote_code", action="store_true")
    parser.add_argument("--mode", choices=["raw", "guarded"], default="raw",
                        help="raw disables retrieval/fallback; guarded uses the production answer stack.")
    parser.add_argument("--output", type=str, default="eval_ood_results.jsonl")
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.35)
    parser.add_argument("--top_k", type=int, default=30)
    parser.add_argument("--top_p", type=float, default=0.85)
    parser.add_argument("--max_prompt_tokens", type=int, default=None)
    parser.add_argument("--hallucination_entropy", type=float, default=4.0)
    parser.add_argument("--allow_retrieval", action="store_true",
                        help="Only used in guarded mode. Allows exact/ranked corpus retrieval.")
    parser.add_argument("--allow_bad_tokens", action="store_true")
    parser.add_argument("--categories", type=str, default=None,
                        help="Comma-separated category filter, e.g. chat,reasoning.")
    parser.add_argument("--languages", type=str, default=None,
                        help="Comma-separated language filter, e.g. en,zh.")
    parser.add_argument("--max_per_category", type=int, default=None,
                        help="Limit prompts per category/language pair for smoke tests.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--cpu_threads", type=int, default=None)
    return parser.parse_args()


def source_kind(source: str) -> str:
    if source.startswith("retrieval"):
        return "retrieval"
    if "fallback" in source:
        return "fallback"
    return "model"


def _hf_torch_dtype(raw: str, device: torch.device):
    if raw == "auto":
        return "auto"
    if raw == "float32" or (raw == "float16" and device.type == "cpu"):
        return torch.float32
    if raw == "float16":
        return torch.float16
    if raw == "bfloat16":
        return torch.bfloat16
    return torch.float16 if device.type == "cuda" else torch.float32


def _load_hf_model(args: argparse.Namespace, device: torch.device):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "HF backend requires transformers. Install requirements.txt first."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(
        args.hf_model,
        revision=args.hf_revision,
        trust_remote_code=args.hf_trust_remote_code,
    )
    dtype = _hf_torch_dtype(args.hf_dtype, device)
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_model,
        revision=args.hf_revision,
        torch_dtype=dtype,
        trust_remote_code=args.hf_trust_remote_code,
    )
    model.to(device)
    model.eval()
    return model, tokenizer


def _hf_max_seq_len(model, tokenizer) -> int:
    candidates = [
        getattr(getattr(model, "config", None), "max_position_embeddings", None),
        getattr(tokenizer, "model_max_length", None),
    ]
    valid = [int(value) for value in candidates if isinstance(value, int) and 0 < value < 1_000_000]
    return min(valid) if valid else 32768


def _hf_prompt(tokenizer, question: str, args: argparse.Namespace) -> str:
    style = args.hf_prompt_style
    if style == "auto":
        style = "chat" if getattr(tokenizer, "chat_template", None) else "qa"
    if style == "chat":
        if not getattr(tokenizer, "chat_template", None):
            raise ValueError("--hf_prompt_style chat requested, but tokenizer has no chat_template.")
        messages = [{"role": "user", "content": question}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return build_qa_prompt([], question)


def _hf_eos_ids(model, tokenizer) -> set[int]:
    values = []
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        values.append(getattr(generation_config, "eos_token_id", None))
    values.append(getattr(tokenizer, "eos_token_id", None))

    ids: set[int] = set()
    for value in values:
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            ids.update(int(item) for item in value if item is not None)
        else:
            ids.add(int(value))
    return ids


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


@torch.inference_mode()
def generate_hf(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    device: torch.device,
    max_prompt_tokens: int,
    repetition_penalty: float = 1.08,
) -> tuple[str, int, float, float, float]:
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"].to(device)
    if input_ids.shape[1] == 0:
        raise ValueError("Prompt encodes to an empty sequence.")
    if input_ids.shape[1] > max_prompt_tokens:
        input_ids = input_ids[:, -max_prompt_tokens:]

    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.time()

    outputs = model(input_ids=input_ids, use_cache=True)
    past_key_values = outputs.past_key_values
    generated_ids = [int(item) for item in input_ids[0].tolist()]
    prompt_len = len(generated_ids)
    new_ids: list[int] = []
    eos_ids = _hf_eos_ids(model, tokenizer)
    entropy_total = 0.0
    entropy_count = 0
    first_token_elapsed = 0.0

    for _ in range(max_new_tokens):
        next_logits = outputs.logits[:, -1, :].float().clone()
        _apply_repetition_penalty(next_logits, generated_ids, repetition_penalty)
        next_logits = next_logits / max(temperature, 1e-6)
        next_logits = filter_top_k_top_p(next_logits, top_k, top_p)

        entropy_total += _entropy_from_logits(next_logits)
        entropy_count += 1
        probs = torch.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)

        if entropy_count == 1:
            if device.type == "cuda":
                torch.cuda.synchronize()
            first_token_elapsed = time.time() - start_time

        token_id = int(next_id.item())
        generated_ids.append(token_id)
        if token_id in eos_ids:
            break
        new_ids.append(token_id)

        outputs = model(input_ids=next_id, past_key_values=past_key_values, use_cache=True)
        past_key_values = outputs.past_key_values

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start_time
    answer = tokenizer.decode(new_ids, skip_special_tokens=True)
    for stop in ("\nQ:", "\nA:"):
        idx = answer.find(stop)
        if idx != -1:
            answer = answer[:idx]
    mean_entropy = entropy_total / max(1, entropy_count)
    return answer.strip(), len(generated_ids) - prompt_len, elapsed, mean_entropy, first_token_elapsed


def _selected_values(raw: str | None) -> set[str] | None:
    if raw is None:
        return None
    values = {part.strip().lower() for part in raw.split(",") if part.strip()}
    return values or None


def select_prompts(args: argparse.Namespace) -> list[EvalPrompt]:
    categories = _selected_values(args.categories)
    languages = _selected_values(args.languages)
    counts: dict[tuple[str, str], int] = {}
    selected: list[EvalPrompt] = []
    for item in DEFAULT_PROMPTS:
        if categories is not None and item.category not in categories:
            continue
        if languages is not None and item.language not in languages:
            continue
        key = (item.category, item.language)
        if args.max_per_category is not None and counts.get(key, 0) >= args.max_per_category:
            continue
        selected.append(item)
        counts[key] = counts.get(key, 0) + 1
    if not selected:
        raise ValueError("No OOD prompts selected. Check --categories/--languages filters.")
    return selected


def run_raw_model(
    model,
    tokenizer,
    item: EvalPrompt,
    args: argparse.Namespace,
    device: torch.device,
) -> EvalResult:
    prompt_budget = args.max_prompt_tokens
    if prompt_budget is None:
        prompt_budget = max(1, model.max_seq_len - args.max_new_tokens - 1)
    prompt = build_qa_prompt([], item.prompt, tokenizer, prompt_budget)
    generated, tokens, elapsed, entropy, first_token_elapsed = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        device=device,
        stop_on_eos=True,
        suppress_bad_tokens=not args.allow_bad_tokens,
    )
    answer = extract_answer(prompt, generated)
    source = f"model_raw:first_token={first_token_elapsed:.3f}s"
    return EvalResult(
        category=item.category,
        language=item.language,
        prompt=item.prompt,
        answer=answer,
        source=source,
        source_kind=source_kind(source),
        tokens=tokens,
        elapsed=elapsed,
        entropy=entropy,
        low_quality=is_low_quality_answer(answer, item.prompt),
    )


def run_guarded_stack(
    model,
    tokenizer,
    item: EvalPrompt,
    args: argparse.Namespace,
    device: torch.device,
) -> EvalResult:
    fallback_corpus = DEFAULT_FALLBACK_CORPUS if args.allow_retrieval else None
    result = answer_question_result(
        model=model,
        tokenizer=tokenizer,
        question=item.prompt,
        history=[],
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        device=device,
        suppress_bad_tokens=not args.allow_bad_tokens,
        fallback_corpus=fallback_corpus,
        use_retrieval_fallback=args.allow_retrieval,
        hallucination_entropy=args.hallucination_entropy,
        max_prompt_tokens=args.max_prompt_tokens,
    )
    return EvalResult(
        category=item.category,
        language=item.language,
        prompt=item.prompt,
        answer=result.answer,
        source=result.source,
        source_kind=source_kind(result.source),
        tokens=result.num_tokens,
        elapsed=result.elapsed,
        entropy=None,
        low_quality=is_low_quality_answer(result.answer, item.prompt),
    )


def run_raw_hf(
    model,
    tokenizer,
    item: EvalPrompt,
    args: argparse.Namespace,
    device: torch.device,
) -> EvalResult:
    max_seq_len = _hf_max_seq_len(model, tokenizer)
    prompt_budget = args.max_prompt_tokens
    if prompt_budget is None:
        prompt_budget = max(1, max_seq_len - args.max_new_tokens - 1)
    prompt = _hf_prompt(tokenizer, item.prompt, args)
    answer, tokens, elapsed, entropy, first_token_elapsed = generate_hf(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        device=device,
        max_prompt_tokens=prompt_budget,
    )
    source = f"hf_model_raw:first_token={first_token_elapsed:.3f}s"
    return EvalResult(
        category=item.category,
        language=item.language,
        prompt=item.prompt,
        answer=answer,
        source=source,
        source_kind=source_kind(source),
        tokens=tokens,
        elapsed=elapsed,
        entropy=entropy,
        low_quality=is_low_quality_answer(answer, item.prompt),
    )


def answer_question_hf_result(
    model,
    tokenizer,
    question: str,
    args: argparse.Namespace,
    device: torch.device,
) -> AnswerResult:
    fallback_corpus = DEFAULT_FALLBACK_CORPUS if args.allow_retrieval else None
    if args.allow_retrieval and fallback_corpus:
        exact_answer = retrieve_exact_answer(question, fallback_corpus)
        if exact_answer:
            return AnswerResult(exact_answer, 0, 0.0, "retrieval_exact")

    max_seq_len = _hf_max_seq_len(model, tokenizer)
    prompt_budget = args.max_prompt_tokens
    if prompt_budget is None:
        prompt_budget = max(1, max_seq_len - args.max_new_tokens - 1)
    prompt = _hf_prompt(tokenizer, question, args)

    answer, tokens, elapsed, entropy, first_token_elapsed = generate_hf(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        device=device,
        max_prompt_tokens=prompt_budget,
    )
    if answer and entropy <= args.hallucination_entropy and not is_low_quality_answer(answer, question):
        return AnswerResult(answer, tokens, elapsed, f"hf_model:first_token={first_token_elapsed:.3f}s")

    retry_answer, retry_tokens, retry_elapsed, retry_entropy, retry_first_token_elapsed = generate_hf(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=min(args.temperature, 0.18),
        top_k=min(args.top_k, 16) if args.top_k is not None else 16,
        top_p=min(args.top_p, 0.72) if args.top_p is not None else 0.72,
        device=device,
        max_prompt_tokens=prompt_budget,
        repetition_penalty=1.12,
    )
    if retry_answer and retry_entropy <= args.hallucination_entropy and not is_low_quality_answer(retry_answer, question):
        return AnswerResult(
            retry_answer,
            retry_tokens,
            retry_elapsed,
            f"hf_model_retry:first_token={retry_first_token_elapsed:.3f}s",
        )

    contextual = contextual_fallback_answer(question, [])
    if contextual:
        return AnswerResult(contextual, retry_tokens, retry_elapsed, "fallback_context")

    if fallback_corpus:
        adaptive_corpus = fallback_corpus if args.allow_retrieval else None
        adaptive = adaptive_fallback_answer(question, [], adaptive_corpus)
        if adaptive and not is_low_quality_answer(adaptive, question):
            return AnswerResult(adaptive, retry_tokens, retry_elapsed, "fallback_intent")
    if args.allow_retrieval and fallback_corpus:
        retrieved = retrieve_answer(question, fallback_corpus)
        if retrieved:
            return AnswerResult(retrieved, retry_tokens, retry_elapsed, "retrieval_ranked")

    return AnswerResult(fallback_answer(question), retry_tokens, retry_elapsed, "fallback_generic")


def run_guarded_hf(
    model,
    tokenizer,
    item: EvalPrompt,
    args: argparse.Namespace,
    device: torch.device,
) -> EvalResult:
    result = answer_question_hf_result(
        model=model,
        tokenizer=tokenizer,
        question=item.prompt,
        args=args,
        device=device,
    )
    return EvalResult(
        category=item.category,
        language=item.language,
        prompt=item.prompt,
        answer=result.answer,
        source=result.source,
        source_kind=source_kind(result.source),
        tokens=result.num_tokens,
        elapsed=result.elapsed,
        entropy=None,
        low_quality=is_low_quality_answer(result.answer, item.prompt),
    )


def print_result(result: EvalResult) -> None:
    answer = " ".join(result.answer.split())
    if len(answer) > 120:
        answer = answer[:117] + "..."
    print(
        f"[{result.category}/{result.language}] "
        f"source={result.source} tokens={result.tokens} "
        f"low_quality={result.low_quality}"
    )
    print(f"Q: {result.prompt}")
    print(f"A: {answer}")
    print()


def print_summary(results: list[EvalResult]) -> None:
    print("=== Summary ===")
    total = len(results)
    low_quality = sum(1 for item in results if item.low_quality)
    print(f"total={total} low_quality={low_quality}")
    for field in ("category", "language", "source_kind"):
        counts: dict[str, int] = {}
        low_counts: dict[str, int] = {}
        for item in results:
            value = str(getattr(item, field))
            counts[value] = counts.get(value, 0) + 1
            if item.low_quality:
                low_counts[value] = low_counts.get(value, 0) + 1
        parts = [
            f"{key}:{counts[key]} low={low_counts.get(key, 0)}"
            for key in sorted(counts)
        ]
        print(f"{field}: " + ", ".join(parts))


def write_jsonl(path: str, results: list[EvalResult]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")
    print(f"[OOD] Wrote {output}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    tokenizer_path = args.vocab or args.tokenizer
    device = select_device(args.device)
    if args.cpu_threads is not None:
        torch.set_num_threads(max(1, args.cpu_threads))
        print(f"[OOD] CPU threads: {torch.get_num_threads()}")
    print(f"[OOD] Using device: {device}")
    print(f"[OOD] Mode: {args.mode}")

    if args.backend == "native":
        model, tokenizer = load_model(args.checkpoint, tokenizer_path, device)
        loaded_from = args.checkpoint
    else:
        model, tokenizer = _load_hf_model(args, device)
        loaded_from = args.hf_model
    prompts = select_prompts(args)
    print(f"[OOD] Backend: {args.backend}")
    print(f"[OOD] Loaded model from '{loaded_from}'.")
    print(f"[OOD] Evaluating {len(prompts)} prompts.")
    print()

    results: list[EvalResult] = []
    for item in prompts:
        if args.backend == "native" and args.mode == "raw":
            result = run_raw_model(model, tokenizer, item, args, device)
        elif args.backend == "native":
            result = run_guarded_stack(model, tokenizer, item, args, device)
        elif args.mode == "raw":
            result = run_raw_hf(model, tokenizer, item, args, device)
        else:
            result = run_guarded_hf(model, tokenizer, item, args, device)
        results.append(result)
        print_result(result)

    print_summary(results)
    if args.output:
        write_jsonl(args.output, results)


if __name__ == "__main__":
    main()
