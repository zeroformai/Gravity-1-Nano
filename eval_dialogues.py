"""
eval_dialogues.py
-----------------
Run short multi-turn smoke tests against a Gravity checkpoint.
"""

import argparse

import torch

from generate import answer_question, load_model


SCENARIOS = [
    [
        "我想聊聊天",
        "有点累",
        "心里紧",
        "说不清楚",
        "谢谢你",
    ],
    [
        "在吗",
        "我今天很烦",
        "事情太多",
        "我不想处理",
    ],
    [
        "我不知道说什么",
        "有点空",
        "没怎么吃饭",
        "我等会儿去",
    ],
    [
        "我睡不着",
        "一直想白天的事",
        "换个轻一点的",
        "晚安",
    ],
    [
        "帮我做明天计划",
        "我必须写报告",
        "怕写不好",
        "那我先列提纲",
    ],
    [
        "我不想处理工作但是必须写报告",
        "帮我变通一个",
        "只有二十分钟",
        "那我先写三个小标题",
    ],
    [
        "are you there",
        "i'm tired",
        "mostly mind tired",
        "thanks for listening",
    ],
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Gravity multi-turn chat")
    parser.add_argument("--checkpoint", type=str, default="checkpoints_latest/model.pt")
    parser.add_argument("--tokenizer", type=str, default="checkpoints_latest/tokenizer.json")
    parser.add_argument("--vocab", type=str, default=None, help="Alias for --tokenizer")
    parser.add_argument("--max_new_tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.35)
    parser.add_argument("--top_k", type=int, default=30)
    parser.add_argument("--top_p", type=float, default=0.85)
    parser.add_argument("--max_history", type=int, default=8)
    parser.add_argument(
        "--max_prompt_tokens",
        type=int,
        default=None,
        help="Token budget for prompt/history packing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer_path = args.vocab or args.tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[EvalDialogue] Using device: {device}")
    model, tokenizer = load_model(args.checkpoint, tokenizer_path, device)
    print(f"[EvalDialogue] Loaded checkpoint from '{args.checkpoint}'.")
    print()

    for index, scenario in enumerate(SCENARIOS, start=1):
        print(f"=== Scenario {index} ===")
        history: list[tuple[str, str]] = []
        for question in scenario:
            prompt_history = history[-args.max_history:] if args.max_history > 0 else []
            answer, num_tokens, elapsed = answer_question(
                model=model,
                tokenizer=tokenizer,
                question=question,
                history=prompt_history,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                device=device,
                max_prompt_tokens=args.max_prompt_tokens,
            )
            speed = num_tokens / elapsed if elapsed > 0 else 0.0
            print(f"Q: {question}")
            print(f"A: {answer}")
            print(f"[Generated {num_tokens} tokens in {elapsed:.2f}s ({speed:.2f} tokens/s)]")
            history.append((question, answer))
        print()


if __name__ == "__main__":
    main()
