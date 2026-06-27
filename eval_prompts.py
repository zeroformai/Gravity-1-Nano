"""
eval_prompts.py
---------------
Run a fixed Q/A smoke test against a Gravity checkpoint.
"""

import argparse

import torch

from generate import answer_question, load_model


DEFAULT_PROMPTS = [
    "hi",
    "你好",
    "who are you?",
    "can you keep me company?",
    "i don't know what to say",
    "comfort me",
    "你是谁？",
    "在吗",
    "陪我说说话",
    "可以陪我吗？",
    "我想聊聊天",
    "我不知道说什么",
    "我睡不着",
    "哄哄我",
    "帮我做明天计划",
    "我有点焦虑",
    "我有些难过",
    "帮我变通一个",
    "我不想处理工作但是必须写报告，帮我变通一个",
    "计划被打乱了，我还得准备面试，怎么办？",
    "我想换个说法安慰朋友",
    "解释一下注意力机制，别太学术",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate fixed Gravity Q/A prompts")
    parser.add_argument("--checkpoint", type=str, default="checkpoints_latest/model.pt")
    parser.add_argument("--tokenizer", type=str, default="checkpoints_latest/tokenizer.json")
    parser.add_argument("--vocab", type=str, default=None, help="Alias for --tokenizer")
    parser.add_argument("--max_new_tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.35)
    parser.add_argument("--top_k", type=int, default=30)
    parser.add_argument("--top_p", type=float, default=0.85)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer_path = args.vocab or args.tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[Eval] Using device: {device}")
    model, tokenizer = load_model(args.checkpoint, tokenizer_path, device)
    print(f"[Eval] Loaded checkpoint from '{args.checkpoint}'.")
    print()

    for prompt in DEFAULT_PROMPTS:
        answer, num_tokens, elapsed = answer_question(
            model=model,
            tokenizer=tokenizer,
            question=prompt,
            history=[],
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=device,
        )
        speed = num_tokens / elapsed if elapsed > 0 else 0.0
        print(f"Q: {prompt}")
        print(f"A: {answer}")
        print(f"[Generated {num_tokens} tokens in {elapsed:.2f}s ({speed:.2f} tokens/s)]")
        print()


if __name__ == "__main__":
    main()
