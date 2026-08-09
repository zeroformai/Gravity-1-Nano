"""
chat_lora.py
------------
Chat with the LoRA fine-tuned local Qwen model (or the raw base for
comparison).

Usage:
    python chat_lora.py                          # interactive, best adapter
    python chat_lora.py --prompt "我今天有点累"   # one-shot
    python chat_lora.py --no_adapter             # raw base model
    python chat_lora.py --adapter checkpoints_lora/qwen35-2b-gravity/adapter_last

Interactive commands: /reset clears history, /exit quits.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

from build_sft_dataset import DEFAULT_SYSTEM_PROMPT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with the fine-tuned model")
    parser.add_argument("--model_dir", type=str,
                        default="hf_models/qwen35-2b")
    parser.add_argument("--adapter", type=str,
                        default="checkpoints_lora/qwen35-2b-gravity/adapter_best")
    parser.add_argument("--no_adapter", action="store_true",
                        help="Chat with the raw base model instead.")
    parser.add_argument("--system", type=str, default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--prompt", type=str, default=None,
                        help="One-shot prompt instead of interactive mode.")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    return parser.parse_args()


def load_model(args: argparse.Namespace):
    base = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base, args.model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else "auto"

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        dtype=dtype,
        attn_implementation="sdpa",
    ).to(device)

    if not args.no_adapter:
        adapter_dir = os.path.join(base, args.adapter)
        if not os.path.isdir(adapter_dir):
            fallback = adapter_dir.replace("adapter_best", "adapter_last")
            if os.path.isdir(fallback):
                adapter_dir = fallback
            else:
                raise SystemExit(
                    f"Adapter not found: {adapter_dir}. "
                    "Run train_lora.py first or pass --no_adapter."
                )
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_dir)
        print(f"[Chat] Loaded adapter: {adapter_dir}")
    else:
        print("[Chat] Using raw base model (no adapter).")

    model.eval()
    return tokenizer, model, device


@torch.no_grad()
def generate_reply(tokenizer, model, device, messages, args, stream: bool) -> str:
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True) if stream else None
    output_ids = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.temperature > 0,
        temperature=max(args.temperature, 1e-5),
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        streamer=streamer,
    )
    reply_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(reply_ids, skip_special_tokens=True).strip()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    tokenizer, model, device = load_model(args)

    if args.prompt is not None:
        messages = [
            {"role": "system", "content": args.system},
            {"role": "user", "content": args.prompt},
        ]
        reply = generate_reply(tokenizer, model, device, messages, args, stream=False)
        print(reply)
        return

    print("[Chat] Interactive mode. /reset clears history, /exit quits.\n")
    messages = [{"role": "system", "content": args.system}]
    while True:
        try:
            user_text = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_text:
            continue
        if user_text == "/exit":
            break
        if user_text == "/reset":
            messages = [{"role": "system", "content": args.system}]
            print("[Chat] History cleared.\n")
            continue

        messages.append({"role": "user", "content": user_text})
        print("Gravity: ", end="", flush=True)
        reply = generate_reply(tokenizer, model, device, messages, args, stream=True)
        messages.append({"role": "assistant", "content": reply})
        print()


if __name__ == "__main__":
    main()
