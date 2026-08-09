"""
eval_lora_ood.py
----------------
A/B smoke test for the LoRA fine-tune: run out-of-dataset prompts through the
base model and the adapter model in one process and print paired outputs.

In-dataset prompts are included at the top as a persona check; everything
under OOD_PROMPTS is intentionally absent from the training corpus.

Usage:
    python eval_lora_ood.py
    python eval_lora_ood.py --adapter_only
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from build_sft_dataset import DEFAULT_SYSTEM_PROMPT

IN_DATASET_PROMPTS = [
    "在吗",
    "我睡不着",
    "who are you?",
]

OOD_PROMPTS = [
    "我最近搬到一个新城市，一个人住，晚上回家觉得有点孤单",
    "我养的猫今天不怎么吃东西，我有点担心它",
    "周末想开始学做饭，完全零基础，从哪道菜入手比较好？",
    "开会的时候被同事当众否定了，到现在心里还堵着",
    "我爸妈总催我找对象，每次打电话都聊不下去",
    "my roommate keeps borrowing my stuff without asking and it's getting on my nerves",
    "高铁上邻座一直外放短视频，我要不要说点什么？",
    "今天终于把拖了两周的报告交了，夸夸我",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OOD A/B eval for the LoRA model")
    parser.add_argument("--model_dir", type=str,
                        default="hf_models/qwen35-2b")
    parser.add_argument("--adapter", type=str,
                        default="checkpoints_lora/qwen35-2b-gravity/adapter_best")
    parser.add_argument("--adapter_only", action="store_true",
                        help="Skip base-model outputs.")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


@torch.no_grad()
def generate(tokenizer, model, device, prompt: str, args) -> str:
    messages = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    reply_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(reply_ids, skip_special_tokens=True).strip()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    base = os.path.dirname(os.path.abspath(__file__))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else "auto"

    tokenizer = AutoTokenizer.from_pretrained(os.path.join(base, args.model_dir))
    model = AutoModelForCausalLM.from_pretrained(
        os.path.join(base, args.model_dir),
        dtype=dtype,
        attn_implementation="sdpa",
    ).to(device)
    model = PeftModel.from_pretrained(model, os.path.join(base, args.adapter))
    model.eval()
    print(f"[Eval] Adapter: {args.adapter}\n")

    sections = [
        ("== 训练集内（人设检查） ==", IN_DATASET_PROMPTS),
        ("== 训练集外（泛化检查） ==", OOD_PROMPTS),
    ]
    for header, prompts in sections:
        print(header)
        for prompt in prompts:
            torch.manual_seed(args.seed)
            print(f"\nQ: {prompt}")
            if not args.adapter_only:
                with model.disable_adapter():
                    base_reply = generate(tokenizer, model, device, prompt, args)
                print(f"  [base ] {base_reply}")
            torch.manual_seed(args.seed)
            tuned_reply = generate(tokenizer, model, device, prompt, args)
            print(f"  [tuned] {tuned_reply}")
        print()


if __name__ == "__main__":
    main()
