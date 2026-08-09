#!/usr/bin/env python3
"""一键训练脚本（跨平台单文件）— Gravity-1 Nano LoRA 训练

流程: 构建 SFT 数据 → SFT 训练 → 构建 DPO 数据 → DPO 训练

用法:
    python train_all.py                  # 全流程（数据已存在则跳过构建）
    python train_all.py --stage sft      # 只跑 SFT
    python train_all.py --stage dpo      # 只跑 DPO（需先有 SFT adapter）
    python train_all.py --force          # 强制重建数据
    python train_all.py -- --device rocm --micro_batch 2   # 透传参数

设备: --device auto 自动探测 CUDA(英伟达) → ROCm(AMD) → CPU
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

GREEN = "\033[1;32m"
YELLOW = "\033[1;33m"
RED = "\033[1;31m"
RESET = "\033[0m"


def say(msg: str) -> None:
    print(f"{GREEN}[train]{RESET} {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"{YELLOW}[train]{RESET} {msg}", flush=True)


def die(msg: str) -> None:
    print(f"{RED}[train]{RESET} {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


def run(cmd: list[str], cwd: Path = ROOT) -> None:
    say("$ " + " ".join(cmd))
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        die(f"命令失败（exit {result.returncode}）: {' '.join(cmd)}")


def check_deps() -> None:
    if shutil.which("python") is None:
        die("缺少 python（推荐 3.11：PyTorch/ROCm wheel 支持最全）")
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        import peft  # noqa: F401
    except ImportError:
        die("缺少依赖，请先执行: pip install -r requirements.txt")


def run_sft(force: bool, extra: list[str]) -> None:
    if force or not (ROOT / "sft_data" / "train.jsonl").exists():
        say("构建 SFT 数据集...")
        run([sys.executable, "build_sft_dataset.py"])
    else:
        warn("SFT 数据已存在，跳过构建（--force 可强制重建）")
    say("开始 SFT 训练...")
    run([sys.executable, "train_lora.py", *extra])
    say("SFT 完成 → checkpoints_lora/qwen35-2b-gravity/adapter_best")


def run_dpo(force: bool, extra: list[str]) -> None:
    adapter = ROOT / "checkpoints_lora" / "qwen35-2b-gravity" / "adapter_best"
    if not adapter.is_dir():
        die(f"找不到 SFT adapter（{adapter}），请先跑 SFT 阶段")
    if force or not (ROOT / "dpo_data" / "train.jsonl").exists():
        say("构建 DPO 数据集...")
        run([sys.executable, "build_dpo_dataset.py"])
    else:
        warn("DPO 数据已存在，跳过构建（--force 可强制重建）")
    say("开始 DPO 训练...")
    run([sys.executable, "train_dpo_lora.py", *extra])
    say("DPO 完成 → checkpoints_lora/qwen35-2b-gravity-dpo/adapter_best")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gravity-1 Nano 一键训练")
    parser.add_argument("--stage", choices=["sft", "dpo", "all"], default="all",
                        help="执行阶段：sft / dpo / all（默认 all）")
    parser.add_argument("--force", action="store_true",
                        help="强制重建数据集")
    parser.add_argument("--", dest="extra", nargs=argparse.REMAINDER,
                        help="透传给 train_lora.py / train_dpo_lora.py 的参数")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    check_deps()
    extra = args.extra or []
    if args.stage in ("sft", "all"):
        run_sft(args.force, extra)
    if args.stage in ("dpo", "all"):
        run_dpo(args.force, extra)
    say("全部完成！对话测试: python chat_lora.py")


if __name__ == "__main__":
    main()
