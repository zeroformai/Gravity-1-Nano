#!/usr/bin/env python3
"""傻瓜式一键训练（零配置，全程回车即可）

双击运行（Windows: 一键训练.bat / Linux: bash 一键训练.sh）
或: python train_wizard.py

流程（全自动）:
    1. 检测 Python 与依赖 → 缺什么自动装什么
    2. 自动探测显卡（NVIDIA / AMD / CPU）
    3. 数据没有就自动构建
    4. SFT 训练 → DPO 训练（全流程）
    5. 训练完自动提示怎么对话测试

所有选项都有默认值，一路回车就是最佳配置。
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GREEN = "\033[1;32m"; YELLOW = "\033[1;33m"; RED = "\033[1;31m"; BLUE = "\033[1;34m"; RESET = "\033[0m"


def c(msg: str, color: str = GREEN) -> str:
    return f"{color}{msg}{RESET}"


def ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    answer = input(f"{c(prompt, BLUE)}{suffix}: ").strip()
    return answer or default


def run(cmd: list[str], check: bool = True) -> int:
    print(c("  $ ") + " ".join(cmd))
    result = subprocess.run(cmd, cwd=ROOT)
    if check and result.returncode != 0:
        print(c(f"命令失败（exit {result.returncode}）: {' '.join(cmd)}", RED))
        sys.exit(1)
    return result.returncode


def banner() -> None:
    print(c("=" * 56))
    print(c("  Gravity-1 Nano 傻瓜式一键训练", GREEN))
    print(c("  全程自动，回车到底就行", YELLOW))
    print(c("=" * 56))


def check_python() -> None:
    print(c("① 检查 Python..."))
    if shutil.which("python") is None:
        print(c("  没找到 python！请先安装 Python 3.11", RED))
        print(c("  下载: https://www.python.org/downloads/release/python-3119/", YELLOW))
        print(c("  安装时记得勾选 Add to PATH", YELLOW))
        sys.exit(1)
    ver = subprocess.run([sys.executable, "-c",
                          "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
                         capture_output=True, text=True).stdout.strip()
    if ver in ("3.10", "3.11", "3.12"):
        print(c(f"  Python {ver} OK"))
    else:
        print(c(f"  Python {ver}（建议 3.10-3.12，PyTorch wheel 更全，但也能跑）", YELLOW))


def install_deps() -> None:
    print(c("② 检查依赖..."))
    missing = []
    for mod in ("torch", "transformers", "peft", "huggingface_hub", "safetensors"):
        ok = subprocess.run([sys.executable, "-c", f"import {mod}"],
                            capture_output=True).returncode == 0
        if not ok:
            missing.append(mod)
    if missing:
        print(c(f"  缺少依赖: {', '.join(missing)}，自动安装中...", YELLOW))
        run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt",
             "huggingface_hub", "--quiet"])
        print(c("  依赖安装完成"))
    else:
        print(c("  依赖齐全"))


def detect_device() -> str:
    print(c("③ 自动探测显卡..."))
    try:
        import torch
    except ImportError:
        print(c("  无法加载 torch，按 CPU 处理", YELLOW))
        return "cpu"
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(c(f"  检测到 NVIDIA GPU: {name} → 用 CUDA (bf16)"))
        return "cuda"
    hip = getattr(torch, "hip", None)
    if hip is not None and hip.is_available():
        print(c("  检测到 AMD GPU → 用 ROCm (bf16)"))
        return "rocm"
    print(c("  未检测到独立显卡 → 用 CPU (fp32)，会比较慢", YELLOW))
    return "cpu"


def check_model() -> bool:
    model = ROOT / "hf_models" / "qwen35-2b" / "model.safetensors-00001-of-00001.safetensors"
    return model.exists()


def download_model(device: str) -> None:
    print(c("④ 下载基座模型 Qwen3.5-2B-Base（约 4.3G）..."))
    run([sys.executable, "download_hf_model.py", "Qwen/Qwen3.5-2B-Base",
         "--local_dir", "hf_models/qwen35-2b"])
    run([sys.executable, "download_weights.py", "Qwen/Qwen3.5-2B-Base",
         "hf_models/qwen35-2b"])


def build_data() -> None:
    print(c("⑤ 检查数据集..."))
    if not (ROOT / "sft_data" / "train.jsonl").exists():
        print(c("  构建 SFT 数据集..."))
        run([sys.executable, "build_sft_dataset.py"])
    else:
        print(c("  SFT 数据已存在，跳过"))
    if not (ROOT / "dpo_data" / "train.jsonl").exists():
        print(c("  构建 DPO 数据集..."))
        run([sys.executable, "build_dpo_dataset.py"])
    else:
        print(c("  DPO 数据已存在，跳过"))


def train_sft(device: str) -> None:
    print(c("⑥ SFT 训练中（耐心等待，日志会实时滚动）..."))
    run([sys.executable, "train_lora.py", "--device", device])


def train_dpo(device: str) -> None:
    print(c("⑦ DPO 训练中..."))
    run([sys.executable, "train_dpo_lora.py", "--device", device])


def done() -> None:
    print(c("=" * 56))
    print(c("🎉 训练全部完成！", GREEN))
    print(c("  对话测试:  python chat_lora.py", YELLOW))
    print(c("  模型位置:  checkpoints_lora/qwen35-2b-gravity-dpo/adapter_best", YELLOW))
    print(c("=" * 56))


def main() -> None:
    banner()
    print()
    print(c("本次训练将依次执行：装依赖 → 探测显卡 → 下载模型(如缺) → 建数据 → SFT → DPO"))
    print(c("按回车开始，随时 Ctrl+C 可以停止", YELLOW))
    input()

    check_python()
    install_deps()
    device = detect_device()
    if not check_model():
        download_model(device)
    else:
        print(c("④ 基座模型已存在，跳过下载"))
    build_data()
    train_sft(device)
    train_dpo(device)
    done()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(c("\n已取消训练。", YELLOW))
        sys.exit(1)
