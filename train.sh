#!/usr/bin/env bash
# ============================================================
# 一键训练脚本（bash 版）— Gravity-1 Nano LoRA 训练
# 流程: 构建 SFT 数据 → SFT 训练 → 构建 DPO 数据 → DPO 训练
#
# 用法:
#   bash train.sh                # 全流程（数据已存在则跳过构建）
#   bash train.sh --stage sft    # 只跑 SFT
#   bash train.sh --stage dpo    # 只跑 DPO（需先有 SFT adapter）
#   bash train.sh --force        # 强制重建数据
#   bash train.sh -- --device rocm --micro_batch 2   # 透传参数给训练脚本
#
# 设备: --device auto 自动探测 CUDA(英伟达) → ROCm(AMD) → CPU
# ============================================================
set -euo pipefail
cd "$(dirname "$0")"

STAGE="all"
FORCE=0
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage) STAGE="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    --) shift; EXTRA=("$@"); break ;;
    -h|--help) grep "^#" "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "未知参数: $1（--stage sft|dpo|all, --force, -- 透传训练参数）"; exit 1 ;;
  esac
done

say()  { printf "\033[1;32m[train]\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m[train]\033[0m %s\n" "$*"; }
die()  { printf "\033[1;31m[train]\033[0m %s\n" "$*" >&2; exit 1; }

# ---------- 前置检查 ----------
command -v python >/dev/null 2>&1 || die "缺少 python（推荐 3.11）"
python -c "import torch, transformers, peft" 2>/dev/null || die "缺少依赖，请先: pip install -r requirements.txt"

# ---------- 1. SFT ----------
run_sft() {
  if [[ "$FORCE" == 1 ]] || [[ ! -f "sft_data/train.jsonl" ]]; then
    say "构建 SFT 数据集..."
    python build_sft_dataset.py
  else
    say "SFT 数据已存在，跳过构建（--force 可强制重建）"
  fi
  say "开始 SFT 训练..."
  python train_lora.py "${EXTRA[@]}"
  say "SFT 完成 → checkpoints_lora/qwen35-2b-gravity/adapter_best"
}

# ---------- 2. DPO ----------
run_dpo() {
  local adapter="checkpoints_lora/qwen35-2b-gravity/adapter_best"
  if [[ ! -d "$adapter" ]]; then
    die "找不到 SFT adapter ($adapter)，请先跑 SFT 阶段"
  fi
  if [[ "$FORCE" == 1 ]] || [[ ! -f "dpo_data/train.jsonl" ]]; then
    say "构建 DPO 数据集..."
    python build_dpo_dataset.py
  else
    say "DPO 数据已存在，跳过构建（--force 可强制重建）"
  fi
  say "开始 DPO 训练..."
  python train_dpo_lora.py "${EXTRA[@]}"
  say "DPO 完成 → checkpoints_lora/qwen35-2b-gravity-dpo/adapter_best"
}

case "$STAGE" in
  sft) run_sft ;;
  dpo) run_dpo ;;
  all) run_sft; run_dpo ;;
  *) die "未知 stage: $STAGE（sft / dpo / all）" ;;
esac

say "全部完成！对话测试: python chat_lora.py"
