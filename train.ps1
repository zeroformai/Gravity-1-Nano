# ============================================================
# 一键训练脚本（PowerShell 版）— Gravity-1 Nano LoRA 训练
# 流程: 构建 SFT 数据 → SFT 训练 → 构建 DPO 数据 → DPO 训练
#
# 用法:
#   powershell -ExecutionPolicy Bypass -File train.ps1              # 全流程
#   powershell -ExecutionPolicy Bypass -File train.ps1 -Stage sft   # 只跑 SFT
#   powershell -ExecutionPolicy Bypass -File train.ps1 -Stage dpo   # 只跑 DPO
#   powershell -ExecutionPolicy Bypass -File train.ps1 -Force       # 强制重建数据
#   透传训练参数: 末尾直接加，如 -ExtraArgs @("--device","rocm","--micro_batch","2")
#
# 设备: --device auto 自动探测 CUDA(英伟达) → ROCm(AMD) → CPU
# ============================================================
param(
  [ValidateSet("all", "sft", "dpo")] [string]$Stage = "all",
  [switch]$Force,
  [string[]]$ExtraArgs = @()
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Say($msg) { Write-Host "[train] $msg" -ForegroundColor Green }
function Warn($msg) { Write-Host "[train] $msg" -ForegroundColor Yellow }
function Die($msg) { Write-Host "[train] $msg" -ForegroundColor Red; exit 1 }

# ---------- 前置检查 ----------
if (-not (Get-Command python -ErrorAction SilentlyContinue)) { Die "缺少 python（推荐 3.11）" }
try { python -c "import torch, transformers, peft" } catch {
  Die "缺少依赖，请先执行: pip install -r requirements.txt"
}

# ---------- 1. SFT ----------
function Run-Sft {
  if ($Force -or -not (Test-Path "sft_data\train.jsonl")) {
    Say "构建 SFT 数据集..."
    python build_sft_dataset.py
  } else {
    Warn "SFT 数据已存在，跳过构建（-Force 可强制重建）"
  }
  Say "开始 SFT 训练..."
  python train_lora.py @ExtraArgs
  Say "SFT 完成 → checkpoints_lora/qwen35-2b-gravity/adapter_best"
}

# ---------- 2. DPO ----------
function Run-Dpo {
  $adapter = "checkpoints_lora\qwen35-2b-gravity\adapter_best"
  if (-not (Test-Path $adapter)) { Die "找不到 SFT adapter ($adapter)，请先跑 SFT 阶段" }
  if ($Force -or -not (Test-Path "dpo_data\train.jsonl")) {
    Say "构建 DPO 数据集..."
    python build_dpo_dataset.py
  } else {
    Warn "DPO 数据已存在，跳过构建（-Force 可强制重建）"
  }
  Say "开始 DPO 训练..."
  python train_dpo_lora.py @ExtraArgs
  Say "DPO 完成 → checkpoints_lora/qwen35-2b-gravity-dpo/adapter_best"
}

switch ($Stage) {
  "sft" { Run-Sft }
  "dpo" { Run-Dpo }
  default { Run-Sft; Run-Dpo }
}

Say "全部完成！对话测试: python chat_lora.py"
