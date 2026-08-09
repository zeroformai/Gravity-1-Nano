#!/usr/bin/env bash
# 一键训练入口（Linux/macOS 双击或命令行运行）
cd "$(dirname "$0")"
echo "启动傻瓜式一键训练..."
python3 train_wizard.py || python train_wizard.py
