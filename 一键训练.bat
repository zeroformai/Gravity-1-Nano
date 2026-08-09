@echo off
chcp 65001 >nul
title Gravity-1 Nano 一键训练
cd /d "%~dp0"
echo 启动傻瓜式一键训练...
python train_wizard.py
if errorlevel 1 (
  echo.
  echo 训练中止或出错，请截图上面的信息给我看。
  pause
) else (
  echo.
  pause
)
