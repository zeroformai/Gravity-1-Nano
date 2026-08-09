#!/usr/bin/env python3
"""下载 Qwen3.5-2B-Base 权重（含分片文件）。

download_hf_model.py 的 allow_patterns 匹配不上分片命名
(model.safetensors-00001-of-00001.safetensors)，本脚本补下这两个文件。

用法: python download_weights.py [model_id] [local_dir]
"""
from __future__ import annotations

import os
import sys

from huggingface_hub import hf_hub_download

MODEL_ID = "Qwen/Qwen3.5-2B-Base"
LOCAL_DIR = "hf_models/qwen35-2b"
WEIGHT_FILES = [
    "model.safetensors-00001-of-00001.safetensors",
    "model.safetensors.index.json",
]


def main() -> None:
    model_id = sys.argv[1] if len(sys.argv) > 1 else MODEL_ID
    local_dir = sys.argv[2] if len(sys.argv) > 2 else LOCAL_DIR
    local_dir = os.path.abspath(local_dir)
    for filename in WEIGHT_FILES:
        target = os.path.join(local_dir, filename)
        if os.path.exists(target):
            print(f"  已存在，跳过: {filename}")
            continue
        print(f"  下载 {filename} ...")
        hf_hub_download(model_id, filename, local_dir=local_dir)
    print("weights done")


if __name__ == "__main__":
    main()
