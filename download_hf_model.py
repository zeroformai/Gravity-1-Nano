"""
download_hf_model.py
--------------------
Download a Hugging Face model snapshot into a local directory.

This helper keeps large baseline model downloads explicit and reproducible
instead of relying on opaque user-level cache state.
"""

from __future__ import annotations

import argparse
import os

from huggingface_hub import snapshot_download


DEFAULT_ALLOW_PATTERNS = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "model.safetensors",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a Hugging Face model snapshot")
    parser.add_argument("model_id")
    parser.add_argument("--local_dir", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--max_workers", type=int, default=1)
    parser.add_argument("--allow_patterns", nargs="*", default=DEFAULT_ALLOW_PATTERNS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    path = snapshot_download(
        repo_id=args.model_id,
        revision=args.revision,
        local_dir=args.local_dir,
        allow_patterns=args.allow_patterns,
        max_workers=args.max_workers,
    )
    print(path)


if __name__ == "__main__":
    main()
