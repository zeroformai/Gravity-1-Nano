"""
tokenizer.py
------------
Tokenizer implementations for MiniGPT.

The default tokenizer uses OpenAI's mature tiktoken BPE vocabulary and adds a
small project-local <eos> token. A simple character tokenizer is kept as a
fallback and for compatibility with older checkpoints.
"""

from __future__ import annotations

from typing import List


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
EOS_TOKEN = "<eos>"

PAD_ID = 0
UNK_ID = 1
EOS_ID = 2

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, EOS_TOKEN]
DEFAULT_TIKTOKEN_ENCODING = "cl100k_base"


class CharTokenizer:
    """Character-level tokenizer with support for atomic special tokens."""

    kind = "char"

    def __init__(self):
        self.token2id: dict[str, int] = {}
        self.id2token: dict[int, str] = {}
        self.vocab_size: int = 0

    def build_vocab(self, text: str) -> None:
        self.token2id = {
            PAD_TOKEN: PAD_ID,
            UNK_TOKEN: UNK_ID,
            EOS_TOKEN: EOS_ID,
        }
        self.id2token = {
            PAD_ID: PAD_TOKEN,
            UNK_ID: UNK_TOKEN,
            EOS_ID: EOS_TOKEN,
        }

        next_id = len(SPECIAL_TOKENS)
        for ch in sorted(set(text)):
            if ch not in self.token2id:
                self.token2id[ch] = next_id
                self.id2token[next_id] = ch
                next_id += 1

        self.vocab_size = next_id
        print(
            f"[Tokenizer] Built char vocabulary with {self.vocab_size} tokens "
            f"({self.vocab_size - len(SPECIAL_TOKENS)} characters + "
            f"{len(SPECIAL_TOKENS)} special tokens)."
        )

    def encode(self, text: str, add_eos: bool = False) -> List[int]:
        ids: list[int] = []
        i = 0
        while i < len(text):
            matched = False
            for token in SPECIAL_TOKENS:
                if text.startswith(token, i):
                    ids.append(self.token2id.get(token, UNK_ID))
                    i += len(token)
                    matched = True
                    break
            if matched:
                continue

            ids.append(self.token2id.get(text[i], UNK_ID))
            i += 1

        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        tokens = []
        for idx in ids:
            token = self.id2token.get(int(idx), "")
            if skip_special and token in SPECIAL_TOKENS:
                continue
            tokens.append(token)
        return "".join(tokens)

    @property
    def eos_id(self) -> int:
        return self.token2id[EOS_TOKEN]

    def save(self, path: str) -> None:
        import json
        import os

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {"kind": self.kind, "token2id": self.token2id},
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"[Tokenizer] Saved to {path}")

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        import json

        tokenizer = cls()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        token2id = data.get("token2id", data.get("char2id"))
        if token2id is None:
            raise ValueError(f"Invalid tokenizer file: {path}")

        tokenizer.token2id = {str(k): int(v) for k, v in token2id.items()}
        if EOS_TOKEN not in tokenizer.token2id:
            tokenizer.token2id[EOS_TOKEN] = len(tokenizer.token2id)

        tokenizer.id2token = {int(v): k for k, v in tokenizer.token2id.items()}
        tokenizer.vocab_size = len(tokenizer.token2id)
        print(
            f"[Tokenizer] Loaded char vocabulary with {tokenizer.vocab_size} "
            f"tokens from {path}"
        )
        return tokenizer

    def __repr__(self) -> str:
        return f"CharTokenizer(vocab_size={self.vocab_size})"


class TiktokenTokenizer:
    """OpenAI tiktoken BPE tokenizer with one local EOS token."""

    kind = "tiktoken"

    def __init__(self, encoding_name: str = DEFAULT_TIKTOKEN_ENCODING):
        try:
            import tiktoken
        except ImportError as exc:
            raise ImportError(
                "TiktokenTokenizer requires the 'tiktoken' package. "
                "Install it with: pip install tiktoken"
            ) from exc

        self.encoding_name = encoding_name
        self.encoding = tiktoken.get_encoding(encoding_name)
        self.base_vocab_size = self.encoding.n_vocab
        self.vocab_size = self.base_vocab_size + 1
        self._eos_id = self.base_vocab_size

    def build_vocab(self, text: str) -> None:
        # tiktoken uses a fixed pretrained BPE vocabulary.
        _ = text
        print(
            f"[Tokenizer] Using tiktoken '{self.encoding_name}' with "
            f"{self.vocab_size} tokens ({self.base_vocab_size} BPE + 1 <eos>)."
        )

    def encode(self, text: str, add_eos: bool = False) -> List[int]:
        ids: list[int] = []
        chunks = text.split(EOS_TOKEN)
        for index, chunk in enumerate(chunks):
            if chunk:
                ids.extend(self.encoding.encode(chunk))
            if index < len(chunks) - 1:
                ids.append(self.eos_id)

        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        parts: list[str] = []
        pending: list[int] = []

        def flush_pending() -> None:
            if pending:
                parts.append(self.encoding.decode(pending))
                pending.clear()

        for raw_id in ids:
            idx = int(raw_id)
            if idx == self.eos_id:
                flush_pending()
                if not skip_special:
                    parts.append(EOS_TOKEN)
            elif 0 <= idx < self.base_vocab_size:
                pending.append(idx)

        flush_pending()
        return "".join(parts)

    @property
    def eos_id(self) -> int:
        return self._eos_id

    def save(self, path: str) -> None:
        import json
        import os

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "kind": self.kind,
                    "encoding_name": self.encoding_name,
                    "eos_token": EOS_TOKEN,
                    "eos_id": self.eos_id,
                    "vocab_size": self.vocab_size,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"[Tokenizer] Saved to {path}")

    @classmethod
    def load(cls, path: str) -> "TiktokenTokenizer":
        import json

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tokenizer = cls(data.get("encoding_name", DEFAULT_TIKTOKEN_ENCODING))

        saved_eos_id = data.get("eos_id")
        if saved_eos_id is not None and int(saved_eos_id) != tokenizer.eos_id:
            raise ValueError(
                f"Tokenizer EOS id mismatch: checkpoint uses {saved_eos_id}, "
                f"current tokenizer uses {tokenizer.eos_id}."
            )

        print(
            f"[Tokenizer] Loaded tiktoken '{tokenizer.encoding_name}' with "
            f"{tokenizer.vocab_size} tokens from {path}"
        )
        return tokenizer

    def __repr__(self) -> str:
        return (
            f"TiktokenTokenizer(encoding_name={self.encoding_name!r}, "
            f"vocab_size={self.vocab_size})"
        )


def create_tokenizer(kind: str = "tiktoken") -> CharTokenizer | TiktokenTokenizer:
    normalized = kind.lower()
    if normalized == "char":
        return CharTokenizer()
    if normalized == "tiktoken":
        return TiktokenTokenizer()
    raise ValueError(f"Unknown tokenizer kind: {kind}")


def load_tokenizer(path: str) -> CharTokenizer | TiktokenTokenizer:
    import json

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    kind = data.get("kind")
    if kind == "tiktoken":
        return TiktokenTokenizer.load(path)
    if kind == "char" or "token2id" in data or "char2id" in data:
        return CharTokenizer.load(path)
    raise ValueError(f"Unknown tokenizer file format: {path}")
