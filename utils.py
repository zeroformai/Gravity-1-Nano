"""
utils.py
--------
Data preparation utilities for the Gravity AI project.

Provides:
  - load_text(path)           : read raw text from disk
  - encode_dataset(text, tok) : tokenise entire corpus into a flat tensor
  - get_batch(data, ...)      : randomly sample (x, y) training pairs
  - estimate_loss(...)        : quick loss estimation over random batches
"""

from __future__ import annotations
import glob
import inspect
import re
import torch
import torch.nn.functional as F
import random
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from tokenizer import EOS_TOKEN


def get_autocast(device_type: str, enabled: bool = False):
    """Return a torch autocast context when mixed precision is enabled."""
    if not enabled:
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device_type, enabled=True)
    if device_type == "cuda" and hasattr(torch, "cuda") and hasattr(torch.cuda, "amp"):
        return torch.cuda.amp.autocast(enabled=True)
    return nullcontext()


def get_grad_scaler(device_type: str, enabled: bool = False):
    """Create a GradScaler only for CUDA AMP; CPU uses a disabled scaler."""
    use_cuda_amp = enabled and device_type == "cuda"
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=use_cuda_amp)
    if hasattr(torch, "cuda") and hasattr(torch.cuda, "amp"):
        return torch.cuda.amp.GradScaler(enabled=use_cuda_amp)
    return _DisabledGradScaler()


def configure_cuda_training(device: torch.device) -> None:
    """Enable CUDA math fast paths that preserve existing training semantics."""
    if device.type != "cuda":
        return
    torch.set_float32_matmul_precision("high")
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = True


def select_device(requested: str) -> torch.device:
    """Resolve a CLI device option to a torch device.

    支持 CUDA (NVIDIA)、ROCm (AMD)、CPU。auto 按可用性探测：
    CUDA 优先，其次 ROCm，最后 CPU。
    """
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is not available.")
        return torch.device("cuda")
    if requested == "rocm":
        if not _rocm_available():
            raise RuntimeError("--device rocm requested but ROCm is not available.")
        return torch.device("rocm")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if _rocm_available():
        return torch.device("rocm")
    return torch.device("cpu")


def _rocm_available() -> bool:
    """ROCm (AMD GPU) 是否可用：torch.hip.is_available() 或 hip 后端的 cuda 兼容层。"""
    hip = getattr(torch, "hip", None)
    if hip is not None:
        try:
            return bool(hip.is_available())
        except Exception:
            return False
    return False


def create_adamw(
    param_groups: list[dict],
    lr: float,
    betas: tuple[float, float],
    eps: float = 1e-8,
    device_type: str = "cpu",
) -> tuple[torch.optim.Optimizer, bool]:
    """Create AdamW and use the fused CUDA implementation when available."""
    kwargs = {
        "lr": lr,
        "betas": betas,
        "eps": eps,
    }
    use_fused = False
    try:
        sig = inspect.signature(torch.optim.AdamW)
        use_fused = "fused" in sig.parameters and device_type == "cuda"
    except (TypeError, ValueError):
        use_fused = False
    if use_fused:
        kwargs["fused"] = True
    return torch.optim.AdamW(param_groups, **kwargs), use_fused


def maybe_compile_model(model: torch.nn.Module, enabled: bool, label: str) -> torch.nn.Module:
    """Compile a model when requested and available."""
    if not enabled:
        return model
    if hasattr(torch, "compile"):
        print(f"[{label}] torch.compile: enabled")
        return torch.compile(model)
    print(f"[{label}] torch.compile requested but unavailable; continuing uncompiled.")
    return model


def unwrap_compiled_model(model: torch.nn.Module) -> torch.nn.Module:
    """Return the original module behind torch.compile wrappers."""
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def model_state_dict(model: torch.nn.Module) -> dict:
    """Return a checkpoint-compatible state dict for compiled or eager models."""
    return unwrap_compiled_model(model).state_dict()


def filter_top_k_top_p(
    logits: torch.Tensor,
    top_k: int | None,
    top_p: float | None,
) -> torch.Tensor:
    """Apply top-k and nucleus filtering to logits."""
    if top_k is not None:
        k = min(top_k, logits.size(-1))
        values, _ = torch.topk(logits, k)
        logits = logits.masked_fill(logits < values[:, -1].unsqueeze(-1), float("-inf"))

    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = False
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, float("-inf"))
    return logits


def contains_any(text: str, needles: tuple[str, ...]) -> bool:
    """Case-insensitive substring check for a group of keywords."""
    lowered = text.lower()
    return any(needle.lower() in lowered for needle in needles)


def lexical_features(
    text: str,
    en_stop_words: set[str] | None = None,
    zh_stop_chars: set[str] | None = None,
    include_cjk_chars: bool = False,
) -> set[str]:
    """Return word and CJK n-gram features for lightweight lexical matching."""
    normalized = re.sub(r"\s+", " ", text.lower()).strip()
    stop_words = en_stop_words or set()
    stop_chars = zh_stop_chars or set()
    words = {
        word
        for word in re.findall(r"[a-z0-9']+", normalized)
        if len(word) > 1 and word not in stop_words
    }
    for segment in re.findall(r"[\u4e00-\u9fff]+", normalized):
        if include_cjk_chars:
            words.update(ch for ch in segment if ch not in stop_chars)
        for size in (2, 3):
            words.update(segment[i:i + size] for i in range(max(0, len(segment) - size + 1)))
    return words


def masked_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Cross entropy over valid target positions only."""
    vocab_size = logits.size(-1)
    flat_targets = targets.reshape(-1)
    valid = flat_targets != ignore_index
    if not bool(valid.any()):
        return logits.sum() * 0.0
    flat_logits = logits.reshape(-1, vocab_size)
    return F.cross_entropy(
        flat_logits[valid],
        flat_targets[valid],
        label_smoothing=label_smoothing,
    )


def masked_sequence_logps(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    """Average log probability per sequence over masked token positions only."""
    mask = loss_mask.to(dtype=torch.bool)
    if not bool(mask.any()):
        return logits.new_zeros(logits.size(0))

    batch_size = logits.size(0)
    row_ids = torch.arange(batch_size, device=logits.device).unsqueeze(1).expand_as(mask)
    valid_rows = row_ids[mask]
    valid_logits = logits[mask]
    valid_targets = targets[mask]
    valid_logps = F.log_softmax(valid_logits, dim=-1).gather(
        -1,
        valid_targets.unsqueeze(-1),
    ).squeeze(-1)

    seq_logps = logits.new_zeros(batch_size)
    seq_logps.scatter_add_(0, valid_rows, valid_logps)
    lengths = mask.sum(dim=-1).clamp_min(1)
    return seq_logps / lengths


def model_sequence_logps(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    loss_mask: torch.Tensor,
    use_amp: bool = False,
) -> torch.Tensor:
    """Forward a model and compute masked average sequence log-probabilities."""
    with get_autocast(x.device.type, enabled=use_amp):
        logits, _ = model(x)
        return masked_sequence_logps(logits, y, loss_mask)


def pad_token_sequences_with_loss_mask(
    sequences: list[list[int]],
    loss_masks: list[list[int]],
    pad_id: int,
    device: torch.device,
    max_sequence_len: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad token-id sequences and matching loss masks for next-token scoring."""
    max_len = max_sequence_len or max(len(ids) for ids in sequences)
    input_len = max_len - 1
    batch = len(sequences)
    x = torch.full((batch, input_len), pad_id, dtype=torch.long)
    y = torch.full((batch, input_len), pad_id, dtype=torch.long)
    mask = torch.zeros((batch, input_len), dtype=torch.bool)

    for row, ids in enumerate(sequences):
        source = ids[:-1]
        target = ids[1:]
        length = len(source)
        x[row, :length] = torch.tensor(source, dtype=torch.long)
        y[row, :length] = torch.tensor(target, dtype=torch.long)
        mask[row, :len(loss_masks[row])] = torch.tensor(loss_masks[row], dtype=torch.bool)
    return x.to(device), y.to(device), mask.to(device)


class _DisabledGradScaler:
    """Small compatibility shim for environments without AMP GradScaler."""

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        return None

    def step(self, optimizer):
        optimizer.step()

    def update(self):
        return None


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_text(path: str) -> str:
    """Read a text file and return its contents as a single string."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"[Utils] Loaded {len(text):,} characters from '{path}'.")
    return text


def load_text_files(spec: str) -> list[tuple[str, str]]:
    """Load one or more UTF-8 text files from comma/semicolon-separated paths or globs."""
    paths: list[str] = []
    for part in re.split(r"[,;]", spec or ""):
        pattern = part.strip()
        if not pattern:
            continue
        matches = sorted(glob.glob(pattern))
        paths.extend(matches or [pattern])

    if not paths:
        raise ValueError("No input files were specified.")

    loaded: list[tuple[str, str]] = []
    for path in paths:
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(path)
        loaded.append((str(file_path), load_text(str(file_path))))
    return loaded


# ---------------------------------------------------------------------------
# Dataset encoding
# ---------------------------------------------------------------------------

def encode_dataset(text: str, tokenizer) -> torch.Tensor:
    """
    Encode the full corpus into a 1-D integer tensor of token ids.

    Args:
        text      : raw text string
        tokenizer : any object with an encode(str) -> List[int] method
    Returns:
        data : 1-D LongTensor of shape (N,)
    """
    ids = tokenizer.encode(text)
    data = torch.tensor(ids, dtype=torch.long)
    print(f"[Utils] Encoded dataset: {len(ids):,} tokens.")
    return data


@dataclass(frozen=True)
class QAExample:
    """A single supervised Q/A training example."""

    question: str
    answer: str

    @property
    def prompt(self) -> str:
        if self.question.lstrip().startswith("Q:"):
            return f"{self.question}\nA:"
        return f"Q: {self.question}\nA:"

    @property
    def text(self) -> str:
        return f"{self.prompt} {self.answer}{EOS_TOKEN}"


@dataclass(frozen=True)
class QABatchTensors:
    """Pre-padded supervised examples for fast random batch sampling."""

    x: torch.Tensor
    y: torch.Tensor
    mask: torch.Tensor

    def __len__(self) -> int:
        return self.x.size(0)


def parse_qa_examples(text: str) -> list[QAExample]:
    """
    Parse a corpus of Q/A records.

    Records must use:
        Q: question
        A: answer<eos>

    The answer may span multiple lines, but every record must end in <eos>.
    """
    if not text.strip():
        raise ValueError("Dataset is empty.")

    pattern = re.compile(r"^Q: (.*?)\nA: (.*?)(?:<eos>)(?:\r?\n|$)", re.MULTILINE | re.DOTALL)
    examples = [
        QAExample(question=q.strip(), answer=a.strip())
        for q, a in pattern.findall(text)
    ]
    if not examples:
        raise ValueError("No Q/A examples found. Expected records ending with <eos>.")

    q_count = len(re.findall(r"^Q: ", text, flags=re.MULTILINE))
    eos_count = text.count(EOS_TOKEN)
    if len(examples) != q_count or len(examples) != eos_count:
        raise ValueError(
            "Malformed Q/A dataset: every 'Q:' record must have an 'A:' answer "
            f"ending with {EOS_TOKEN}. Parsed {len(examples)} examples, found "
            f"{q_count} questions and {eos_count} EOS markers."
        )

    print(f"[Utils] Parsed {len(examples):,} supervised Q/A examples.")
    return examples


def parse_qa_conversations(text: str) -> list[list[QAExample]]:
    """Parse blank-line separated Q/A conversations."""
    conversations: list[list[QAExample]] = []
    for chunk in re.split(r"(?:\r?\n){2,}", text.strip()):
        if not chunk.strip():
            continue
        conversations.append(parse_qa_examples(chunk + "\n"))
    return conversations


def build_history_examples(
    conversations: list[list[QAExample]],
    max_history: int = 2,
) -> list[QAExample]:
    """Convert multi-turn conversations into supervised examples with prior turns in the prompt."""
    examples: list[QAExample] = []
    for conversation in conversations:
        for index, turn in enumerate(conversation):
            start = max(0, index - max_history)
            history = conversation[start:index]
            if not history:
                examples.append(turn)
                continue
            prompt_parts = [
                f"Q: {previous.question}\nA: {previous.answer}{EOS_TOKEN}"
                for previous in history
            ]
            prompt_parts.append(f"Q: {turn.question}")
            examples.append(QAExample(question="\n".join(prompt_parts), answer=turn.answer))
    print(f"[Utils] Built {len(examples):,} history-aware dialogue examples.")
    return examples


def _qa_language_key(example: QAExample) -> str:
    text = f"{example.question} {example.answer}"
    has_cjk = bool(re.search(r"[\u4e00-\u9fff]", text))
    has_latin = bool(re.search(r"[A-Za-z]", text))
    if has_cjk and has_latin:
        return "mixed"
    if has_cjk:
        return "zh"
    if has_latin:
        return "en"
    return "other"


def _qa_stratify_key(example: QAExample) -> tuple[str, str]:
    answer = re.sub(r"\s+", " ", example.answer.strip().lower())
    return _qa_language_key(example), answer


def _fractional_split_counts(
    total: int,
    val_fraction: float,
    test_fraction: float,
) -> tuple[int, int, int]:
    test_count = int(round(total * test_fraction))
    val_count = int(round(total * val_fraction))
    if total > 1 and test_fraction > 0 and test_count == 0:
        test_count = 1
    if total > 2 and val_fraction > 0 and val_count == 0:
        val_count = 1

    while total - val_count - test_count < 1 and (val_count > 0 or test_count > 0):
        if test_count >= val_count and test_count > 0:
            test_count -= 1
        elif val_count > 0:
            val_count -= 1
    train_count = total - val_count - test_count
    return train_count, val_count, test_count


def _split_shuffled_items(
    items: list[QAExample],
    val_fraction: float,
    test_fraction: float,
    rng: random.Random,
) -> tuple[list[QAExample], list[QAExample], list[QAExample]]:
    shuffled = list(items)
    rng.shuffle(shuffled)
    train_count, val_count, test_count = _fractional_split_counts(
        len(shuffled),
        val_fraction,
        test_fraction,
    )
    test_examples = shuffled[:test_count]
    val_examples = shuffled[test_count:test_count + val_count]
    train_examples = shuffled[test_count + val_count:test_count + val_count + train_count]
    return train_examples, val_examples, test_examples


def split_qa_examples(
    examples: list[QAExample],
    val_fraction: float = 0.05,
    test_fraction: float = 0.05,
    seed: int = 42,
    stratify: bool = True,
    min_stratum_size: int = 10,
) -> tuple[list[QAExample], list[QAExample], list[QAExample]]:
    """Randomly split raw Q/A examples, stratifying repeated answer templates."""
    if not examples:
        raise ValueError("No Q/A examples available for splitting.")
    if val_fraction < 0 or test_fraction < 0:
        raise ValueError("Split fractions must be non-negative.")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("--val_frac + --test_frac must be less than 1.0.")

    rng = random.Random(seed)
    if not stratify:
        train_examples, val_examples, test_examples = _split_shuffled_items(
            examples,
            val_fraction,
            test_fraction,
            rng,
        )
    else:
        groups: dict[tuple[str, str], list[QAExample]] = {}
        for example in examples:
            groups.setdefault(_qa_stratify_key(example), []).append(example)

        train_examples = []
        val_examples = []
        test_examples = []
        residual_examples = []
        group_items = list(groups.values())
        rng.shuffle(group_items)
        for group in group_items:
            if len(group) < min_stratum_size:
                residual_examples.extend(group)
                continue
            group_train, group_val, group_test = _split_shuffled_items(
                group,
                val_fraction,
                test_fraction,
                rng,
            )
            train_examples.extend(group_train)
            val_examples.extend(group_val)
            test_examples.extend(group_test)

        if residual_examples:
            residual_train, residual_val, residual_test = _split_shuffled_items(
                residual_examples,
                val_fraction,
                test_fraction,
                rng,
            )
            train_examples.extend(residual_train)
            val_examples.extend(residual_val)
            test_examples.extend(residual_test)

        rng.shuffle(train_examples)
        rng.shuffle(val_examples)
        rng.shuffle(test_examples)

    print(
        f"[Utils] QA split: train {len(train_examples):,} | "
        f"val {len(val_examples):,} | test {len(test_examples):,} "
        f"(seed={seed}, stratify={stratify})"
    )
    return train_examples, val_examples, test_examples


def encode_qa_examples(
    examples: list[QAExample],
    tokenizer,
    seq_len: int,
) -> list[tuple[list[int], list[int]]]:
    """
    Encode examples for supervised answer-only loss.

    Inputs are full Q/A records. Labels are masked with -100 until the first
    answer token, so the loss trains answer generation rather than prompt copy.
    """
    encoded = []
    skipped = 0
    for example in examples:
        prompt_ids = tokenizer.encode(example.prompt)
        text_ids = tokenizer.encode(example.text)
        if len(text_ids) < 2:
            skipped += 1
            continue
        if len(text_ids) > seq_len:
            skipped += 1
            continue

        labels = text_ids[1:] + [tokenizer.eos_id]
        prompt_token_count = max(0, len(prompt_ids) - 1)
        labels[:prompt_token_count] = [-100] * prompt_token_count
        encoded.append((text_ids, labels))

    if not encoded:
        raise ValueError(
            f"No Q/A examples fit seq_len={seq_len}. Increase --seq_len or shorten data."
        )
    if skipped:
        print(f"[Utils] Skipped {skipped:,} Q/A examples longer than seq_len={seq_len}.")
    print(f"[Utils] Encoded supervised Q/A examples: {len(encoded):,}.")
    return encoded


def split_examples(
    examples: list[tuple[list[int], list[int]]],
    val_fraction: float = 0.1,
) -> tuple[list[tuple[list[int], list[int]]], list[tuple[list[int], list[int]]]]:
    """Split encoded examples into train and validation sets."""
    split = int(len(examples) * (1.0 - val_fraction))
    split = min(max(split, 1), len(examples))
    train_examples = examples[:split]
    val_examples = examples[split:] or examples[-1:]
    print(
        f"[Utils] Train examples: {len(train_examples):,} | "
        f"Val examples: {len(val_examples):,}"
    )
    return train_examples, val_examples


def prepare_qa_batch_tensors(
    examples: list[tuple[list[int], list[int]]],
    seq_len: int,
    device: torch.device,
) -> QABatchTensors:
    """Pre-pad encoded Q/A examples for fast indexed batching."""
    prepared = [
        (ids, labels)
        for ids, labels in examples
        if ids and labels and len(ids) <= seq_len and len(labels) <= seq_len
    ]
    if not prepared:
        raise ValueError("No encoded Q/A examples are available after preparation.")

    x = torch.full((len(prepared), seq_len), 0, dtype=torch.long, device=device)
    y = torch.full((len(prepared), seq_len), -100, dtype=torch.long, device=device)
    mask = torch.zeros((len(prepared), seq_len), dtype=torch.bool, device=device)
    for row, (ids, labels) in enumerate(prepared):
        length = min(len(ids), seq_len)
        x[row, :length] = torch.as_tensor(ids[:length], dtype=torch.long, device=device)
        y[row, :length] = torch.as_tensor(labels[:length], dtype=torch.long, device=device)
        mask[row, :length] = True
    return QABatchTensors(x=x, y=y, mask=mask)


# ---------------------------------------------------------------------------
# Train / validation split
# ---------------------------------------------------------------------------

def train_val_split(
    data: torch.Tensor,
    val_fraction: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Split a 1-D token tensor into train and validation sets.

    Args:
        data         : 1-D token tensor
        val_fraction : fraction of tokens reserved for validation
    Returns:
        (train_data, val_data)
    """
    split = int(len(data) * (1.0 - val_fraction))
    train_data = data[:split]
    val_data = data[split:]
    print(f"[Utils] Train tokens: {len(train_data):,} | Val tokens: {len(val_data):,}")
    return train_data, val_data


# ---------------------------------------------------------------------------
# Batch sampling
# ---------------------------------------------------------------------------

def get_batch(
    data: torch.Tensor,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Randomly sample a batch of (input, target) sequence pairs.

    For each sample the input is a slice of length `seq_len` and the
    target is the same slice shifted by one position (next-token prediction).

    Args:
        data       : 1-D token tensor (the whole split, e.g. train_data)
        batch_size : number of sequences per batch
        seq_len    : length of each sequence (context window)
        device     : torch device to move tensors to

    Returns:
        x : (batch_size, seq_len)  — input token ids
        y : (batch_size, seq_len)  — target token ids (shifted by 1)
    """
    # Maximum valid starting index so we always have seq_len + 1 tokens
    max_start = len(data) - seq_len - 1
    if max_start <= 0:
        raise ValueError(
            f"Dataset too small ({len(data)} tokens) for seq_len={seq_len}. "
            "Use a larger dataset or reduce seq_len."
        )

    if data.device != device:
        data = data.to(device)

    starts = torch.randint(0, max_start, (batch_size,), device=device)
    offsets = torch.arange(seq_len, device=device)
    positions = starts[:, None] + offsets[None, :]

    x = data[positions]
    y = data[positions + 1]
    return x, y


def get_qa_batch(
    examples: list[tuple[list[int], list[int]]] | QABatchTensors,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    pad_id: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample and right-pad supervised Q/A examples with an attention mask."""
    if not examples:
        raise ValueError("No Q/A examples available for batching.")

    if isinstance(examples, QABatchTensors):
        indices = torch.randint(0, len(examples), (batch_size,), device=examples.x.device)
        return (
            examples.x.index_select(0, indices),
            examples.y.index_select(0, indices),
            examples.mask.index_select(0, indices),
        )

    sampled = random.choices(examples, k=batch_size)
    x = torch.full((batch_size, seq_len), pad_id, dtype=torch.long)
    y = torch.full((batch_size, seq_len), -100, dtype=torch.long)
    mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)

    for row, (ids, labels) in enumerate(sampled):
        length = min(len(ids), seq_len)
        x[row, :length] = torch.tensor(ids[:length], dtype=torch.long)
        y[row, :length] = torch.tensor(labels[:length], dtype=torch.long)
        mask[row, :length] = True

    return x.to(device), y.to(device), mask.to(device)


# ---------------------------------------------------------------------------
# Loss estimation (no-gradient evaluation pass)
# ---------------------------------------------------------------------------

@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    eval_iters: int = 40,
    use_amp: bool = False,
) -> dict:
    """
    Estimate mean loss over `eval_iters` random batches for both splits.

    Returns:
        {"train": float, "val": float}
    """
    model.eval()
    out = {}
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = []
        for _ in range(eval_iters):
            x, y = get_batch(data, batch_size, seq_len, device)
            with get_autocast(device.type, enabled=use_amp):
                logits, _ = model(x)
                # Flatten for cross-entropy: (B*T, vocab_size) vs (B*T,)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), y.view(-1)
                )
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out


@torch.no_grad()
def estimate_qa_loss(
    model: torch.nn.Module,
    train_examples: list[tuple[list[int], list[int]]],
    val_examples: list[tuple[list[int], list[int]]],
    batch_size: int,
    seq_len: int,
    device: torch.device,
    eval_iters: int = 40,
    use_amp: bool = False,
    pad_id: int = 0,
    use_padding_mask: bool = True,
) -> dict:
    """Estimate answer-only supervised loss over random Q/A batches."""
    model.eval()
    out = {}
    for split, examples in [("train", train_examples), ("val", val_examples)]:
        losses = []
        for _ in range(eval_iters):
            x, y, mask = get_qa_batch(examples, batch_size, seq_len, device, pad_id)
            with get_autocast(device.type, enabled=use_amp):
                logits, _ = model(x, padding_mask=mask if use_padding_mask else None)
                loss = masked_cross_entropy(logits, y)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out


# ---------------------------------------------------------------------------
# Reproducibility helper
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
