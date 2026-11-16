"""Transformer classifier: loading, batching and scoring helpers (plain PyTorch)."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from .data import ROOT, Example

MODEL_NAME = "distilbert-base-multilingual-cased"
MODEL_DIR = ROOT / "models" / "transformer"
MAX_LEN = 256


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class TextDataset(Dataset):
    def __init__(self, examples: list[Example]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int) -> Example:
        return self.examples[i]


def make_collate(tokenizer, max_len: int = MAX_LEN):
    def collate(batch: list[Example]) -> dict:
        enc = tokenizer(
            [e.text for e in batch], padding=True, truncation=True, max_length=max_len, return_tensors="pt"
        )
        enc["labels"] = torch.tensor([e.label for e in batch], dtype=torch.long)
        return enc

    return collate


@torch.no_grad()
def score_loader(model, loader: DataLoader, device: torch.device) -> list[float]:
    """Return P(injection) for every example in the loader, in order."""
    model.eval()
    out: list[float] = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        logits = model(**batch).logits
        out.extend(torch.softmax(logits.float(), dim=-1)[:, 1].cpu().tolist())
    return out


def weights_available(model_dir: Path = MODEL_DIR) -> bool:
    return (model_dir / "config.json").exists() and (model_dir / "guard_config.json").exists()


def load(model_dir: Path = MODEL_DIR, device: torch.device | None = None):
    """Load fine-tuned model, tokenizer and the validation-chosen threshold."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if not weights_available(model_dir):
        raise FileNotFoundError(f"no fine-tuned model in {model_dir}; run `python scripts/train_transformer.py`")
    device = device or pick_device()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()
    cfg = json.loads((model_dir / "guard_config.json").read_text())
    return model, tokenizer, float(cfg["threshold"]), device
