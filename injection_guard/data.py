"""Dataset download and loading.

The dataset is treated as untrusted data: it is only ever parsed as JSON lines,
never executed. Downloaded files live under data/.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

DATASET_NAME = "deepset/prompt-injections"
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SEED = 42
VAL_FRACTION = 0.15


@dataclass(frozen=True)
class Example:
    id: str
    text: str
    label: int  # 1 = injection, 0 = benign


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def download(data_dir: Path = DATA_DIR) -> None:
    """Fetch the official splits from Hugging Face and write them as JSONL."""
    from datasets import load_dataset

    data_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(DATASET_NAME, cache_dir=str(data_dir / "hf_cache"))
    for split in ("train", "test"):
        with open(data_dir / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for i, row in enumerate(ds[split]):
                rec = {"id": f"{split}-{i:04d}", "text": row["text"], "label": int(row["label"])}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[Example]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            label = int(rec["label"])
            if label not in (0, 1):
                raise ValueError(f"unexpected label {label!r} in {path}")
            out.append(Example(id=str(rec["id"]), text=str(rec["text"]), label=label))
    return out


def stratified_split(
    examples: list[Example], val_fraction: float = VAL_FRACTION, seed: int = SEED
) -> tuple[list[Example], list[Example]]:
    """Deterministic stratified train/validation split."""
    rng = random.Random(seed)
    train, val = [], []
    for label in (0, 1):
        group = [e for e in examples if e.label == label]
        rng.shuffle(group)
        n_val = round(len(group) * val_fraction)
        val.extend(group[:n_val])
        train.extend(group[n_val:])
    train.sort(key=lambda e: e.id)
    val.sort(key=lambda e: e.id)
    return train, val


def load_splits(data_dir: Path = DATA_DIR) -> dict[str, list[Example]]:
    """Return {'train', 'val', 'test'}. 'test' is the official held-out split."""
    train_path, test_path = data_dir / "train.jsonl", data_dir / "test.jsonl"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"{train_path} / {test_path} missing; run `python scripts/download_data.py` first"
        )
    full_train = read_jsonl(train_path)
    test = read_jsonl(test_path)
    train, val = stratified_split(full_train)
    return {"train": train, "val": val, "test": test}
