"""Fine-tune a multilingual transformer for prompt-injection detection.

The training loop is plain PyTorch: DataLoader, AdamW, linear warmup/decay
scheduler, gradient clipping, early stopping on validation F1, fixed seeds.
The decision threshold is chosen on the validation split only.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from injection_guard.data import ROOT, SEED, load_splits  # noqa: E402
from injection_guard.metrics import binary_metrics, choose_threshold  # noqa: E402
from injection_guard.transformer import (  # noqa: E402
    MAX_LEN,
    MODEL_DIR,
    MODEL_NAME,
    TextDataset,
    make_collate,
    pick_device,
    score_loader,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", type=Path, default=MODEL_DIR)
    args = ap.parse_args()

    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    set_seed(args.seed)
    device = pick_device()
    splits = load_splits()
    train, val = splits["train"], splits["val"]
    print(f"device={device} train={len(train)} val={len(val)} model={args.model}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=2, id2label={0: "benign", 1: "injection"}, label2id={"benign": 0, "injection": 1}
    ).to(device)

    collate = make_collate(tokenizer, MAX_LEN)
    gen = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(TextDataset(train), batch_size=args.batch_size, shuffle=True, collate_fn=collate, generator=gen)
    val_loader = DataLoader(TextDataset(val), batch_size=32, shuffle=False, collate_fn=collate)

    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    params = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(args.warmup_frac * total_steps), total_steps)
    loss_fn = torch.nn.CrossEntropyLoss()

    y_val = [e.label for e in val]
    best_f1, best_epoch, best_state, bad_epochs = -1.0, -1, None, 0
    history = []
    t0 = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, n = 0.0, 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            logits = model(**batch).logits
            loss = loss_fn(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running += loss.item() * labels.size(0)
            n += labels.size(0)

        val_scores = score_loader(model, val_loader, device)
        m = binary_metrics(y_val, [int(s >= 0.5) for s in val_scores])
        history.append({"epoch": epoch, "train_loss": running / n, "val_f1@0.5": m["f1"], "val_acc@0.5": m["accuracy"],
                        "elapsed_s": time.perf_counter() - t0})
        print(json.dumps(history[-1]), flush=True)

        if m["f1"] > best_f1:
            best_f1, best_epoch, bad_epochs = m["f1"], epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"early stop at epoch {epoch} (best epoch {best_epoch})", flush=True)
                break

    train_seconds = time.perf_counter() - t0
    model.load_state_dict(best_state)
    val_scores = score_loader(model, val_loader, device)
    threshold = choose_threshold(y_val, val_scores, candidates=np.round(np.arange(0.05, 0.96, 0.05), 2))
    val_at_thr = binary_metrics(y_val, [int(s >= threshold) for s in val_scores])

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    cfg = {
        "base_model": args.model,
        "threshold": threshold,
        "best_epoch": best_epoch,
        "best_val_f1@0.5": best_f1,
        "val_metrics_at_threshold": val_at_thr,
        "train_seconds": round(train_seconds, 1),
        "device": str(device),
        "hardware": platform.machine() + " " + platform.platform(),
        "torch": torch.__version__,
        "hparams": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "max_len": MAX_LEN,
        "history": history,
    }
    (args.out / "guard_config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    log_name = "transformer_training.json" if args.out.resolve() == MODEL_DIR.resolve() else f"seed_runs/seed{args.seed}_training.json"
    log_path = ROOT / "results" / log_name
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"saved to {args.out}; threshold={threshold}; train_seconds={train_seconds:.1f}", flush=True)


if __name__ == "__main__":
    main()
