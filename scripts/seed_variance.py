"""Score extra-seed fine-tuning runs on the test split (each with its own val-chosen threshold).

Usage: python scripts/seed_variance.py models/seed43 models/seed44 ...
Writes results/seed_runs/summary.json. The main reported model is still models/transformer.
"""

import json
import sys
from pathlib import Path

from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from injection_guard import transformer as tr  # noqa: E402
from injection_guard.data import ROOT, load_splits  # noqa: E402
from injection_guard.metrics import binary_metrics  # noqa: E402


def main() -> None:
    test = load_splits()["test"]
    y = [e.label for e in test]
    rows = []
    for d in map(Path, sys.argv[1:]):
        model, tok, thr, device = tr.load(d)
        loader = DataLoader(tr.TextDataset(test), batch_size=32, collate_fn=tr.make_collate(tok))
        scores = tr.score_loader(model, loader, device)
        cfg = json.loads((d / "guard_config.json").read_text())
        m = binary_metrics(y, [int(s >= thr) for s in scores])
        rows.append({"model_dir": d.name, "seed": cfg["hparams"]["seed"], "threshold": thr, "best_epoch": cfg["best_epoch"],
                     "train_seconds": cfg["train_seconds"], "test": m})
        print(d.name, thr, {k: round(m[k], 4) for k in ("precision", "recall", "f1", "accuracy")}, m["confusion_matrix"])
    out = ROOT / "results" / "seed_runs" / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
