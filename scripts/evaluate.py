"""Fit/score all three models, choose thresholds on validation, evaluate on test.

Writes:
  results/metrics.json            per-model test metrics, thresholds, val metrics
  results/predictions_test.jsonl  one row per test example with every model's score/prediction
  results/errors_transformer.json false positives/negatives of the fine-tuned model
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from injection_guard import rules, tfidf  # noqa: E402
from injection_guard.data import DATA_DIR, DATASET_NAME, ROOT, SEED, VAL_FRACTION, load_splits, text_hash  # noqa: E402
from injection_guard.metrics import binary_metrics, choose_threshold  # noqa: E402
from injection_guard import transformer as tr  # noqa: E402

RESULTS = ROOT / "results"


def split_stats(examples) -> dict:
    n = len(examples)
    pos = sum(e.label for e in examples)
    return {"n": n, "injection": pos, "benign": n - pos, "injection_rate": round(pos / n, 4)}


def run_rules(splits) -> dict:
    val_scores = [rules.score(e.text) for e in splits["val"]]
    thr = choose_threshold([e.label for e in splits["val"]], val_scores, candidates=range(1, len(rules.RULES) + 1))
    test_scores = [rules.score(e.text) for e in splits["test"]]
    return {"threshold": thr, "val_scores": val_scores, "test_scores": test_scores,
            "notes": "score = number of regex rules that fire; threshold chosen on validation"}


def run_tfidf(splits) -> dict:
    X_tr = [e.text for e in splits["train"]]
    y_tr = [e.label for e in splits["train"]]
    X_va = [e.text for e in splits["val"]]
    y_va = [e.label for e in splits["val"]]
    # choose C on validation F1 at 0.5, then the threshold on validation scores
    best = None
    c_search = {}
    for C in (0.1, 1.0, 10.0, 100.0):
        pipe = tfidf.build(SEED)
        pipe.set_params(clf__C=C)
        pipe.fit(X_tr, y_tr)
        s = pipe.predict_proba(X_va)[:, 1]
        f1 = binary_metrics(y_va, (s >= 0.5).astype(int).tolist())["f1"]
        c_search[str(C)] = f1
        if best is None or f1 > best[0]:
            best = (f1, C, pipe, s)
    _, C, pipe, val_scores = best
    thr = choose_threshold(y_va, val_scores, candidates=np.round(np.arange(0.05, 0.96, 0.05), 2))
    test_scores = pipe.predict_proba([e.text for e in splits["test"]])[:, 1]
    return {"threshold": thr, "C": C, "val_f1_by_C@0.5": c_search,
            "val_scores": val_scores.tolist(), "test_scores": test_scores.tolist(),
            "notes": "word 1-2gram + char_wb 2-5gram TF-IDF, logistic regression (class_weight=balanced); C and threshold chosen on validation"}


def run_transformer(splits) -> dict | None:
    if not tr.weights_available():
        print("no fine-tuned transformer found; skipping", file=sys.stderr)
        return None
    model, tokenizer, thr, device = tr.load()
    collate = tr.make_collate(tokenizer)

    def scores(exs):
        return tr.score_loader(model, DataLoader(tr.TextDataset(exs), batch_size=32, collate_fn=collate), device)

    val_scores = scores(splits["val"])
    t0 = time.perf_counter()
    test_scores = scores(splits["test"])
    infer_s = time.perf_counter() - t0
    cfg = json.loads((tr.MODEL_DIR / "guard_config.json").read_text())
    return {"threshold": thr, "val_scores": val_scores, "test_scores": test_scores,
            "base_model": cfg["base_model"], "best_epoch": cfg["best_epoch"], "train_seconds": cfg["train_seconds"],
            "device": cfg["device"], "test_inference_seconds": round(infer_s, 2),
            "notes": "fine-tuned in plain PyTorch; early stopping on val F1; threshold chosen on validation"}


def main() -> None:
    splits = load_splits()
    meta = json.loads((DATA_DIR / "dataset_meta.json").read_text()) if (DATA_DIR / "dataset_meta.json").exists() else {}
    y_val = [e.label for e in splits["val"]]
    y_test = [e.label for e in splits["test"]]

    models = {"rules": run_rules(splits), "tfidf_logreg": run_tfidf(splits)}
    t = run_transformer(splits)
    if t is not None:
        models["transformer"] = t

    out = {
        "dataset": {**meta, "name": DATASET_NAME, "val_fraction_of_train": VAL_FRACTION, "split_seed": SEED,
                    "splits": {k: split_stats(v) for k, v in splits.items()}},
        "models": {},
    }
    for name, r in models.items():
        thr = r["threshold"]
        val_pred = [int(s >= thr) for s in r["val_scores"]]
        test_pred = [int(s >= thr) for s in r["test_scores"]]
        r["test_pred"] = test_pred
        extra = {k: v for k, v in r.items() if k not in ("val_scores", "test_scores", "test_pred")}
        out["models"][name] = {**extra, "val": binary_metrics(y_val, val_pred), "test": binary_metrics(y_test, test_pred)}

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "metrics.json").write_text(json.dumps(out, indent=2) + "\n")

    with open(RESULTS / "predictions_test.jsonl", "w", encoding="utf-8") as f:
        for i, e in enumerate(splits["test"]):
            row = {"id": e.id, "text_sha256": text_hash(e.text), "label": e.label}
            for name, r in models.items():
                row[f"{name}_score"] = float(r["test_scores"][i])
                row[f"{name}_pred"] = r["test_pred"][i]
            f.write(json.dumps(row) + "\n")

    if "transformer" in models:
        r = models["transformer"]
        errs = {"false_positives": [], "false_negatives": []}
        for i, e in enumerate(splits["test"]):
            p = r["test_pred"][i]
            if p != e.label:
                key = "false_positives" if p == 1 else "false_negatives"
                errs[key].append({"id": e.id, "score": round(float(r["test_scores"][i]), 4), "text": e.text,
                                  "rules_fired": rules.matched_rules(e.text)})
        (RESULTS / "errors_transformer.json").write_text(json.dumps(errs, indent=2, ensure_ascii=False) + "\n")

    print(f"{'model':<14}{'thr':>6}{'P':>8}{'R':>8}{'F1':>8}{'Acc':>8}   confusion (tn fp fn tp)")
    for name, m in out["models"].items():
        t = m["test"]
        cm = t["confusion_matrix"]
        print(f"{name:<14}{m['threshold']:>6}{t['precision']:>8.4f}{t['recall']:>8.4f}{t['f1']:>8.4f}{t['accuracy']:>8.4f}   "
              f"{cm['tn']} {cm['fp']} {cm['fn']} {cm['tp']}")


if __name__ == "__main__":
    main()
