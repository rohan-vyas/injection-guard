"""Re-score results/predictions_test.jsonl and add bootstrap 95% CIs plus an exact McNemar test.

Reads only the predictions file, so it also checks that the saved predictions reproduce metrics.json.
Writes results/bootstrap_ci.json.
"""

import json
import sys
from math import comb
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent
N_BOOT = 10_000
SEED = 42


def mcnemar_exact(a_correct, b_correct) -> dict:
    b = int(np.sum(a_correct & ~b_correct))
    c = int(np.sum(~a_correct & b_correct))
    n = b + c
    p = 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, k) for k in range(0, min(b, c) + 1)) / 2**n)
    return {"only_first_correct": b, "only_second_correct": c, "p_value": p}


def main() -> None:
    rows = [json.loads(l) for l in open(ROOT / "results" / "predictions_test.jsonl")]
    metrics = json.loads((ROOT / "results" / "metrics.json").read_text())
    y = np.array([r["label"] for r in rows])
    names = [k[:-5] for k in rows[0] if k.endswith("_pred")]
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(y), size=(N_BOOT, len(y)))
    out = {"n_boot": N_BOOT, "seed": SEED, "models": {}, "mcnemar": {}}
    preds = {}
    for n in names:
        p = np.array([r[f"{n}_pred"] for r in rows])
        preds[n] = p
        f1 = f1_score(y, p)
        assert abs(f1 - metrics["models"][n]["test"]["f1"]) < 1e-9, f"{n}: predictions do not reproduce metrics.json"
        f1s = np.array([f1_score(y[i], p[i], zero_division=0) for i in idx])
        accs = np.array([(y[i] == p[i]).mean() for i in idx])
        out["models"][n] = {"f1": f1, "f1_ci95": [float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))],
                            "accuracy": float((y == p).mean()),
                            "accuracy_ci95": [float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))]}
        print(f"{n:<14} F1 {f1:.4f} [{out['models'][n]['f1_ci95'][0]:.4f}, {out['models'][n]['f1_ci95'][1]:.4f}]")
    if "transformer" in preds:
        for n in names:
            if n != "transformer":
                r = mcnemar_exact(preds["transformer"] == y, preds[n] == y)
                out["mcnemar"][f"transformer_vs_{n}"] = r
                print(f"McNemar transformer vs {n}: {r}")
    (ROOT / "results" / "bootstrap_ci.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    sys.exit(main())
