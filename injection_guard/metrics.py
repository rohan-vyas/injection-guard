"""Metric and threshold helpers shared by all models."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def binary_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n": int(len(y_true)),
    }


def choose_threshold(y_true: Sequence[int], scores: Sequence[float], candidates=None) -> float:
    """Pick the threshold that maximises F1 on the given (validation) data.

    Ties are broken towards the threshold closest to the middle of the tied
    range, which is less brittle than taking an extreme.
    """
    y = np.asarray(y_true)
    s = np.asarray(scores, dtype=float)
    if candidates is None:
        candidates = np.unique(s)
    candidates = np.asarray(sorted(set(float(c) for c in candidates)))
    f1s = np.array([f1_score(y, (s >= t).astype(int), zero_division=0) for t in candidates])
    best = f1s.max()
    tied = candidates[np.isclose(f1s, best)]
    return float(tied[len(tied) // 2])
