"""Imbalance-aware metrics, validation-side threshold choice, cost view, bootstrap and drift.

Every function takes labels y (0/1) and scores s where higher means more likely fraud.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def _check(y: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y, s = np.asarray(y).astype(int), np.asarray(s, dtype=float)
    if y.shape != s.shape or y.ndim != 1:
        raise ValueError("y and scores must be 1-D arrays of the same length.")
    if y.sum() == 0 or y.sum() == len(y):
        raise ValueError("Metric needs both classes present in the labels.")
    return y, s


def base_rate(y: np.ndarray) -> float:
    """Fraction of positives. A PR-AUC is only interpretable against this number."""
    return float(np.mean(y))


def pr_auc(y: np.ndarray, s: np.ndarray) -> float:
    """Average precision, the headline metric. Random scoring gives about the base rate."""
    y, s = _check(y, s)
    return float(average_precision_score(y, s))


def roc_auc(y: np.ndarray, s: np.ndarray) -> float:
    """ROC-AUC. Reported but optimistic under imbalance: the many negatives keep the FPR small."""
    y, s = _check(y, s)
    return float(roc_auc_score(y, s))


def precision_at_recall(y: np.ndarray, s: np.ndarray, level: float) -> float:
    """Best precision among operating points that reach at least `level` recall."""
    y, s = _check(y, s)
    if not 0 < level <= 1:
        raise ValueError("Recall level must be in (0, 1].")
    precision, recall, _ = precision_recall_curve(y, s)
    return float(precision[recall >= level].max())


def recall_at_budget(y: np.ndarray, s: np.ndarray, fraction: float) -> float:
    """Recall when only the top `fraction` of transactions by score are flagged.

    An alert budget mirrors reality: analysts can review a fixed number of cases per day.
    Ties are broken by input order (stable sort).
    """
    y, s = _check(y, s)
    if not 0 < fraction <= 1:
        raise ValueError("Alert budget must be a fraction in (0, 1].")
    k = max(1, math.ceil(fraction * len(y)))
    top = np.argsort(-s, kind="stable")[:k]
    return float(y[top].sum() / y.sum())


def fbeta_threshold(y: np.ndarray, s: np.ndarray, beta: float) -> tuple[float, float]:
    """Threshold maximising F-beta on the given data (call this on VALIDATION only).

    Flagging rule everywhere in this package: score >= threshold.
    """
    y, s = _check(y, s)
    precision, recall, thresholds = precision_recall_curve(y, s)
    p, r = precision[:-1], recall[:-1]
    denom = beta**2 * p + r
    f = np.divide((1 + beta**2) * p * r, denom, out=np.zeros_like(denom), where=denom > 0)
    best = int(np.argmax(f))
    return float(thresholds[best]), float(f[best])


def expected_cost(false_pos: int, false_neg: int, cost_fp: float, cost_fn: float) -> float:
    """Total cost of an operating point under illustrative per-error costs."""
    return float(false_pos * cost_fp + false_neg * cost_fn)


def cost_threshold(y: np.ndarray, s: np.ndarray, cost_fp: float, cost_fn: float) -> float:
    """Threshold minimising total cost on the given data (VALIDATION only).

    Returns +inf when flagging nothing is cheapest.
    """
    y, s = _check(y, s)
    fpr, tpr, thresholds = roc_curve(y, s, drop_intermediate=False)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    cost = fpr * n_neg * cost_fp + (1 - tpr) * n_pos * cost_fn
    return float(thresholds[int(np.argmin(cost))])


def threshold_report(
    y: np.ndarray, s: np.ndarray, threshold: float, beta: float, cost_fp: float, cost_fn: float
) -> dict[str, Any]:
    """Confusion counts, precision, recall, F-beta and cost at a fixed threshold."""
    y, s = _check(y, s)
    flagged = s >= threshold
    tp = int(np.sum(flagged & (y == 1)))
    fp = int(np.sum(flagged & (y == 0)))
    fn = int(np.sum(~flagged & (y == 1)))
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn)
    if tp + fp and tp:
        fbeta = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
    else:
        fbeta = 0.0
    cost = expected_cost(fp, fn, cost_fp, cost_fn)
    return {
        "threshold": float(threshold),
        "flagged": int(flagged.sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "fbeta": float(fbeta),
        "cost": cost,
        "cost_per_1000": cost / len(y) * 1000,
        "cost_flag_nothing_per_1000": expected_cost(0, int(y.sum()), cost_fp, cost_fn)
        / len(y)
        * 1000,
    }


def bootstrap_ci(
    y: np.ndarray, s: np.ndarray, n_boot: int, seed: int, alpha: float = 0.05
) -> dict[str, Any]:
    """Percentile bootstrap CI for PR-AUC, resampling transactions with replacement.

    Resamples without any positive are skipped (and counted). With few frauds the interval
    is wide, which is the honest picture.
    """
    y, s = _check(y, s)
    rng = np.random.default_rng(seed)
    n = len(y)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if y[idx].sum() == 0:
            continue
        values.append(average_precision_score(y[idx], s[idx]))
    if not values:
        raise ValueError("All bootstrap resamples lacked positives; cannot form an interval.")
    lo, hi = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"low": float(lo), "high": float(hi), "n_valid": len(values), "n_boot": n_boot}


def ks_shift(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Two-sample Kolmogorov-Smirnov statistic between two score samples."""
    res = stats.ks_2samp(np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    return {"statistic": float(res.statistic), "p_value": float(res.pvalue)}


def evaluate_scores(
    y: np.ndarray, s: np.ndarray, cfg: dict[str, Any], threshold: float
) -> dict[str, Any]:
    """All threshold-free metrics plus the fixed-threshold report for one model and window."""
    ev = cfg["evaluation"]
    return {
        "n": len(y),
        "n_fraud": int(np.sum(y)),
        "base_rate": base_rate(y),
        "pr_auc": pr_auc(y, s),
        "roc_auc": roc_auc(y, s),
        "precision_at_recall": {
            str(level): precision_at_recall(y, s, level) for level in ev["recall_levels"]
        },
        "recall_at_budget": {
            str(frac): recall_at_budget(y, s, frac) for frac in ev["alert_budgets"]
        },
        "at_fbeta_threshold": threshold_report(
            y, s, threshold, ev["beta"], ev["cost_fp"], ev["cost_fn"]
        ),
    }
