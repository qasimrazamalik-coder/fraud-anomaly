"""Isolation Forest (unsupervised) and the two supervised reference models.

All scorers return one float per row where higher means more likely fraud, so metrics
treat every model identically.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression


def fit_isolation_forest(x_normal: np.ndarray, cfg: dict[str, Any], seed: int) -> IsolationForest:
    """Fit on normal rows only; the forest never sees a label."""
    model = IsolationForest(
        n_estimators=int(cfg["n_estimators"]),
        max_samples=cfg["max_samples"],
        max_features=float(cfg.get("max_features", 1.0)),
        random_state=seed,
        n_jobs=int(cfg.get("n_jobs", 2)),
    )
    return model.fit(x_normal)


def isolation_scores(model: IsolationForest, x: np.ndarray) -> np.ndarray:
    """Anomaly score with higher = more anomalous.

    scikit-learn's `score_samples` is the opposite (higher = more normal), so it is negated
    here. Getting this sign wrong silently inverts every metric.
    """
    return -model.score_samples(x)


def fit_logistic_regression(
    x: np.ndarray, y: np.ndarray, cfg: dict[str, Any], seed: int
) -> LogisticRegression:
    """Supervised reference: class weights counter the ~0.17% positive rate."""
    model = LogisticRegression(
        C=float(cfg["C"]), max_iter=int(cfg["max_iter"]), class_weight="balanced", random_state=seed
    )
    return model.fit(x, y)


def fit_hist_gradient_boosting(
    x: np.ndarray, y: np.ndarray, cfg: dict[str, Any], seed: int
) -> HistGradientBoostingClassifier:
    """Supervised ceiling: gradient boosting with balanced class weights."""
    model = HistGradientBoostingClassifier(
        max_iter=int(cfg["max_iter"]),
        learning_rate=float(cfg["learning_rate"]),
        max_leaf_nodes=int(cfg.get("max_leaf_nodes", 31)),
        class_weight="balanced",
        early_stopping=False,
        random_state=seed,
    )
    return model.fit(x, y)


def positive_probability(model: Any, x: np.ndarray) -> np.ndarray:
    """P(fraud) from a fitted scikit-learn classifier."""
    return model.predict_proba(x)[:, 1]
