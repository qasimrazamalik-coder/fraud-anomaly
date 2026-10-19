"""Loading, schema checks, the time-based split, the train-only scaler and a synthetic fixture.

The real data is the Kaggle 'Credit Card Fraud Detection' CSV (Time, V1..V28, Amount, Class).
The synthetic generator exists for tests and the smoke run only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PCA_COLUMNS = [f"V{i}" for i in range(1, 29)]
FEATURE_COLUMNS = [*PCA_COLUMNS, "Amount"]
TIME_COLUMN = "Time"
LABEL_COLUMN = "Class"
SECONDS_PER_TWO_DAYS = 172_800.0


class SchemaError(ValueError):
    """The CSV does not match the expected transaction schema."""


@dataclass(frozen=True)
class Dataset:
    """A transaction table plus whether it is the synthetic fixture."""

    frame: pd.DataFrame
    name: str
    synthetic: bool


@dataclass(frozen=True)
class Splits:
    """Chronological train, validation and test windows (original row index preserved)."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def validate_schema(frame: pd.DataFrame, require_label: bool = False) -> None:
    """Check columns, dtypes and values; raise SchemaError with a message fit for display.

    Time and Class are optional for scoring but required for training and evaluation
    (pass require_label=True there). Everything the models read must be finite numbers.
    """
    if frame.empty:
        raise SchemaError("The table has no rows.")
    needed = list(FEATURE_COLUMNS) + ([LABEL_COLUMN, TIME_COLUMN] if require_label else [])
    missing = [c for c in needed if c not in frame.columns]
    if missing:
        shown = ", ".join(missing[:8]) + (" ..." if len(missing) > 8 else "")
        raise SchemaError(
            f"Missing {len(missing)} expected column(s): {shown}. "
            f"Expected {TIME_COLUMN} (optional), {PCA_COLUMNS[0]}..{PCA_COLUMNS[-1]}, Amount "
            f"and, optionally, {LABEL_COLUMN}."
        )
    check = FEATURE_COLUMNS + [c for c in (TIME_COLUMN, LABEL_COLUMN) if c in frame.columns]
    for col in check:
        if not pd.api.types.is_numeric_dtype(frame[col]):
            raise SchemaError(f"Column '{col}' must be numeric.")
        values = frame[col].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise SchemaError(f"Column '{col}' contains missing or infinite values.")
    if LABEL_COLUMN in frame.columns and not set(frame[LABEL_COLUMN].unique()) <= {0, 1}:
        raise SchemaError(f"Column '{LABEL_COLUMN}' must contain only 0 and 1.")
    if (frame["Amount"] < 0).any():
        raise SchemaError("Column 'Amount' must be non-negative.")


def read_transactions(path: str | Path, require_label: bool = True) -> pd.DataFrame:
    """Read a transactions CSV and validate it. Fails with the path in the message."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run `python scripts/download_data.py` "
            "(see the README for the Kaggle download)."
        )
    frame = pd.read_csv(path)
    validate_schema(frame, require_label=require_label)
    return frame


def synthetic_transactions(
    n: int = 6000, fraud_rate: float = 0.02, seed: int = 0, drift: float = 0.5
) -> pd.DataFrame:
    """SYNTHETIC imbalanced transactions: Gaussian-mixture normals, one shifted rare cluster.

    The normal mixture drifts slowly along Time so validation and test windows differ, which
    lets tests and the smoke run exercise the drift report. Not a model of real fraud.
    """
    rng = np.random.default_rng(seed)
    dim = len(PCA_COLUMNS)
    time = np.sort(rng.uniform(0.0, SECONDS_PER_TWO_DAYS, size=n))
    comp_means = rng.normal(0.0, 1.0, size=(3, dim))
    comp = rng.integers(0, 3, size=n)
    direction = rng.normal(size=dim)
    direction /= np.linalg.norm(direction)
    x = comp_means[comp] + rng.normal(0.0, 1.0, size=(n, dim))
    x += drift * 3.0 * (time / SECONDS_PER_TWO_DAYS)[:, None] * direction[None, :]

    n_fraud = max(10, round(n * fraud_rate))
    fraud_idx = rng.choice(n, size=n_fraud, replace=False)
    shift = np.zeros(dim)
    active = rng.choice(dim, size=10, replace=False)
    shift[active] = rng.choice([-1.0, 1.0], size=10) * 2.0
    x[fraud_idx] = shift + rng.normal(0.0, 1.5, size=(n_fraud, dim))

    amount = rng.lognormal(3.5, 1.2, size=n)
    amount[fraud_idx] = rng.lognormal(4.2, 1.5, size=n_fraud)
    label = np.zeros(n, dtype=int)
    label[fraud_idx] = 1

    frame = pd.DataFrame(x, columns=PCA_COLUMNS)
    frame.insert(0, TIME_COLUMN, time)
    frame["Amount"] = amount
    frame[LABEL_COLUMN] = label
    return frame


def load_dataset(cfg: dict[str, Any]) -> Dataset:
    """Load the dataset named in the config's `data` block (csv or synthetic)."""
    source = cfg.get("source")
    if source == "synthetic":
        frame = synthetic_transactions(
            n=int(cfg["n"]), fraud_rate=float(cfg["fraud_rate"]), seed=int(cfg.get("seed", 0))
        )
        return Dataset(frame, "SYNTHETIC fixture", True)
    if source == "csv":
        return Dataset(read_transactions(cfg["path"]), "Credit Card Fraud Detection", False)
    raise ValueError(f"data.source must be 'csv' or 'synthetic', got {source!r}")


def time_split(frame: pd.DataFrame, train_frac: float, val_frac: float) -> Splits:
    """Split by Time: earliest rows train, next validation, latest test.

    A random split lets a model see transactions from the same hours it is tested on and
    hides temporal drift; a deployed detector only ever scores the future. Boundaries are
    moved forward past tied Time values so no timestamp straddles two windows.
    """
    if not (train_frac > 0 and val_frac > 0 and train_frac + val_frac < 1):
        raise ValueError("train_frac and val_frac must be positive and sum to less than 1.")
    ordered = frame.sort_values(TIME_COLUMN, kind="stable")
    times = ordered[TIME_COLUMN].to_numpy()
    n = len(ordered)

    def boundary(frac: float) -> int:
        i = math.floor(n * frac)
        while 0 < i < n and times[i] == times[i - 1]:
            i += 1
        return i

    cut1, cut2 = boundary(train_frac), boundary(train_frac + val_frac)
    parts = ordered.iloc[:cut1], ordered.iloc[cut1:cut2], ordered.iloc[cut2:]
    if any(len(p) == 0 for p in parts):
        raise ValueError("Time split produced an empty window; use more rows or other fractions.")
    return Splits(*parts)


class FeatureScaler:
    """log1p(Amount) then z-score, with statistics from the training window only.

    Time is not a model input: it grows monotonically, so a model could only use it to
    memorise the window it was trained on.
    """

    def __init__(self) -> None:
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    @staticmethod
    def _raw(frame: pd.DataFrame) -> np.ndarray:
        x = frame[FEATURE_COLUMNS].to_numpy(dtype=np.float64).copy()
        x[:, -1] = np.log1p(x[:, -1])
        return x

    def fit(self, frame: pd.DataFrame) -> FeatureScaler:
        x = self._raw(frame)
        self.mean = x.mean(axis=0)
        std = x.std(axis=0)
        self.std = np.where(std > 1e-12, std, 1.0)
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("FeatureScaler.transform called before fit.")
        return ((self._raw(frame) - self.mean) / self.std).astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        if self.mean is None or self.std is None:
            raise RuntimeError("FeatureScaler is not fitted.")
        return {"columns": FEATURE_COLUMNS, "mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FeatureScaler:
        if payload.get("columns") != FEATURE_COLUMNS:
            raise ValueError("Scaler file was written for different feature columns.")
        scaler = cls()
        scaler.mean = np.asarray(payload["mean"], dtype=np.float64)
        scaler.std = np.asarray(payload["std"], dtype=np.float64)
        return scaler
