"""Save, load and apply the trained unsupervised detectors (used by the app).

Weights are written next to a scaler and the validation-chosen thresholds so a loaded bundle
scores new transactions exactly as the evaluation did.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import IsolationForest

from fraud_anomaly.autoencoder import MLPAutoencoder, reconstruction_error
from fraud_anomaly.data import FeatureScaler
from fraud_anomaly.detectors import isolation_scores

AE_FILE = "autoencoder.pt"
IF_FILE = "isolation_forest.joblib"
SCALER_FILE = "scaler.json"
META_FILE = "thresholds.json"


class ModelNotFoundError(FileNotFoundError):
    """No trained bundle in the model directory."""


@dataclass
class ModelBundle:
    """Scaler, both detectors and their validation-chosen thresholds."""

    scaler: FeatureScaler
    autoencoder: MLPAutoencoder
    forest: IsolationForest
    thresholds: dict[str, float]
    synthetic: bool
    beta: float

    def score(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        """Anomaly scores per model, higher = more anomalous."""
        x = self.scaler.transform(frame)
        return {
            "autoencoder": reconstruction_error(self.autoencoder, x),
            "isolation_forest": isolation_scores(self.forest, x),
        }


def save_bundle(
    model_dir: str | Path,
    scaler: FeatureScaler,
    autoencoder: MLPAutoencoder,
    forest: IsolationForest,
    thresholds: dict[str, float],
    synthetic: bool,
    beta: float,
) -> None:
    """Write the bundle. `joblib` pickles the forest, so only load folders you trained."""
    out = Path(model_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"spec": autoencoder.spec, "state_dict": autoencoder.state_dict()}, out / AE_FILE)
    joblib.dump(forest, out / IF_FILE)
    (out / SCALER_FILE).write_text(json.dumps(scaler.to_dict()))
    meta: dict[str, Any] = {"thresholds": thresholds, "synthetic": synthetic, "beta": beta}
    (out / META_FILE).write_text(json.dumps(meta, indent=2))


def load_bundle(model_dir: str | Path) -> ModelBundle:
    """Load a bundle or raise ModelNotFoundError naming what is missing."""
    d = Path(model_dir)
    missing = [f for f in (AE_FILE, IF_FILE, SCALER_FILE, META_FILE) if not (d / f).is_file()]
    if missing:
        raise ModelNotFoundError(f"No trained models in {d} (missing: {', '.join(missing)}).")
    payload = torch.load(d / AE_FILE, map_location="cpu", weights_only=True)
    ae = MLPAutoencoder(**payload["spec"])
    ae.load_state_dict(payload["state_dict"])
    ae.eval()
    meta = json.loads((d / META_FILE).read_text())
    return ModelBundle(
        scaler=FeatureScaler.from_dict(json.loads((d / SCALER_FILE).read_text())),
        autoencoder=ae,
        forest=joblib.load(d / IF_FILE),
        thresholds={k: float(v) for k, v in meta["thresholds"].items()},
        synthetic=bool(meta["synthetic"]),
        beta=float(meta["beta"]),
    )
