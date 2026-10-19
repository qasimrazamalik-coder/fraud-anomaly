import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from fraud_anomaly.data import synthetic_transactions


@pytest.fixture(scope="session")
def synth():
    """SYNTHETIC transactions used by the tests."""
    return synthetic_transactions(n=2500, fraud_rate=0.04, seed=3)


@pytest.fixture()
def tiny_cfg(tmp_path):
    """Smallest config that exercises the whole pipeline on the SYNTHETIC fixture."""
    return {
        "seed": 5,
        "data": {"source": "synthetic", "n": 2500, "fraud_rate": 0.04},
        "split": {"train_frac": 0.6, "val_frac": 0.2},
        "autoencoder": {
            "hidden_dims": [8],
            "bottleneck": 3,
            "lr": 0.003,
            "batch_size": 128,
            "max_epochs": 3,
            "patience": 2,
            "denoising_std": 0.0,
            "num_threads": 1,
        },
        "isolation_forest": {"n_estimators": 20, "max_samples": 128, "n_jobs": 1},
        "logistic_regression": {"C": 1.0, "max_iter": 100},
        "hist_gradient_boosting": {"max_iter": 10, "learning_rate": 0.2, "max_leaf_nodes": 7},
        "evaluation": {
            "beta": 2.0,
            "recall_levels": [0.5, 0.9],
            "alert_budgets": [0.001, 0.01],
            "cost_fp": 5.0,
            "cost_fn": 100.0,
            "n_bootstrap": 20,
        },
        "output": {
            "dir": str(tmp_path / "results"),
            "name": "run",
            "model_dir": str(tmp_path / "models"),
        },
    }
