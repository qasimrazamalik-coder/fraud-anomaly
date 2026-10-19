"""Seeding, config loading, JSON output and environment logging."""

from __future__ import annotations

import json
import os
import platform
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def seed_everything(seed: int) -> None:
    """Seed python, numpy and torch in one place so every run is reproducible on CPU."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str | Path) -> dict[str, Any]:
    """Read a YAML config. Fails loudly if the file is missing or not a mapping."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open() as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config {path} must contain a YAML mapping at the top level")
    return cfg


def package_versions() -> dict[str, str]:
    """Versions of the libraries that affect the numbers, written into every results JSON."""
    import pandas
    import scipy
    import sklearn
    import torch

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pandas.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
        "torch": torch.__version__,
    }


def utc_now() -> str:
    """ISO timestamp in UTC, second resolution."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Cannot serialise {type(obj).__name__}")


def _clean(obj: Any) -> Any:
    """Replace NaN and infinity with None so the output is strict JSON."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, (float, np.floating)) and not np.isfinite(obj):
        return None
    return obj


def write_json(obj: Any, path: str | Path) -> Path:
    """Write strict JSON (NaN becomes null), creating parent folders."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(_clean(obj), fh, indent=2, default=_json_default)
        fh.write("\n")
    return path
