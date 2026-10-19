"""Run the full protocol: time split, fit, validation thresholds, test metrics, drift.

    python -m fraud_anomaly.train --config configs/default.yaml

Writes <output.dir>/<output.name>.json, <output.dir>/<output.name>_scores.npz (for plots)
and the unsupervised model bundle in output.model_dir.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from fraud_anomaly.autoencoder import fit_autoencoder, reconstruction_error
from fraud_anomaly.data import (
    LABEL_COLUMN,
    TIME_COLUMN,
    FeatureScaler,
    Splits,
    load_dataset,
    time_split,
)
from fraud_anomaly.detectors import (
    fit_hist_gradient_boosting,
    fit_isolation_forest,
    fit_logistic_regression,
    isolation_scores,
    positive_probability,
)
from fraud_anomaly.inference import save_bundle
from fraud_anomaly.metrics import (
    bootstrap_ci,
    cost_threshold,
    evaluate_scores,
    fbeta_threshold,
    ks_shift,
    threshold_report,
)
from fraud_anomaly.utils import (
    load_config,
    package_versions,
    seed_everything,
    utc_now,
    write_json,
)

SETTINGS = {
    "autoencoder": "unsupervised",
    "isolation_forest": "unsupervised",
    "logistic_regression": "supervised reference",
    "hist_gradient_boosting": "supervised reference",
}


def window_summary(frame: Any) -> dict[str, Any]:
    """Size, time range and fraud count of one window, for the results JSON."""
    return {
        "n": len(frame),
        "n_fraud": int(frame[LABEL_COLUMN].sum()),
        "base_rate": float(frame[LABEL_COLUMN].mean()),
        "time_start": float(frame[TIME_COLUMN].min()),
        "time_end": float(frame[TIME_COLUMN].max()),
    }


def check_windows(splits: Splits) -> None:
    """Fail early if a window cannot support the metrics (no frauds to learn a threshold on)."""
    for name, part in (("train", splits.train), ("validation", splits.val), ("test", splits.test)):
        frauds = int(part[LABEL_COLUMN].sum())
        if frauds == 0 or frauds == len(part):
            raise ValueError(
                f"The {name} window has {frauds} fraud rows out of {len(part)}; "
                "metrics and thresholds need both classes. Use more data or other split fractions."
            )


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    """Execute the protocol and return the results dict (also written by `main`)."""
    seed = int(cfg["seed"])
    seed_everything(seed)
    data = load_dataset({**cfg["data"], "seed": seed})
    splits = time_split(data.frame, cfg["split"]["train_frac"], cfg["split"]["val_frac"])
    check_windows(splits)

    scaler = FeatureScaler().fit(splits.train)
    frames = {"train": splits.train, "val": splits.val, "test": splits.test}
    x = {n: scaler.transform(f) for n, f in frames.items()}
    y = {n: f[LABEL_COLUMN].to_numpy() for n, f in frames.items()}

    normal_train = x["train"][y["train"] == 0]
    normal_val = x["val"][y["val"] == 0]
    ae, history = fit_autoencoder(normal_train, normal_val, cfg["autoencoder"], seed)
    forest = fit_isolation_forest(normal_train, cfg["isolation_forest"], seed)
    logreg = fit_logistic_regression(x["train"], y["train"], cfg["logistic_regression"], seed)
    hgb = fit_hist_gradient_boosting(x["train"], y["train"], cfg["hist_gradient_boosting"], seed)

    scorers = {
        "autoencoder": lambda a: reconstruction_error(ae, a),
        "isolation_forest": lambda a: isolation_scores(forest, a),
        "logistic_regression": lambda a: positive_probability(logreg, a),
        "hist_gradient_boosting": lambda a: positive_probability(hgb, a),
    }
    ev = cfg["evaluation"]
    models: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {"y_val": y["val"], "y_test": y["test"]}
    for name, scorer in scorers.items():
        s_val, s_test = scorer(x["val"]), scorer(x["test"])
        arrays[f"{name}__val"], arrays[f"{name}__test"] = s_val, s_test
        thr, val_f = fbeta_threshold(y["val"], s_val, ev["beta"])
        cost_thr = cost_threshold(y["val"], s_val, ev["cost_fp"], ev["cost_fn"])
        test = evaluate_scores(y["test"], s_test, cfg, thr)
        test["pr_auc_ci"] = bootstrap_ci(y["test"], s_test, int(ev["n_bootstrap"]), seed)
        test["at_cost_threshold"] = threshold_report(
            y["test"], s_test, cost_thr, ev["beta"], ev["cost_fp"], ev["cost_fn"]
        )
        normal_val_scores = s_val[y["val"] == 0]
        normal_test_scores = s_test[y["test"] == 0]
        models[name] = {
            "setting": SETTINGS[name],
            "fbeta_threshold": thr,
            "cost_threshold": cost_thr,
            "val_fbeta": val_f,
            "val_pr_auc": evaluate_scores(y["val"], s_val, cfg, thr)["pr_auc"],
            "test": test,
            "drift": {
                "all_rows": ks_shift(s_val, s_test),
                "normal_rows_only": ks_shift(normal_val_scores, normal_test_scores),
            },
        }

    model_dir = cfg["output"]["model_dir"]
    save_bundle(
        model_dir,
        scaler,
        ae,
        forest,
        {k: models[k]["fbeta_threshold"] for k in ("autoencoder", "isolation_forest")},
        data.synthetic,
        float(ev["beta"]),
    )
    results = {
        "meta": {
            "dataset": data.name,
            "synthetic": data.synthetic,
            "seed": seed,
            "created_utc": utc_now(),
            "versions": package_versions(),
            "config": cfg,
            "windows": {n: window_summary(f) for n, f in frames.items()},
            "autoencoder_history": history,
            "model_dir": str(model_dir),
        },
        "models": models,
    }
    out_dir = Path(cfg["output"]["dir"])
    name = cfg["output"]["name"]
    write_json(results, out_dir / f"{name}.json")
    np.savez_compressed(out_dir / f"{name}_scores.npz", **arrays)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    results = run(cfg)
    label = "SYNTHETIC fixture" if results["meta"]["synthetic"] else results["meta"]["dataset"]
    print(f"Finished on {label}. Wrote {cfg['output']['dir']}/{cfg['output']['name']}.json")
    for name, info in results["models"].items():
        t = info["test"]
        print(
            f"  {name:24s} test PR-AUC {t['pr_auc']:.4f} (base rate {t['base_rate']:.4f})"
            "  [pipeline check only]" * results["meta"]["synthetic"]
        )


if __name__ == "__main__":
    main()
