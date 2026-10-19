import json

import numpy as np
import pytest

from fraud_anomaly import evaluate as ev
from fraud_anomaly.data import time_split
from fraud_anomaly.inference import ModelNotFoundError, load_bundle
from fraud_anomaly.train import check_windows, run


@pytest.fixture()
def result(tiny_cfg):
    return run(tiny_cfg)


def test_run_writes_results_scores_and_models(tiny_cfg, result, tmp_path):
    out = tmp_path / "results"
    saved = json.loads((out / "run.json").read_text())
    assert saved["meta"]["synthetic"] is True
    assert saved["meta"]["seed"] == 5 and "torch" in saved["meta"]["versions"]
    assert set(saved["models"]) == {
        "autoencoder",
        "isolation_forest",
        "logistic_regression",
        "hist_gradient_boosting",
    }
    w = saved["meta"]["windows"]
    assert w["train"]["time_end"] < w["val"]["time_start"] < w["test"]["time_start"]
    for info in saved["models"].values():
        assert 0 < info["test"]["base_rate"] < 1
        assert info["test"]["pr_auc_ci"]["low"] <= info["test"]["pr_auc_ci"]["high"]
        assert "normal_rows_only" in info["drift"]
    assert (out / "run_scores.npz").is_file()
    assert (tmp_path / "models" / "autoencoder.pt").is_file()


def test_thresholds_come_from_validation_not_test(tiny_cfg, result, tmp_path):
    scores = np.load(tmp_path / "results" / "run_scores.npz")
    from fraud_anomaly.metrics import fbeta_threshold

    for name, info in result["models"].items():
        thr, _ = fbeta_threshold(scores["y_val"], scores[f"{name}__val"], 2.0)
        assert info["fbeta_threshold"] == pytest.approx(thr)
        assert info["test"]["at_fbeta_threshold"]["threshold"] == pytest.approx(thr)


def test_same_seed_gives_same_numbers(tiny_cfg, result):
    again = run(tiny_cfg)
    for name in result["models"]:
        assert again["models"][name]["test"]["pr_auc"] == pytest.approx(
            result["models"][name]["test"]["pr_auc"]
        )


def test_bundle_round_trip_matches_training_scores(tiny_cfg, result, tmp_path):
    from fraud_anomaly.data import synthetic_transactions

    bundle = load_bundle(tmp_path / "models")
    frame = synthetic_transactions(n=2500, fraud_rate=0.04, seed=5)
    splits = time_split(frame, 0.6, 0.2)
    scores = bundle.score(splits.test)
    saved = np.load(tmp_path / "results" / "run_scores.npz")
    assert np.allclose(scores["autoencoder"], saved["autoencoder__test"], rtol=1e-4)
    assert np.allclose(scores["isolation_forest"], saved["isolation_forest__test"], rtol=1e-6)
    assert bundle.synthetic is True


def test_load_bundle_missing_dir(tmp_path):
    with pytest.raises(ModelNotFoundError, match="missing"):
        load_bundle(tmp_path / "nope")


def test_check_windows_rejects_window_without_fraud(synth):
    s = time_split(synth, 0.6, 0.2)
    s.val.loc[:, "Class"] = 0
    with pytest.raises(ValueError, match="validation window has 0 fraud"):
        check_windows(s)


def test_render_markdown_labels_synthetic_and_keeps_base_rate(result):
    md = ev.render_markdown(result)
    assert md.startswith("> SYNTHETIC fixture")
    assert "Test base rate" in md and "Score drift" in md and "Illustrative parameters" in md


def test_readme_update_refuses_synthetic_and_requires_markers(result, tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(f"intro\n{ev.START}\nold\n{ev.END}\noutro\n")
    with pytest.raises(ValueError, match="SYNTHETIC"):
        ev.update_readme(readme, "table", synthetic=True)
    ev.update_readme(readme, "new table", synthetic=False)
    assert readme.read_text() == f"intro\n{ev.START}\nnew table\n{ev.END}\noutro\n"
    bare = tmp_path / "bare.md"
    bare.write_text("no markers")
    with pytest.raises(ValueError, match="must contain"):
        ev.update_readme(bare, "x", synthetic=False)


def test_figures_are_written(result, tmp_path):
    scores = np.load(tmp_path / "results" / "run_scores.npz")
    for fn, name in (
        (ev.plot_pr_curves, "pr.png"),
        (ev.plot_budget_curves, "budget.png"),
        (ev.plot_drift, "drift.png"),
    ):
        fn(scores, result, tmp_path / name)
        assert (tmp_path / name).stat().st_size > 1000
