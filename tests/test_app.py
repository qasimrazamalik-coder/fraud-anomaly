import pytest

import app as app_module  # app/app.py, path added in conftest
from fraud_anomaly.data import time_split
from fraud_anomaly.train import run


def test_app_builds_without_weights(monkeypatch, tmp_path):
    monkeypatch.setattr(app_module, "MODEL_DIR", tmp_path / "none")
    assert app_module.build_app() is not None


def test_empty_state_when_no_models(tmp_path, synth):
    path = tmp_path / "tx.csv"
    synth.to_csv(path, index=False)
    status, thr, top, readout = app_module.score_upload(str(path), model_dir=tmp_path / "none")
    assert "No trained models" in status and "make train" in status
    assert thr is None and top is None and readout is None


def test_rejects_wrong_extension_empty_and_oversize(tmp_path, monkeypatch):
    txt = tmp_path / "a.txt"
    txt.write_text("a,b\n1,2\n")
    with pytest.raises(app_module.InputError, match=r"\.csv"):
        app_module.check_upload(str(txt))
    empty = tmp_path / "e.csv"
    empty.write_bytes(b"")
    with pytest.raises(app_module.InputError, match="empty"):
        app_module.check_upload(str(empty))
    with pytest.raises(app_module.InputError, match="No file"):
        app_module.check_upload(None)
    monkeypatch.setattr(app_module, "MAX_UPLOAD_BYTES", 10)
    big = tmp_path / "big.csv"
    big.write_text("a,b\n" + "1,2\n" * 50)
    with pytest.raises(app_module.InputError, match="20 MB"):
        app_module.check_upload(str(big))


@pytest.fixture()
def trained(tiny_cfg, tmp_path):
    run(tiny_cfg)
    return tmp_path / "models"


def test_schema_error_is_shown_clearly(trained, tmp_path, synth):
    path = tmp_path / "bad.csv"
    synth.drop(columns=["V5"]).to_csv(path, index=False)
    status, thr, *_ = app_module.score_upload(str(path), model_dir=trained)
    assert status.startswith("Cannot score") and "V5" in status and thr is None


def test_scores_csv_with_labels_and_shows_synthetic_banner(trained, tmp_path, synth):
    test = time_split(synth, 0.6, 0.2).test
    path = tmp_path / "tx.csv"
    test.to_csv(path, index=False)
    status, thr, top, readout = app_module.score_upload(str(path), "isolation_forest", trained)
    assert "SYNTHETIC" in status
    assert list(thr["Model"]) == ["Autoencoder", "Isolation Forest"]
    assert len(top) == app_module.TOP_ROWS
    assert top["Isolation Forest score"].is_monotonic_decreasing
    assert readout is not None and set(readout["Model"]) == {"Autoencoder", "Isolation Forest"}


def test_no_readout_without_class_column(trained, tmp_path, synth):
    path = tmp_path / "tx.csv"
    synth.drop(columns=["Class", "Time"]).to_csv(path, index=False)
    status, _, top, readout = app_module.score_upload(str(path), model_dir=trained)
    assert readout is None and top is not None and "Class" not in top.columns
    assert "Scored" in status
