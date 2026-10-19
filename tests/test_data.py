import numpy as np
import pandas as pd
import pytest

from fraud_anomaly.data import (
    FEATURE_COLUMNS,
    FeatureScaler,
    SchemaError,
    load_dataset,
    synthetic_transactions,
    time_split,
    validate_schema,
)


def test_time_split_is_ordered_and_disjoint(synth):
    s = time_split(synth, 0.6, 0.2)
    assert s.train["Time"].max() < s.val["Time"].min()
    assert s.val["Time"].max() < s.test["Time"].min()
    idx = [set(p.index) for p in (s.train, s.val, s.test)]
    assert not (idx[0] & idx[1] or idx[1] & idx[2] or idx[0] & idx[2])
    assert sum(len(p) for p in (s.train, s.val, s.test)) == len(synth)


def test_time_split_works_on_shuffled_input(synth):
    shuffled = synth.sample(frac=1.0, random_state=0)
    s = time_split(shuffled, 0.6, 0.2)
    assert s.train["Time"].max() < s.test["Time"].min()


def test_tied_timestamps_never_straddle_windows():
    frame = synthetic_transactions(n=300, seed=1)
    frame["Time"] = np.repeat(np.arange(30), 10).astype(float)
    s = time_split(frame, 0.55, 0.2)
    assert s.train["Time"].max() < s.val["Time"].min() < s.test["Time"].min()


def test_time_split_rejects_bad_fractions(synth):
    with pytest.raises(ValueError, match="sum to less than 1"):
        time_split(synth, 0.8, 0.3)


def test_scaler_uses_train_statistics_only(synth):
    s = time_split(synth, 0.6, 0.2)
    scaler = FeatureScaler().fit(s.train)
    train_x = scaler.transform(s.train)
    assert np.allclose(train_x.mean(axis=0), 0, atol=1e-4)
    assert np.allclose(train_x.std(axis=0), 1, atol=1e-3)
    full = FeatureScaler().fit(synth)
    assert not np.allclose(scaler.mean, full.mean)
    # the test window is NOT re-centred: the drifting synthetic mean must remain visible
    assert np.abs(scaler.transform(s.test).mean(axis=0)).max() > 0.05


def test_scaler_is_unchanged_by_transform_and_round_trips(synth):
    s = time_split(synth, 0.6, 0.2)
    scaler = FeatureScaler().fit(s.train)
    before = scaler.mean.copy()
    scaler.transform(s.test)
    assert np.array_equal(before, scaler.mean)
    clone = FeatureScaler.from_dict(scaler.to_dict())
    assert np.allclose(clone.transform(s.test), scaler.transform(s.test))


def test_scaler_requires_fit(synth):
    with pytest.raises(RuntimeError, match="before fit"):
        FeatureScaler().transform(synth)


def test_synthetic_is_deterministic_and_imbalanced():
    a, b = synthetic_transactions(n=800, seed=2), synthetic_transactions(n=800, seed=2)
    pd.testing.assert_frame_equal(a, b)
    assert 0 < a["Class"].mean() < 0.1
    assert list(a.columns) == ["Time", *FEATURE_COLUMNS[:-1], "Amount", "Class"]


def test_load_dataset_flags_synthetic_and_rejects_unknown():
    ds = load_dataset({"source": "synthetic", "n": 300, "fraud_rate": 0.05})
    assert ds.synthetic and "SYNTHETIC" in ds.name
    with pytest.raises(ValueError, match=r"data\.source"):
        load_dataset({"source": "parquet"})


def test_schema_accepts_valid_with_and_without_label(synth):
    validate_schema(synth, require_label=True)
    validate_schema(synth.drop(columns=["Class", "Time"]))


def test_schema_reports_missing_columns(synth):
    with pytest.raises(SchemaError, match="Missing 2 expected column"):
        validate_schema(synth.drop(columns=["V3", "Amount"]))
    with pytest.raises(SchemaError, match="Class"):
        validate_schema(synth.drop(columns=["Class"]), require_label=True)


def test_schema_rejects_bad_values(synth):
    with pytest.raises(SchemaError, match="no rows"):
        validate_schema(synth.iloc[:0])
    bad = synth.copy()
    bad["V1"] = "x"
    with pytest.raises(SchemaError, match="numeric"):
        validate_schema(bad)
    bad = synth.copy()
    bad.loc[bad.index[0], "V2"] = np.nan
    with pytest.raises(SchemaError, match="missing or infinite"):
        validate_schema(bad)
    bad = synth.copy()
    bad["Class"] = 2
    with pytest.raises(SchemaError, match="only 0 and 1"):
        validate_schema(bad)
    bad = synth.copy()
    bad.loc[bad.index[0], "Amount"] = -1.0
    with pytest.raises(SchemaError, match="non-negative"):
        validate_schema(bad)
