import math

import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from fraud_anomaly import metrics as m

Y = np.array([0, 0, 1, 1])
S = np.array([0.1, 0.4, 0.35, 0.8])


def test_pr_auc_matches_sklearn_and_hand_value():
    assert m.pr_auc(Y, S) == pytest.approx(average_precision_score(Y, S))
    # hand computation: hits at ranks 1 and 3 -> (1/1 + 2/3) / 2
    assert m.pr_auc(Y, S) == pytest.approx((1 + 2 / 3) / 2)


def test_roc_auc_hand_value():
    # positive/negative pairs ordered correctly: 3 of 4
    assert m.roc_auc(Y, S) == pytest.approx(0.75)


def test_metrics_need_both_classes():
    with pytest.raises(ValueError, match="both classes"):
        m.pr_auc(np.zeros(5), np.arange(5.0))


def test_random_scores_pr_auc_is_near_base_rate():
    rng = np.random.default_rng(0)
    y = (rng.random(20000) < 0.01).astype(int)
    assert m.pr_auc(y, rng.random(20000)) == pytest.approx(m.base_rate(y), abs=0.01)


def test_recall_at_budget_hand_case():
    y = np.array([1, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    s = np.array([0.9, 0.8, 0.1, 0.7, 0.2, 0.3, 0.05, 0.04, 0.03, 0.02])
    assert m.recall_at_budget(y, s, 0.1) == pytest.approx(0.5)  # top 1 row: the 0.9 fraud
    assert m.recall_at_budget(y, s, 0.2) == pytest.approx(0.5)  # top 2: 0.9, 0.8
    assert m.recall_at_budget(y, s, 0.3) == pytest.approx(1.0)  # top 3 adds the 0.7 fraud
    assert m.recall_at_budget(y, s, 0.0001) == pytest.approx(0.5)  # at least one row is flagged


def test_precision_at_recall_hand_case():
    # ranked: 0.8(+) 0.4(-) 0.35(+) 0.1(-): precision 1 at recall .5; 2/3 at recall 1
    assert m.precision_at_recall(Y, S, 0.5) == pytest.approx(1.0)
    assert m.precision_at_recall(Y, S, 1.0) == pytest.approx(2 / 3)
    with pytest.raises(ValueError):
        m.precision_at_recall(Y, S, 0.0)


def test_expected_cost():
    assert m.expected_cost(3, 2, cost_fp=5.0, cost_fn=100.0) == 215.0


def test_threshold_report_hand_case():
    r = m.threshold_report(Y, S, threshold=0.35, beta=1.0, cost_fp=5.0, cost_fn=100.0)
    # flagged: 0.8(+), 0.4(-), 0.35(+) -> tp 2, fp 1, fn 0
    assert (r["tp"], r["fp"], r["fn"], r["flagged"]) == (2, 1, 0, 3)
    assert r["precision"] == pytest.approx(2 / 3)
    assert r["recall"] == pytest.approx(1.0)
    assert r["fbeta"] == pytest.approx(0.8)
    assert r["cost"] == pytest.approx(5.0)
    assert r["cost_per_1000"] == pytest.approx(1250.0)
    assert r["cost_flag_nothing_per_1000"] == pytest.approx(50000.0)


def test_threshold_report_nothing_flagged():
    r = m.threshold_report(Y, S, threshold=2.0, beta=2.0, cost_fp=1.0, cost_fn=1.0)
    assert r["flagged"] == 0 and r["recall"] == 0.0 and r["fbeta"] == 0.0
    assert math.isnan(r["precision"])


def test_fbeta_threshold_picks_best_cut():
    thr, f = m.fbeta_threshold(Y, S, beta=1.0)
    assert thr == pytest.approx(0.35)
    assert f == pytest.approx(0.8)
    # a recall-heavy beta must not pick a stricter threshold than F1 does
    thr2, _ = m.fbeta_threshold(Y, S, beta=4.0)
    assert thr2 <= thr


def test_cost_threshold_follows_the_costs():
    thr_cheap_fp, thr_dear_fp = (
        m.cost_threshold(Y, S, cost_fp=1.0, cost_fn=100.0),
        m.cost_threshold(Y, S, cost_fp=100.0, cost_fn=1.0),
    )
    assert thr_cheap_fp == pytest.approx(0.35)  # catch both frauds, accept one false alarm
    assert thr_dear_fp == pytest.approx(0.8)  # only the surest row is worth a false-alarm risk


def test_bootstrap_is_seeded_and_brackets_estimate():
    rng = np.random.default_rng(1)
    y = (rng.random(3000) < 0.05).astype(int)
    s = y * 1.5 + rng.normal(size=3000)
    a = m.bootstrap_ci(y, s, n_boot=100, seed=4)
    b = m.bootstrap_ci(y, s, n_boot=100, seed=4)
    assert a == b
    assert a["low"] < m.pr_auc(y, s) < a["high"]
    assert a["n_valid"] == 100


def test_ks_shift_zero_for_equal_and_large_for_shifted():
    x = np.random.default_rng(0).normal(size=2000)
    assert m.ks_shift(x, x)["statistic"] == 0.0
    assert m.ks_shift(x, x + 3)["statistic"] > 0.8


def test_evaluate_scores_contains_expected_keys():
    cfg = {
        "evaluation": {
            "beta": 2.0,
            "recall_levels": [0.5],
            "alert_budgets": [0.25],
            "cost_fp": 1.0,
            "cost_fn": 10.0,
        }
    }
    out = m.evaluate_scores(Y, S, cfg, threshold=0.35)
    assert out["base_rate"] == 0.5 and out["n_fraud"] == 2
    assert set(out["recall_at_budget"]) == {"0.25"}
    assert out["at_fbeta_threshold"]["threshold"] == 0.35
