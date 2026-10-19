import numpy as np
import pytest
import torch

from fraud_anomaly.autoencoder import MLPAutoencoder, fit_autoencoder, reconstruction_error
from fraud_anomaly.detectors import (
    fit_hist_gradient_boosting,
    fit_isolation_forest,
    fit_logistic_regression,
    isolation_scores,
    positive_probability,
)

AE_CFG = {
    "hidden_dims": [8],
    "bottleneck": 3,
    "lr": 0.01,
    "batch_size": 64,
    "max_epochs": 2,
    "patience": 5,
    "num_threads": 1,
}


def toy_normal(n=600, dim=10, seed=0):
    """SYNTHETIC low-rank normals so a small bottleneck can reconstruct them."""
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(n, 3))
    mix = rng.normal(size=(3, dim))
    return (latent @ mix + 0.05 * rng.normal(size=(n, dim))).astype(np.float32)


def test_autoencoder_trains_two_epochs_and_lowers_loss():
    x = toy_normal(800, seed=0)  # one mixing matrix, split into train and validation
    x_train, x_val = x[:600], x[600:]
    model, hist = fit_autoencoder(x_train, x_val, AE_CFG, seed=0)
    assert hist["epochs_run"] == 2
    assert hist["train_loss"][1] < hist["train_loss"][0]
    untrained = MLPAutoencoder(10, [8], 3)
    assert reconstruction_error(model, x_val).mean() < reconstruction_error(untrained, x_val).mean()


def test_autoencoder_is_seeded():
    x = toy_normal(400)
    a, _ = fit_autoencoder(x[:300], x[300:], AE_CFG, seed=3)
    b, _ = fit_autoencoder(x[:300], x[300:], AE_CFG, seed=3)
    assert np.allclose(reconstruction_error(a, x), reconstruction_error(b, x))


def test_early_stopping_restores_best_epoch():
    x = toy_normal(400)
    cfg = {**AE_CFG, "max_epochs": 30, "patience": 1, "lr": 0.5}  # unstable lr: stops early
    model, hist = fit_autoencoder(x[:300], x[300:], cfg, seed=0)
    assert hist["epochs_run"] < 30
    assert reconstruction_error(model, x[300:]).mean() == pytest.approx(
        hist["best_val_loss"], rel=1e-4
    )


def test_denoising_variant_trains():
    x = toy_normal(400)
    _, hist = fit_autoencoder(x[:300], x[300:], {**AE_CFG, "denoising_std": 0.1}, seed=0)
    assert np.isfinite(hist["val_loss"]).all()


def test_autoencoder_rejects_empty_input():
    with pytest.raises(ValueError, match="non-empty"):
        fit_autoencoder(np.zeros((0, 10), dtype=np.float32), toy_normal(), AE_CFG, seed=0)


def test_reconstruction_error_is_higher_for_off_manifold_rows():
    x = toy_normal(800)
    model, _ = fit_autoencoder(x[:600], x[600:], {**AE_CFG, "max_epochs": 25}, seed=0)
    off = np.random.default_rng(9).normal(0, 6, size=(100, 10)).astype(np.float32)
    assert reconstruction_error(model, off).mean() > reconstruction_error(model, x[600:]).mean()
    assert torch.is_tensor(next(model.parameters()))


def test_isolation_forest_higher_means_more_anomalous():
    rng = np.random.default_rng(0)
    normal = rng.normal(0, 1, size=(500, 5))
    outliers = rng.normal(0, 1, size=(20, 5)) + 8
    forest = fit_isolation_forest(normal, {"n_estimators": 50, "max_samples": 128, "n_jobs": 1}, 0)
    assert isolation_scores(forest, outliers).mean() > isolation_scores(forest, normal).mean()
    # sklearn's own convention is the reverse; the wrapper must flip it
    assert forest.score_samples(outliers).mean() < forest.score_samples(normal).mean()


def test_supervised_references_return_probabilities(synth):
    from fraud_anomaly.data import FeatureScaler

    scaler = FeatureScaler().fit(synth)
    x, y = scaler.transform(synth), synth["Class"].to_numpy()
    lr = fit_logistic_regression(x, y, {"C": 1.0, "max_iter": 100}, 0)
    hgb = fit_hist_gradient_boosting(
        x, y, {"max_iter": 10, "learning_rate": 0.2, "max_leaf_nodes": 7}, 0
    )
    for model in (lr, hgb):
        p = positive_probability(model, x)
        assert p.shape == (len(x),) and p.min() >= 0 and p.max() <= 1
        assert p[y == 1].mean() > p[y == 0].mean()
