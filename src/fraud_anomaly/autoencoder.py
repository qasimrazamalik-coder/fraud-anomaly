"""Compact MLP autoencoder; the anomaly score is per-row reconstruction error."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
from torch import nn


class MLPAutoencoder(nn.Module):
    """Symmetric MLP with a narrow bottleneck.

    The bottleneck is what forces the network to learn the structure of normal traffic
    instead of copying its input, so fraud that breaks that structure reconstructs badly.
    """

    def __init__(self, input_dim: int, hidden_dims: list[int], bottleneck: int) -> None:
        super().__init__()
        enc: list[nn.Module] = []
        prev = input_dim
        for width in hidden_dims:
            enc += [nn.Linear(prev, width), nn.ReLU()]
            prev = width
        enc.append(nn.Linear(prev, bottleneck))
        dec: list[nn.Module] = []
        prev = bottleneck
        for width in reversed(hidden_dims):
            dec += [nn.Linear(prev, width), nn.ReLU()]
            prev = width
        dec.append(nn.Linear(prev, input_dim))
        self.encoder = nn.Sequential(*enc)
        self.decoder = nn.Sequential(*dec)
        self.spec = {
            "input_dim": input_dim,
            "hidden_dims": list(hidden_dims),
            "bottleneck": bottleneck,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def reconstruction_error(
    model: MLPAutoencoder, x: np.ndarray, batch_size: int = 4096
) -> np.ndarray:
    """Mean squared reconstruction error per row. Higher means more anomalous."""
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.from_numpy(np.ascontiguousarray(x[start : start + batch_size]))
            out.append(((model(batch) - batch) ** 2).mean(dim=1).numpy())
    return np.concatenate(out).astype(np.float64)


def fit_autoencoder(
    x_train: np.ndarray, x_val: np.ndarray, cfg: dict[str, Any], seed: int
) -> tuple[MLPAutoencoder, dict[str, Any]]:
    """Train on normal rows, stop on validation reconstruction loss of normal rows.

    Both inputs must contain normal transactions only. With `denoising_std` > 0 the training
    inputs get Gaussian noise but the target stays clean (denoising variant); validation is
    always clean so the stopping criterion is comparable across variants.
    Returns the best-epoch model and a history dict.
    """
    if len(x_train) == 0 or len(x_val) == 0:
        raise ValueError("Autoencoder needs non-empty normal train and validation sets.")
    torch.set_num_threads(int(cfg.get("num_threads", 2)))
    torch.manual_seed(seed)
    model = MLPAutoencoder(x_train.shape[1], cfg["hidden_dims"], int(cfg["bottleneck"]))
    opt = torch.optim.Adam(
        model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg.get("weight_decay", 0.0))
    )
    loss_fn = nn.MSELoss()
    noise = float(cfg.get("denoising_std", 0.0))
    batch = int(cfg["batch_size"])
    train_t = torch.from_numpy(np.ascontiguousarray(x_train))
    gen = torch.Generator().manual_seed(seed)

    best_state = copy.deepcopy(model.state_dict())
    best_val, best_epoch, stale = float("inf"), 0, 0
    history: dict[str, Any] = {"train_loss": [], "val_loss": []}
    for epoch in range(1, int(cfg["max_epochs"]) + 1):
        model.train()
        perm = torch.randperm(len(train_t), generator=gen)
        total = 0.0
        for start in range(0, len(perm), batch):
            target = train_t[perm[start : start + batch]]
            inp = target + noise * torch.randn(target.shape, generator=gen) if noise > 0 else target
            opt.zero_grad()
            loss = loss_fn(model(inp), target)
            loss.backward()
            opt.step()
            total += loss.item() * len(target)
        val_loss = float(reconstruction_error(model, x_val).mean())
        history["train_loss"].append(total / len(train_t))
        history["val_loss"].append(val_loss)
        if val_loss < best_val - 1e-9:
            best_val, best_epoch, stale = val_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= int(cfg["patience"]):
                break
    model.load_state_dict(best_state)
    history.update(best_epoch=best_epoch, best_val_loss=best_val, epochs_run=epoch)
    return model, history
