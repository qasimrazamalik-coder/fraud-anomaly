"""Render results JSON into markdown tables and figures, optionally into the README.

    python -m fraud_anomaly.evaluate --config configs/default.yaml --update-readme

The README block between the RESULTS markers is only ever written from a real-data run;
results from the synthetic fixture are refused.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_curve

from fraud_anomaly.utils import load_config

START = "<!-- RESULTS:START -->"
END = "<!-- RESULTS:END -->"
LABELS = {
    "autoencoder": "Autoencoder",
    "isolation_forest": "Isolation Forest",
    "logistic_regression": "Logistic regression",
    "hist_gradient_boosting": "Hist. gradient boosting",
}
# Restrained palette: two warm/cool accents for the unsupervised models, greys for the
# supervised references (dashed), so the problem setting is visible without a legend lookup.
STYLE = {
    "autoencoder": {"color": "#a8441f", "linestyle": "-"},
    "isolation_forest": {"color": "#3d5a6c", "linestyle": "-"},
    "logistic_regression": {"color": "#8a8478", "linestyle": "--"},
    "hist_gradient_boosting": {"color": "#1d1b18", "linestyle": "--"},
}


def _num(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _pct(frac: str) -> str:
    return f"{float(frac) * 100:g}%"


def headline_table(results: dict[str, Any]) -> str:
    """PR-AUC with bootstrap CI, ROC-AUC and recall at the alert budgets, base rate included."""
    budgets = list(next(iter(results["models"].values()))["test"]["recall_at_budget"])
    head = "| Model | Setting | Test base rate | PR-AUC (95% CI) | ROC-AUC |"
    head += "".join(f" Recall @ top {_pct(b)} flagged |" for b in budgets)
    lines = [head, "| --- | --- | ---: | ---: | ---: |" + " ---: |" * len(budgets)]
    for name, info in results["models"].items():
        t = info["test"]
        ci = t["pr_auc_ci"]
        row = (
            f"| {LABELS[name]} | {info['setting']} | {t['base_rate']:.4f} | "
            f"{t['pr_auc']:.3f} ({ci['low']:.3f} to {ci['high']:.3f}) | {t['roc_auc']:.3f} |"
        )
        row += "".join(f" {t['recall_at_budget'][b]:.3f} |" for b in budgets)
        lines.append(row)
    return "\n".join(lines)


def threshold_table(results: dict[str, Any]) -> str:
    """Test results at the F-beta threshold fixed on validation, plus the illustrative cost."""
    beta = results["meta"]["config"]["evaluation"]["beta"]
    lines = [
        f"| Model | Test base rate | Threshold (validation) | Flagged | Precision | Recall "
        f"| F{beta:g} | Cost per 1,000 | Cost per 1,000 if nothing flagged |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, info in results["models"].items():
        t = info["test"]
        r = t["at_fbeta_threshold"]
        lines.append(
            f"| {LABELS[name]} | {t['base_rate']:.4f} | {r['threshold']:.4g} | {r['flagged']} | "
            f"{_num(r['precision'])} | {r['recall']:.3f} | {r['fbeta']:.3f} | "
            f"{r['cost_per_1000']:.2f} | {r['cost_flag_nothing_per_1000']:.2f} |"
        )
    return "\n".join(lines)


def precision_table(results: dict[str, Any]) -> str:
    """Best attainable precision at fixed recall levels (test)."""
    levels = list(next(iter(results["models"].values()))["test"]["precision_at_recall"])
    head = "| Model | Test base rate |" + "".join(f" Precision @ recall >= {x} |" for x in levels)
    lines = [head, "| --- | ---: |" + " ---: |" * len(levels)]
    for name, info in results["models"].items():
        t = info["test"]
        cells = "".join(f" {t['precision_at_recall'][x]:.3f} |" for x in levels)
        lines.append(f"| {LABELS[name]} | {t['base_rate']:.4f} |{cells}")
    return "\n".join(lines)


def cost_table(results: dict[str, Any]) -> str:
    """Test cost when the threshold is chosen to minimise cost on validation."""
    ev = results["meta"]["config"]["evaluation"]
    lines = [
        f"Illustrative parameters: false positive costs {ev['cost_fp']:g}, false negative costs "
        f"{ev['cost_fn']:g} (arbitrary units, not real business costs).",
        "",
        "| Model | Flagged | Precision | Recall | Cost per 1,000 | Cost if nothing flagged |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, info in results["models"].items():
        r = info["test"]["at_cost_threshold"]
        lines.append(
            f"| {LABELS[name]} | {r['flagged']} | {_num(r['precision'])} | {r['recall']:.3f} | "
            f"{r['cost_per_1000']:.2f} | {r['cost_flag_nothing_per_1000']:.2f} |"
        )
    return "\n".join(lines)


def drift_table(results: dict[str, Any]) -> str:
    """KS statistic between validation and test score distributions."""
    lines = [
        "| Model | KS, all rows | KS, normal rows only | p-value (normal rows) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, info in results["models"].items():
        d = info["drift"]
        lines.append(
            f"| {LABELS[name]} | {d['all_rows']['statistic']:.3f} | "
            f"{d['normal_rows_only']['statistic']:.3f} | {d['normal_rows_only']['p_value']:.2g} |"
        )
    return "\n".join(lines)


def windows_line(results: dict[str, Any]) -> str:
    w = results["meta"]["windows"]
    parts = [
        f"{n}: {w[n]['n']:,} rows, {w[n]['n_fraud']} frauds ({w[n]['base_rate']:.4f})"
        for n in ("train", "val", "test")
    ]
    return "Windows (time ordered) - " + "; ".join(parts) + "."


def readme_block(results: dict[str, Any]) -> str:
    """The compact tables that go into the README."""
    return (
        f"{windows_line(results)}\n\n"
        "Threshold-free metrics on the test window. The unsupervised models never see fraud "
        "labels when fitting; the supervised rows are a different problem setting.\n\n"
        f"{headline_table(results)}\n\n"
        "At the F-beta threshold chosen on the validation window (never on test), with the "
        "illustrative cost view:\n\n"
        f"{threshold_table(results)}"
    )


def render_markdown(results: dict[str, Any]) -> str:
    meta = results["meta"]
    banner = (
        "> SYNTHETIC fixture. These numbers test the pipeline and are not results.\n\n"
        if meta["synthetic"]
        else ""
    )
    return (
        f"{banner}# Results: {meta['dataset']}\n\n"
        f"Seed {meta['seed']}, created {meta['created_utc']}. {windows_line(results)}\n\n"
        f"## Headline metrics (test)\n\n{headline_table(results)}\n\n"
        "ROC-AUC is optimistic under extreme imbalance; read PR-AUC against the base rate.\n\n"
        f"## Precision at fixed recall (test)\n\n{precision_table(results)}\n\n"
        f"## F-beta threshold chosen on validation (test)\n\n{threshold_table(results)}\n\n"
        f"## Cost-sensitive view (test)\n\n{cost_table(results)}\n\n"
        f"## Score drift, validation vs test\n\n{drift_table(results)}\n"
    )


def update_readme(readme: str | Path, block: str, synthetic: bool) -> None:
    """Replace the text between the RESULTS markers; refuse synthetic results."""
    if synthetic:
        raise ValueError("Refusing to write SYNTHETIC fixture numbers into the README.")
    path = Path(readme)
    text = path.read_text(encoding="utf-8")
    if START not in text or END not in text or text.index(START) > text.index(END):
        raise ValueError(f"{path} must contain {START} before {END}")
    before, rest = text.split(START, 1)
    _, after = rest.split(END, 1)
    path.write_text(f"{before}{START}\n{block}\n{END}{after}", encoding="utf-8")


def _style_axes(ax: Any) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(alpha=0.25, linewidth=0.6)


def plot_pr_curves(scores: Any, results: dict[str, Any], path: Path) -> None:
    """Test PR curves; the dotted line is the base rate (a random scorer)."""
    y = scores["y_test"]
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for name in results["models"]:
        p, r, _ = precision_recall_curve(y, scores[f"{name}__test"])
        ax.plot(r, p, label=LABELS[name], linewidth=1.6, **STYLE[name])
    ax.axhline(y.mean(), color="#b5afa3", linestyle=":", linewidth=1.2, label="Base rate")
    ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02))
    ax.set_title("Precision-recall, test window", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_budget_curves(scores: Any, results: dict[str, Any], path: Path) -> None:
    """Recall as a function of the fraction of transactions flagged (alert budget)."""
    y = scores["y_test"]
    n_pos = y.sum()
    fig, ax = plt.subplots(figsize=(6, 4.2))
    fractions = np.arange(1, len(y) + 1) / len(y)
    for name in results["models"]:
        order = np.argsort(-scores[f"{name}__test"], kind="stable")
        recall = np.cumsum(y[order]) / n_pos
        ax.plot(fractions, recall, label=LABELS[name], linewidth=1.6, **STYLE[name])
    for frac in results["meta"]["config"]["evaluation"]["alert_budgets"]:
        ax.axvline(frac, color="#b5afa3", linestyle=":", linewidth=1.0)
    ax.plot([0, 1], [0, 1], color="#b5afa3", linestyle=":", linewidth=1.2)
    ax.set(
        xscale="log",
        xlim=(fractions[0], 1),
        ylim=(0, 1.02),
        xlabel="Fraction of transactions flagged (log scale)",
        ylabel="Recall of frauds",
    )
    ax.set_title("Recall vs alert budget, test window", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_drift(scores: Any, results: dict[str, Any], path: Path) -> None:
    """Score distributions of NORMAL transactions in validation vs test, per unsupervised model."""
    names = [n for n, i in results["models"].items() if i["setting"] == "unsupervised"]
    fig, axes = plt.subplots(1, len(names), figsize=(5.2 * len(names), 3.8), squeeze=False)
    for ax, name in zip(axes[0], names, strict=True):
        val = scores[f"{name}__val"][scores["y_val"] == 0]
        test = scores[f"{name}__test"][scores["y_test"] == 0]
        lo, hi = np.percentile(np.concatenate([val, test]), [0.5, 99.5])
        bins = np.linspace(lo, hi, 60)
        ax.hist(val, bins=bins, density=True, histtype="step", color="#3d5a6c", label="validation")
        ax.hist(test, bins=bins, density=True, histtype="step", color="#a8441f", label="test")
        ks = results["models"][name]["drift"]["normal_rows_only"]["statistic"]
        ax.set_title(f"{LABELS[name]} (KS {ks:.3f})", loc="left", fontsize=10)
        ax.set_xlabel("Anomaly score, normal rows only")
        ax.legend(frameon=False, fontsize=8)
        _style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--readme", default="README.md")
    parser.add_argument("--update-readme", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    out_dir, name = Path(cfg["output"]["dir"]), cfg["output"]["name"]
    json_path = out_dir / f"{name}.json"
    if not json_path.is_file():
        raise FileNotFoundError(f"{json_path} not found. Run `make train` first.")
    results = json.loads(json_path.read_text())
    scores = np.load(out_dir / f"{name}_scores.npz")

    (out_dir / f"{name}.md").write_text(render_markdown(results), encoding="utf-8")
    plot_pr_curves(scores, results, out_dir / "pr_curves.png")
    plot_budget_curves(scores, results, out_dir / "recall_vs_alert_budget.png")
    plot_drift(scores, results, out_dir / "score_drift.png")
    print(f"Wrote {out_dir / f'{name}.md'} and three figures in {out_dir}/")
    if args.update_readme:
        update_readme(args.readme, readme_block(results), results["meta"]["synthetic"])
        print(f"Updated {args.readme}")


if __name__ == "__main__":
    main()
