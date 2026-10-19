"""Gradio app: upload transactions, score them with the trained detectors, inspect the top rows.

    python app/app.py

Research demo on anonymised public data. It reads models from models/default (override with
the FRAUD_MODEL_DIR environment variable) and shows an empty state when none exist.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")  # no telemetry calls

import gradio as gr
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fraud_anomaly.data import (
    FEATURE_COLUMNS,
    LABEL_COLUMN,
    TIME_COLUMN,
    SchemaError,
    validate_schema,
)
from fraud_anomaly.inference import ModelBundle, ModelNotFoundError, load_bundle
from fraud_anomaly.metrics import pr_auc
from theme import CSS, build_theme

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(os.environ.get("FRAUD_MODEL_DIR", ROOT / "models" / "default"))
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
TOP_ROWS = 25
LABELS = {"autoencoder": "Autoencoder", "isolation_forest": "Isolation Forest"}
NOTICE = (
    "Research demo on anonymised public data. It is not a fraud-prevention system and must "
    "not be used to block, flag or investigate real customers or transactions."
)
EMPTY_MESSAGE = (
    f"No trained models found in `{MODEL_DIR}`. Download the data with "
    "`python scripts/download_data.py`, then run `make train`. "
    "`make smoke` trains on a synthetic fixture only to check the pipeline."
)


class InputError(ValueError):
    """A problem with user input, worded for display."""


def check_upload(path: str | None) -> Path:
    """Validate extension and size of an uploaded file before reading it."""
    if not path:
        raise InputError("No file uploaded.")
    file = Path(path)
    if file.suffix.lower() != ".csv":
        raise InputError(f"Only .csv files are accepted (got '{file.suffix or 'no extension'}').")
    size = os.path.getsize(file)
    if size == 0:
        raise InputError("The file is empty.")
    if size > MAX_UPLOAD_BYTES:
        raise InputError(f"File is {size / 1e6:.1f} MB; the limit is 20 MB.")
    return file


def read_upload(path: str | None) -> pd.DataFrame:
    """Read and schema-check the CSV; every failure becomes an InputError."""
    file = check_upload(path)
    try:
        frame = pd.read_csv(file)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise InputError(f"Could not parse the CSV: {exc}") from exc
    try:
        validate_schema(frame)
    except SchemaError as exc:
        raise InputError(str(exc)) from exc
    return frame


def flagged_table(
    frame: pd.DataFrame, scores: dict[str, np.ndarray], bundle: ModelBundle, model: str
) -> tuple[pd.DataFrame, str]:
    """Top rows by the chosen model's score, with both scores and flag status."""
    order = np.argsort(-scores[model], kind="stable")
    above = scores[model][order] >= bundle.thresholds[model]
    n_flagged = int(above.sum())
    if n_flagged:
        note = f"{n_flagged} of {len(frame)} rows are at or above the threshold."
    else:
        note = "No rows reach the threshold; showing the highest scores anyway."
    top = order[:TOP_ROWS]
    table = pd.DataFrame({"CSV row": top + 2})  # +2: header line and 1-based numbering
    for name, label in LABELS.items():
        table[f"{label} score"] = scores[name][top]
        table[f"{label} flagged"] = np.where(
            scores[name][top] >= bundle.thresholds[name], "yes", "no"
        )
    table["Amount"] = frame["Amount"].to_numpy()[top]
    for col in (TIME_COLUMN, LABEL_COLUMN):
        if col in frame.columns:
            table[col] = frame[col].to_numpy()[top]
    return table, note


def threshold_frame(scores: dict[str, np.ndarray], bundle: ModelBundle) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Model": [LABELS[m] for m in LABELS],
            "Threshold (chosen on validation)": [bundle.thresholds[m] for m in LABELS],
            "Rows flagged": [int((scores[m] >= bundle.thresholds[m]).sum()) for m in LABELS],
            "Rows scored": [len(scores[m]) for m in LABELS],
        }
    )


def readout_frame(
    frame: pd.DataFrame, scores: dict[str, np.ndarray], bundle: ModelBundle
) -> pd.DataFrame | None:
    """Precision and recall at the stored thresholds; only when the CSV has a Class column."""
    if LABEL_COLUMN not in frame.columns:
        return None
    y = frame[LABEL_COLUMN].to_numpy().astype(int)
    rows = []
    for name, label in LABELS.items():
        flagged = scores[name] >= bundle.thresholds[name]
        tp = int(np.sum(flagged & (y == 1)))
        has_both = 0 < y.sum() < len(y)
        rows.append(
            {
                "Model": label,
                "Base rate in file": f"{y.mean():.4f}",
                "Precision": f"{tp / flagged.sum():.3f}" if flagged.sum() else "n/a",
                "Recall": f"{tp / y.sum():.3f}" if y.sum() else "n/a",
                "PR-AUC": f"{pr_auc(y, scores[name]):.3f}" if has_both else "n/a",
            }
        )
    return pd.DataFrame(rows)


def score_upload(
    path: str | None, model: str = "autoencoder", model_dir: Path | None = None
) -> tuple[str, pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
    """Validate, score and shape results. Returns (status, thresholds, top rows, readout)."""
    try:
        bundle = load_bundle(model_dir or MODEL_DIR)
    except ModelNotFoundError:
        return EMPTY_MESSAGE, None, None, None
    if model not in LABELS:
        return "Choose one of the two models.", None, None, None
    try:
        frame = read_upload(path)
    except InputError as exc:
        return f"Cannot score: {exc}", None, None, None
    scores = bundle.score(frame)
    top, note = flagged_table(frame, scores, bundle, model)
    banner = (
        "These models were trained on the SYNTHETIC fixture; scores mean nothing. "
        if bundle.synthetic
        else ""
    )
    status = f"{banner}Scored {len(frame)} rows. {note} Table sorted by {LABELS[model]} score."
    return status, threshold_frame(scores, bundle), top, readout_frame(frame, scores, bundle)


def build_app() -> gr.Blocks:
    with gr.Blocks(title="fraud-anomaly") as demo:
        gr.Markdown(
            "# fraud-anomaly\nAutoencoder and Isolation Forest scores for a CSV of transactions."
        )
        gr.Markdown(NOTICE, elem_classes="notice")
        try:
            load_bundle(MODEL_DIR)
            first = "Upload a CSV and press Score."
        except ModelNotFoundError:
            first = EMPTY_MESSAGE
        status = gr.Markdown(first)
        with gr.Row():
            with gr.Column(scale=2):
                upload = gr.File(label="CSV file (max 20 MB)", file_types=[".csv"], type="filepath")
                gr.Markdown(
                    f"Expected columns: `{FEATURE_COLUMNS[0]}` to `{FEATURE_COLUMNS[-2]}` and "
                    f"`Amount`. `{TIME_COLUMN}` and `{LABEL_COLUMN}` are optional; with "
                    f"`{LABEL_COLUMN}` a precision and recall readout is added."
                )
            with gr.Column(scale=1):
                model = gr.Radio(
                    [(label, key) for key, label in LABELS.items()],
                    value="autoencoder",
                    label="Rank rows by",
                )
                run = gr.Button("Score", variant="primary")
        gr.Markdown("### Thresholds used")
        thresholds = gr.Dataframe(interactive=False, wrap=True)
        gr.Markdown(f"### Top {TOP_ROWS} rows by score")
        top = gr.Dataframe(interactive=False, wrap=True)
        gr.Markdown("### Precision and recall (only when the CSV has a Class column)")
        readout = gr.Dataframe(interactive=False, wrap=True)

        with gr.Accordion("How it works", open=False):
            gr.Markdown(
                "Both models were fitted on normal transactions only. The autoencoder compresses "
                "the 29 inputs through a narrow bottleneck; the score is the mean squared "
                "reconstruction error. The Isolation Forest scores how quickly a row is isolated "
                "by random splits. Higher means more anomalous for both. Each threshold was "
                "chosen on a validation window, not on your data, so it may flag too many or too "
                "few rows on a different time period."
            )
        with gr.Accordion("Limitations", open=False):
            gr.Markdown(
                "V1 to V28 are anonymised PCA components, so no score can be explained in "
                "business terms. The models learned from two days of 2013 card transactions from "
                "one dataset and will not transfer to other data without retraining. Scores are "
                "not probabilities. Precision depends on the share of fraud in the file."
            )

        run.click(score_upload, [upload, model], [status, thresholds, top, readout])
    return demo


if __name__ == "__main__":
    build_app().launch(theme=build_theme(), css=CSS)
