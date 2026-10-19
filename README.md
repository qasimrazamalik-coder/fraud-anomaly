# fraud-anomaly

Credit-card fraud detection under extreme class imbalance: an autoencoder and an Isolation Forest, both fitted on normal transactions only, compared with two supervised references on a strictly time-ordered split, scored with PR-AUC, recall at a fixed alert budget and an illustrative cost view.

## Results

Real-data results only. Nothing here is typed by hand: `make eval` rewrites the block between the markers from `results/run.json`, and refuses to do so for synthetic-fixture runs.

<!-- RESULTS:START -->
| Model | Setting | Test base rate | PR-AUC (95% CI) | ROC-AUC | Recall @ top 0.1% flagged | Recall @ top 1% flagged |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Autoencoder | unsupervised | not yet run | not yet run | not yet run | not yet run | not yet run |
| Isolation Forest | unsupervised | not yet run | not yet run | not yet run | not yet run | not yet run |
| Logistic regression | supervised reference | not yet run | not yet run | not yet run | not yet run | not yet run |
| Hist. gradient boosting | supervised reference | not yet run | not yet run | not yet run | not yet run | not yet run |

Fill it with: `make data && make train && make eval` (data step: see Quickstart).
<!-- RESULTS:END -->

`make eval` also writes `results/run.md` (precision at fixed recall, the F-beta and cost views, drift table) and three figures: `results/pr_curves.png`, `results/recall_vs_alert_budget.png`, `results/score_drift.png`. `make smoke` runs the same code on a **synthetic** fixture; those numbers only prove the pipeline runs and are not reported anywhere.

## Demo

Screenshot added after first run: `docs/demo.png`.

## What question this answers

If you can only train on normal traffic, how much of the fraud do you still catch when you review a fixed number of alerts per day, and how far behind a model that was given the fraud labels? The comparison is made on a time-ordered split, because a deployed detector only ever scores the future. PR-AUC is the headline and is always shown next to the base rate; ROC-AUC is reported but not trusted on its own.

## Quickstart

```bash
git clone https://github.com/qasimrazamalik-coder/fraud-anomaly.git
cd fraud-anomaly
python -m venv .venv && source .venv/bin/activate
make install       # package and dev tools (installs PyTorch from PyPI)
make test          # unit tests on CPU, about a minute
make smoke         # end-to-end run on a SYNTHETIC fixture (not a result)
# get the data (Kaggle login needed, see "Data" below), then:
python scripts/download_data.py --csv path/to/creditcard.csv
make train         # writes results/run.json and models/default/
make eval          # writes results/run.md, three figures, and updates the table above
make app           # Gradio demo at http://127.0.0.1:7860
```

If `make install` pulls a PyTorch build that does not suit your machine, install torch first from <https://pytorch.org/get-started/locally/>.

## Method

### Data

Kaggle "Credit Card Fraud Detection": 284,807 card transactions made by European cardholders over two days, 492 of them fraud (about 0.17%), collected by Worldline and the Machine Learning Group of ULB. Columns are `Time` (seconds since the first transaction), `V1` to `V28`, `Amount` and `Class`. **V1 to V28 are PCA components of undisclosed original features, so no domain interpretation of them is possible**; this project does not attempt one. Kaggle requires a login, so the script does not download anything on its own; see the docstring of `scripts/download_data.py` for the browser and `kaggle datasets download` routes. The licence is not restated here: check the dataset page before redistributing or publishing results, and follow its citation request.

### Protocol

- **Time-based split.** Rows are ordered by `Time`: the earliest 60% train, the next 20% validation, the latest 20% test (configurable). A random split lets the model train on transactions from the same hours it is tested on, which leaks temporal structure and inflates scores; it also hides drift. Window boundaries never cut through tied timestamps.
- **Unsupervised models** (autoencoder, Isolation Forest) are fitted on the **normal** training transactions only. Labels are used for nothing else there except filtering normals and, later, choosing a threshold and scoring.
- **Supervised references** (logistic regression with balanced class weights, histogram gradient boosting with balanced class weights) use the labelled training window. They solve an easier, different problem and act as a ceiling, not as a like-for-like rival.
- **Scaling.** `log1p(Amount)` then z-score, statistics from the training window only. `Time` is not a model input.
- **Thresholds are chosen on validation**, never on test: the F-beta-optimal threshold (default beta 2, so recall counts more than precision) and a cost-minimising threshold. The test window is touched only to report.
- **The one thing that keeps the comparison fair:** every model sees the same three windows, the same scaler and the same thresholding rule, and the test window is scored once.

### Models

- **Autoencoder** (PyTorch): MLP 29 -> 20 -> 12 -> 6 -> 12 -> 20 -> 29 by default, trained with MSE on normal rows. Score is per-row mean squared reconstruction error. Early stopping on the reconstruction loss of *normal validation* rows; the best epoch is restored. Seeded and config-driven. `autoencoder.denoising_std > 0` switches on a denoising variant (Gaussian noise on the inputs, clean target); it is off by default and its effect has not been measured.
- **Isolation Forest** (scikit-learn): score is the negated `score_samples`, so higher means more anomalous for every model. A unit test pins that sign.
- A one-class SVM is not included: on 170k training rows it is not cheap.

### Metrics

- **PR-AUC (average precision)** is the headline, with a 95% percentile bootstrap interval over test transactions (resamples without a fraud are skipped). A random scorer's PR-AUC equals the base rate, which is why the base rate is in every table.
- **ROC-AUC** is reported but optimistic under imbalance: with hundreds of thousands of negatives, even a large number of false alarms keeps the false-positive rate tiny.
- **Precision at fixed recall** (0.5, 0.7, 0.9): best precision among operating points reaching that recall.
- **Recall at an alert budget**: recall when only the top 0.1% and top 1% of test transactions by score are flagged.
- **F-beta at the validation-chosen threshold**, with precision and recall.
- **Cost view.** Cost = FP x `cost_fp` + FN x `cost_fn`, reported per 1,000 transactions next to the cost of flagging nothing. The defaults (5 and 100) are illustrative parameters, not real business costs; change them in `configs/default.yaml`.
- **Drift note.** A two-sample Kolmogorov-Smirnov statistic between validation and test scores, for all rows and for normal rows only (the latter isolates score drift from a change in fraud mix). A larger statistic means thresholds learned on validation transfer less well. `results/score_drift.png` shows the distributions.

### Results JSON

`results/run.json` holds the seed, package versions, full config, window sizes and time ranges, the autoencoder training history, and per model the validation thresholds, test metrics and drift statistics. Scores for plotting go to `results/run_scores.npz`.

## Repo structure

```
configs/    default.yaml (real data), smoke.yaml (synthetic fixture)
src/fraud_anomaly/
  data.py        schema check, time split, train-only scaler, synthetic fixture
  autoencoder.py detectors.py   the four models
  metrics.py     imbalance metrics, thresholds, cost, bootstrap, KS
  train.py evaluate.py inference.py utils.py
app/        Gradio demo and theme
scripts/    download_data.py
tests/      split, scaler, metrics, models, pipeline, app
results/    outputs from real runs
```

## App

`python app/app.py` accepts a `.csv` up to 20 MB with columns `V1`..`V28` and `Amount` (`Time` and `Class` optional). It scores with the trained autoencoder and Isolation Forest, shows the validation-chosen thresholds, the top 25 rows by score, and precision and recall only when the file has a `Class` column. With no trained models it shows an empty state telling you what to run. Models load from `models/default` (set `FRAUD_MODEL_DIR` to change). The Isolation Forest is stored with joblib, which uses pickle: only load model folders you trained yourself.

## Limitations

- **Anonymised features.** V1 to V28 cannot be interpreted, so there is no way to check that a flagged row looks like fraud for a sensible reason.
- **Two days of data** from one set of cardholders. Daily and weekly cycles, seasonality and new fraud patterns are not represented, and a 20% test window covers well under a day.
- **Few frauds.** 492 in total leaves on the order of a hundred in the test window (the exact count is printed by `make train`); PR-AUC intervals are wide and small differences between models are not meaningful.
- **Concept drift.** Fraud adapts. Thresholds chosen on validation degrade on later data, which the KS table is meant to make visible; this project does not implement recalibration.
- **Label delay.** Real fraud labels arrive days or weeks late (chargebacks), so a clean "normal only" training set and a fresh validation window with trustworthy labels are optimistic assumptions here.
- **Age.** The data was collected in 2013 and has been public for years; current fraud tactics, card technology and anti-fraud controls differ.
- The cost parameters are illustrative. The alert-budget numbers assume analysts can review exactly that share of transactions.
- GPU runs are not bit-for-bit deterministic even with fixed seeds.
- The app is a research demo on anonymised public data. It is not a fraud-prevention system and should not be used to act on real customers or transactions.
- The synthetic fixture (Gaussian mixtures, a shifted rare cluster, a slow drift over time) is for tests and smoke runs only. Results on it say nothing about real data.

## Running on free Colab or Kaggle

1. New notebook; a GPU is optional. The models are small and CPU is enough.
2. `!git clone https://github.com/qasimrazamalik-coder/fraud-anomaly.git && cd fraud-anomaly && pip install -e ".[dev]"`
3. Upload `creditcard.csv` (or use the Kaggle dataset attached to a Kaggle notebook), then `!cd fraud-anomaly && python scripts/download_data.py --csv /path/to/creditcard.csv && make train && make eval`.
4. Download `results/run.json`, `results/run.md` and the PNGs; copy the PNGs to `docs/`.

Runtime is approximate and depends on the hardware: expect minutes to a few tens of minutes on a laptop CPU, dominated by the autoencoder epochs and the 500 bootstrap resamples per model. Lower `max_epochs` or `n_bootstrap` in the config to shorten it.

## License and citation

MIT. See `LICENSE`. The dataset keeps its own licence and citation requirements (see the Kaggle dataset page). References: Liu, Ting and Zhou, "Isolation Forest", IEEE ICDM 2008; Saito and Rehmsmeier, "The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets", PLoS ONE 2015.
