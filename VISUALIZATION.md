# Presentation charts

`visualization.py` creates six charts with a light navy/teal/coral theme:

- Confusion matrix: counts and percentages within each actual class, with recall,
  precision and false-positive rate highlighted.
- ROC curve: AUC, the saved operating point, and an optional FPR limit.
- Precision–recall curve: average precision, the operating point and prevalence
  baseline. This helps explain performance when fraud is rare.
- Score distributions: normalized separately for fraud and non-fraud.
- Calibration: observed fraud rates versus scores in equal-frequency bins.
- Cumulative gains: fraud captured when reviewing the highest-scoring applications.

Every figure has a fixed 16:9 layout. Defaults are 2560 × 1440 PNGs and SVGs;
PDF is also supported via `formats=("png", "svg", "pdf")`. Use PNG for easy slide
insertion and SVG for scalable graphics. The output includes `metrics.json`.
An existing output directory is reused and matching filenames are replaced.

## After validation

Both `train.py` and `tune.py` now generate charts automatically after evaluation.
They reuse the same prediction scores and selected threshold as the logged metrics.
MLflow artifacts appear under **charts/validation** in the corresponding run.
Local copies are under `<training output>/charts/<run ID>/validation/`.

To skip the export overhead during a large search, pass `--no-charts`. You can
generate charts for the winning saved model afterward. Existing Optuna studies
check source hashes, so these changes require a new study name when tuning.

With predictions you already have:

```python
from visualization import create_evaluation_report

report = create_evaluation_report(
    y_val,
    fraud_scores,                 # model.predict_proba(X_val)[:, 1]
    threshold=saved_threshold,     # the threshold selected during validation
    dataset_name="Validation",
    model_name="CatBoost v1",
    output_dir="work/reports/validation",
    fpr_limit=0.05,                # optional reference line; does not tune anything
)
```

For MLflow logging, call inside your active run with `log_to_mlflow=True`:

```python
import mlflow
from train import setup_mlflow

setup_mlflow(experiment="fraud-reports")
with mlflow.start_run(run_name="validation-charts"):
    report = create_evaluation_report(
        y_val, fraud_scores, threshold=saved_threshold,
        output_dir="work/reports/validation", log_to_mlflow=True,
    )
```

The standalone function logs the chart files and `metrics.json`. Training already
logs its native `validation_*` MLflow metrics. The function requires an active run
when logging is requested; it never starts an implicit run or changes your server.
To attach charts to an existing run, select that run's experiment before resuming:

```python
existing = mlflow.get_run(existing_run_id)
mlflow.set_experiment(experiment_id=existing.info.experiment_id)
with mlflow.start_run(run_id=existing_run_id):
    # Call create_evaluation_report(..., log_to_mlflow=True) here.
    pass
```

## An already trained model

From the project directory, supply the saved model bundle:

```powershell
python report.py --bundle work/tuning/catboost-gpu-v2/118401d16e4e444d8b21bdc17d04bd01 --data data/splits/val.csv --output work/reports/validation
```

This command loads the model, checks that preprocessing matches the saved bundle,
scores the requested labelled CSV, and uses `training_config.json`'s saved threshold.
It does not fit the model or select a new threshold. `provenance.json` records the
model and dataset hashes. Add `--log-mlflow` to log to a new report run, or add both
`--log-mlflow --run-id <existing-run-id>` to attach to a particular run. The
`--tracking-uri` and `--experiment` options configure the destination.

The same command works for one later labelled batch by changing `--data`, `--name`
and `--output`. For delayed-label monitoring, prefer the original served scores
over rescoring an old batch with a newer model.

## The 12 monthly sets

Keep the saved validation threshold fixed when evaluating one model over time.
Pass months in chronological insertion order; names are not sorted alphabetically.
The helper accepts any number of months, so it also works incrementally.

```python
import json
from pathlib import Path
import mlflow
import pandas as pd
from train import setup_mlflow
from visualization import EvaluationReport, create_monthly_reports

bundle = Path("work/tuning/catboost-gpu-v2/118401d16e4e444d8b21bdc17d04bd01")
config = json.loads((bundle / "training_config.json").read_text())
monthly_sets = {}
for month in range(1, 13):
    # Future input files: labels joined to the original served scores by source_id.
    batch = pd.read_csv(f"data/monthly_scored/month_{month:02d}.csv")
    monthly_sets[f"Month {month:02d}"] = (
        batch["fraud_bool"], batch["fraud_score"],
    )

setup_mlflow(experiment="fraud-monthly-monitoring")
with mlflow.start_run(run_name="catboost-v1-months-01-12"):
    reports, table = create_monthly_reports(
        monthly_sets,
        threshold=config["threshold"],
        output_dir="work/reports/monthly",
        model_name="CatBoost v1",
        fpr_limit=(config.get("threshold_selection") or {}).get("max_fpr"),
        baseline=EvaluationReport.load("work/reports/validation"),  # optional
        log_to_mlflow=True,
    )
```

This produces all six charts for each month, a four-panel monthly comparison,
and `summary/monthly_metrics.csv`. MLflow stores artifacts under
`charts/month_01`, etc., plus `charts/monthly_summary`. Native `month_*` metric
histories use steps 1–12. The CSV maps steps to names, model versions, thresholds
and sample counts. A validation baseline should use the matching model and threshold.

If you score fresh monthly datasets directly, use the matching model and
`prepare_dataset(batch)` from `preprocessing.py`, then pass
`(labels, model.predict_proba(features)[:, 1])`. The example paths above are future
inputs; this reporting code does not create the monthly simulation datasets.

For reports produced by separate jobs, including changing model versions:

```python
from visualization import EvaluationReport, create_monthly_summary

reports = [EvaluationReport.load(f"work/reports/monthly/month_{m:02d}")
           for m in range(1, 13)]
table = create_monthly_summary(reports, output_dir="work/reports/monthly/summary")
```

The comparison records the model and threshold for each month and discloses when
thresholds differ. It never chooses thresholds from monthly labels. Evaluate only
labels available at the simulated reporting date, joined to their original scores
in the same row order. There is no automatic label-delay logic in these functions.

## Theme and metric details

Pass `theme=ChartTheme(accent="#087F8C", warning="#C65C42", ink="#172B46")`
to customize colors (`from visualization import ChartTheme`). Styling stays local
to the report and does not modify notebook chart defaults. Figures are rendered
without a GUI and are released after export.

Inputs must be nonempty aligned 0/1 labels and finite scores in [0, 1]. Use fraud
probabilities, not predicted classes or the full two-column probability matrix.
Pandas Series must also have matching indices. Predictions use `score >= threshold`.
The special `np.nextafter(1.0, np.inf)` threshold means no alerts, including at score 1.

One-class batches still produce reports. ROC AUC is undefined without both classes;
recall/AP need positive labels, and FPR needs negative labels. Undefined values are
JSON `null`, blank CSV cells and gaps in trend lines. Precision is recorded as zero
when no alerts are raised, consistent with training. Calibration is a diagnostic,
not a probability-calibration procedure. Top-review budgets break equal-score ties
in input order; they do not represent the fixed decision threshold's alert volume.
An FPR limit is a reference for future batches, not a guarantee.

Exports use [Matplotlib's figure saving API](https://matplotlib.org/stable/api/_as_gen/matplotlib.figure.Figure.savefig.html)
and [MLflow's artifact logging API](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.html#mlflow.log_artifact).

## Charts of the simulated year (after all runs)

`simulation/plot_results.py` makes every result chart from saved result folders
only (the output of `simulation.summarize`). It needs no API, MLflow or model, so
the runs can finish first and the charts are made afterwards, as often as needed.
Copy each run's summarize output to `docs/results/all_labels` and
`docs/results/realistic_labels/<name>`, then:

```powershell
python -m simulation.plot_results                      # all folders in docs/results
python -m simulation.plot_results --run all_labels=docs/results/all_labels `
    --run audit_a=docs/results/realistic_labels/audit_sample_a `
    --compare all_labels:system,audit_a:system --no-monthly-matrices
```

The three models are `system` (served scores), `fast_loop` (frozen v1 + weekly
threshold) and `no_mlops` (frozen v1, fixed threshold); a series is `run:model`.
The first series of a comparison is orange, the second black (dashed, square
markers). Output in `docs/presentation/results/`:

- per run: a confusion matrix per month and model plus a 12-month overview,
  "what the system saw" (realistic runs: labelled rows only), labels per month,
  fraud caught/missed, belief vs true recall, feature PSI heatmap, model timeline,
  weekly review share and threshold;
- per comparison: recall with 95 % band, caught fraud, FPR, precision, workload,
  recall at equal workload, FPR ratio by age, frauds only one model caught (with
  McNemar test), cumulative caught fraud, a 2 × 2 overview and `monthly_numbers.csv`;
- with one all-label run and at least two realistic runs: recall with the range
  of the audit samples.

`summarize` saves `scenario.json` and `monitoring.json` with each run, so the event
lines, audit share and review plan stay correct when the configs change later.

## Verification

```powershell
python -m unittest discover -s tests -t . -v
```

The current test suite (see TRAINING.md) does not have separate tests for the
chart functions; they are exercised by `train.py` and `simulation/summarize.py`.
For the simulated year, `python -m simulation.summarize` creates the
adaptive-vs-frozen comparison charts from the prediction log.
