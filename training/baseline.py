"""Baseline: logistic regression, trained and judged exactly like the CatBoost model.

    python -m training.baseline   # -> work/reports/baseline/ and docs/results/baseline.json

"Why CatBoost and not something simpler?" should be answered with a number,
so this script trains a logistic regression on the same training split, picks
its threshold with the same rule (highest recall with FPR < 5 % on the
validation split) and compares both models on validation and on the test set.

Logistic regression needs more preprocessing than CatBoost: missing numbers
are filled with the median (plus a 0/1 "was missing" column, because missing
values are informative in BAF), numbers are scaled, and the five categorical
features are one-hot encoded. prepare_features() is still used first, so both
models start from the same cleaned data.

The test set was already used once for the final check of v1. Here it is only
used for this comparison table - nothing is chosen with it.
"""

import argparse
import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fraud import model_registry
from fraud.fairness import fpr_by_age
from fraud.preprocessing import CATEGORICAL_FEATURES, FEATURE_COLUMNS, prepare_dataset
from fraud.scoring import FraudScorer
from training.train import select_threshold_at_fpr, setup_mlflow
from training.visualization import calculate_metrics


PROJECT_DIR = Path(__file__).resolve().parents[1]
MAX_FPR = 0.05
NUMERIC_FEATURES = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_FEATURES]


def build_baseline():
    """Preprocessing + logistic regression as one scikit-learn pipeline."""
    numeric = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler())
    categorical = OneHotEncoder(handle_unknown="ignore")
    columns = ColumnTransformer([("numeric", numeric, NUMERIC_FEATURES),
                                 ("categorical", categorical, CATEGORICAL_FEATURES)])
    # class_weight="balanced" plays the same role as the class weights in CatBoost:
    # otherwise the 1 % fraud cases hardly influence the fit.
    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    return Pipeline([("preprocessing", columns), ("model", model)])


def evaluate(y, scores, threshold, ages):
    metrics = calculate_metrics(y, scores, threshold)
    metrics.update(fpr_by_age(y, scores >= threshold, ages))
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=PROJECT_DIR / "work" / "reports" / "baseline")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    splits = {name: pd.read_csv(PROJECT_DIR / "data" / "splits" / f"{name}.csv", float_precision="round_trip")
              for name in ("train", "val", "test")}
    data = {name: prepare_dataset(frame) for name, frame in splits.items()}

    print("Training the logistic regression on 400,000 rows ...")
    baseline = build_baseline()
    baseline.fit(*data["train"])
    val_scores = baseline.predict_proba(data["val"][0])[:, 1]
    threshold = select_threshold_at_fpr(data["val"][1], val_scores, MAX_FPR)

    # The deployed CatBoost model v1 (see package_model.py) with its saved threshold.
    catboost = FraudScorer(model_registry.version_dir(model_registry.default_models_dir(), "v1"))

    results = {}
    for name in ("val", "test"):
        X, y = data[name]
        ages = splits[name]["customer_age"].to_numpy()
        results[name] = {
            "logistic_regression": evaluate(y.to_numpy(), baseline.predict_proba(X)[:, 1], threshold, ages),
            "catboost_v1": evaluate(y.to_numpy(), catboost.predict_proba(splits[name]), catboost.threshold, ages),
        }

    args.output.mkdir(parents=True, exist_ok=True)
    report = {"max_fpr": MAX_FPR, "logistic_regression_threshold": threshold,
              "catboost_threshold": catboost.threshold, "results": results}
    for path in (args.output / "baseline.json", PROJECT_DIR / "docs" / "results" / "baseline.json"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, default=float) + "\n", encoding="utf-8")

    if not args.no_mlflow:
        # Next to the tuning runs, so all model comparisons are in one place.
        setup_mlflow(experiment="fraud-baseline")
        with mlflow.start_run(run_name="logistic-regression"):
            mlflow.log_params({"model": "LogisticRegression", "class_weight": "balanced", "max_fpr": MAX_FPR,
                               "threshold": threshold})
            for name, block in results.items():
                for model_name, metrics in block.items():
                    mlflow.log_metrics({f"{name}_{model_name}_{k}": v for k, v in metrics.items()
                                        if isinstance(v, (int, float)) and v is not None and np.isfinite(v)})

    keys = ["recall", "false_positive_rate", "precision", "average_precision", "roc_auc", "recall_at_top_5pct",
            "fpr_ratio_age"]
    for name in ("val", "test"):
        table = pd.DataFrame({model: {k: results[name][model].get(k) for k in keys} for model in results[name]})
        print(f"\n{'Validation' if name == 'val' else 'Test'} (threshold: highest recall with FPR < 5 % on validation)")
        print(table.round(4).to_string())
    print(f"\nSaved to {args.output} and docs/results/baseline.json")


if __name__ == "__main__":
    main()
