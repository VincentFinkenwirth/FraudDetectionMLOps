"""Train one CatBoost model with fixed settings and record the result in MLflow.

The tuning script imports train_model(), so both commands fit models in the
same way. Importing this file does not read the datasets or start training.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
from copy import deepcopy
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import mlflow
import mlflow.catboost
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_curve

from drift import build_reference_profile
from fairness import fpr_by_age
from preprocessing import CATEGORICAL_FEATURES, FEATURE_COLUMNS, prepare_dataset, prepare_features
from scoring import CALIBRATION_METHOD, class_weight_ratio, fraud_probabilities
from visualization import calculate_metrics, create_evaluation_report


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_DIR / "configs" / "training.json"
DEFAULT_TRAIN = PROJECT_DIR / "data" / "splits" / "train.csv"
DEFAULT_VAL = PROJECT_DIR / "data" / "splits" / "val.csv"
# A finite threshold just above 1 means "flag nothing", even for a score of 1.
NO_ALERT_THRESHOLD = float(np.nextafter(1.0, np.inf))


def file_hash(path):
    """Record which file was used, even if its name stays the same later."""
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def write_json(path, values):
    Path(path).write_text(json.dumps(values, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config.get("model_params"), dict):
        raise ValueError("The config needs a model_params dictionary.")
    if not isinstance(config.get("fit_params"), dict):
        raise ValueError("The config needs a fit_params dictionary.")
    if not 0 <= config["threshold"] <= NO_ALERT_THRESHOLD:
        raise ValueError("The threshold must be between 0 and 1, or the no-alert threshold.")
    selection = config.get("threshold_selection")
    if selection is not None:
        if not isinstance(selection, dict) or selection.get("method") != "recall_at_fpr":
            raise ValueError("threshold_selection.method must be recall_at_fpr.")
        if not 0 < selection.get("max_fpr", 0) < 1:
            raise ValueError("max_fpr must be strictly between 0 and 1.")
    return config


def use_cpu(config):
    """Return a copy of the config that trains on the CPU.

    The tuned configs use my GPU, but the Jenkins container (and most people who
    want to reproduce the project) do not have one. border_count is fixed at 128
    because that is the GPU default; the CPU default would be 254.
    """
    config = deepcopy(config)
    params = config["model_params"]
    params["task_type"] = "CPU"
    params.pop("devices", None)
    params.setdefault("border_count", 128)
    return config


def get_selection_metric(config):
    """Use the operating objective when configured; older configs still use AP."""
    return "recall_at_fpr_limit" if config.get("threshold_selection") else "average_precision"


def load_training_data(train_path, val_path):
    """Only load the requested training and validation files, never the test set."""
    train_path, val_path = Path(train_path).resolve(), Path(val_path).resolve()
    if train_path == val_path:
        raise ValueError("Training and validation must be different files.")

    train_data = pd.read_csv(train_path, float_precision="round_trip")
    val_data = pd.read_csv(val_path, float_precision="round_trip")
    data_info = {}
    for name, path, data in [("train", train_path, train_data), ("validation", val_path, val_data)]:
        data_info[name] = {"path": str(path), "sha256": file_hash(path), "rows": len(data)}
    return train_data, val_data, data_info


def get_model_params(training_config, labels):
    """Adjust the metric syntax for GPU training without changing the config."""
    params = training_config["model_params"].copy()
    metric = params.get("eval_metric", "")
    if params.get("task_type", "CPU") != "GPU" or not metric.startswith("PRAUC:"):
        return params

    # CatBoost rejects an explicit use_weights option on GPU if every training
    # weight is 1. Balanced/SqrtBalanced also give all ones for balanced labels.
    counts = labels.value_counts()
    has_weights = (
        params.get("auto_class_weights", "None") in ("Balanced", "SqrtBalanced")
        and counts.nunique() > 1
    )
    class_weights = params.get("class_weights")
    if class_weights is not None:
        values = class_weights.values() if isinstance(class_weights, dict) else class_weights
        has_weights = has_weights or any(weight != 1 for weight in values)
    has_weights = has_weights or params.get("scale_pos_weight", 1) != 1

    if not has_weights:
        name, options = metric.split(":", 1)
        options = [option for option in options.split(";") if not option.startswith("use_weights=")]
        params["eval_metric"] = name + (":" + ";".join(options) if options else "")
    # With real class weights, keep use_weights=False so early stopping still
    # evaluates the original class distribution rather than a reweighted one.
    return params


def row_weights(data):
    """Weights of the rows, or None if every row counts once.

    Only the realistic label mode has a "weight" column: an audited
    application stands for 1 / audit share unreviewed ones (inverse
    probability weighting), a reviewed (flagged) one only for itself.
    """
    if "weight" not in data or np.allclose(data["weight"].to_numpy(dtype=float), 1.0):
        return None
    return data["weight"].to_numpy(dtype=float)


def train_model(training_data, validation_data, training_config):
    """Fit a fresh model. Optuna and later Jenkins job can use this.

    validation_data can be None: the retraining uses a fixed tree count, so
    the validation data would not change the model at all (see pipeline/retrain.py).
    """
    uses_early_stopping = "early_stopping_rounds" in training_config["fit_params"]
    if validation_data is None:
        if uses_early_stopping:
            raise ValueError("Early stopping needs validation data.")
        return _fit_without_validation(training_data, training_config)

    X_train, y_train = prepare_dataset(training_data)
    X_val, y_val = prepare_dataset(validation_data)
    train_weights, val_weights = row_weights(training_data), row_weights(validation_data)

    # AP and ROC AUC need meaningful validation labels. A one-class training
    # batch is also not suitable for this fraud classifier.
    for name, y in [("Training", y_train), ("Validation", y_val)]:
        if y.nunique() != 2:
            raise ValueError(f"{name} data must contain both fraud and non-fraud examples.")

    # source_id is available in our splits and helps catch accidental leakage.
    if "source_id" in training_data and "source_id" in validation_data:
        for data in [training_data, validation_data]:
            if data["source_id"].isna().any() or data["source_id"].duplicated().any():
                raise ValueError("source_id must be present and unique within each dataset.")
        if training_data["source_id"].isin(validation_data["source_id"]).any():
            raise ValueError("Training and validation contain overlapping source_ids.")

    model = CatBoostClassifier(
        **get_model_params(training_config, y_train),
        cat_features=CATEGORICAL_FEATURES,
    )
    # Validation controls early stopping. The final test set must never go here.
    # A Pool is only needed to give the validation rows their weights.
    eval_set = (X_val, y_val) if val_weights is None else Pool(
        X_val, y_val, cat_features=CATEGORICAL_FEATURES, weight=val_weights)
    # Without early stopping (fixed number of trees, see pipeline/retrain.py)
    # the model must keep all trees and not jump back to the "best" iteration.
    model.fit(
        X_train,
        y_train,
        sample_weight=train_weights,
        eval_set=eval_set,
        use_best_model=uses_early_stopping,
        **training_config["fit_params"],
    )
    return model


def _fit_without_validation(training_data, training_config):
    """Same model, but all trees are kept and nothing is evaluated during training."""
    X_train, y_train = prepare_dataset(training_data)
    if y_train.nunique() != 2:
        raise ValueError("Training data must contain both fraud and non-fraud examples.")
    model = CatBoostClassifier(
        **get_model_params(training_config, y_train),
        cat_features=CATEGORICAL_FEATURES,
    )
    model.fit(X_train, y_train, sample_weight=row_weights(training_data), **training_config["fit_params"])
    return model


def select_threshold_at_fpr(labels, scores, max_fpr, weights=None):
    """Find the most recall possible with FPR strictly below the chosen limit."""
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    if not 0 < max_fpr < 1:
        raise ValueError("max_fpr must be strictly between 0 and 1.")
    if labels.ndim != 1 or scores.ndim != 1 or len(labels) != len(scores):
        raise ValueError("Labels and scores must be matching one-dimensional arrays.")
    if not np.array_equal(np.unique(labels), [0, 1]):
        raise ValueError("Threshold selection needs both fraud and non-fraud labels.")
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Scores must be finite probabilities between 0 and 1.")

    # Keep every real threshold: interpolation could suggest an operating point
    # that cannot be reached, especially when several rows have the same score.
    fprs, recalls, thresholds = roc_curve(labels, scores, sample_weight=weights, drop_intermediate=False)
    allowed = np.flatnonzero(fprs < max_fpr)
    # ROC thresholds are descending. The first maximum recall therefore uses
    # the fewest false positives if several thresholds catch the same frauds.
    best = allowed[np.argmax(recalls[allowed])]
    threshold = thresholds[best]
    return float(threshold) if np.isfinite(threshold) else NO_ALERT_THRESHOLD


def evaluate_model(model, validation_data, threshold, threshold_selection=None, *, return_predictions=False):
    """Use unweighted metrics, even when the training uses class weights.

    The scores are corrected for the class weights first (see scoring.py), so
    the threshold and the metrics are on a real probability scale. Because the
    correction keeps the order of the scores, the selected operating point is
    the same as with the raw scores.
    """
    X_val, y_val = prepare_dataset(validation_data)
    weights = row_weights(validation_data)
    scores = fraud_probabilities(model, X_val)
    if threshold_selection is not None:
        threshold = select_threshold_at_fpr(y_val, scores, threshold_selection["max_fpr"], weights)
    metrics = calculate_metrics(y_val, scores, threshold, sample_weight=weights)
    if threshold_selection is not None:
        metrics.update({
            "recall_at_fpr_limit": metrics["recall"],
            "precision_at_fpr_limit": metrics["precision"],
            "fpr_limit": threshold_selection["max_fpr"],
        })
    # Fairness between the age groups, like in the BAF benchmark.
    metrics.update(fpr_by_age(y_val, scores >= threshold, X_val["customer_age"], weights=weights))

    return (metrics, y_val, scores) if return_predictions else metrics


def numeric_metrics(metrics):
    """MLflow cannot log None, e.g. an undefined fairness ratio."""
    return {key: value for key, value in metrics.items() if value is not None}


def setup_mlflow(tracking_uri=None, experiment="fraud-catboost"):
    # A local database is enough to start. In Docker this points to the MLflow server.
    work_dir = PROJECT_DIR / "work"
    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        # Only create the folder for the local database. In Jenkins the project
        # folder is mounted read-only, so I must not write there.
        work_dir.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(uri or f"sqlite:///{(work_dir / 'mlflow.db').as_posix()}")
    if mlflow.get_experiment_by_name(experiment) is None:
        if uri and uri.startswith(("http://", "https://")):
            # A remote server manages its own artifact storage.
            mlflow.create_experiment(experiment)
        else:
            artifacts = Path(os.environ.get("MLFLOW_ARTIFACT_DIR", work_dir / "mlflow_artifacts")).resolve()
            mlflow.create_experiment(experiment, artifact_location=artifacts.as_uri())
    mlflow.set_experiment(experiment)


def calibrated_config(config, model, threshold):
    """The config that is saved next to the model.

    The threshold is on the corrected probability scale, and the odds divisor
    is stored so the API does exactly the same correction as the evaluation.
    """
    saved_config = deepcopy(config)
    saved_config["threshold"] = float(threshold)
    saved_config["threshold_scale"] = "probability"
    saved_config["probability_calibration"] = {
        "method": CALIBRATION_METHOD,
        "odds_divisor": class_weight_ratio(model),
    }
    return saved_config


def write_bundle(bundle_dir, model, saved_config, metrics, reference_data, reference_scores, metadata):
    """Save everything the API needs for one model version into one folder.

    Used by train.py, and later by the retraining pipeline, so every model
    version looks the same. reference_data/scores describe the "normal" data
    for drift monitoring (the validation data of this model).
    """
    bundle_dir = Path(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    selection = saved_config.get("threshold_selection")

    # The native model is used by the API with predict_proba().
    # The preprocessing source is saved too, because it is part of the model.
    model.save_model(str(bundle_dir / "model.cbm"))
    write_json(bundle_dir / "training_config.json", saved_config)
    write_json(bundle_dir / "validation_metrics.json", metrics)
    write_json(bundle_dir / "feature_schema.json", {
        "feature_columns": FEATURE_COLUMNS,
        "categorical_features": CATEGORICAL_FEATURES,
        "preprocessing_sha256": file_hash(PROJECT_DIR / "preprocessing.py"),
    })
    shutil.copy2(PROJECT_DIR / "preprocessing.py", bundle_dir / "preprocessing.py")

    importance = dict(zip(model.feature_names_, model.get_feature_importance()))
    write_json(bundle_dir / "reference_profile.json", build_reference_profile(
        prepare_features(reference_data), importance, reference_scores, saved_config["threshold"],
    ))

    write_json(bundle_dir / "metadata.json", {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tree_count": int(model.tree_count_),
        "best_iteration": model.get_best_iteration(),
        "catboost_best_scores": model.get_best_score(),
        "effective_model_params": model.get_params(),
        "training_code_sha256": file_hash(Path(__file__)),
        "visualization_code_sha256": file_hash(PROJECT_DIR / "visualization.py"),
        "library_versions": {name: version(name) for name in [
            "catboost", "pandas", "numpy", "scikit-learn", "mlflow", "matplotlib",
        ]},
        "threshold_selection": selection,
        "threshold_note": (
            "Maximise validation recall with FPR strictly below max_fpr. Predictions use "
            "score >= threshold. The limit is empirical on validation, not guaranteed on "
            "future data. Scores are corrected for the class weights, so the threshold "
            "is on the probability scale."
            if selection is not None else
            "Configured threshold on the corrected probability scale; not tuned automatically."
        ),
        **metadata,
    })


def run_training(training_data, validation_data, config, data_info, output_dir, run_name, *, create_charts=True):
    """Run training with logging and save everything needed to use the model."""
    with mlflow.start_run(run_name=run_name) as run:
        # Every run gets its own folder; previous model versions are kept.
        bundle_dir = Path(output_dir).resolve() / run.info.run_id
        bundle_dir.mkdir(parents=True, exist_ok=False)
        mlflow.log_params(config["model_params"])
        mlflow.log_params({f"fit_{key}": value for key, value in config["fit_params"].items()})
        mlflow.log_dict(data_info, "data_info.json")
        mlflow.set_tag("selection_metric", f"validation_{get_selection_metric(config)}")
        selection = config.get("threshold_selection")
        if selection is not None:
            mlflow.log_param("max_fpr", selection["max_fpr"])
            mlflow.log_param("threshold_method", selection["method"])

        model = train_model(training_data, validation_data, config)
        mlflow.log_param("effective_eval_metric", model.get_param("eval_metric"))
        metrics, labels, scores = evaluate_model(
            model, validation_data, config["threshold"], selection, return_predictions=True,
        )
        # Save the threshold belonging to THIS fitted model. The policy stays
        # in the config so retraining selects a new threshold on its validation set.
        saved_config = calibrated_config(config, model, metrics["decision_threshold"])
        mlflow.log_param("threshold", saved_config["threshold"])
        mlflow.log_param("odds_divisor", saved_config["probability_calibration"]["odds_divisor"])
        mlflow.log_metrics({f"validation_{key}": value for key, value in numeric_metrics(metrics).items()})
        mlflow.log_metric("tree_count", model.tree_count_)

        write_bundle(bundle_dir, model, saved_config, metrics, validation_data, scores, {
            "run_id": run.info.run_id,
            "model_version": f"run-{run.info.run_id[:8]}",
            "data": data_info,
        })
        mlflow.log_artifacts(str(bundle_dir), artifact_path="bundle")
        mlflow.catboost.log_model(
            model,
            name="model",
            input_example=prepare_features(validation_data.head(3)),
            pip_requirements=[f"{name}=={version(name)}" for name in ["catboost", "numpy", "pandas"]],
        )
        # Reuse the exact validation scores and selected threshold. Store charts
        # next to the bundle so model artifacts are not duplicated in MLflow.
        chart_dir = None
        if create_charts:
            chart_dir = bundle_dir.parent / "charts" / run.info.run_id / "validation"
            create_evaluation_report(
                labels, scores, threshold=saved_config["threshold"],
                output_dir=chart_dir, dataset_name="Validation",
                fpr_limit=selection["max_fpr"] if selection else None,
            )
            mlflow.log_artifacts(str(chart_dir), artifact_path="charts/validation")
        return {
            "run_id": run.info.run_id, "bundle_dir": str(bundle_dir),
            "metrics": metrics, "training_config": saved_config,
            "charts_dir": str(chart_dir) if chart_dir else None,
        }


def add_common_arguments(parser):
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN, help="Labelled training CSV.")
    parser.add_argument("--val", type=Path, default=DEFAULT_VAL, help="Separate labelled validation CSV.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Complete training configuration.")
    parser.add_argument("--tracking-uri", help="MLflow URI; defaults to MLFLOW_TRACKING_URI or local SQLite.")
    parser.add_argument("--experiment", default="fraud-catboost", help="MLflow experiment name.")
    parser.add_argument("--no-charts", action="store_true", help="Skip validation charts (e.g. during a large tuning search).")
    parser.add_argument("--cpu", action="store_true", help="Train on the CPU even if the config says GPU.")


def main():
    # MLflow servers make MLflow print an emoji; a cp1252 Windows console would crash on it.
    sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--output", type=Path, default=PROJECT_DIR / "work" / "training")
    parser.add_argument("--run-name", default="catboost-training")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.cpu:
        config = use_cpu(config)
    training_data, validation_data, data_info = load_training_data(args.train, args.val)
    setup_mlflow(args.tracking_uri, args.experiment)
    result = run_training(training_data, validation_data, config, data_info, args.output, args.run_name,
                          create_charts=not args.no_charts)
    print(f"Validation average precision: {result['metrics']['average_precision']:.4f}")
    if config.get("threshold_selection"):
        print(f"Validation recall: {result['metrics']['recall']:.2%}")
        print(f"Validation FPR: {result['metrics']['false_positive_rate']:.4%}")
        print(f"Selected threshold: {result['metrics']['decision_threshold']:.8f}")
    print(f"Saved model bundle: {result['bundle_dir']}")


if __name__ == "__main__":
    main()
