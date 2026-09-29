"""Turn the model selected by Optuna into the first deployable version (v1).

The tuned bundle (copied to initial_model/) has two problems for production:

1. Its threshold (0.369) is on the raw, class-weighted score scale, so the
   API would return "probabilities" that are about ten times too high.
2. It has no reference profile, so drift monitoring would not know what
   "normal" data looks like.

This script does NOT retrain anything. It loads the saved model, applies the
class-weight correction from scoring.py, selects the threshold again with the
same rule (max recall with FPR < 5 %) on the corrected validation scores and
checks that exactly the same applications are flagged as before. Then it
writes a complete bundle to models/versions/v1.

    python package_model.py            # creates v1 and deploys it if nothing is deployed yet
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import brier_score_loss, log_loss

import model_registry
from preprocessing import prepare_features
from scoring import FraudScorer, class_weight_ratio, text_hash
from train import PROJECT_DIR, calibrated_config, evaluate_model, file_hash, write_bundle


DEFAULT_SOURCE = PROJECT_DIR / "initial_model"
DEFAULT_VAL = PROJECT_DIR / "data" / "splits" / "val.csv"


def load_legacy_bundle(source):
    source = Path(source)
    config = json.loads((source / "training_config.json").read_text(encoding="utf-8"))
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    if config.get("probability_calibration"):
        raise ValueError("This bundle is already calibrated; use it directly.")
    if text_hash(source / "preprocessing.py") != text_hash(PROJECT_DIR / "preprocessing.py"):
        raise ValueError("The bundle used different preprocessing code than the current project.")
    model = CatBoostClassifier()
    model.load_model(str(source / "model.cbm"))
    return model, config, metadata


def package(source, val_path, models_dir, *, deploy=True):
    model, config, old_metadata = load_legacy_bundle(source)
    validation_data = pd.read_csv(val_path, float_precision="round_trip")
    selection = config.get("threshold_selection")

    # Same evaluation code as during training, now with corrected probabilities.
    metrics, labels, probabilities = evaluate_model(
        model, validation_data, config["threshold"], selection, return_predictions=True,
    )

    # The correction must not change which applications are flagged.
    raw_scores = model.predict_proba(prepare_features(validation_data))[:, 1]
    old_flags = raw_scores >= config["threshold"]
    new_flags = probabilities >= metrics["decision_threshold"]
    if not np.array_equal(old_flags, new_flags):
        raise RuntimeError("The calibrated threshold flags different applications than the original one.")

    calibration_check = {
        "odds_divisor": class_weight_ratio(model),
        "raw_threshold": config["threshold"],
        "probability_threshold": metrics["decision_threshold"],
        "validation_fraud_rate": float(np.mean(labels)),
        "mean_raw_score": float(raw_scores.mean()),
        "mean_probability": float(probabilities.mean()),
        "brier_raw": float(brier_score_loss(labels, raw_scores)),
        "brier_corrected": float(brier_score_loss(labels, probabilities)),
        "brier_constant_base_rate": float(brier_score_loss(labels, np.full(len(labels), labels.mean()))),
        "log_loss_raw": float(log_loss(labels, raw_scores)),
        "log_loss_corrected": float(log_loss(labels, probabilities)),
        "same_applications_flagged": True,
    }

    staging = Path(models_dir) / "staging" / "v1"
    if staging.exists():
        shutil.rmtree(staging)
    write_bundle(staging, model, calibrated_config(config, model, metrics["decision_threshold"]),
                 metrics, validation_data, probabilities, {
                     "model_version": "v1",
                     "run_id": old_metadata.get("run_id"),
                     "data": old_metadata.get("data"),
                     "trained_at": old_metadata.get("created_at"),
                     "training_as_of": None,   # trained before the simulation starts
                     "source_bundle": str(Path(source).resolve()),
                     "source_model_sha256": file_hash(Path(source) / "model.cbm"),
                     "calibration_check": calibration_check,
                     "trained_on": "GPU (Optuna study catboost-fpr-v1); later versions are retrained on CPU",
                 })

    version = model_registry.publish(staging, models_dir, "v1")
    shutil.rmtree(staging.parent)
    FraudScorer(model_registry.version_dir(models_dir, version))   # fails loudly if something is missing

    if deploy and model_registry.read_current(models_dir) is None:
        model_registry.set_current(models_dir, version, reason="initial model from tuning")
    return version, metrics, calibration_check


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Tuned bundle without calibration.")
    parser.add_argument("--val", type=Path, default=DEFAULT_VAL, help="Validation CSV the threshold was selected on.")
    parser.add_argument("--models-dir", type=Path, default=model_registry.default_models_dir())
    parser.add_argument("--no-deploy", action="store_true", help="Only create v1, do not write current.json.")
    args = parser.parse_args()

    version, metrics, check = package(args.source, args.val, args.models_dir, deploy=not args.no_deploy)
    print(f"Created model {version} in {model_registry.version_dir(args.models_dir, version)}")
    print(f"Mean score before/after correction: {check['mean_raw_score']:.2%} / {check['mean_probability']:.2%} "
          f"(real fraud rate {check['validation_fraud_rate']:.2%})")
    print(f"Brier score: raw {check['brier_raw']:.5f} -> corrected {check['brier_corrected']:.5f} "
          f"(constant base rate {check['brier_constant_base_rate']:.5f})")
    print(f"Threshold: raw {check['raw_threshold']:.4f} -> probability {check['probability_threshold']:.4f}")
    print(f"Validation recall {metrics['recall']:.2%}, FPR {metrics['false_positive_rate']:.2%}, "
          f"FPR ratio between age groups {metrics['fpr_ratio_age']:.2f}")
    current = model_registry.read_current(args.models_dir)
    print(f"Currently deployed: {current['version'] if current else 'nothing'}")


if __name__ == "__main__":
    main()
