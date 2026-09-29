"""Create presentation charts from an existing CatBoost bundle, without fitting."""

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path

from catboost import CatBoostClassifier
import mlflow
import pandas as pd

from preprocessing import prepare_dataset
from scoring import fraud_probabilities
from train import PROJECT_DIR, file_hash, setup_mlflow
from visualization import _slug, create_evaluation_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True, help="Folder with model.cbm and training_config.json.")
    parser.add_argument("--data", type=Path, default=PROJECT_DIR / "data" / "splits" / "val.csv")
    parser.add_argument("--name", default="Validation", help="Dataset label displayed on the slides.")
    parser.add_argument("--output", type=Path, default=PROJECT_DIR / "work" / "reports" / "validation")
    parser.add_argument("--log-mlflow", action="store_true", help="Log charts to MLflow as well as saving locally.")
    parser.add_argument("--run-id", help="Existing MLflow run to receive charts (requires --log-mlflow).")
    parser.add_argument("--tracking-uri")
    parser.add_argument("--experiment", default="fraud-reports")
    args = parser.parse_args()
    if args.run_id and not args.log_mlflow:
        parser.error("--run-id requires --log-mlflow.")

    config = json.loads((args.bundle / "training_config.json").read_text(encoding="utf-8"))
    schema = json.loads((args.bundle / "feature_schema.json").read_text(encoding="utf-8"))
    if schema["preprocessing_sha256"] != file_hash(PROJECT_DIR / "preprocessing.py"):
        parser.error("This bundle used different preprocessing. Use its matching preprocessing code to generate scores.")
    model = CatBoostClassifier()
    model.load_model(str(args.bundle / "model.cbm"))
    data = pd.read_csv(args.data, float_precision="round_trip")
    X, labels = prepare_dataset(data)
    calibration = config.get("probability_calibration")
    # New bundles save the threshold as a probability, so the scores need the
    # same class-weight correction. Old tuning bundles still use raw scores.
    if calibration:
        scores = fraud_probabilities(model, X, calibration["odds_divisor"])
    else:
        scores = model.predict_proba(X)[:, 1]
    if args.log_mlflow and args.run_id:
        # Resume in the original experiment, regardless of the CLI default.
        uri = args.tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
        mlflow.set_tracking_uri(uri or f"sqlite:///{(PROJECT_DIR / 'work' / 'mlflow.db').as_posix()}")
        existing = mlflow.get_run(args.run_id)
        mlflow.set_experiment(experiment_id=existing.info.experiment_id)
    elif args.log_mlflow:
        setup_mlflow(args.tracking_uri, args.experiment)
    context = (mlflow.start_run(run_id=args.run_id) if args.run_id else
               mlflow.start_run(run_name=f"report-{args.name}")) if args.log_mlflow else nullcontext()
    with context:
        report = create_evaluation_report(
            labels, scores, threshold=config["threshold"], output_dir=args.output,
            dataset_name=args.name, model_name=f"CatBoost · {args.bundle.name[:12]}",
            fpr_limit=(config.get("threshold_selection") or {}).get("max_fpr"),
            log_to_mlflow=args.log_mlflow,
        )
        provenance = {
            "bundle": str(args.bundle.resolve()), "model_sha256": file_hash(args.bundle / "model.cbm"),
            "data": str(args.data.resolve()), "data_sha256": file_hash(args.data),
            "threshold": config["threshold"],
        }
        (report.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
        if args.log_mlflow:
            # Dataset identity accompanies the chart without modifying training metrics.
            mlflow.log_dict(provenance, f"charts/{_slug(args.name)}/provenance.json")
    print(f"Saved six charts (PNG + SVG) and metrics to {report.output_dir}")


if __name__ == "__main__":
    main()
