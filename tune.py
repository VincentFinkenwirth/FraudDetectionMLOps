"""Use Optuna to select CatBoost settings through the shared training code.

Each trial is saved in MLflow. The winning fitted model is kept, so there is
no extra training run after tuning. Nothing runs when this file is imported.
"""

import argparse
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path

import optuna

from train import (
    PROJECT_DIR,
    add_common_arguments,
    file_hash,
    get_selection_metric,
    load_config,
    load_training_data,
    run_training,
    setup_mlflow,
    use_cpu,
    write_json,
)


def suggest_config(trial, base_config):
    """Only change the settings being searched; retain the fixed settings."""
    config = deepcopy(base_config)
    config["model_params"].update({
        "depth": trial.suggest_int("depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 60.0, log=True),
        "border_count": trial.suggest_categorical("border_count", [128, 254]),
        "auto_class_weights": trial.suggest_categorical(
            "auto_class_weights", ["None", "SqrtBalanced", "Balanced"],
        ),
    })
    params = config["model_params"]
    # These are alternative sampling methods, so do not mix their parameters.
    params.pop("bagging_temperature", None)
    params.pop("subsample", None)
    params["bootstrap_type"] = trial.suggest_categorical(
        "bootstrap_type", ["Bayesian", "Bernoulli"],
    )
    if params["bootstrap_type"] == "Bayesian":
        params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0.0, 3.0)
    else:
        params["subsample"] = trial.suggest_float("subsample", 0.66, 1.0)
    return config


def queue_starting_trials(study):
    """Give each class-weight option four initial trials before adaptive search."""
    if study.trials:
        return
    for _ in range(4):
        for weights in ["None", "SqrtBalanced", "Balanced"]:
            # Only fix weighting. Optuna samples the other parameters normally.
            study.enqueue_trial({"auto_class_weights": weights})


def objective(trial, training_data, validation_data, base_config, data_info, output_dir, *, create_charts=True):
    """Compare trials using the configured, unweighted validation objective."""
    config = suggest_config(trial, base_config)
    result = run_training(
        training_data, validation_data, config, data_info, output_dir,
        run_name=f"catboost-trial-{trial.number}",
        create_charts=create_charts,
    )
    # Optuna stores parameters and scores, but does not save fitted models.
    # These attributes connect each trial with the model saved by train.py.
    trial.set_user_attr("mlflow_run_id", result["run_id"])
    trial.set_user_attr("bundle_dir", result["bundle_dir"])
    trial.set_user_attr("training_config", result["training_config"])
    trial.set_user_attr("validation_metrics", result["metrics"])
    metric = get_selection_metric(config)
    trial.set_user_attr("selection_metric", metric)
    return result["metrics"][metric]


def check_study_context(study, config, data_info):
    """Prevent an old study from being resumed with different data or code."""
    context = {
        "base_config": config,
        "data_hashes": {name: details["sha256"] for name, details in data_info.items()},
        "code_hashes": {name: file_hash(PROJECT_DIR / name) for name in [
            "preprocessing.py", "train.py", "tune.py", "visualization.py",
        ]},
        "versions": {name: version(name) for name in [
            "catboost", "pandas", "numpy", "scikit-learn", "optuna", "mlflow", "matplotlib",
        ]},
        "objective": f"validation_{get_selection_metric(config)}",
    }
    previous = study.user_attrs.get("context")
    if previous is None and not study.trials:
        study.set_user_attr("context", context)
    elif previous != context:
        raise ValueError("This study used different data, settings or code. Choose a new --study-name.")


def save_best_result(study, output_dir):
    """Export settings for retraining and point to the already fitted model."""
    best = study.best_trial
    metrics = best.user_attrs["validation_metrics"]
    metric = best.user_attrs["selection_metric"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "best_config.json", best.user_attrs["training_config"])
    write_json(output_dir / "best_result.json", {
        "study_name": study.study_name,
        "trial_number": best.number,
        "selection_metric": f"validation_{metric}",
        "selection_score": best.value,
        "validation_average_precision": metrics["average_precision"],
        "validation_recall": metrics["recall"],
        "validation_precision": metrics["precision"],
        "validation_false_positive_rate": metrics["false_positive_rate"],
        "threshold": metrics["decision_threshold"],
        "threshold_selection": best.user_attrs["training_config"].get("threshold_selection"),
        "mlflow_run_id": best.user_attrs["mlflow_run_id"],
        "bundle_dir": best.user_attrs["bundle_dir"],
    })
    study.trials_dataframe().to_csv(output_dir / "trials.csv", index=False)
    print(f"Best validation {metric}: {best.value:.4f}")
    print(f"Precision: {metrics['precision']:.2%} | Recall: {metrics['recall']:.2%}")
    print(f"FPR: {metrics['false_positive_rate']:.4%} | Threshold: {metrics['decision_threshold']:.8f}")
    print(f"Winning model bundle: {best.user_attrs['bundle_dir']}")
    print(f"Retraining configuration: {output_dir / 'best_config.json'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.set_defaults(
        config=PROJECT_DIR / "configs" / "training_fpr.json",
        experiment="fraud-catboost-fpr",
    )
    parser.add_argument("--trials", type=int, default=60, help="Number of additional trials to run.")
    parser.add_argument("--timeout", type=int, default=3600, help="Seconds before starting no further trials.")
    parser.add_argument("--study-name", default="catboost-fpr-v1")
    parser.add_argument("--storage", help="Optuna database URL; defaults to local work/optuna.db.")
    parser.add_argument("--output", type=Path, default=PROJECT_DIR / "work" / "tuning")
    args = parser.parse_args()
    if args.trials < 1 or args.timeout < 1:
        parser.error("--trials and --timeout must be positive.")
    # The name is also used as a folder name, so keep it portable.
    if not args.study_name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.study_name):
        parser.error("--study-name may contain only letters, numbers, underscores and hyphens.")

    config = load_config(args.config)
    if args.cpu:
        config = use_cpu(config)
    if any(key in config["model_params"] for key in ("class_weights", "scale_pos_weight")):
        parser.error("This search tunes auto_class_weights; remove class_weights and scale_pos_weight.")
    training_data, validation_data, data_info = load_training_data(args.train, args.val)
    setup_mlflow(args.tracking_uri, args.experiment)
    storage = args.storage or f"sqlite:///{(PROJECT_DIR / 'work' / 'optuna.db').as_posix()}"
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            seed=config["model_params"]["random_seed"], n_startup_trials=12,
        ),
    )
    check_study_context(study, config, data_info)
    queue_starting_trials(study)
    output_dir = args.output.resolve() / args.study_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # PRAUC still controls CatBoost early stopping. Optuna compares the fitted
    # models by recall below the FPR limit when using the new default config.
    try:
        study.optimize(
            lambda trial: objective(trial, training_data, validation_data, config, data_info, output_dir,
                                    create_charts=not args.no_charts),
            n_trials=args.trials,
            timeout=args.timeout,
            n_jobs=1,
            gc_after_trial=True,
        )
    finally:
        # Keep the best completed trial even when a later trial is interrupted.
        completed = study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))
        if completed:
            save_best_result(study, output_dir)


if __name__ == "__main__":
    main()
