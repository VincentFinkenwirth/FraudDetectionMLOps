"""Weekly monitoring job: is the model still fine, and do we need to retrain?

    python -m pipeline.monitor --as-of 2025-02-03T00:00:00 --output work/monitoring

Jenkins runs this every week (in the simulation: once per simulated week).
It looks at two things:

1. Data (no labels needed): the served applications of the last 7 days are
   compared with the reference profile of the deployed model -> drift share,
   alert rate, missing values and unknown categories.
2. Performance (needs the delayed labels): the labels that arrived during the
   last 28 days are joined with the scores that were actually served.

It also runs the "fast loop": the served threshold is reset so that the
planned share of applications (review capacity) is sent to review. This needs
no labels, so the workload is back to normal one week after a sudden change,
while retraining (the "slow loop") has to wait for the labels.

The task says: retrain after one month OR when the data changed by a threshold.
I added the performance trigger because the "camouflage" concept drift is
almost invisible in the data (fraud is only 1 % of the rows).

Label delay problem: when the data drifts, a retraining only helps once the
drifted applications have labels AND those labels land in the fit data (the
newest labelled days are the holdout of the retraining). A drift alarm
therefore waits (decision "deferred_waiting_for_labels") until enough drifted
applications older than the holdout are labelled. A performance trigger waits
during that time too; the monthly schedule does not.

Possible decisions (written to <output>/decision.txt for Jenkins):
  retrain                      Jenkins starts the retraining job
  deferred_waiting_for_labels  a trigger fired, but there are not enough new labels yet
  blocked_quality              data-quality incident: fix the data first, do not retrain on it
  quality_incident             incident without a retraining request
  no_action / insufficient_data
"""

import argparse
import os
from datetime import timedelta
from pathlib import Path

import mlflow
import numpy as np
from sklearn.metrics import average_precision_score

import model_registry
from drift import compare_to_reference
from fairness import fpr_by_age
from pipeline.common import (
    PROJECT_DIR, api_settings, effective_count, load_json, models_dir, parse_time, reload_api, to_jsonable,
    week_number, workload_threshold, write_json,
)
from preprocessing import prepare_features
from scoring import FraudScorer, calibration_summary
from service import db
from train import setup_mlflow


DEFAULT_CONFIG = "configs/monitoring.json"


def served_performance(labelled):
    """Metrics of the served predictions (the scores the reviewers actually saw).

    With realistic labels only reviewed and audited applications have a label,
    so every rate is weighted: an audited case counts 1 / audit share times.
    """
    y = labelled["fraud_bool"].to_numpy().astype(int)
    flagged = labelled["flagged"].astype(bool).to_numpy()
    scores = labelled["fraud_probability"].to_numpy()
    w = labelled["weight"].to_numpy(dtype=float) if "weight" in labelled else np.ones(len(y))
    tp, fp = w[flagged & (y == 1)].sum(), w[flagged & (y == 0)].sum()
    fraud_weight, legit_weight = w[y == 1].sum(), w[y == 0].sum()
    both = bool((y == 1).any() and (y == 0).any())
    # The API returns probabilities, so I also check whether they fit the real
    # fraud rate (Brier score, mean probability / fraud rate, ECE).
    calibration = calibration_summary(y, scores, w) if len(y) else {}
    return {
        "rows": int(len(y)),
        "fraud_cases": int(y.sum()),
        "effective_fraud_cases": effective_count(w[y == 1]),
        # Labelled fraud cases that the served model did NOT flag. Only these show
        # the missed fraud; with realistic labels they come from the audit only.
        "missed_fraud_labels": int(((y == 1) & ~flagged).sum()),
        "audited_rows": int((labelled.get("label_source") == "audit").sum()) if "label_source" in labelled else 0,
        "fraud_rate": float(fraud_weight / w.sum()) if len(y) else None,
        "recall": float(tp / fraud_weight) if fraud_weight else None,
        "false_positive_rate": float(fp / legit_weight) if legit_weight else None,
        "precision": float(tp / (tp + fp)) if tp + fp else None,
        "flagged_rate": float(np.average(flagged, weights=w)) if len(y) else None,
        "average_precision": float(average_precision_score(y, scores, sample_weight=w)) if both else None,
        **fpr_by_age(y, flagged, labelled["customer_age"].to_numpy(), weights=w),
        **{key: value for key, value in calibration.items() if key != "observed_rate"},   # = fraud_rate
    }


def quality_problem(drift, config):
    """Missing values in normally complete features, or categories the model has never seen."""
    return drift is not None and (
        drift["max_missing_increase_complete_features"] >= config["missing_increase_threshold"]
        or drift["max_unseen_category_share"] >= config["unseen_category_threshold"])


def data_drift_detected(drift, config):
    """The "data changed by a threshold value" rule from the task (no labels needed).

    Any of these counts as drift:
    - features that carry >= 25 % of the model importance have PSI >= 0.1
    - one important feature (>= 5 % importance) has PSI >= 0.25
    - the alert rate moved by 4 or more standard errors
    A quality incident also changes the PSI, so it is not counted twice.
    """
    if drift is None or quality_problem(drift, config):
        return False
    z = drift.get("flag_rate_z")
    return (drift["drift_share"] >= config["drift_share_threshold"]
            or drift["max_psi_important_feature"] >= config["important_feature_psi_threshold"]
            or (z is not None and abs(z) >= config["flag_rate_z_threshold"]))


def drifted_labels_in_fit(engine, drift_since, label_cutoff, as_of, holdout_days):
    """Labelled drifted applications that a retraining started now would put into the FIT data.

    A retraining keeps the newest `holdout_days` of labelled applications as its
    holdout. Before, I counted all drifted labels - but the first drifted labels
    are exactly the newest ones, so they all ended up in the holdout and the
    "drift" retraining learned nothing about the drift (week 22 in my runs).
    """
    newest = db.newest_labelled_event(engine, as_of)
    if newest is None:
        return 0
    return db.count_labelled_since(engine, drift_since, label_cutoff, as_of,
                                   event_until=newest - timedelta(days=holdout_days))


def decide(config, as_of, drift, performance, max_fpr, last_training, new_labels, drifted_labels):
    """Turn the measurements into one decision plus a list of reasons."""
    start = parse_time(config["simulation_start"])
    last_time = last_training["as_of"] if last_training else start
    days_since_training = (as_of - last_time).total_seconds() / 86400
    in_cooldown = last_training is not None and days_since_training < config["cooldown_days"]
    reasons, warnings = [], []
    waiting_for_drift_labels = False

    # 1) Time trigger (the "one month" from the task)
    if days_since_training >= config["retrain_every_days"]:
        reasons.append("schedule")

    # 2) Data quality: we should fix the data, not retrain on it
    bad_quality = quality_problem(drift, config)
    if bad_quality:
        warnings.append("data-quality incident: missing values in a normally complete feature or unknown categories")

    # 3) Data drift. Retraining only helps once the NEW kind of data has labels.
    #    Because labels arrive 4 weeks late, I wait until enough applications
    #    from the drifted period are labelled; otherwise the "new" model would
    #    only be trained on the old data again.
    if data_drift_detected(drift, config):
        if in_cooldown:
            warnings.append("data drift detected, but the model was retrained recently (cooldown)")
        elif drifted_labels < config["min_drifted_labels"]:
            warnings.append(f"data drift detected; waiting for labels of the drifted period "
                            f"({drifted_labels} of {config['min_drifted_labels']})")
            waiting_for_drift_labels = True
        else:
            reasons.append("data_drift")

    # 4) Performance of the served predictions whose labels arrived in the last 28 days.
    #    The FPR only needs honest labels, and there are always plenty of them.
    #    Recall also needs labels of fraud cases the model did NOT flag. With
    #    realistic labels those only come from the small audit. In my first
    #    realistic run the monitor once had 117 reviewed fraud labels but not a
    #    single audited fraud, and it reported a recall of 100 % (true: 44 %).
    #    The effective sample size does not notice this (all 117 labels have the
    #    same weight), so recall is only judged when enough missed frauds are labelled.
    #    The recall target is FIXED (min_served_recall in monitoring.json). Before,
    #    each model was compared with its own holdout recall, which jumped between
    #    0.34 and 0.64 from model to model (only ~100 frauds per holdout) and fell
    #    with the camouflage drift. In weeks 38-46 of my v3 run the served recall
    #    was 36-42 % (v1 started at 59 %), but the trigger never fired.
    if performance is not None and performance["fraud_cases"] >= config["min_fraud_labels_for_performance"]:
        recall_measurable = (performance["effective_fraud_cases"] >= config["min_fraud_labels_for_performance"]
                             and performance["missed_fraud_labels"] >= config["min_missed_fraud_labels"])
        recall_low = recall_measurable and (
            performance["recall"] < config["min_served_recall"])
        fpr_high = performance["false_positive_rate"] > max_fpr + config["fpr_tolerance"]
        if not recall_measurable:
            warnings.append(f"recall not measurable: only {performance['missed_fraud_labels']} labelled frauds "
                            f"that the model missed ({performance['effective_fraud_cases']:.0f} effective fraud "
                            "labels); missed fraud is only found by the audit")
        if (recall_low or fpr_high) and not in_cooldown:
            reasons.append("performance")
        elif recall_low or fpr_high:
            warnings.append("performance below the reference, but the model was retrained recently (cooldown)")
    if performance is not None and performance["fraud_cases"] >= config["min_fraud_labels_for_performance"]:
        # Probabilities that are clearly too high or too low. Only a warning: the
        # ranking (and so the review list) can still be fine. Each retraining
        # recalibrates the new model on its holdout.
        calibration = performance.get("calibration_ratio")
        low, high = config["calibration_ratio_band"]
        if calibration is not None and not low <= calibration <= high:
            warnings.append(f"calibration: the mean fraud probability is {calibration:.2f}x the real fraud rate")
    if performance is not None:
        ratio = performance.get("fpr_ratio_age")
        if ratio is not None and ratio < config["min_fpr_ratio_age_warning"]:
            warnings.append(f"fairness: FPR ratio between age groups is only {ratio:.2f}")

    # 5) While a detected drift still waits for labels in the fit data, a
    #    performance trigger waits too: a retraining now could not learn the
    #    drift either (in my runs this gave a useless retraining in week 22).
    #    The monthly schedule still retrains, the task asks for it.
    if waiting_for_drift_labels and "performance" in reasons and "schedule" not in reasons:
        reasons.remove("performance")
        warnings.append("performance below the reference; waiting for labels of the drifted period first")

    if bad_quality:
        decision = "blocked_quality" if reasons else "quality_incident"
    elif not reasons:
        if waiting_for_drift_labels:
            decision = "deferred_waiting_for_labels"
        else:
            decision = "insufficient_data" if drift is None and performance is None else "no_action"
    elif new_labels < config["min_new_labels"]:
        decision = "deferred_waiting_for_labels"
    else:
        decision = "retrain"

    return {
        "decision": decision,
        "reasons": reasons,
        "warnings": warnings,
        "days_since_training": days_since_training,
        "in_cooldown": in_cooldown,
        "new_labels_since_last_training": new_labels,
        "drifted_labels_since_last_training": drifted_labels,
    }


def workload_step(config, as_of, window, champion, models_folder, bad_quality):
    """Fast loop without labels: keep the review workload at the planned share.

    Every week the served threshold is set so that `review_share` of last
    week's applications would have been flagged. In my first runs the threshold
    stayed at its validation value until the next retraining, and after the new
    credit-bureau scale the old models sent 12-19 % of all applications to
    review for up to 8 weeks (labels arrive 4 weeks late). An offline replay
    showed that this rule brings the false-positive rate back after one week.
    I do not adapt to broken data: during a quality incident the threshold stays.
    """
    current = model_registry.read_current(models_folder) or {}
    before = current.get("threshold", champion.threshold)
    result = {"threshold_before": before, "threshold_after": before,
              "review_share_last_week": float(window["flagged"].mean()) if len(window) else None}
    if bad_quality:
        result["action"] = "kept: data-quality incident"
    elif len(window) < config["min_window_rows"]:
        result["action"] = "kept: too few applications"
    else:
        # I score the window again with the current model, because after a
        # deployment the stored scores can still come from the previous version.
        after = workload_threshold(champion.predict_proba(window), config["review_share"])
        model_registry.set_threshold(models_folder, champion.version, after,
                                     reason="weekly workload threshold", as_of=as_of)
        result.update(action="reset", threshold_after=after)
        # Make sure the API really uses it before the next applications arrive.
        api_url, admin_key = api_settings()
        if api_url:
            try:
                reload_api(api_url, admin_key, champion.version, after)
                result["api_confirmed"] = True
            except Exception as error:   # noqa: BLE001 - a warning in the report, the job goes on
                result["api_confirmed"] = False
                result["api_error"] = f"{type(error).__name__}: {error}"
    return result


def run_monitoring(as_of, config, engine, models_folder, log_to_mlflow=True):
    as_of = parse_time(as_of)
    bundle = model_registry.current_bundle_dir(models_folder)
    champion = FraudScorer(bundle)
    max_fpr = (champion.config.get("threshold_selection") or {}).get("max_fpr", 0.05)

    # 1) Data drift on the applications of the last week
    window = db.read_predictions(engine, as_of - timedelta(days=config["window_days"]), as_of)
    drift = None
    if len(window) >= config["min_window_rows"] and champion.reference_profile:
        own_scores = window.loc[window["model_version"] == champion.version, "fraud_probability"].to_numpy()
        drift = compare_to_reference(
            champion.reference_profile, prepare_features(window),
            own_scores if len(own_scores) >= config["min_window_rows"] else None,
        )

    # 2) Performance: labels that arrived in the last 28 days. With monthly
    #    retraining and a 4-week label delay the current model is often
    #    replaced before its own labels arrive, so the trigger uses all served
    #    predictions; the current model's own numbers are kept for information.
    labelled = db.read_labelled(engine, labels_until=as_of,
                                labels_after=as_of - timedelta(days=config["performance_window_days"]))
    all_served = served_performance(labelled) if len(labelled) else None
    own = labelled[labelled["model_version"] == champion.version]
    current_model = served_performance(own) if len(own) else None

    # 3) Since when is the data drifting? (carried over from last week's report)
    window_start = as_of - timedelta(days=config["window_days"])
    previous = db.last_event(engine, "monitoring", before=as_of - timedelta(seconds=1))
    drift_since = None
    if data_drift_detected(drift, config):
        drift_since = (previous or {}).get("details", {}).get("drift_since") or window_start

    # 4) Decision
    last_training = db.last_event(engine, "training", statuses=["promoted", "rejected"], before=as_of)
    label_cutoff = (parse_time(last_training["details"]["label_cutoff"]) if last_training
                    else parse_time(config["simulation_start"]))
    new_labels = db.count_labels(engine, after=label_cutoff, until=as_of)
    drifted_labels = (drifted_labels_in_fit(engine, parse_time(drift_since), label_cutoff, as_of,
                                            config["holdout_days"])
                      if drift_since else 0)
    decision = decide(config, as_of, drift, all_served, max_fpr, last_training, new_labels, drifted_labels)

    # 5) Fast loop (no labels): threshold for next week's applications
    threshold = None
    if config.get("threshold_policy") == "workload":
        threshold = workload_step(config, as_of, window, champion, models_folder, quality_problem(drift, config))
        if threshold.get("api_confirmed") is False:
            decision["warnings"].append(f"the API did not confirm the new threshold: {threshold['api_error']}")

    report = to_jsonable({
        "as_of": as_of,
        "week": week_number(as_of, config["simulation_start"]),
        "model_version": champion.version,
        "window_rows": len(window),
        "drift": drift,
        "data_drift_detected": drift_since is not None,
        "drift_since": drift_since,
        "performance_all_served": all_served,
        "performance_current_model": current_model,
        "target_recall": config["min_served_recall"],
        "threshold": threshold,
        **decision,
    })

    db.add_event(engine, "monitoring", report["decision"], as_of, champion.version, report)
    if report["decision"] in ("blocked_quality", "quality_incident"):
        db.add_event(engine, "incident", "open", as_of, champion.version,
                     {"warnings": report["warnings"], "drift": drift and {
                         k: drift[k] for k in ("max_missing_increase_complete_features",
                                               "max_unseen_category_share")}})
    if log_to_mlflow:
        log_report(report, config)
    return report


def log_report(report, config):
    """Append this week to one long MLflow run, so the charts show the whole year."""
    setup_mlflow(experiment=config["mlflow_experiment"])
    stream = os.environ.get("MONITORING_STREAM", "simulation")
    experiment = mlflow.get_experiment_by_name(config["mlflow_experiment"])
    runs = mlflow.search_runs([experiment.experiment_id], filter_string=f"tags.stream = '{stream}'",
                              max_results=1, output_format="list")
    metrics = {"window_rows": report["window_rows"],
               "retrain_requested": float(report["decision"] == "retrain"),
               "new_labels_since_last_training": report["new_labels_since_last_training"]}
    version = report["model_version"]
    if version.startswith("v") and version[1:].isdigit():
        metrics["model_version_number"] = int(version[1:])
    metrics["data_drift_detected"] = float(report["data_drift_detected"])
    if report.get("threshold"):
        metrics["served_threshold"] = report["threshold"]["threshold_after"]
        if report["threshold"]["review_share_last_week"] is not None:
            metrics["review_share_last_week"] = report["threshold"]["review_share_last_week"]
    for key in ("drift_share", "weighted_psi", "flag_rate", "flag_rate_z", "max_missing_increase",
                "max_missing_increase_complete_features", "max_unseen_category_share",
                "max_psi_important_feature"):
        if report["drift"] and report["drift"].get(key) is not None:
            metrics[key] = report["drift"][key]
    for prefix, block in (("current", report["performance_current_model"]), ("served", report["performance_all_served"])):
        for key, value in (block or {}).items():
            if value is not None:
                metrics[f"{prefix}_{key}"] = value

    with mlflow.start_run(run_id=runs[0].info.run_id if runs else None,
                          run_name=None if runs else f"monitoring-{stream}"):
        mlflow.set_tag("stream", stream)
        mlflow.log_metrics(metrics, step=report["week"])
        mlflow.log_dict(report, f"weekly/week_{report['week']:03d}.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--as-of", default="", help="End of the monitored week (ISO time). Default: now.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=PROJECT_DIR / "work" / "monitoring")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    config = load_json(args.config)
    report = run_monitoring(args.as_of, config, db.get_engine(), models_dir(), log_to_mlflow=not args.no_mlflow)
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "monitoring_report.json", report)
    (args.output / "decision.txt").write_text(report["decision"] + "\n", encoding="utf-8")
    (args.output / "retrain_reason.txt").write_text(",".join(report["reasons"]) + "\n", encoding="utf-8")

    drift = report["drift"] or {}
    print(f"Week {report['week']} | model {report['model_version']} | rows {report['window_rows']} | "
          f"drift share {drift.get('drift_share', float('nan')):.2f} | "
          f"flag rate z {drift.get('flag_rate_z') if drift.get('flag_rate_z') is not None else float('nan'):.1f}")
    if report["performance_all_served"]:
        p = report["performance_all_served"]
        print(f"Served predictions with new labels: recall {p['recall'] if p['recall'] is not None else float('nan'):.2%} "
              f"FPR {p['false_positive_rate'] if p['false_positive_rate'] is not None else float('nan'):.2%} "
              f"({p['fraud_cases']} fraud cases)")
    if report["threshold"]:
        t = report["threshold"]
        print(f"Threshold: {t['action']} ({t['threshold_before']:.4f} -> {t['threshold_after']:.4f})")
    for warning in report["warnings"]:
        print(f"WARNING: {warning}")
    print(f"Decision: {report['decision']} {report['reasons']}")


if __name__ == "__main__":
    main()
