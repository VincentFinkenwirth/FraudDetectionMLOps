"""Retraining pipeline: snapshot -> train -> evaluate -> register -> deploy.

Every stage is a sub-command, so Jenkins can show them as separate stages:

    python -m pipeline.retrain snapshot --run-dir run --as-of 2025-03-03T00:00:00
    python -m pipeline.retrain train    --run-dir run --reason schedule
    python -m pipeline.retrain evaluate --run-dir run
    python -m pipeline.retrain register --run-dir run
    python -m pipeline.retrain deploy   --run-dir run

`python -m pipeline.retrain all ...` runs all stages after each other (used by
the local simulation without Jenkins). The stages pass files to each other
through the run directory, which also documents what happened.

Data policy (configs/retraining.json):
- Only labels that were known at `as_of` are used (delayed labels!).
- The newest 14 days of labelled applications are the holdout, on which the
  new model must not be clearly worse than the deployed model ("champion").
  They are NOT used for fitting. There is no validation split (fixed tree
  count, see stage_train).
- Fitting data = a fixed sample of the original training data (so older fraud
  patterns are not forgotten) + the labelled simulation data of the last 24
  weeks before the holdout.
- Drift reference of the new model = all applications (labelled or not) of
  the last 14 days of the fit window.
- Applications from weeks with a data-quality incident are quarantined (not
  used for fitting, validation, holdout or the drift reference).

Promotion gate:
- Both models are compared at the SAME workload ("threshold_policy":
  "workload"): each gets the threshold that sends `review_share` of all
  applications of the holdout weeks to review, because that is how the weekly
  job runs the served model. Before, each model used its own validation
  threshold, and with realistic labels that threshold came from ~570 rows.
- Recall is only compared when the holdout contains enough labelled frauds
  that the served model missed (see promotion_decision).
"""

import argparse
import json
import sys
from datetime import timedelta
from pathlib import Path

import mlflow
import mlflow.catboost
import numpy as np
import pandas as pd
import requests

import model_registry
from fairness import fpr_by_age
from pipeline.common import (
    PROJECT_DIR, api_settings, effective_count, load_json, models_dir, parse_time, to_jsonable, workload_threshold,
    write_json,
)
from preprocessing import FEATURE_COLUMNS, TARGET_COLUMN, prepare_features
from scoring import FraudScorer, fraud_probabilities
from service import db
from train import (
    calibrated_config, evaluate_model, load_config, numeric_metrics, row_weights, setup_mlflow, train_model,
    use_cpu, write_bundle,
)
from visualization import calculate_metrics


DEFAULT_CONFIG = "configs/retraining.json"
# "weight" = how many applications a labelled row stands for (see service/db.py).
# "flagged" = what the served model decided (1/0; empty for the history sample).
FRAME_COLUMNS = ["application_id", "event_time", *FEATURE_COLUMNS, TARGET_COLUMN, "weight", "flagged"]


class NotEnoughData(RuntimeError):
    """Raised when the snapshot has too few labelled fraud cases to train or evaluate."""


# ---------------------------------------------------------------- snapshot

def load_history(config):
    """A fixed random sample of the original labelled data (train + val + test)."""
    frames = [pd.read_csv(PROJECT_DIR / path, float_precision="round_trip") for path in config["history_files"]]
    history = pd.concat(frames, ignore_index=True)
    history = history.sample(n=min(config["history_rows"], len(history)), random_state=config["history_seed"])
    history["application_id"] = "HIST-" + history["source_id"].astype(str)
    history["event_time"] = pd.NaT
    history["weight"] = 1.0          # the original data is fully labelled
    history["flagged"] = np.nan      # these applications were never served
    return history[FRAME_COLUMNS].reset_index(drop=True)


def quarantine_mask(labelled, engine, as_of, days):
    """True for applications sent during a week in which monitoring found a quality incident."""
    mask = pd.Series(False, index=labelled.index)
    incidents = db.read_events(engine, "incident")
    if incidents.empty or labelled.empty:
        return mask
    for end in incidents.loc[incidents["as_of"] <= as_of, "as_of"]:
        start = end - timedelta(days=days)
        mask |= (labelled["event_time"] >= start) & (labelled["event_time"] < end)
    return mask


def stage_snapshot(run_dir, as_of, config, engine):
    labelled = db.read_labelled(engine, labels_until=as_of)
    quarantined = quarantine_mask(labelled, engine, as_of, config["quarantine_days"])
    # Applications from weeks with a data-quality incident are not used at all.
    # In my first full run they ended up in the validation set, so the new
    # model's drift reference contained the broken data and normal weeks
    # afterwards looked like drift.
    labelled = labelled[~quarantined]
    if labelled.empty:
        raise NotEnoughData("There are no usable labelled applications yet.")
    labelled = labelled.sort_values(["event_time", "id"]).reset_index(drop=True)
    # As a number, so it can share one parquet column with the empty history values.
    labelled["flagged"] = labelled["flagged"].astype(float)

    # Holdout = the newest labelled days; fit = everything older. There is no
    # validation split any more: with the fixed tree count it did not change
    # the model, and it took the two freshest weeks away from the fit data.
    # After a drift those are exactly the weeks the new model has to learn from.
    newest = labelled["event_time"].max()
    holdout_start = newest - timedelta(days=config["holdout_days"])
    holdout = labelled[labelled["event_time"] > holdout_start]
    older = labelled[labelled["event_time"] <= holdout_start]
    recent = older[older["event_time"] > holdout_start - timedelta(days=config["fit_window_days"])]
    fit = pd.concat([load_history(config), recent[FRAME_COLUMNS]], ignore_index=True)

    if holdout[TARGET_COLUMN].sum() < config["min_eval_frauds"]:
        raise NotEnoughData(f"The holdout set has only {int(holdout[TARGET_COLUMN].sum())} fraud cases.")
    # The same application must never be used twice (leakage check).
    if set(fit["application_id"]) & set(holdout["application_id"]):
        raise RuntimeError("An application appears in more than one data set.")

    # Drift reference for the new model = ALL applications (labelled or not) of
    # the last days of the fit window. With realistic labels the labelled rows
    # are mostly flagged cases and would not look like normal applications.
    # I take it from the fit data and not from the holdout: if a retraining
    # happens before drifted data is in fit, a holdout reference would already
    # be drifted and the drift alarm would stop although the model never
    # learned the new data.
    reference = db.read_predictions(engine, holdout_start - timedelta(days=config["reference_days"]),
                                    holdout_start + timedelta(microseconds=1))
    reference = reference[(reference["event_time"] > holdout_start - timedelta(days=config["reference_days"]))
                          & ~quarantine_mask(reference, engine, as_of, config["quarantine_days"])]
    if reference.empty:
        # Without a reference the new model could not be monitored for drift at all.
        raise NotEnoughData("There are no applications before the holdout for the drift reference.")

    run_dir.mkdir(parents=True, exist_ok=True)
    fit.to_parquet(run_dir / "fit.parquet", index=False)
    reference[FEATURE_COLUMNS].to_parquet(run_dir / "reference.parquet", index=False)
    holdout[FRAME_COLUMNS].to_parquet(run_dir / "holdout.parquet", index=False)

    def describe(frame):
        times = frame["event_time"].dropna()
        frauds = frame[TARGET_COLUMN] == 1
        return {"rows": len(frame), "fraud_cases": int(frauds.sum()),
                "effective_fraud_cases": effective_count(frame.loc[frauds, "weight"]),
                "missed_fraud_labels": int((frauds & (frame["flagged"] == 0)).sum()),
                "fraud_rate": float(np.average(frame[TARGET_COLUMN], weights=frame["weight"])),
                "first_event": times.min() if len(times) else None,
                "last_event": times.max() if len(times) else None}

    snapshot = {
        "as_of": as_of,
        "label_cutoff": as_of,
        "history_rows": int((fit["application_id"].str.startswith("HIST-")).sum()),
        "simulation_rows_in_fit": len(recent),
        "quarantined_rows": int(quarantined.sum()),
        "reference_rows": len(reference),
        "fit": describe(fit),
        "holdout": describe(holdout),
        "reference": {"rows": len(reference),
                      "first_event": reference["event_time"].min() if len(reference) else None,
                      "last_event": reference["event_time"].max() if len(reference) else None},
    }
    write_json(run_dir / "snapshot.json", snapshot)
    return snapshot


# ---------------------------------------------------------------- train

def stage_train(run_dir, reason, config):
    snapshot = load_json(run_dir / "snapshot.json")
    fit = pd.read_parquet(run_dir / "fit.parquet")
    holdout = pd.read_parquet(run_dir / "holdout.parquet")
    reference = pd.read_parquet(run_dir / "reference.parquet")
    model_config = use_cpu(load_config(PROJECT_DIR / config["model_config"]))
    # No early stopping: every retrained model gets the tree count of the tuned
    # model. With realistic labels the validation set consisted mostly of
    # applications the old model flagged, so early stopping measured the ranking
    # INSIDE that group - and stopped after 1 or 2 trees in about a third of my
    # retrainings (one of these models was promoted and caught only ~20 % of the
    # fraud for six months). Without early stopping nothing is chosen on the
    # retraining data, so one holdout is enough and there is no validation set.
    model_config["model_params"]["iterations"] = config["fixed_tree_count"]
    model_config["fit_params"] = {k: v for k, v in model_config["fit_params"].items()
                                  if k != "early_stopping_rounds"}
    selection = model_config.get("threshold_selection")

    setup_mlflow(experiment=config["mlflow_experiment"])
    with mlflow.start_run(run_name=f"retrain-{snapshot['as_of'][:10]}") as run:
        mlflow.set_tags({"trigger_reason": reason, "as_of": snapshot["as_of"], "stage": "trained"})
        mlflow.log_params({k: v for k, v in model_config["model_params"].items()})
        mlflow.log_params({"fit_rows": snapshot["fit"]["rows"], "holdout_rows": snapshot["holdout"]["rows"],
                           "reference_rows": snapshot["reference_rows"], "label_cutoff": snapshot["label_cutoff"]})
        mlflow.log_dict(snapshot, "snapshot.json")

        model = train_model(fit, None, model_config)
        # The documented threshold (recall at FPR < 5 %) and the reference
        # metrics of the bundle are measured on the holdout. Neither is used
        # by the gate (it sets both models to the same workload) and the trees
        # never see the holdout, so this does not bias the comparison.
        metrics, labels, scores = evaluate_model(
            model, holdout, model_config["threshold"], selection, return_predictions=True)
        saved_config = calibrated_config(model_config, model, metrics["decision_threshold"])
        mlflow.log_metrics({f"holdout_{k}": v for k, v in numeric_metrics(metrics).items()})
        mlflow.log_metric("tree_count", model.tree_count_)

        reference_scores = fraud_probabilities(model, prepare_features(reference))
        write_bundle(run_dir / "candidate", model, saved_config, metrics, reference, reference_scores, {
            "run_id": run.info.run_id,
            "model_version": "candidate",
            "training_as_of": snapshot["as_of"],
            "label_cutoff": snapshot["label_cutoff"],
            "trigger_reason": reason,
            "snapshot": snapshot,
        })
        mlflow.log_artifacts(str(run_dir / "candidate"), artifact_path="bundle")
        model_info = mlflow.catboost.log_model(model, name="model")

    result = {"run_id": run.info.run_id, "model_uri": model_info.model_uri, "reason": reason,
              "holdout_metrics": metrics, "threshold": saved_config["threshold"]}
    write_json(run_dir / "candidate.json", result)
    return result


# ---------------------------------------------------------------- evaluate

def paired_recall_difference(labels, champion_flags, candidate_flags, samples, seed, weights=None):
    """Recall(candidate) - recall(champion) with a 90 % bootstrap interval.

    Both models are scored on the SAME fraud cases, so I resample the cases
    and compare the two models on every resample (paired comparison). This is
    much less noisy than comparing two independent recall numbers.
    With weights (realistic labels) a case is drawn with a probability
    proportional to its weight, which gives the weighted recall.
    The interval is reported (evaluation.json, MLflow); the gate decides on
    the point estimate (see promotion_decision).
    """
    fraud = np.flatnonzero(np.asarray(labels) == 1)
    w = np.ones(len(labels)) if weights is None else np.asarray(weights, dtype=float)
    difference = (np.average(candidate_flags[fraud], weights=w[fraud])
                  - np.average(champion_flags[fraud], weights=w[fraud]))
    rng = np.random.default_rng(seed)
    if weights is None:
        picks = rng.integers(0, len(fraud), size=(samples, len(fraud)))
    else:
        picks = rng.choice(len(fraud), size=(samples, len(fraud)), p=w[fraud] / w[fraud].sum())
    resampled = candidate_flags[fraud][picks].mean(axis=1) - champion_flags[fraud][picks].mean(axis=1)
    low, high = np.percentile(resampled, [5, 95])
    return float(difference), float(low), float(high)


def promotion_decision(candidate, champion, difference, gate, max_fpr, missed_frauds, same_workload=False,
                       known_fraud_shares=None):
    """Tolerance gate on the point estimate: the candidate must not be clearly worse.

    With only ~100 fraud cases in a holdout, "must be better" would reject
    almost every candidate by chance. The task wants a regularly retrained
    model, so I accept a candidate whose measured recall is at most
    `recall_margin` lower, as long as it keeps the FPR limit. This is NOT a
    statistical non-inferiority test: the paired bootstrap interval is logged
    for information only. With ~100 frauds it is about +-3 to 5 percentage
    points wide, so a rule on its lower bound would reject nearly every
    candidate; the gate therefore only blocks clearly worse models.

    same_workload: both models flag the same share of applications (workload
    policy). Then the FPR check says nothing new - at a fixed review share a
    model with more false alarms automatically catches less fraud, and even a
    model that catches nothing stays at ~5.6 % FPR. With realistic labels the
    FPR estimate of the candidate is also noisy (its extra flags are only seen
    through the audit), and it rejected good candidates, so I skip it.

    `missed_frauds` = labelled fraud cases in the holdout that the served model
    did not flag. Only these show whether a model finds the fraud that the
    champion misses. With realistic labels there are only one or two (from the
    audit), and my first realistic run rejected a candidate because its recall
    was "100 % worse" - 0 vs 1 audited case. So without enough of them I do not
    compare recall at all: the newer model (trained on newer data) is promoted.

    Because the newer model can now be promoted without a recall comparison,
    there is a sanity check first: a model that ranks no better than chance has
    an average precision of about the fraud rate, so the candidate needs at
    least `min_ap_lift` times the fraud rate.

    known_fraud_shares = (candidate, champion): share of the KNOWN fraud cases
    (the ones the served model flagged, so they were reviewed and all have a
    label) that each model flags. A gain of the candidate is invisible without
    missed-fraud labels, but a loss is not: if it misses clearly more of the
    known frauds, it is rejected. A candidate with 2 trees was promoted in my
    runs before this check (24 % vs 75 % of the known frauds).
    """
    fpr_limit = max_fpr + gate["fpr_tolerance"]
    if candidate["fraud_cases"] < gate["min_holdout_frauds"]:
        return False, "too few fraud cases in the holdout to compare the models"
    ap, base_rate = candidate.get("average_precision"), candidate.get("fraud_rate")
    if ap is None or base_rate is None or ap < gate["min_ap_lift"] * base_rate:
        return False, "sanity check failed: the candidate ranks hardly better than chance"
    if not same_workload and candidate["false_positive_rate"] > fpr_limit:
        return False, f"candidate FPR {candidate['false_positive_rate']:.2%} is above {fpr_limit:.0%}"
    if missed_frauds < gate["min_missed_frauds"]:
        if known_fraud_shares is not None:
            candidate_share, champion_share = known_fraud_shares
            if candidate_share < champion_share - gate["max_known_fraud_loss"]:
                return False, (f"candidate flags only {candidate_share:.0%} of the known fraud cases "
                               f"(champion {champion_share:.0%})")
        return True, (f"recall not comparable (only {missed_frauds} labelled missed frauds); "
                      "the newer model is promoted")
    if difference >= -gate["recall_margin"]:
        return True, f"recall difference {difference:+.2%} is within the margin of -{gate['recall_margin']:.0%}"
    if not same_workload and champion["false_positive_rate"] > fpr_limit:
        return True, "the champion breaks the FPR limit and the candidate does not"
    return False, f"candidate recall is lower by {-difference:.2%}"


def holdout_metrics(scorer, holdout, threshold):
    probabilities = scorer.predict_proba(holdout)
    flagged = probabilities >= threshold
    labels = holdout[TARGET_COLUMN].to_numpy()
    weights = row_weights(holdout)
    metrics = calculate_metrics(labels, probabilities, threshold, sample_weight=weights)
    metrics.update(fpr_by_age(labels, flagged, holdout["customer_age"].to_numpy(), weights=weights))
    metrics["fraud_cases"] = int(labels.sum())
    metrics["effective_fraud_cases"] = effective_count(holdout.loc[holdout[TARGET_COLUMN] == 1, "weight"])
    return metrics, flagged


def stage_evaluate(run_dir, config, engine, models_folder):
    snapshot = load_json(run_dir / "snapshot.json")
    candidate_info = load_json(run_dir / "candidate.json")
    holdout = pd.read_parquet(run_dir / "holdout.parquet")
    candidate = FraudScorer(run_dir / "candidate")
    champion = FraudScorer(model_registry.current_bundle_dir(models_folder))
    max_fpr = candidate.config["threshold_selection"]["max_fpr"]
    gate = config["gate"]

    if config.get("threshold_policy") == "workload":
        # Same workload for both: each threshold sends `review_share` of ALL
        # applications of the holdout weeks to review (labelled or not).
        applications = db.read_predictions(engine, parse_time(snapshot["holdout"]["first_event"]),
                                           parse_time(snapshot["holdout"]["last_event"]) + timedelta(microseconds=1))
        applications = applications.drop_duplicates("application_id")
        candidate_threshold = workload_threshold(candidate.predict_proba(applications), config["review_share"])
        champion_threshold = workload_threshold(champion.predict_proba(applications), config["review_share"])
    else:
        # Old behaviour: each model uses the threshold fixed on its own validation data.
        candidate_threshold, champion_threshold = candidate.threshold, champion.threshold
    candidate_metrics, candidate_flags = holdout_metrics(candidate, holdout, candidate_threshold)
    champion_metrics, champion_flags = holdout_metrics(champion, holdout, champion_threshold)
    difference, low, high = paired_recall_difference(
        holdout[TARGET_COLUMN].to_numpy(), champion_flags, candidate_flags, gate["bootstrap_samples"], gate["seed"],
        weights=row_weights(holdout))
    missed_frauds = int(((holdout[TARGET_COLUMN] == 1) & (holdout["flagged"] == 0)).sum())
    # Fraud cases the served model flagged: reviewed, so their labels are known exactly.
    known = ((holdout[TARGET_COLUMN] == 1) & (holdout["flagged"] == 1)).to_numpy()
    known_shares = (float(candidate_flags[known].mean()), float(champion_flags[known].mean())) if known.any() else None
    promoted, explanation = promotion_decision(candidate_metrics, champion_metrics, difference, gate, max_fpr,
                                               missed_frauds,
                                               same_workload=config.get("threshold_policy") == "workload",
                                               known_fraud_shares=known_shares)

    evaluation = {
        "promoted": promoted,
        "explanation": explanation,
        "champion_version": champion.version,
        "threshold_policy": config.get("threshold_policy", "validation"),
        "candidate_threshold": candidate_threshold,
        "champion_threshold": champion_threshold,
        "holdout_missed_fraud_labels": missed_frauds,
        "known_fraud_shares_candidate_champion": known_shares,
        "recall_difference": difference,
        "recall_difference_90ci": [low, high],
        "candidate": candidate_metrics,
        "champion": champion_metrics,
    }
    write_json(run_dir / "evaluation.json", evaluation)

    setup_mlflow(experiment=config["mlflow_experiment"])
    with mlflow.start_run(run_id=candidate_info["run_id"]):
        mlflow.log_metrics({f"holdout_candidate_{k}": v for k, v in numeric_metrics(candidate_metrics).items()})
        mlflow.log_metrics({f"holdout_champion_{k}": v for k, v in numeric_metrics(champion_metrics).items()})
        mlflow.log_metrics({"recall_difference": difference, "recall_difference_ci_low": low,
                            "recall_difference_ci_high": high})
        mlflow.set_tags({"promoted": str(promoted), "gate_explanation": explanation,
                         "champion_version": champion.version, "stage": "evaluated"})
        mlflow.log_dict(to_jsonable(evaluation), "evaluation.json")

    # The monitoring job uses this event for the "one month since the last training" rule.
    db.add_event(engine, "training", "promoted" if promoted else "rejected", snapshot["as_of"], None, {
        "label_cutoff": snapshot["label_cutoff"],
        "reason": candidate_info["reason"],
        "run_id": candidate_info["run_id"],
        "champion_version": champion.version,
        "explanation": explanation,
        "recall_difference": difference,
        "candidate_recall": candidate_metrics["recall"],
        "champion_recall": champion_metrics["recall"],
        "candidate_fpr": candidate_metrics["false_positive_rate"],
        "champion_fpr": champion_metrics["false_positive_rate"],
    })
    return evaluation


# ---------------------------------------------------------------- register

def stage_register(run_dir, config, models_folder):
    """Every candidate goes into the MLflow registry; only accepted ones into models/."""
    evaluation = load_json(run_dir / "evaluation.json")
    candidate_info = load_json(run_dir / "candidate.json")
    version = model_registry.publish(run_dir / "candidate", models_folder) if evaluation["promoted"] else None

    setup_mlflow(experiment=config["mlflow_experiment"])
    registered = mlflow.register_model(candidate_info["model_uri"], config["registered_model_name"],
                                       tags={"promoted": str(evaluation["promoted"]),
                                             "local_version": version or "not deployed"})
    result = {"local_version": version, "mlflow_model": config["registered_model_name"],
              "mlflow_version": registered.version, "promoted": evaluation["promoted"]}
    write_json(run_dir / "registered.json", result)
    return result


# ---------------------------------------------------------------- deploy

def check_api(api_url, admin_key, expected_version, sample, expected_threshold=None):
    """Reload the API and check that it serves the expected version and gives sane answers."""
    headers = {"X-API-Key": admin_key}
    response = requests.post(f"{api_url}/admin/reload", headers=headers, timeout=60)
    response.raise_for_status()
    info = requests.get(f"{api_url}/model-info", headers=headers, timeout=10).json()
    if info["model_version"] != expected_version:
        raise RuntimeError(f"The API serves {info['model_version']} instead of {expected_version}.")
    if expected_threshold is not None and abs(info["threshold"] - expected_threshold) > 1e-12:
        raise RuntimeError(f"The API uses threshold {info['threshold']} instead of {expected_threshold}.")
    # Smoke test with real rows; store=false keeps them out of the monitoring data.
    # to_json turns NaN into null, which is how the API expects missing values.
    rows = json.loads(sample[FEATURE_COLUMNS].to_json(orient="records"))
    applications = [{"application_id": f"SMOKE-{i}", "features": row} for i, row in enumerate(rows)]
    response = requests.post(f"{api_url}/predict/batch?store=false", json={"applications": applications},
                             headers=headers, timeout=30)
    response.raise_for_status()
    probabilities = [p["fraud_probability"] for p in response.json()["predictions"]]
    if not all(0.0 <= p <= 1.0 for p in probabilities):
        raise RuntimeError("The API returned probabilities outside [0, 1].")
    return info


def stage_deploy(run_dir, config, engine, models_folder):
    snapshot = load_json(run_dir / "snapshot.json")
    registered = load_json(run_dir / "registered.json")
    api_url, admin_key = api_settings()
    sample = pd.read_parquet(run_dir / "holdout.parquet").head(5)

    explanation = load_json(run_dir / "evaluation.json")["explanation"]
    summary_file = run_dir / "summary.txt"

    if not registered["promoted"]:
        # Nothing to switch, but the task wants the service to stay reachable after every run.
        current = model_registry.read_current(models_folder)["version"]
        if api_url:
            check_api(api_url, admin_key, current, sample)
        db.add_event(engine, "deployment", "kept_champion", snapshot["as_of"], current,
                     {"reason": "candidate rejected by the gate"})
        summary_file.write_text(f"rejected ({explanation}); API still serves {current}\n", encoding="utf-8")
        return {"deployed": False, "serving": current}

    version = registered["local_version"]
    # Model and threshold are one deployment state. I first prepare the new
    # threshold and only then switch both in one step. Before, the model was
    # switched first, and if the threshold step crashed the new model was
    # already live with its bundle threshold and nothing was rolled back.
    threshold = None
    try:
        if config.get("threshold_policy") == "workload":
            # The new model starts with a workload threshold right away (not with its
            # bundle threshold): I score the applications of the last 7 days with it.
            as_of = parse_time(snapshot["as_of"])
            last_week = db.read_predictions(engine, as_of - timedelta(days=7), as_of)
            if len(last_week):
                scorer = FraudScorer(model_registry.version_dir(models_folder, version))
                threshold = workload_threshold(scorer.predict_proba(last_week), config["review_share"])
    except Exception as error:
        # Nothing was switched yet, so the champion simply keeps serving.
        current = model_registry.read_current(models_folder)["version"]
        db.add_event(engine, "deployment", "failed_before_switch", snapshot["as_of"], current,
                     {"failed_version": version, "error": str(error)})
        summary_file.write_text(f"deployment of {version} failed before the switch; API still serves {current}\n",
                                encoding="utf-8")
        raise

    # set_current either writes current.json in one atomic step or fails before
    # writing, so a failure here needs no rollback (a rollback would go back one
    # version too far).
    model_registry.set_current(models_folder, version, threshold=threshold,
                               threshold_reason="workload threshold of the new model",
                               reason="retraining", as_of=snapshot["as_of"])
    try:
        if api_url:
            check_api(api_url, admin_key, version, sample, expected_threshold=threshold)
    except Exception as error:
        # Roll back to the previous version (with the threshold it was serving)
        # and make the API load it again.
        entry = model_registry.rollback(models_folder, as_of=snapshot["as_of"])
        if api_url:
            requests.post(f"{api_url}/admin/reload", headers={"X-API-Key": admin_key}, timeout=60)
        db.add_event(engine, "deployment", "rolled_back", snapshot["as_of"], entry["version"],
                     {"failed_version": version, "error": str(error)})
        summary_file.write_text(f"deployment of {version} failed, rolled back to {entry['version']}\n",
                                encoding="utf-8")
        raise

    setup_mlflow(experiment=config["mlflow_experiment"])
    client = mlflow.MlflowClient()
    client.set_registered_model_alias(config["registered_model_name"], "champion", registered["mlflow_version"])
    db.add_event(engine, "deployment", "deployed", snapshot["as_of"], version,
                 {"mlflow_version": registered["mlflow_version"]})
    summary_file.write_text(f"promoted and deployed {version} ({explanation})\n", encoding="utf-8")
    return {"deployed": True, "serving": version}


# ---------------------------------------------------------------- command line

def run_all(run_dir, as_of, reason, config, engine, models_folder):
    """All stages in one go (local simulation). Returns a short summary."""
    try:
        snapshot = stage_snapshot(run_dir, as_of, config, engine)
    except NotEnoughData as error:
        return {"status": "skipped", "message": str(error)}
    stage_train(run_dir, reason, config)
    evaluation = stage_evaluate(run_dir, config, engine, models_folder)
    stage_register(run_dir, config, models_folder)
    deployment = stage_deploy(run_dir, config, engine, models_folder)
    return {"status": "promoted" if evaluation["promoted"] else "rejected",
            "message": evaluation["explanation"], "serving": deployment["serving"],
            "fit_rows": snapshot["fit"]["rows"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["snapshot", "train", "evaluate", "register", "deploy", "all"])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--as-of", default="", help="Simulated time of the run (ISO). Default: now.")
    parser.add_argument("--reason", default="manual")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()

    config = load_json(args.config)
    engine, folder = db.get_engine(), models_dir()
    as_of = parse_time(args.as_of)

    if args.stage == "all":
        print(to_jsonable(run_all(args.run_dir, as_of, args.reason, config, engine, folder)))
        return
    if args.stage == "snapshot":
        try:
            result = stage_snapshot(args.run_dir, as_of, config, engine)
        except NotEnoughData as error:
            # Exit code 3 tells Jenkins to stop without marking the build as failed.
            print(f"Not enough data: {error}")
            sys.exit(3)
    elif args.stage == "train":
        result = stage_train(args.run_dir, args.reason, config)
    elif args.stage == "evaluate":
        result = stage_evaluate(args.run_dir, config, engine, folder)
    elif args.stage == "register":
        result = stage_register(args.run_dir, config, folder)
    else:
        result = stage_deploy(args.run_dir, config, engine, folder)
    print(to_jsonable(result))


if __name__ == "__main__":
    main()
