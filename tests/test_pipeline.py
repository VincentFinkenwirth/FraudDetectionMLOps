import json
import os
import shutil
import unittest
from datetime import datetime, timedelta
from unittest import mock

import numpy as np
import pandas as pd

import model_registry
from pipeline.common import load_json
from pipeline.common import effective_count
from pipeline.monitor import decide, drifted_labels_in_fit, served_performance
from pipeline.retrain import (
    NotEnoughData, paired_recall_difference, promotion_decision, run_all, stage_deploy, stage_snapshot,
)
from preprocessing import FEATURE_COLUMNS, TARGET_COLUMN
from scoring import FraudScorer
from service import db
from simulation.replay import select_labels
from tests.helpers import make_applications, make_bundle, temp_dir

CONFIG = load_json("configs/monitoring.json")
START = datetime.fromisoformat(CONFIG["simulation_start"])
NO_DRIFT = {"drift_share": 0.0, "max_psi_important_feature": 0.0, "flag_rate_z": 0.0,
            "max_missing_increase_complete_features": 0.0, "max_unseen_category_share": 0.0}


def run_decide(as_of_days, drift=NO_DRIFT, performance=None, last_training_days=None,
               new_labels=50_000, drifted_labels=0):
    as_of = START + timedelta(days=as_of_days)
    last = None if last_training_days is None else {"as_of": START + timedelta(days=last_training_days)}
    return decide(CONFIG, as_of, drift, performance, 0.05, last, new_labels, drifted_labels)


class MonitorDecisionTests(unittest.TestCase):
    def test_nothing_happens_in_a_normal_week(self):
        self.assertEqual(run_decide(7)["decision"], "no_action")

    def test_schedule_after_one_month(self):
        result = run_decide(28)
        self.assertEqual((result["decision"], result["reasons"]), ("retrain", ["schedule"]))

    def test_schedule_waits_for_labels(self):
        self.assertEqual(run_decide(28, new_labels=100)["decision"], "deferred_waiting_for_labels")

    def test_drift_waits_until_drifted_period_has_labels(self):
        drift = {**NO_DRIFT, "max_psi_important_feature": 3.0}
        waiting = run_decide(14, drift=drift, drifted_labels=10)
        self.assertEqual(waiting["decision"], "deferred_waiting_for_labels")
        ready = run_decide(14, drift=drift, drifted_labels=5000)
        self.assertEqual((ready["decision"], ready["reasons"]), ("retrain", ["data_drift"]))

    LOW_RECALL = {"fraud_cases": 200, "effective_fraud_cases": 200, "missed_fraud_labels": 120,
                  "recall": 0.40, "false_positive_rate": 0.04, "fpr_ratio_age": 0.5}

    def test_performance_waits_while_a_drift_waits_for_labels(self):
        # A retraining now could not learn the drift (its labels are not in the fit data yet).
        drift = {**NO_DRIFT, "max_psi_important_feature": 3.0}
        result = run_decide(14, drift=drift, performance=self.LOW_RECALL, drifted_labels=10)
        self.assertEqual(result["decision"], "deferred_waiting_for_labels")
        self.assertNotIn("performance", result["reasons"])
        self.assertTrue(any("performance" in w for w in result["warnings"]))

    def test_schedule_still_retrains_while_a_drift_waits_for_labels(self):
        drift = {**NO_DRIFT, "max_psi_important_feature": 3.0}
        result = run_decide(28, drift=drift, performance=self.LOW_RECALL, drifted_labels=10)
        self.assertEqual(result["decision"], "retrain")
        self.assertIn("schedule", result["reasons"])

    def test_drift_during_cooldown_is_only_a_warning(self):
        drift = {**NO_DRIFT, "drift_share": 0.5}
        result = run_decide(20, drift=drift, last_training_days=14, drifted_labels=5000)
        self.assertEqual(result["decision"], "no_action")
        self.assertTrue(result["warnings"])

    def test_quality_incident_blocks_retraining(self):
        drift = {**NO_DRIFT, "max_unseen_category_share": 0.3}
        self.assertEqual(run_decide(28, drift=drift)["decision"], "blocked_quality")
        self.assertEqual(run_decide(7, drift=drift)["decision"], "quality_incident")

    def test_performance_trigger(self):
        performance = {"fraud_cases": 200, "effective_fraud_cases": 200, "missed_fraud_labels": 120,
                       "recall": 0.40, "false_positive_rate": 0.04, "fpr_ratio_age": 0.5}
        result = run_decide(14, performance=performance)
        self.assertEqual((result["decision"], result["reasons"]), ("retrain", ["performance"]))
        few_cases = {**performance, "fraud_cases": 20, "effective_fraud_cases": 20, "missed_fraud_labels": 12}
        self.assertEqual(run_decide(14, performance=few_cases)["decision"], "no_action")

    def test_recall_target_is_fixed(self):
        # The target comes from monitoring.json, not from the served model's own holdout:
        # 0.45 is below it, 0.55 is not.
        self.assertEqual(CONFIG["min_served_recall"], 0.49)
        below = {**self.LOW_RECALL, "recall": 0.45}
        self.assertEqual(run_decide(14, performance=below)["reasons"], ["performance"])
        above = {**self.LOW_RECALL, "recall": 0.55}
        self.assertEqual(run_decide(14, performance=above)["decision"], "no_action")

    def test_calibration_warning(self):
        off = {**self.LOW_RECALL, "recall": 0.55, "calibration_ratio": 1.9}
        result = run_decide(14, performance=off)
        self.assertEqual(result["decision"], "no_action")      # a warning, not a trigger
        self.assertTrue(any("calibration" in w for w in result["warnings"]))
        fine = run_decide(14, performance={**off, "calibration_ratio": 1.1})
        self.assertFalse(any("calibration" in w for w in fine["warnings"]))

    def test_recall_not_measurable_with_realistic_labels(self):
        # 200 reviewed fraud cases, but the missed ones are only seen through a few audits
        performance = {"fraud_cases": 200, "effective_fraud_cases": 9, "missed_fraud_labels": 2,
                       "recall": 0.30, "false_positive_rate": 0.04, "fpr_ratio_age": 0.5}
        result = run_decide(14, performance=performance)
        self.assertEqual(result["decision"], "no_action")
        self.assertTrue(any("not measurable" in w for w in result["warnings"]))
        # The false-positive rate can still be judged and still triggers retraining.
        result = run_decide(14, performance={**performance, "false_positive_rate": 0.09})
        self.assertEqual(result["reasons"], ["performance"])


    def test_no_audited_fraud_means_recall_is_not_measurable(self):
        # Week 18 of my first realistic run: 117 reviewed fraud labels (all weight 1,
        # so the effective count is also 117) and no missed fraud at all -> the
        # estimated recall was 100 %, the true one 44 %. This must not be judged.
        performance = {"fraud_cases": 117, "effective_fraud_cases": 117.0, "missed_fraud_labels": 0,
                       "recall": 1.0, "false_positive_rate": 0.04, "fpr_ratio_age": 0.5}
        result = run_decide(14, performance=performance)
        self.assertTrue(any("not measurable" in w for w in result["warnings"]))
        low_recall = run_decide(14, performance={**performance, "recall": 0.1})
        self.assertNotIn("performance", low_recall["reasons"])


class RealisticLabelTests(unittest.TestCase):
    def test_only_flagged_and_audited_applications_get_labels(self):
        week = pd.DataFrame({"application_id": ["A", "B", "C", "D"], "fraud_bool": [1, 0, 1, 0],
                             "label_available_at": ["2025-02-10"] * 4, "audited": [False, False, True, True]})
        full = select_labels(week, "full", {"A"}, 0.02)
        self.assertEqual(len(full), 4)
        self.assertTrue((full["weight"] == 1).all())
        realistic = select_labels(week, "realistic", {"A"}, 0.02).set_index("application_id")
        self.assertEqual(list(realistic.index), ["A", "C", "D"])      # B: not flagged, not audited
        self.assertEqual(realistic.loc["A", "label_source"], "review")
        self.assertEqual(realistic.loc["C", "label_source"], "audit")
        self.assertAlmostEqual(realistic.loc["C", "weight"], 50.0)

    def test_effective_count(self):
        self.assertAlmostEqual(effective_count([1] * 100), 100)
        self.assertAlmostEqual(effective_count([50, 50]), 2)
        self.assertLess(effective_count([1] * 100 + [50] * 2), 10)

    def test_weighted_performance_estimates_recall(self):
        # 10 flagged frauds (reviewed) + 1 audited missed fraud standing for 50 -> recall 10/60
        labelled = pd.DataFrame({
            "fraud_bool": [1] * 10 + [1] + [0] * 5,
            "flagged": [True] * 10 + [False] + [True] * 5,
            "fraud_probability": [0.9] * 10 + [0.01] + [0.5] * 5,
            "weight": [1.0] * 10 + [50.0] + [1.0] * 5,
            "customer_age": [30] * 16,
            "label_source": ["review"] * 10 + ["audit"] + ["review"] * 5,
        })
        performance = served_performance(labelled)
        self.assertAlmostEqual(performance["recall"], 10 / 60)
        self.assertEqual(performance["audited_rows"], 1)
        self.assertEqual(performance["missed_fraud_labels"], 1)
        # Calibration is weighted the same way: mean probability (9 + 0.5 + 2.5) / 65 vs fraud rate 60 / 65.
        self.assertAlmostEqual(performance["calibration_ratio"], 12.0 / 60.0)


class GateTests(unittest.TestCase):
    GATE = {"min_holdout_frauds": 20, "min_missed_frauds": 10, "min_ap_lift": 2.0, "max_known_fraud_loss": 0.15,
            "recall_margin": 0.02, "fpr_tolerance": 0.01}

    def metrics(self, recall, fpr, frauds=100, ap=0.15):
        return {"recall": recall, "false_positive_rate": fpr, "fraud_cases": frauds,
                "average_precision": ap, "fraud_rate": 0.012}

    def test_equal_candidate_is_promoted(self):
        promoted, _ = promotion_decision(self.metrics(0.6, 0.05), self.metrics(0.6, 0.05), 0.0, self.GATE, 0.05, 50)
        self.assertTrue(promoted)

    def test_worse_candidate_is_rejected(self):
        promoted, _ = promotion_decision(self.metrics(0.5, 0.05), self.metrics(0.6, 0.05), -0.1, self.GATE, 0.05, 50)
        self.assertFalse(promoted)

    def test_candidate_with_too_many_false_alarms_is_rejected(self):
        promoted, _ = promotion_decision(self.metrics(0.8, 0.09), self.metrics(0.6, 0.05), 0.2, self.GATE, 0.05, 50)
        self.assertFalse(promoted)

    def test_champion_breaking_fpr_limit_is_replaced(self):
        promoted, _ = promotion_decision(self.metrics(0.5, 0.05), self.metrics(0.6, 0.09), -0.1, self.GATE, 0.05, 50)
        self.assertTrue(promoted)

    def test_without_missed_frauds_recall_is_not_compared(self):
        # Realistic labels: 0 vs 1 audited case looked like "100 % worse" before.
        promoted, explanation = promotion_decision(self.metrics(0.0, 0.05), self.metrics(1.0, 0.05), -1.0,
                                                   self.GATE, 0.05, 1)
        self.assertTrue(promoted)
        self.assertIn("not comparable", explanation)
        # ... but the FPR limit still counts
        promoted, _ = promotion_decision(self.metrics(0.9, 0.09), self.metrics(0.5, 0.05), 0.4, self.GATE, 0.05, 1)
        self.assertFalse(promoted)

    def test_same_workload_skips_the_fpr_check(self):
        # Both models flag 5.5 %: a noisy FPR estimate of 7 % must not reject the candidate.
        promoted, _ = promotion_decision(self.metrics(0.6, 0.07), self.metrics(0.6, 0.05), 0.0, self.GATE, 0.05, 50,
                                         same_workload=True)
        self.assertTrue(promoted)

    def test_loss_on_known_frauds_is_visible_without_missed_frauds(self):
        # The 2-tree model from my runs: 24 % vs 75 % of the reviewed fraud cases -> rejected
        promoted, explanation = promotion_decision(self.metrics(0.24, 0.05), self.metrics(0.75, 0.05), -0.5,
                                                   self.GATE, 0.05, 0, same_workload=True,
                                                   known_fraud_shares=(0.24, 0.75))
        self.assertFalse(promoted)
        self.assertIn("known fraud", explanation)
        # A normal small difference is fine (the newer model is promoted)
        promoted, _ = promotion_decision(self.metrics(0.85, 0.05), self.metrics(0.9, 0.05), -0.05,
                                         self.GATE, 0.05, 0, same_workload=True, known_fraud_shares=(0.85, 0.9))
        self.assertTrue(promoted)

    def test_candidate_ranking_like_chance_is_rejected(self):
        # AP of about the fraud rate = no better than random, even if recall cannot be compared
        promoted, explanation = promotion_decision(self.metrics(0.05, 0.05, ap=0.013), self.metrics(0.6, 0.05), 0.0,
                                                   self.GATE, 0.05, 1, same_workload=True)
        self.assertFalse(promoted)
        self.assertIn("sanity", explanation)

    def test_paired_bootstrap(self):
        labels = np.array([1] * 100 + [0] * 100)
        champion = np.array([True] * 50 + [False] * 150)
        candidate = np.array([True] * 60 + [False] * 140)
        difference, low, high = paired_recall_difference(labels, champion, candidate, 500, 0)
        self.assertAlmostEqual(difference, 0.10)
        self.assertLessEqual(low, difference)
        self.assertGreaterEqual(high, difference)


class RetrainingIntegrationTest(unittest.TestCase):
    """Whole pipeline on synthetic data: snapshot -> train -> evaluate -> register -> deploy."""

    def setUp(self):
        self.folder = temp_dir()
        self.models = self.folder / "models"
        make_bundle(self.folder / "b1", seed=1)
        model_registry.publish(self.folder / "b1", self.models, "v1")
        model_registry.set_current(self.models, "v1")
        self.engine = db.get_engine(f"sqlite:///{(self.folder / 'test.db').as_posix()}")

        # History file like data/splits/train.csv
        make_applications(3000, seed=11).to_csv(self.folder / "history.csv", index=False)
        model_config = json.loads(open("configs/training_cpu.json").read())
        model_config["model_params"].update(iterations=40, thread_count=2)
        (self.folder / "model_config.json").write_text(json.dumps(model_config))
        self.config = {**load_json("configs/retraining.json"),
                       "model_config": str(self.folder / "model_config.json"),
                       "history_files": [str(self.folder / "history.csv")],
                       "history_rows": 2000, "min_eval_frauds": 5, "fixed_tree_count": 30,
                       "gate": {**load_json("configs/retraining.json")["gate"], "min_holdout_frauds": 5,
                                "bootstrap_samples": 200}}

        # Four weeks of served predictions with labels
        scorer = FraudScorer(self.models / "versions" / "v1")
        applications = make_applications(4000, seed=12)
        probabilities, flagged = scorer.score(applications)
        times = [START + timedelta(minutes=10 * i) for i in range(len(applications))]
        rows = [{"prediction_id": f"p{i}", "application_id": f"APP-{i}", "payload_hash": "h", "event_time": t,
                 "scored_at": t, "model_version": "v1", "fraud_probability": float(p), "flagged": bool(f),
                 "threshold": scorer.threshold, **{c: row[c] for c in FEATURE_COLUMNS}}
                for i, (t, p, f, (_, row)) in enumerate(zip(times, probabilities, flagged, applications.iterrows()))]
        labels = [{"application_id": f"APP-{i}", "fraud_bool": int(y), "label_available_at": t + timedelta(days=28),
                   "label_source": "test", "received_at": t} for i, (t, y) in enumerate(zip(times, applications[TARGET_COLUMN]))]
        with self.engine.begin() as conn:
            conn.execute(db.predictions.insert(), [{k: (v.item() if hasattr(v, "item") else v) for k, v in r.items()} for r in rows])
            conn.execute(db.labels.insert(), labels)

    def tearDown(self):
        self.engine.dispose()
        shutil.rmtree(self.folder, ignore_errors=True)

    def test_incident_weeks_are_quarantined(self):
        # Monitoring found a quality incident in the second simulated week.
        db.add_event(self.engine, "incident", "open", START + timedelta(days=14), "v1", {})
        snapshot = stage_snapshot(self.folder / "run", START + timedelta(days=70), self.config, self.engine)
        self.assertGreater(snapshot["quarantined_rows"], 0)
        used = pd.concat([pd.read_parquet(self.folder / "run" / f"{name}.parquet") for name in ("fit", "holdout")])
        times = used["event_time"].dropna()
        in_incident_week = (times >= START + timedelta(days=7)) & (times < START + timedelta(days=14))
        self.assertFalse(in_incident_week.any())

    def test_holdout_is_the_newest_labelled_data_and_fit_is_older(self):
        snapshot = stage_snapshot(self.folder / "run", START + timedelta(days=70), self.config, self.engine)
        fit = pd.read_parquet(self.folder / "run" / "fit.parquet")
        holdout = pd.read_parquet(self.folder / "run" / "holdout.parquet")
        holdout_start = holdout["event_time"].max() - timedelta(days=self.config["holdout_days"])
        self.assertTrue((holdout["event_time"] > holdout_start).all())
        self.assertTrue((fit["event_time"].dropna() <= holdout_start).all())
        # No validation split any more: every labelled application is either fit or holdout.
        self.assertFalse((self.folder / "run" / "validation.parquet").exists())
        self.assertEqual(snapshot["simulation_rows_in_fit"] + snapshot["holdout"]["rows"], 4000)

    def test_drift_reference_covers_the_last_fit_days_including_unlabelled_rows(self):
        # 100 applications near the end of the fit window never get a label (like with realistic labels)
        with self.engine.begin() as conn:
            conn.execute(db.labels.delete().where(db.labels.c.application_id.in_(
                [f"APP-{i}" for i in range(1000, 1100)])))
        config = {**self.config, "reference_days": 7}
        snapshot = stage_snapshot(self.folder / "run", START + timedelta(days=70), config, self.engine)
        times = pd.Series([START + timedelta(minutes=10 * i) for i in range(4000)])
        holdout_start = times.max() - timedelta(days=config["holdout_days"])
        expected = ((times > holdout_start - timedelta(days=7)) & (times <= holdout_start)).sum()
        self.assertEqual(snapshot["reference"]["rows"], expected)

    def test_newest_labelled_event(self):
        # Labels arrive 28 days after the application.
        self.assertEqual(db.newest_labelled_event(self.engine, START + timedelta(days=28, hours=5)),
                         START + timedelta(hours=5))
        self.assertIsNone(db.newest_labelled_event(self.engine, START))

    def test_drifted_labels_only_count_what_lands_in_the_fit_data(self):
        # All applications (~28 days) are labelled at day 70. A retraining now puts
        # the newest 14 labelled days into the holdout, so only older drifted rows count.
        as_of, drift_since = START + timedelta(days=70), START + timedelta(days=7)
        times = pd.Series([START + timedelta(minutes=10 * i) for i in range(4000)])
        expected = ((times >= drift_since) & (times <= times.max() - timedelta(days=14))).sum()
        self.assertEqual(drifted_labels_in_fit(self.engine, drift_since, START, as_of, 14), expected)
        # A drift that started inside the holdout period has nothing in the fit data yet.
        self.assertEqual(drifted_labels_in_fit(self.engine, times.max() - timedelta(days=3), START, as_of, 14), 0)

    def test_no_applications_for_the_drift_reference_stops_the_run(self):
        # All labelled data falls into a 30-day holdout, so nothing is left before it.
        with self.assertRaises(NotEnoughData):
            stage_snapshot(self.folder / "run", START + timedelta(days=70), {**self.config, "holdout_days": 30},
                           self.engine)

    def test_too_few_frauds_in_the_holdout_stops_the_run(self):
        with self.assertRaises(NotEnoughData):
            stage_snapshot(self.folder / "run", START + timedelta(days=70),
                           {**self.config, "min_eval_frauds": 10_000}, self.engine)

    def test_full_run(self):
        as_of = START + timedelta(days=70)
        env = {"MLFLOW_TRACKING_URI": f"sqlite:///{(self.folder / 'mlflow.db').as_posix()}",
               "MLFLOW_ARTIFACT_DIR": str(self.folder / "mlflow_artifacts"), "API_URL": ""}
        with mock.patch.dict(os.environ, env):
            result = run_all(self.folder / "run", as_of, "schedule", self.config, self.engine, self.models)
        self.assertIn(result["status"], ("promoted", "rejected"))
        # Fit comes before the holdout in time, and nothing overlaps.
        fit = pd.read_parquet(self.folder / "run" / "fit.parquet")
        holdout = pd.read_parquet(self.folder / "run" / "holdout.parquet")
        self.assertLess(fit["event_time"].dropna().max(), holdout["event_time"].min())   # history has no time
        self.assertFalse(set(fit["application_id"]) & set(holdout["application_id"]))
        # No early stopping: the candidate has exactly the fixed number of trees.
        metadata = json.loads((self.folder / "run" / "candidate" / "metadata.json").read_text())
        self.assertEqual(metadata["tree_count"], 30)
        # Lineage: data hashes in the snapshot (and so in the bundle), code version in the bundle.
        snapshot = json.loads((self.folder / "run" / "snapshot.json").read_text())
        self.assertEqual(set(snapshot["file_sha256"]), {"fit.parquet", "holdout.parquet", "reference.parquet",
                                                        *self.config["history_files"]})
        self.assertEqual(metadata["snapshot"]["file_sha256"], snapshot["file_sha256"])
        self.assertIn("git_commit", metadata["lineage"])
        # Recalibration: on the holdout the mean probability equals the fraud rate.
        calibration = json.loads((self.folder / "run" / "candidate" / "training_config.json").read_text())[
            "probability_calibration"]
        self.assertAlmostEqual(calibration["odds_divisor"],
                               calibration["class_weight_ratio"] * calibration["prior_adjustment"])
        holdout = pd.read_parquet(self.folder / "run" / "holdout.parquet")
        probabilities = FraudScorer(self.folder / "run" / "candidate").predict_proba(holdout)
        if 0.25 < calibration["prior_adjustment"] < 4:
            self.assertAlmostEqual(np.average(probabilities, weights=holdout["weight"]),
                                   np.average(holdout[TARGET_COLUMN], weights=holdout["weight"]), places=5)
        evaluation = json.loads((self.folder / "run" / "evaluation.json").read_text())
        self.assertIn("calibration_ratio", evaluation["candidate"])
        # The training event is what the monitor uses for the monthly schedule.
        self.assertIsNotNone(db.last_event(self.engine, "training"))
        if result["status"] == "promoted":
            self.assertEqual(model_registry.read_current(self.models)["version"], "v2")
            FraudScorer(self.models / "versions" / "v2")

    # ------------------------------------------------------------ deployment

    def prepare_deployment(self, as_of):
        """A run folder as the register stage leaves it: v2 is published but not served yet."""
        run = self.folder / "deploy"
        run.mkdir()
        make_bundle(self.folder / "b2", seed=2)
        version = model_registry.publish(self.folder / "b2", self.models)
        (run / "snapshot.json").write_text(json.dumps({"as_of": as_of.isoformat()}))
        (run / "registered.json").write_text(json.dumps({"promoted": True, "local_version": version,
                                                         "mlflow_version": "1"}))
        (run / "evaluation.json").write_text(json.dumps({"explanation": "test"}))
        make_applications(10, seed=3).to_parquet(run / "holdout.parquet")
        return run, version

    def deploy_without_api(self, run):
        with mock.patch("pipeline.retrain.api_settings", return_value=("", "")), \
                mock.patch("pipeline.retrain.setup_mlflow"), mock.patch("pipeline.retrain.mlflow.MlflowClient"):
            return stage_deploy(run, self.config, self.engine, self.models)

    def test_model_and_threshold_are_switched_together(self):
        run, version = self.prepare_deployment(START + timedelta(days=28))
        self.deploy_without_api(run)
        switch = json.loads((self.models / "deployments.jsonl").read_text().splitlines()[-1])
        self.assertEqual(switch["version"], version)
        self.assertAlmostEqual(switch["threshold"], model_registry.read_current(self.models)["threshold"])

    def test_failed_threshold_computation_keeps_the_champion(self):
        model_registry.set_threshold(self.models, "v1", 0.123)
        before = model_registry.read_current(self.models)
        run, _ = self.prepare_deployment(START + timedelta(days=28))
        with mock.patch("pipeline.retrain.workload_threshold", side_effect=ValueError("no scores")):
            with self.assertRaises(ValueError):
                self.deploy_without_api(run)
        self.assertEqual(model_registry.read_current(self.models), before)
        self.assertEqual(db.last_event(self.engine, "deployment")["status"], "failed_before_switch")

    def test_failed_api_check_restores_model_and_threshold(self):
        model_registry.set_threshold(self.models, "v1", 0.123)
        run, _ = self.prepare_deployment(START + timedelta(days=28))
        with mock.patch("pipeline.retrain.api_settings", return_value=("http://api", "key")), \
                mock.patch("pipeline.retrain.check_api", side_effect=RuntimeError("wrong answers")), \
                mock.patch("pipeline.retrain.requests.post"):
            with self.assertRaises(RuntimeError):
                stage_deploy(run, self.config, self.engine, self.models)
        current = model_registry.read_current(self.models)
        self.assertEqual((current["version"], current["threshold"]), ("v1", 0.123))
        self.assertEqual(db.last_event(self.engine, "deployment")["status"], "rolled_back")

    def test_without_recent_applications_the_bundle_threshold_is_used(self):
        run, version = self.prepare_deployment(START + timedelta(days=70))
        result = self.deploy_without_api(run)
        current = model_registry.read_current(self.models)
        self.assertEqual((result["serving"], current["version"]), (version, version))
        self.assertNotIn("threshold", current)


if __name__ == "__main__":
    unittest.main()
