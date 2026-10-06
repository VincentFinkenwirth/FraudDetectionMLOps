"""Tests for the "fast loop": the label-free workload threshold."""

import json
import os
import shutil
import unittest
from datetime import datetime
from unittest import mock

import numpy as np

import model_registry
from pipeline.common import load_json, workload_threshold
from pipeline.monitor import workload_step
from scoring import FraudScorer
from tests.helpers import make_applications, make_bundle, temp_dir


class WorkloadThresholdTests(unittest.TestCase):
    def test_threshold_flags_the_planned_share(self):
        scores = np.random.default_rng(0).random(10_000)
        threshold = workload_threshold(scores, 0.055)
        self.assertAlmostEqual(np.mean(scores >= threshold), 0.055, places=3)

    def test_needs_scores_and_a_valid_share(self):
        with self.assertRaises(ValueError):
            workload_threshold([], 0.055)
        with self.assertRaises(ValueError):
            workload_threshold([0.1, 0.2], 1.5)

    def test_both_jobs_use_the_same_policy(self):
        # The monitor sets the threshold, the gate compares models with it:
        # both must use the same review share.
        monitoring = load_json("configs/monitoring.json")
        retraining = load_json("configs/retraining.json")
        # holdout_days: the monitor must know which labels a retraining would put into fit.
        for key in ("threshold_policy", "review_share", "holdout_days"):
            self.assertEqual(monitoring[key], retraining[key], key)


class RegistryThresholdTests(unittest.TestCase):
    def setUp(self):
        self.folder = temp_dir()
        self.models = self.folder / "models"
        make_bundle(self.folder / "b1", seed=1)
        model_registry.publish(self.folder / "b1", self.models, "v1")
        model_registry.publish(self.folder / "b1", self.models, "v2")
        model_registry.set_current(self.models, "v1")

    def tearDown(self):
        shutil.rmtree(self.folder, ignore_errors=True)

    def test_set_threshold_changes_only_current_json(self):
        bundle_config = (self.models / "versions" / "v1" / "training_config.json").read_text()
        model_registry.set_threshold(self.models, "v1", 0.07, reason="test", as_of=datetime(2025, 3, 3))
        current = model_registry.read_current(self.models)
        self.assertAlmostEqual(current["threshold"], 0.07)
        self.assertEqual(current["threshold_simulated_time"], "2025-03-03T00:00:00")
        # The bundle is never changed, the history is logged.
        self.assertEqual((self.models / "versions" / "v1" / "training_config.json").read_text(), bundle_config)
        log = [json.loads(line) for line in (self.models / "thresholds.jsonl").read_text().splitlines()]
        self.assertEqual(log[-1]["threshold"], 0.07)

    def test_only_the_served_version_can_get_a_threshold(self):
        with self.assertRaises(RuntimeError):
            model_registry.set_threshold(self.models, "v2", 0.07)

    def test_new_deployment_without_threshold_uses_the_bundle_threshold(self):
        model_registry.set_threshold(self.models, "v1", 0.07)
        model_registry.set_current(self.models, "v2")
        current = model_registry.read_current(self.models)
        self.assertNotIn("threshold", current)
        # The threshold v1 was really serving is kept for a rollback.
        self.assertAlmostEqual(current["previous_threshold"], 0.07)

    def test_rollback_restores_the_threshold_that_was_served(self):
        # Before, a rollback brought v1 back with its bundle threshold, which after the
        # credit-scale change sent 17 % of the applications to review instead of 5.5 %.
        model_registry.set_threshold(self.models, "v1", 0.07)
        model_registry.set_current(self.models, "v2", threshold=0.05, threshold_reason="workload")
        model_registry.rollback(self.models)
        current = model_registry.read_current(self.models)
        self.assertEqual(current["version"], "v1")
        self.assertAlmostEqual(current["threshold"], 0.07)

    def test_deployment_writes_model_and_threshold_together(self):
        model_registry.set_current(self.models, "v2", threshold=0.05, threshold_reason="workload",
                                   as_of=datetime(2025, 3, 3))
        current = model_registry.read_current(self.models)
        self.assertEqual((current["version"], current["threshold"]), ("v2", 0.05))
        self.assertEqual(current["threshold_reason"], "workload")
        log = [json.loads(line) for line in (self.models / "thresholds.jsonl").read_text().splitlines()]
        self.assertEqual((log[-1]["version"], log[-1]["threshold"]), ("v2", 0.05))

    def test_rollback_to_a_version_without_threshold_uses_its_bundle_threshold(self):
        model_registry.set_current(self.models, "v2", threshold=0.05)
        model_registry.rollback(self.models)
        current = model_registry.read_current(self.models)
        self.assertEqual(current["version"], "v1")
        self.assertNotIn("threshold", current)


class WeeklyStepTests(unittest.TestCase):
    CONFIG = {"review_share": 0.055, "min_window_rows": 500}

    def setUp(self):
        self.folder = temp_dir()
        self.models = self.folder / "models"
        make_bundle(self.folder / "b1", seed=1)
        model_registry.publish(self.folder / "b1", self.models, "v1")
        model_registry.set_current(self.models, "v1")
        self.champion = FraudScorer(self.models / "versions" / "v1")
        self.window = make_applications(3000, seed=7)
        self.window["flagged"] = self.champion.score(self.window)[1]

    def tearDown(self):
        shutil.rmtree(self.folder, ignore_errors=True)

    def test_threshold_is_reset_to_the_planned_share(self):
        result = workload_step(self.CONFIG, datetime(2025, 2, 3), self.window, self.champion, self.models, False)
        self.assertEqual(result["action"], "reset")
        threshold = model_registry.read_current(self.models)["threshold"]
        share = np.mean(self.champion.predict_proba(self.window) >= threshold)
        self.assertAlmostEqual(share, 0.055, delta=0.002)

    def test_no_change_during_a_quality_incident(self):
        result = workload_step(self.CONFIG, datetime(2025, 2, 3), self.window, self.champion, self.models, True)
        self.assertTrue(result["action"].startswith("kept"))
        self.assertNotIn("threshold", model_registry.read_current(self.models))

    def test_the_api_is_told_and_must_confirm_the_threshold(self):
        env = {"API_URL": "http://api", "API_KEY_ADMIN": "admin-key-for-tests-123456"}
        with mock.patch.dict(os.environ, env), mock.patch("pipeline.monitor.reload_api") as reload:
            result = workload_step(self.CONFIG, datetime(2025, 2, 3), self.window, self.champion, self.models, False)
            reload.assert_called_once_with("http://api", env["API_KEY_ADMIN"], "v1", result["threshold_after"])
            self.assertTrue(result["api_confirmed"])
            reload.side_effect = RuntimeError("API uses the old threshold")
            result = workload_step(self.CONFIG, datetime(2025, 2, 3), self.window, self.champion, self.models, False)
            self.assertFalse(result["api_confirmed"])

    def test_no_change_with_too_few_applications(self):
        result = workload_step(self.CONFIG, datetime(2025, 2, 3), self.window.head(100), self.champion,
                               self.models, False)
        self.assertEqual(result["action"], "kept: too few applications")


if __name__ == "__main__":
    unittest.main()
