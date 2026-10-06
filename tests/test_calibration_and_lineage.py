import json
import os
import shutil
import subprocess
import unittest
from unittest import mock

import numpy as np

import lineage
from scoring import FraudScorer, calibration_summary, correct_for_class_weights, fit_prior_adjustment
from tests.helpers import make_applications, make_bundle, temp_dir


class CalibrationSummaryTests(unittest.TestCase):
    def test_well_calibrated_scores(self):
        rng = np.random.default_rng(0)
        p = rng.uniform(0, 0.2, 200_000)
        y = (rng.random(len(p)) < p).astype(int)
        summary = calibration_summary(y, p)
        self.assertAlmostEqual(summary["calibration_ratio"], 1.0, delta=0.03)
        self.assertLess(summary["ece"], 0.005)
        self.assertLess(summary["brier"], summary["brier_base_rate"])

    def test_twice_too_high(self):
        y = np.array([1] + [0] * 99)
        summary = calibration_summary(y, np.full(100, 0.02))
        self.assertAlmostEqual(summary["mean_predicted"], 0.02)
        self.assertAlmostEqual(summary["observed_rate"], 0.01)
        self.assertAlmostEqual(summary["calibration_ratio"], 2.0)

    def test_weights_count_like_repeated_rows(self):
        y, p = np.array([1, 0, 0]), np.array([0.5, 0.1, 0.2])
        weighted = calibration_summary(y, p, weights=np.array([1.0, 3.0, 1.0]))
        repeated = calibration_summary(np.array([1, 0, 0, 0, 0]), np.array([0.5, 0.1, 0.1, 0.1, 0.2]))
        for key in ("brier", "mean_predicted", "observed_rate", "calibration_ratio"):
            self.assertAlmostEqual(weighted[key], repeated[key])

    def test_no_fraud_gives_no_ratio(self):
        self.assertIsNone(calibration_summary(np.zeros(10), np.full(10, 0.1))["calibration_ratio"])


class PriorAdjustmentTests(unittest.TestCase):
    def test_mean_matches_the_observed_rate_and_order_is_kept(self):
        rng = np.random.default_rng(1)
        p = rng.uniform(0, 0.1, 5000)
        y = (rng.random(len(p)) < p / 2).astype(int)        # true rate is half of the scores
        c = fit_prior_adjustment(p, y)
        adjusted = correct_for_class_weights(p, c)
        self.assertAlmostEqual(adjusted.mean(), y.mean(), places=6)
        np.testing.assert_array_equal(np.argsort(p, kind="stable"), np.argsort(adjusted, kind="stable"))
        self.assertGreater(c, 1.0)

    def test_weighted_rate(self):
        p, y, w = np.full(4, 0.1), np.array([1, 0, 0, 0]), np.array([1.0, 1.0, 1.0, 7.0])
        c = fit_prior_adjustment(p, y, w)
        self.assertAlmostEqual(np.average(correct_for_class_weights(p, c), weights=w), 0.1, places=6)

    def test_factor_is_clamped(self):
        self.assertEqual(fit_prior_adjustment(np.full(100, 0.5), np.array([1] + [0] * 99)), 4.0)
        self.assertEqual(fit_prior_adjustment(np.full(100, 0.001), np.array([1] * 60 + [0] * 40)), 0.25)


class ScorerChecksTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = temp_dir()
        cls.bundle = make_bundle(cls.folder / "bundle")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.folder, ignore_errors=True)

    def copy_with(self, name, config_change=None, metadata_change=None):
        copy = self.folder / name
        shutil.copytree(self.bundle, copy)
        for file, change in (("training_config.json", config_change), ("metadata.json", metadata_change)):
            if change:
                values = json.loads((copy / file).read_text())
                change(values)
                (copy / file).write_text(json.dumps(values))
        return copy

    def test_prior_adjustment_is_applied(self):
        def adjust(config):
            calibration = config["probability_calibration"]
            calibration["prior_adjustment"] = 2.0
            calibration["class_weight_ratio"] = calibration["odds_divisor"]
            calibration["odds_divisor"] *= 2.0
        plain, adjusted = FraudScorer(self.bundle), FraudScorer(self.copy_with("adjusted", adjust))
        data = make_applications(200, seed=5)
        np.testing.assert_allclose(adjusted.predict_proba(data),
                                   correct_for_class_weights(plain.predict_proba(data), 2.0))

    def test_divisor_that_does_not_match_the_adjustment_is_refused(self):
        def adjust(config):
            calibration = config["probability_calibration"]
            calibration["prior_adjustment"] = 2.0
            calibration["class_weight_ratio"] = calibration["odds_divisor"]   # but divisor not multiplied
        with self.assertRaises(ValueError):
            FraudScorer(self.copy_with("wrong_divisor", adjust))

    def test_other_catboost_version_is_refused(self):
        def change(metadata):
            metadata["library_versions"]["catboost"] = "0.0.1"
        with self.assertRaises(ValueError):
            FraudScorer(self.copy_with("old_catboost", metadata_change=change))

    def test_bundle_metadata_has_lineage(self):
        metadata = json.loads((self.bundle / "metadata.json").read_text())
        self.assertIn("git_commit", metadata["lineage"])
        self.assertIn("python_version", metadata["lineage"])


class LineageTests(unittest.TestCase):
    def test_commit_from_git(self):
        info = lineage.code_lineage()
        expected = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                  cwd=lineage.PROJECT_DIR).stdout.strip()
        self.assertEqual(info["git_commit"], expected)
        self.assertIn(info["git_dirty"], (True, False))

    def test_without_git_the_jenkins_variable_is_used(self):
        env = {"GIT_COMMIT": "abc123", "BUILD_TAG": "jenkins-fraud-retrain-7"}
        with mock.patch.dict(os.environ, env), mock.patch("lineage.run_git", return_value=None):
            info = lineage.code_lineage()
        self.assertEqual((info["git_commit"], info["git_dirty"]), ("abc123", None))
        self.assertEqual(info["trained_by"], "jenkins-fraud-retrain-7")

    def test_nothing_known(self):
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch("lineage.run_git", return_value=None):
            info = lineage.code_lineage()
        self.assertIsNone(info["git_commit"])
        self.assertEqual(info["trained_by"], "local")


if __name__ == "__main__":
    unittest.main()
