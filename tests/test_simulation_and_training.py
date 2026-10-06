import json
import unittest

import numpy as np
import pandas as pd

from baseline import build_baseline
from preprocessing import TARGET_COLUMN, prepare_dataset
from simulation.build_months import build_all, keep_probability
from simulation.replay import redraw_audit
from simulation.summarize import EXPORT_COLUMNS, applications_table, frozen_workload_flags
from train import calibrated_config, select_threshold_at_fpr, train_model, use_cpu
from visualization import calculate_metrics
from tests.helpers import make_applications, make_bundle, temp_dir

SCENARIO = {
    "seed": 1, "start_date": "2025-01-06T00:00:00", "weeks_per_month": 4, "rows_per_month": 1000,
    "label_delay_weeks": 4,
    "camouflage_features": ["device_os", "has_other_cards"],
    "months": [
        {"month": 1},
        {"month": 2, "downturn": 0.6, "drop_legit": 0.3},
        {"month": 3, "camouflage": 0.5, "score_offset": 600},
        {"month": 4, "incident": 0.3},
    ],
}


class SimulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The downturn month needs about twice as many pool rows as it keeps.
        pool = make_applications(8000, seed=3, fraud_rate=0.1)
        pool["month"] = 4
        cls.pool = pool
        cls.months, cls.summary = build_all(pool, SCENARIO)

    def test_months_do_not_overlap(self):
        ids = [set(batch["application_id"]) for batch in self.months.values()]
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                self.assertFalse(ids[i] & ids[j])

    def test_downturn_and_prior_shift(self):
        # Every month is filled up to its planned size; the downturn month just
        # rejects more rows and therefore uses more of the pool.
        self.assertEqual(len(self.months[1]), 1000)
        self.assertEqual(len(self.months[2]), 1000)
        used = {item["month"]: item["pool_rows_used"] for item in self.summary}
        self.assertEqual(used[1], 1000)
        self.assertGreater(used[2], 1300)
        self.assertGreater(self.months[2][TARGET_COLUMN].mean(), self.months[1][TARGET_COLUMN].mean())
        # Richer applicants are dropped more often -> lower average income
        self.assertLess(self.months[2]["income"].mean(), self.months[1]["income"].mean())

    def test_downturn_only_drops_the_upper_half(self):
        # Applicants below the median income / time at the address are never dropped by
        # the downturn, the richest ones are dropped with probability "downturn".
        batch = pd.DataFrame({"income_rank": [0.1, 0.5, 0.75, 1.0], "address_rank": [0.2, 0.5, 0.75, 1.0],
                              TARGET_COLUMN: 0})
        np.testing.assert_allclose(keep_probability(batch, {"downturn": 0.8}), [1.0, 1.0, 0.6, 0.2])

    def test_planned_volume_per_month(self):
        scenario = {**SCENARIO, "months": [{"month": 1, "rows": 500}, {"month": 2, "rows": 700}]}
        months, _ = build_all(self.pool, scenario)
        self.assertEqual([len(months[1]), len(months[2])], [500, 700])

    def test_too_small_pool_is_an_error(self):
        scenario = {**SCENARIO, "months": [{"month": 1, "rows": 9000}]}
        with self.assertRaises(ValueError):
            build_all(self.pool, scenario)

    def test_camouflage_changes_only_fraud_rows_and_keeps_labels(self):
        batch = self.months[3]
        original = self.pool.set_index("source_id").loc[batch["source_id"]]
        changed = batch["camouflaged"].to_numpy()
        self.assertTrue(changed.any())
        self.assertTrue((batch.loc[changed, TARGET_COLUMN] == 1).all())
        np.testing.assert_array_equal(batch[TARGET_COLUMN].to_numpy(), original[TARGET_COLUMN].to_numpy())
        np.testing.assert_array_equal(batch["credit_risk_score"].to_numpy(), original["credit_risk_score"].to_numpy() + 600)

    def test_incident(self):
        batch = self.months[4]
        share = batch["incident"].mean()
        self.assertTrue(0.2 < share < 0.4)
        self.assertTrue((batch.loc[batch["incident"], "device_os"] == "android").all())

    def test_labels_arrive_four_weeks_later(self):
        batch = self.months[2]
        self.assertTrue((batch["label_week"] == batch["sim_week"] + 4).all())
        self.assertEqual(set(batch["sim_week"]), {5, 6, 7, 8})


class AuditAndWorkloadTests(unittest.TestCase):
    def test_other_audit_seed_changes_only_the_audit(self):
        labels = pd.DataFrame({"application_id": [f"A{i}" for i in range(4000)], "fraud_bool": 0,
                               "sim_week": np.repeat([1, 2, 5, 6], 1000), "audited": False})
        first = redraw_audit(labels, 1, 0.02, 4)
        second = redraw_audit(labels, 2, 0.02, 4)
        self.assertTrue(0.01 < first["audited"].mean() < 0.03)
        self.assertFalse((first["audited"] == second["audited"]).all())
        self.assertTrue((redraw_audit(labels, 1, 0.02, 4)["audited"] == first["audited"]).all())   # repeatable
        pd.testing.assert_frame_equal(first.drop(columns="audited"), labels.drop(columns="audited"))

    def test_frozen_model_with_workload_threshold(self):
        rng = np.random.default_rng(0)
        served = pd.DataFrame({"sim_week": np.repeat([1, 2, 3, 4], 2000)})
        probabilities = rng.random(8000)
        probabilities[served["sim_week"] >= 3] += 0.5        # scores jump in week 3 (like a new scale)
        flags, thresholds = frozen_workload_flags(served, probabilities, 0.95, 0.05, incident_weeks=set())
        share = pd.Series(flags).groupby(served["sim_week"]).mean()
        self.assertAlmostEqual(share[2], 0.05, delta=0.015)   # set from week 1
        self.assertGreater(share[3], 0.5)                      # one week too late ...
        self.assertAlmostEqual(share[4], 0.05, delta=0.015)   # ... and then back to plan
        # The threshold of every row is kept for the export (week 1 = saved v1 threshold)
        self.assertTrue((thresholds[served["sim_week"] == 1] == 0.95).all())
        self.assertAlmostEqual(thresholds[served["sim_week"] == 2][0],
                               np.quantile(probabilities[served["sim_week"] == 1], 0.95))
        # After an incident week the threshold is not changed
        flags, _ = frozen_workload_flags(served, probabilities, 0.95, 0.05, incident_weeks={3})
        self.assertGreater(pd.Series(flags)[served["sim_week"] == 4].mean(), 0.5)

    def test_applications_table_has_one_row_per_application(self):
        served = pd.DataFrame({
            "application_id": ["A", "B", "C"],
            "event_time": pd.to_datetime(["2025-01-06", "2025-01-07", "2025-01-14"]),
            "sim_week": [1, 1, 2], "month": [1, 1, 1], "fraud_bool": [1, 0, 0],
            "label_available_at": pd.to_datetime(["2025-02-03", "2025-02-04", "2025-02-11"]),
            "customer_age": [30, 50, 40], "model_version": ["v1", "v1", "v2"],
            "fraud_probability": [0.9, 0.1, 0.3], "threshold": [0.5, 0.5, 0.2], "flagged": [True, False, True]})
        labels_seen = pd.DataFrame({"application_id": ["A", "C"], "label_source": ["review", "audit"]})
        table = applications_table(served, np.array([0.8, 0.2, 0.1]), 0.5, np.array([True, False, False]),
                                   np.array([0.6, 0.6, 0.15]), labels_seen)
        self.assertEqual(len(table), 3)
        self.assertEqual(list(table.columns), EXPORT_COLUMNS)
        never_labelled = table.set_index("application_id").loc["B"]
        self.assertTrue(pd.isna(never_labelled["label_source"]))
        self.assertTrue(pd.isna(never_labelled["label_available_at"]))
        self.assertEqual(table["label_source"].tolist()[2], "audit")
        self.assertEqual(table["frozen_v1_flagged"].tolist(), [True, False, False])
        self.assertEqual(table["served_model_version"].tolist(), ["v1", "v1", "v2"])


class TrainingHelperTests(unittest.TestCase):
    def test_threshold_keeps_fpr_strictly_below_limit(self):
        labels = np.array([0] * 100 + [1] * 10)
        scores = np.concatenate([np.linspace(0, 0.5, 100), np.linspace(0.3, 1.0, 10)])
        threshold = select_threshold_at_fpr(labels, scores, 0.05)
        fpr = np.mean(scores[labels == 0] >= threshold)
        self.assertLess(fpr, 0.05)

    def test_weights_of_one_change_nothing(self):
        rng = np.random.default_rng(0)
        y, scores = rng.integers(0, 2, 500), rng.random(500)
        plain = calculate_metrics(y, scores, 0.5)
        weighted = calculate_metrics(y, scores, 0.5, sample_weight=np.ones(500))
        for key in ("recall", "false_positive_rate", "precision", "average_precision", "recall_at_top_5pct"):
            self.assertAlmostEqual(plain[key], weighted[key])

    def test_weight_two_is_like_a_duplicated_row(self):
        y, scores = np.array([1, 0, 1, 0, 0]), np.array([0.9, 0.8, 0.2, 0.1, 0.6])
        weighted = calculate_metrics(y, scores, 0.5, sample_weight=np.array([1, 1, 2, 1, 1]))
        duplicated = calculate_metrics(np.append(y, 1), np.append(scores, 0.2), 0.5)
        self.assertAlmostEqual(weighted["recall"], duplicated["recall"])
        self.assertAlmostEqual(weighted["false_positive_rate"], duplicated["false_positive_rate"])

    def test_baseline_pipeline_gives_probabilities(self):
        X, y = prepare_dataset(make_applications(1500, seed=4))
        baseline = build_baseline().fit(X, y)
        scores = baseline.predict_proba(prepare_dataset(make_applications(300, seed=5))[0])[:, 1]
        self.assertTrue(((scores >= 0) & (scores <= 1)).all())

    def test_use_cpu(self):
        config = {"model_params": {"task_type": "GPU", "devices": "0"}, "fit_params": {}, "threshold": 0.5}
        cpu = use_cpu(config)
        self.assertEqual(cpu["model_params"]["task_type"], "CPU")
        self.assertNotIn("devices", cpu["model_params"])
        self.assertEqual(config["model_params"]["task_type"], "GPU")   # original unchanged

    def small_config(self, early_stopping):
        config = json.loads(open("configs/training_cpu.json").read())
        config["model_params"].update(iterations=25, thread_count=2)
        if not early_stopping:
            config["fit_params"] = {k: v for k, v in config["fit_params"].items() if k != "early_stopping_rounds"}
        return config

    def test_training_without_validation_data_keeps_all_trees(self):
        # The retraining has no validation split any more (fixed tree count).
        model = train_model(make_applications(1500, seed=6), None, self.small_config(early_stopping=False))
        self.assertEqual(model.tree_count_, 25)

    def test_early_stopping_needs_validation_data(self):
        with self.assertRaises(ValueError):
            train_model(make_applications(1500, seed=6), None, self.small_config(early_stopping=True))

    def test_bundle_contains_everything_for_the_api(self):
        folder = make_bundle(temp_dir() / "bundle")
        for name in ["model.cbm", "training_config.json", "feature_schema.json", "preprocessing.py",
                     "metadata.json", "reference_profile.json", "validation_metrics.json"]:
            self.assertTrue((folder / name).exists(), name)
        config = json.loads((folder / "training_config.json").read_text())
        self.assertEqual(config["threshold_scale"], "probability")
        self.assertGreater(config["probability_calibration"]["odds_divisor"], 1.0)


if __name__ == "__main__":
    unittest.main()
