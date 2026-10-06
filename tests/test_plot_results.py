import json
import sys
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from simulation import plot_results
from simulation.summarize import EXPORT_COLUMNS
from tests.helpers import temp_dir

SCENARIO = {"weeks_per_month": 4, "audit_share": 0.05,
            "months": [{"month": 1}, {"month": 2, "downturn": 0.5}, {"month": 3, "incident": 0.3}]}
MONITORING = {"review_share": 0.055, "min_fraud_labels_for_performance": 100, "min_missed_fraud_labels": 20,
              "min_served_recall": 0.49, "min_fpr_ratio_age_warning": 0.2}


def make_result_folder(folder, realistic, seed=0):
    """A small result folder with the same files that simulation.summarize writes."""
    rng = np.random.default_rng(seed)
    n = 3000
    week = np.repeat(np.arange(1, 13), n // 12)
    fraud = rng.random(n) < 0.05
    served = np.clip(fraud * 0.3 + rng.random(n) * 0.7, 0, 1)
    frozen = np.clip(fraud * 0.2 + rng.random(n) * 0.8, 0, 1)
    apps = pd.DataFrame({
        "application_id": [f"app-{i}" for i in range(n)],
        "event_time": pd.Timestamp("2025-01-06") + pd.to_timedelta((week - 1) * 7, unit="D"),
        "sim_week": week, "month": 1 + (week - 1) // 4, "fraud_bool": fraud.astype(int),
        "label_source": "simulation", "label_available_at": pd.Timestamp("2025-03-01"),
        "customer_age": rng.choice([20, 30, 40, 50, 60], n).astype(float),
        "served_model_version": np.where(week < 7, "v1", "v2"),
        "served_probability": served, "served_threshold": 0.6, "served_flagged": served >= 0.6,
        "frozen_v1_probability": frozen, "frozen_v1_threshold": 0.6, "frozen_v1_flagged": frozen >= 0.6,
        "frozen_v1_workload_threshold": 0.55, "frozen_v1_workload_flagged": frozen >= 0.55,
    })
    if realistic:
        # only flagged applications and a few random ones get a label
        audited = rng.random(n) < 0.05
        apps["label_source"] = np.where(apps["served_flagged"], "review", np.where(audited, "audit", None))
    apps[EXPORT_COLUMNS].to_parquet(folder / "applications.parquet", index=False)

    weeks = range(1, 13)
    pd.DataFrame({"week": weeks, "decision": "no_action", "served_recall_last_28_days_of_labels": 0.5,
                  "true_recall_last_28_days": 0.45, "effective_fraud_labels": 120,
                  "missed_fraud_labels": 5}).to_csv(folder / "weekly_monitoring.csv", index=False)
    pd.DataFrame({"week": [6], "status": ["promoted"], "reason": ["data_drift"],
                  "deployed_version": ["v2"]}).to_csv(folder / "trainings.csv", index=False)
    pd.DataFrame({"week": weeks, "model_version": "v1", "income": np.linspace(0.01, 0.5, 12),
                  "credit_risk_score": 0.02}).to_csv(folder / "weekly_feature_psi.csv", index=False)
    (folder / "scenario.json").write_text(json.dumps(SCENARIO))
    (folder / "monitoring.json").write_text(json.dumps(MONITORING))
    return apps


class NumberTests(unittest.TestCase):
    def test_wilson_interval(self):
        low, high = plot_results.wilson(50, 100)
        self.assertAlmostEqual(low, 0.404, places=3)
        self.assertAlmostEqual(high, 0.596, places=3)
        self.assertTrue(all(np.isnan(plot_results.wilson(0, 0))))

    def test_monthly_counts(self):
        frame = pd.DataFrame({"application_id": list("abcdef"), "month": 1, "sim_week": 1,
                              "fraud_bool": [1, 1, 1, 0, 0, 0], "customer_age": [20, 30, 50, 20, 30, 50],
                              "flagged": [True, True, False, True, False, False],
                              "probability": [0.9, 0.8, 0.1, 0.7, 0.2, 0.3]})
        row = plot_results.monthly_metrics(frame).iloc[0]
        self.assertEqual((row["tp"], row["fn"], row["fp"], row["tn"]), (2, 1, 1, 2))
        self.assertAlmostEqual(row["recall"], 2 / 3)
        self.assertAlmostEqual(row["false_positive_rate"], 1 / 3)
        self.assertAlmostEqual(row["precision"], 2 / 3)

    def test_paired_frauds_only_counts_disagreements(self):
        base = pd.DataFrame({"application_id": list("abcd"), "month": 1, "fraud_bool": [1, 1, 1, 0]})
        a = base.assign(flagged=[True, True, False, True])
        b = base.assign(flagged=[True, False, True, False])
        row = plot_results.paired_frauds(a, b).iloc[0]
        self.assertEqual((row["only_a"], row["only_b"]), (1, 1))

    def test_events_come_from_the_scenario(self):
        starts, incident = plot_results.scenario_events(SCENARIO)
        self.assertEqual(starts, [(2, "downturn")])
        self.assertEqual(incident, [3])


class PipelineTests(unittest.TestCase):
    def test_all_charts_from_saved_folders_only(self):
        root = temp_dir()
        folders = {}
        for name, realistic, seed in (("full", False, 0), ("real_a", True, 1), ("real_b", True, 2)):
            folders[name] = root / name
            folders[name].mkdir()
            make_result_folder(folders[name], realistic, seed)
        output = root / "charts"
        argv = ["plot_results", "--output", str(output), "--no-monthly-matrices"]
        for name, folder in folders.items():
            argv += ["--run", f"{name}={folder}"]
        with mock.patch.object(sys, "argv", argv):
            plot_results.main()

        expected = [
            "full/confusion/system_year.png", "full/confusion/no_mlops_year.png", "full/outcomes_per_month.png",
            "full/model_timeline.png", "full/feature_psi_heatmap.png", "full/review_share_and_threshold.png",
            "real_a/confusion/seen_by_system_year.png", "real_a/labels_per_month.png",
            "compare/full-system_vs_full-fast_loop/recall.png",
            "compare/full-system_vs_full-fast_loop/paired_difference.png",
            "compare/full-system_vs_full-fast_loop/monthly_numbers.csv",
            "compare/full-system_vs_real_a-system/cumulative_caught.png",
            "compare/label_modes_recall.png",
        ]
        for path in expected:
            self.assertTrue((output / path).exists(), path)
        # only the full-label run has no "seen by the system" chart
        self.assertFalse((output / "full" / "confusion" / "seen_by_system_year.png").exists())
        numbers = pd.read_csv(output / "compare" / "full-system_vs_full-fast_loop" / "monthly_numbers.csv")
        self.assertEqual(len(numbers), 3)


if __name__ == "__main__":
    unittest.main()
