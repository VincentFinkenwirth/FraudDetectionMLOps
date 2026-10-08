import unittest

import numpy as np

from fraud.drift import build_reference_profile, compare_to_reference, psi
from fraud.fairness import fpr_by_age
from fraud.preprocessing import FEATURE_COLUMNS, prepare_features
from tests.helpers import make_applications


def equal_importance():
    # credit_risk_score gets a larger share (10 / 38 = 26 %) so it counts as an important feature.
    return {column: 10.0 if column == "credit_risk_score" else 1.0 for column in FEATURE_COLUMNS}


class PsiTests(unittest.TestCase):
    def test_identical_distributions(self):
        self.assertAlmostEqual(psi([0.2, 0.3, 0.5], [0.2, 0.3, 0.5]), 0.0)

    def test_shift_is_positive(self):
        self.assertGreater(psi([0.2, 0.3, 0.5], [0.5, 0.3, 0.2]), 0.3)

    def test_empty_bins_do_not_crash(self):
        self.assertTrue(np.isfinite(psi([0.0, 1.0], [1.0, 0.0])))


class ReferenceProfileTests(unittest.TestCase):
    def setUp(self):
        self.reference = prepare_features(make_applications(4000, seed=1))
        scores = np.linspace(0, 1, len(self.reference))
        self.profile = build_reference_profile(self.reference, equal_importance(), scores, threshold=0.9)

    def test_same_data_has_no_drift(self):
        result = compare_to_reference(self.profile, self.reference)
        self.assertAlmostEqual(result["drift_share"], 0.0)
        self.assertEqual(result["max_unseen_category_share"], 0.0)

    def test_new_sample_of_same_process_has_little_drift(self):
        other = prepare_features(make_applications(4000, seed=2))
        self.assertLess(compare_to_reference(self.profile, other)["drift_share"], 0.1)

    def test_scale_change_is_found(self):
        shifted = self.reference.copy()
        shifted["credit_risk_score"] += 600
        result = compare_to_reference(self.profile, shifted)
        self.assertEqual(result["most_drifted_important_feature"], "credit_risk_score")
        self.assertGreater(result["max_psi_important_feature"], 1.0)

    def test_unknown_category_and_missing_values(self):
        broken = self.reference.copy()
        broken.loc[broken.index[:400], "device_os"] = "android"
        broken.loc[broken.index[:400], "session_length_in_minutes"] = np.nan
        result = compare_to_reference(self.profile, broken)
        self.assertAlmostEqual(result["unseen_category_share"]["device_os"], 0.1)
        self.assertAlmostEqual(result["max_missing_increase_complete_features"], 0.1, places=3)

    def test_naturally_missing_feature_is_not_a_quality_problem(self):
        # prev_address_months_count is missing in ~30 % of the reference rows
        more_missing = self.reference.copy()
        more_missing.loc[more_missing.index[:800], "prev_address_months_count"] = np.nan
        result = compare_to_reference(self.profile, more_missing)
        self.assertGreater(result["max_missing_increase"], 0.05)
        self.assertLess(result["max_missing_increase_complete_features"], 0.01)

    def test_flag_rate_z(self):
        result = compare_to_reference(self.profile, self.reference, probabilities=np.ones(1000))
        self.assertEqual(result["flag_rate"], 1.0)
        self.assertGreater(result["flag_rate_z"], 10)


class FairnessTests(unittest.TestCase):
    def test_rates_and_ratio(self):
        labels = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        flagged = np.array([1, 0, 0, 0, 1, 1, 0, 0])
        ages = np.array([20, 20, 20, 20, 60, 60, 60, 60])
        result = fpr_by_age(labels, flagged, ages)
        self.assertAlmostEqual(result["fpr_age_under_50"], 0.25)
        self.assertAlmostEqual(result["fpr_age_50_plus"], 0.5)
        self.assertAlmostEqual(result["fpr_ratio_age"], 0.5)

    def test_group_without_negatives(self):
        result = fpr_by_age([1, 0], [1, 0], [60, 20])
        self.assertIsNone(result["fpr_age_50_plus"])
        self.assertIsNone(result["fpr_ratio_age"])


if __name__ == "__main__":
    unittest.main()
