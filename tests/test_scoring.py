import json
import shutil
import unittest

import numpy as np

from fraud.preprocessing import prepare_features
from fraud.scoring import FraudScorer, class_weight_ratio, correct_for_class_weights
from tests.helpers import make_applications, make_bundle, temp_dir


class CalibrationTests(unittest.TestCase):
    def test_no_weights_changes_nothing(self):
        scores = np.array([0.0, 0.2, 0.9, 1.0])
        np.testing.assert_array_equal(correct_for_class_weights(scores, 1.0), scores)

    def test_known_value(self):
        # odds 1:1 divided by 10 -> odds 1:10 -> probability 1/11
        self.assertAlmostEqual(correct_for_class_weights([0.5], 10.0)[0], 1 / 11)

    def test_order_is_kept(self):
        scores = np.linspace(0, 1, 101)
        corrected = correct_for_class_weights(scores, 10.05)
        self.assertTrue(np.all(np.diff(corrected) > 0))
        self.assertEqual(corrected[0], 0.0)
        self.assertEqual(corrected[-1], 1.0)

    def test_invalid_divisor(self):
        with self.assertRaises(ValueError):
            correct_for_class_weights([0.5], 0)


class ScorerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = temp_dir()
        cls.bundle = make_bundle(cls.folder / "bundle")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.folder, ignore_errors=True)

    def test_ratio_matches_sqrt_balanced(self):
        scorer = FraudScorer(self.bundle)
        # SqrtBalanced: weight(fraud) = sqrt(#non-fraud / #fraud)
        self.assertGreater(scorer.odds_divisor, 1.0)
        self.assertAlmostEqual(scorer.odds_divisor, class_weight_ratio(scorer.model), places=5)

    def test_probabilities_are_corrected(self):
        scorer = FraudScorer(self.bundle)
        data = make_applications(500, seed=7)
        raw = scorer.model.predict_proba(prepare_features(data))[:, 1]
        corrected = scorer.predict_proba(data)
        self.assertTrue(np.all(corrected <= raw + 1e-12))
        np.testing.assert_array_equal(np.argsort(raw, kind="stable"), np.argsort(corrected, kind="stable"))

    def test_bundle_without_calibration_is_refused(self):
        copy = self.folder / "old_bundle"
        shutil.copytree(self.bundle, copy)
        config = json.loads((copy / "training_config.json").read_text())
        del config["probability_calibration"]
        (copy / "training_config.json").write_text(json.dumps(config))
        with self.assertRaises(ValueError):
            FraudScorer(copy)

    def test_changed_preprocessing_is_refused(self):
        copy = self.folder / "changed_preprocessing"
        shutil.copytree(self.bundle, copy)
        with (copy / "preprocessing.py").open("a") as file:
            file.write("\n# changed\n")
        with self.assertRaises(ValueError):
            FraudScorer(copy)

    def test_line_endings_do_not_matter(self):
        copy = self.folder / "crlf"
        shutil.copytree(self.bundle, copy)
        text = (copy / "preprocessing.py").read_bytes().replace(b"\n", b"\r\n")
        (copy / "preprocessing.py").write_bytes(text)
        FraudScorer(copy)   # must not raise


if __name__ == "__main__":
    unittest.main()
