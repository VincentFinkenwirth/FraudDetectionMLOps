import unittest

from pipeline.check_environment import PROJECT_DIR, mismatches, read_lock


class LockFileTests(unittest.TestCase):
    def test_api_lock_uses_the_same_versions_as_the_main_lock(self):
        main = read_lock(PROJECT_DIR / "requirements.lock")
        api = read_lock(PROJECT_DIR / "requirements-api.lock")
        self.assertTrue(api)
        for name, version in api.items():
            self.assertEqual(main.get(name), version, f"{name} differs between the two lock files")

    def test_model_library_is_the_one_that_trained_v1(self):
        # Changing CatBoost means retraining/repackaging, so it is a deliberate step.
        self.assertEqual(read_lock(PROJECT_DIR / "requirements.lock")["catboost"], "1.2.10")

    def test_read_lock_normalises_names_and_skips_comments(self):
        lock = read_lock(lines=["# comment", "Scikit_Learn==1.9.1", "    # via mlflow", "", "numpy==2.5.3"])
        self.assertEqual(lock, {"scikit-learn": "1.9.1", "numpy": "2.5.3"})

    def test_mismatches(self):
        installed = {"numpy": "2.5.3", "pandas": "3.0.5"}
        found = mismatches({"numpy": "2.5.3", "pandas": "3.0.6", "catboost": "1.2.10"}, installed.get)
        self.assertEqual(found, {"pandas": ("3.0.6", "3.0.5"), "catboost": ("1.2.10", None)})


if __name__ == "__main__":
    unittest.main()
