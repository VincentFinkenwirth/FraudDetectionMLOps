import json
import math
import shutil
import unittest

from fastapi.testclient import TestClient

from fraud import model_registry
from fraud.preprocessing import FEATURE_COLUMNS
from service.api import ModelHolder, Settings, create_app
from service.schemas import ApplicationFeatures
from tests.helpers import make_applications, make_bundle, temp_dir

KEYS = {
    "API_KEY_PREDICT": "predict-key-for-tests-123",
    "API_KEY_LABELS": "labels-key-for-tests-1234",
    "API_KEY_MONITOR": "monitor-key-for-tests-123",
    "API_KEY_ADMIN": "admin-key-for-tests-123456",
}


def as_requests(frame, prefix="APP"):
    items = []
    for i, row in frame.reset_index(drop=True).iterrows():
        features = {}
        for column in FEATURE_COLUMNS:
            value = row[column]
            value = value.item() if hasattr(value, "item") else value
            features[column] = None if isinstance(value, float) and math.isnan(value) else value
        items.append({"application_id": f"{prefix}-{i}", "event_time": "2025-01-06T12:00:00Z", "features": features})
    return items


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = temp_dir()
        cls.models = cls.folder / "models"
        make_bundle(cls.folder / "b1", seed=1)
        make_bundle(cls.folder / "b2", seed=2)
        model_registry.publish(cls.folder / "b1", cls.models, "v1")
        model_registry.publish(cls.folder / "b2", cls.models, "v2")
        model_registry.set_current(cls.models, "v1")
        settings = Settings({**KEYS, "MODEL_DIR": str(cls.models),
                             "DATABASE_URL": f"sqlite:///{(cls.folder / 'api.db').as_posix()}",
                             "MODEL_CHECK_SECONDS": "3600"})
        cls.client_context = TestClient(create_app(settings))
        cls.client = cls.client_context.__enter__()
        cls.applications = as_requests(make_applications(20, seed=5))

    @classmethod
    def tearDownClass(cls):
        cls.client_context.__exit__(None, None, None)
        shutil.rmtree(cls.folder, ignore_errors=True)

    def headers(self, role):
        return {"X-API-Key": KEYS[f"API_KEY_{role.upper()}"]}

    def test_schema_matches_feature_list(self):
        self.assertEqual(list(ApplicationFeatures.model_fields), FEATURE_COLUMNS)

    def test_health_needs_no_key(self):
        self.assertEqual(self.client.get("/health").status_code, 200)
        self.assertEqual(self.client.get("/ready").json()["status"], "ready")

    def test_keys_and_roles(self):
        item = self.applications[0]
        self.assertEqual(self.client.post("/predict", json=item).status_code, 401)
        self.assertEqual(self.client.post("/predict", json=item, headers={"X-API-Key": "wrong-key-123456789"}).status_code, 401)
        self.assertEqual(self.client.post("/predict", json=item, headers=self.headers("labels")).status_code, 403)
        self.assertEqual(self.client.post("/predict", json=item, headers=self.headers("admin")).status_code, 200)

    def test_batch_prediction_and_retry(self):
        batch = {"applications": self.applications[1:6]}
        first = self.client.post("/predict/batch", json=batch, headers=self.headers("predict")).json()
        self.assertEqual(len(first["predictions"]), 5)
        for prediction in first["predictions"]:
            self.assertTrue(0.0 <= prediction["fraud_probability"] <= 1.0)
            self.assertEqual(prediction["flagged_for_review"],
                             prediction["fraud_probability"] >= prediction["threshold"])
        # Sending the same batch again must return the stored predictions.
        again = self.client.post("/predict/batch", json=batch, headers=self.headers("predict")).json()
        self.assertTrue(all(p["duplicate"] for p in again["predictions"]))
        self.assertEqual([p["prediction_id"] for p in first["predictions"]],
                         [p["prediction_id"] for p in again["predictions"]])

    def test_invalid_input(self):
        item = json.loads(json.dumps(self.applications[7]))
        item["features"]["incme"] = 0.3                       # typo
        self.assertEqual(self.client.post("/predict", json=item, headers=self.headers("predict")).status_code, 422)
        item = json.loads(json.dumps(self.applications[7]))
        del item["features"]["income"]                          # missing field
        self.assertEqual(self.client.post("/predict", json=item, headers=self.headers("predict")).status_code, 422)
        item = json.loads(json.dumps(self.applications[7]))
        item["features"]["income"] = None                       # null = missing value is allowed
        self.assertEqual(self.client.post("/predict", json=item, headers=self.headers("predict")).status_code, 200)

    def test_store_false_only_for_admin(self):
        batch = {"applications": self.applications[8:10]}
        self.assertEqual(self.client.post("/predict/batch?store=false", json=batch,
                                          headers=self.headers("predict")).status_code, 403)
        response = self.client.post("/predict/batch?store=false", json=batch, headers=self.headers("admin"))
        self.assertEqual(response.json()["predictions"][0]["prediction_id"], "not-stored")

    def test_labels(self):
        self.client.post("/predict", json=self.applications[10], headers=self.headers("predict"))
        body = {"labels": [{"application_id": "APP-10", "fraud_bool": 1},
                           {"application_id": "UNKNOWN-1", "fraud_bool": 0}]}
        self.assertEqual(self.client.post("/labels", json=body, headers=self.headers("predict")).status_code, 403)
        result = self.client.post("/labels", json=body, headers=self.headers("labels")).json()
        self.assertEqual(result, {"stored": 2, "matched_predictions": 1})
        audit = {"labels": [{"application_id": "APP-10", "fraud_bool": 0, "label_source": "audit", "weight": 50}]}
        self.assertEqual(self.client.post("/labels", json=audit, headers=self.headers("labels")).status_code, 200)
        too_small = {"labels": [{"application_id": "APP-10", "fraud_bool": 0, "weight": 0.5}]}
        self.assertEqual(self.client.post("/labels", json=too_small, headers=self.headers("labels")).status_code, 422)

    def test_monitoring_summary_role(self):
        self.assertEqual(self.client.get("/monitoring/summary", headers=self.headers("predict")).status_code, 403)
        self.assertEqual(self.client.get("/monitoring/summary", headers=self.headers("monitor")).status_code, 200)

    def test_reload_switches_version_and_keeps_old_model_on_error(self):
        self.assertEqual(self.client.get("/model-info", headers=self.headers("admin")).json()["model_version"], "v1")
        model_registry.set_current(self.models, "v2")
        self.assertEqual(self.client.post("/admin/reload", headers=self.headers("admin")).json()["model_version"], "v2")

        # A broken version must not replace the working one.
        broken = self.models / "versions" / "v3"
        shutil.copytree(self.models / "versions" / "v2", broken)
        (broken / "model.cbm").write_bytes(b"this is not a model")
        model_registry.set_current(self.models, "v3")
        self.assertEqual(self.client.post("/admin/reload", headers=self.headers("admin")).status_code, 500)
        self.assertEqual(self.client.get("/model-info", headers=self.headers("admin")).json()["model_version"], "v2")
        model_registry.set_current(self.models, "v1")
        self.client.post("/admin/reload", headers=self.headers("admin"))


    def test_new_workload_threshold_is_used_without_new_model(self):
        info = self.client.get("/model-info", headers=self.headers("admin")).json()
        model_registry.set_threshold(self.models, info["model_version"], 0.99, reason="test")
        info = self.client.post("/admin/reload", headers=self.headers("admin")).json()
        self.assertAlmostEqual(info["threshold"], 0.99)
        self.assertEqual(info["threshold_source"], "test")
        # A threshold of 0.99 flags (almost) nothing, and the stored prediction keeps it.
        answer = self.client.post("/predict", json={**self.applications[0], "application_id": "THRESHOLD-1"},
                                  headers=self.headers("predict")).json()
        self.assertAlmostEqual(answer["threshold"], 0.99)
        # A new deployment starts again with the threshold from its bundle.
        model_registry.set_current(self.models, info["model_version"])
        info = self.client.post("/admin/reload", headers=self.headers("admin")).json()
        self.assertAlmostEqual(info["threshold"], info["bundle_threshold"])


class ModelHolderTests(unittest.TestCase):
    def test_threshold_change_does_not_touch_the_scorer_in_use(self):
        # A request keeps the scorer it got at the start. If the weekly threshold
        # changed that object, flags and stored threshold could disagree.
        folder = temp_dir()
        self.addCleanup(shutil.rmtree, folder, True)
        make_bundle(folder / "b1", seed=1)
        model_registry.publish(folder / "b1", folder / "models", "v1")
        model_registry.set_current(folder / "models", "v1")
        holder = ModelHolder(folder / "models", 3600)
        holder.refresh(force=True)
        in_use = holder.scorer
        old_threshold = in_use.threshold
        model_registry.set_threshold(folder / "models", "v1", 0.99, reason="test")
        self.assertTrue(holder.refresh())
        self.assertEqual(in_use.threshold, old_threshold)
        self.assertAlmostEqual(holder.scorer.threshold, 0.99)
        self.assertIs(holder.scorer.model, in_use.model)


class SettingsTests(unittest.TestCase):
    def test_admin_key_is_required(self):
        with self.assertRaises(RuntimeError):
            Settings({"API_KEY_PREDICT": KEYS["API_KEY_PREDICT"]})

    def test_short_keys_are_refused(self):
        with self.assertRaises(ValueError):
            Settings({"API_KEY_ADMIN": "short"})


if __name__ == "__main__":
    unittest.main()
