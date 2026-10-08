"""REST API that returns a fraud probability for incoming applications.

Start it locally (from the project folder):

    uvicorn service.api:create_app --factory --port 8000

and open http://localhost:8000/docs for the interactive documentation.

Access control: every request (except /health and /ready) needs an X-API-Key
header. There are four roles with separate keys, so for example the team that
enters investigation results cannot request predictions:

    predict  - the application system that sends new applications
    labels   - the investigation team that reports confirmed fraud later
    monitor  - people who may look at the monitoring summary
    admin    - Jenkins (reload the model after a deployment) and me

The model is never retrained inside the API. Jenkins writes a new version into
the models folder and changes models/current.json. The API checks this file
every few seconds (and immediately on POST /admin/reload) and swaps the model
only after the new one has loaded successfully, so the service stays reachable.
"""

import hashlib
import json
import logging
import os
import secrets
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Security, status
from fastapi.security import APIKeyHeader
from sqlalchemy import delete, distinct, func, select, text
from sqlalchemy.exc import IntegrityError

from fraud import model_registry
from fraud.preprocessing import FEATURE_COLUMNS
from fraud.scoring import FraudScorer
from service import db
from service.schemas import (
    Application, BatchRequest, BatchResponse, LabelBatch, LabelResponse, Prediction,
)


log = logging.getLogger("fraud_api")
ROLES = ("predict", "labels", "monitor", "admin")


class Settings:
    """Configuration from environment variables (see .env.example)."""

    def __init__(self, env=None):
        env = os.environ if env is None else env
        self.models_dir = Path(env.get("MODEL_DIR", model_registry.default_models_dir()))
        self.database_url = env.get("DATABASE_URL", db.database_url())
        self.model_check_seconds = float(env.get("MODEL_CHECK_SECONDS", "5"))
        self.api_keys = {}
        for role in ROLES:
            key = env.get(f"API_KEY_{role.upper()}")
            if key:
                if len(key) < 16:
                    raise ValueError(f"API_KEY_{role.upper()} is too short (use at least 16 characters).")
                self.api_keys[role] = key
        if "admin" not in self.api_keys:
            # I do not want a default password in the code, so the API refuses to start.
            raise RuntimeError("Set API_KEY_ADMIN (and the other API_KEY_* variables) before starting the API.")

    def role_for(self, key):
        if not key:
            return None
        for role, expected in self.api_keys.items():
            # compare_digest takes the same time for every wrong key (no timing attacks)
            if secrets.compare_digest(key.encode(), expected.encode()):
                return role
        return None


class ModelHolder:
    """Keeps the current model in memory and notices when current.json changes."""

    def __init__(self, models_dir, check_seconds):
        self.models_dir = Path(models_dir)
        self.check_seconds = check_seconds
        self.scorer = None
        self.loaded_at = None
        self.last_error = None
        self._last_check = 0.0
        self._lock = threading.Lock()

    def get(self):
        if time.monotonic() - self._last_check >= self.check_seconds:
            self.refresh()
        if self.scorer is None:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "No model is loaded yet.")
        return self.scorer

    def refresh(self, force=False):
        """Load the version from current.json if it differs from the loaded one."""
        with self._lock:
            self._last_check = time.monotonic()
            try:
                current = model_registry.read_current(self.models_dir)
                if current is None:
                    raise FileNotFoundError(f"{self.models_dir / 'current.json'} does not exist.")
                wanted_threshold = current.get("threshold")
                if not force and self.scorer is not None and self.scorer.version == current["version"]:
                    # Same model, but the weekly job may have set a new workload threshold.
                    # Changing one number does not need a reload of the model.
                    if wanted_threshold is not None and wanted_threshold != self.scorer.threshold:
                        old_threshold = self.scorer.threshold
                        self.scorer.use_threshold(wanted_threshold, current.get("threshold_reason", "current.json"))
                        log.info("Threshold of %s: %.5f -> %.5f", self.scorer.version, old_threshold, wanted_threshold)
                        return True
                    return False
                new_scorer = FraudScorer(model_registry.version_dir(self.models_dir, current["version"]))
                if wanted_threshold is not None:
                    new_scorer.use_threshold(wanted_threshold, current.get("threshold_reason", "current.json"))
                # Swap only after the new model loaded completely. If loading fails,
                # the old model simply keeps serving.
                old_version = self.scorer.version if self.scorer else None
                self.scorer, self.loaded_at, self.last_error = new_scorer, db.now_utc(), None
                log.info("Loaded model %s (before: %s)", new_scorer.version, old_version)
                return True
            except Exception as error:   # noqa: BLE001 - any loading problem keeps the old model
                self.last_error = f"{type(error).__name__}: {error}"
                log.error("Could not load the model: %s", self.last_error)
                return False


def payload_hash(application):
    """Fingerprint of the full request item, used to recognise retries."""
    content = json.dumps(application.model_dump(mode="json"), sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()


def create_app(settings=None):
    settings = settings or Settings()
    holder = ModelHolder(settings.models_dir, settings.model_check_seconds)
    key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

    @asynccontextmanager
    async def lifespan(app):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
        app.state.engine = db.get_engine(settings.database_url)
        holder.refresh(force=True)
        yield

    app = FastAPI(
        title="Fraud detection API",
        description="Returns the probability that a benefit application is fraudulent. "
                    "The score only prioritises manual reviews; it never rejects an application.",
        version="1.0",
        lifespan=lifespan,
    )

    def require(*allowed_roles):
        """Dependency that checks the API key; admin may use every endpoint."""
        def check(api_key: str | None = Security(key_header)):
            role = settings.role_for(api_key)
            if role is None:
                raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid API key.")
            if role != "admin" and role not in allowed_roles:
                raise HTTPException(status.HTTP_403_FORBIDDEN, f"The role '{role}' may not use this endpoint.")
            return role
        return check

    def score_applications(items, store=True):
        """Score a list of applications, store them and return the predictions."""
        scorer = holder.get()
        engine = app.state.engine
        keys = [(item.application_id, payload_hash(item)) for item in items]

        if not store:
            # Smoke test after a deployment: score, but keep it out of the prediction log.
            features = pd.DataFrame([item.features.model_dump() for item in items], columns=FEATURE_COLUMNS)
            probabilities, flagged = scorer.score(features)
            now = db.now_utc()
            return scorer.version, [Prediction(
                prediction_id="not-stored", application_id=item.application_id,
                fraud_probability=float(p), flagged_for_review=bool(f), threshold=scorer.threshold,
                model_version=scorer.version, scored_at=now, duplicate=False,
            ) for item, p, f in zip(items, probabilities, flagged)]

        # 1) Retries: return the stored result instead of scoring twice.
        with engine.connect() as conn:
            stored = conn.execute(
                select(db.predictions).where(db.predictions.c.application_id.in_({k[0] for k in keys}))
            ).mappings().all()
        known = {(row["application_id"], row["payload_hash"]): row for row in stored}

        # 2) Score everything new in one call (much faster than row by row).
        new_positions = {}
        for position, key in enumerate(keys):
            if key not in known and key not in new_positions:
                new_positions[key] = position
        new_rows = []
        if new_positions:
            batch = [items[p] for p in new_positions.values()]
            features = pd.DataFrame([item.features.model_dump() for item in batch], columns=FEATURE_COLUMNS)
            probabilities, flagged = scorer.score(features)
            scored_at = db.now_utc()
            for item, key, probability, is_flagged in zip(batch, new_positions, probabilities, flagged):
                row = {
                    "prediction_id": str(uuid.uuid4()),
                    "application_id": item.application_id,
                    "payload_hash": key[1],
                    "event_time": db.utc_naive(item.event_time) or scored_at,
                    "scored_at": scored_at,
                    "model_version": scorer.version,
                    "fraud_probability": float(probability),
                    "flagged": bool(is_flagged),
                    "threshold": scorer.threshold,
                    **item.features.model_dump(),
                }
                new_rows.append(row)
            try:
                with engine.begin() as conn:
                    conn.execute(db.predictions.insert(), new_rows)
            except IntegrityError:
                # Two identical requests at the same moment; the client can simply retry.
                raise HTTPException(status.HTTP_409_CONFLICT, "Duplicate request in progress, please retry.")

        new_by_key = {(row["application_id"], row["payload_hash"]): row for row in new_rows}
        results = []
        for position, key in enumerate(keys):
            row = known.get(key) or new_by_key[key]
            first_time = key in new_by_key and new_positions[key] == position
            results.append(Prediction(
                prediction_id=row["prediction_id"],
                application_id=row["application_id"],
                fraud_probability=row["fraud_probability"],
                flagged_for_review=row["flagged"],
                threshold=row["threshold"],
                model_version=row["model_version"],
                scored_at=row["scored_at"],
                duplicate=not first_time,
            ))
        return scorer.version, results

    @app.get("/health", tags=["status"])
    def health():
        """Liveness check without authentication (used by Docker)."""
        return {"status": "ok"}

    @app.get("/ready", tags=["status"])
    def ready():
        """Ready = a model is loaded and the database answers."""
        try:
            with app.state.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
        except Exception as error:   # noqa: BLE001
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, f"Database not reachable: {error}")
        if holder.scorer is None:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, f"No model loaded: {holder.last_error}")
        return {"status": "ready", "model_version": holder.scorer.version}

    @app.get("/model-info", tags=["status"])
    def model_info(role: str = Depends(require(*ROLES))):
        scorer = holder.get()
        return {**scorer.info(), "loaded_at": holder.loaded_at, "last_reload_error": holder.last_error}

    @app.post("/predict", response_model=Prediction, tags=["predictions"])
    def predict(application: Application, role: str = Depends(require("predict"))):
        return score_applications([application])[1][0]

    @app.post("/predict/batch", response_model=BatchResponse, tags=["predictions"])
    def predict_batch(request: BatchRequest, store: bool = True, role: str = Depends(require("predict"))):
        if not store and role != "admin":
            raise HTTPException(status.HTTP_403_FORBIDDEN, "Only the admin role may score without storing.")
        version, results = score_applications(request.applications, store=store)
        return BatchResponse(model_version=version, predictions=results)

    @app.post("/labels", response_model=LabelResponse, tags=["labels"])
    def add_labels(batch: LabelBatch, role: str = Depends(require("labels"))):
        """Store confirmed outcomes. A second label for the same application replaces the first."""
        now = db.now_utc()
        rows = [{
            "application_id": label.application_id,
            "fraud_bool": label.fraud_bool,
            "label_available_at": db.utc_naive(label.label_available_at) or now,
            "label_source": label.label_source,
            "weight": label.weight,
            "received_at": now,
        } for label in batch.labels]
        ids = [row["application_id"] for row in rows]
        if len(set(ids)) != len(ids):
            raise HTTPException(422, "Each application may appear only once per request.")
        with app.state.engine.begin() as conn:
            conn.execute(delete(db.labels).where(db.labels.c.application_id.in_(ids)))
            conn.execute(db.labels.insert(), rows)
            matched = conn.execute(
                select(func.count(distinct(db.predictions.c.application_id)))
                .where(db.predictions.c.application_id.in_(ids))
            ).scalar_one()
        return LabelResponse(stored=len(rows), matched_predictions=matched)

    @app.post("/admin/reload", tags=["admin"])
    def reload_model(role: str = Depends(require("admin"))):
        """Called by Jenkins right after a deployment (or a rollback)."""
        holder.refresh(force=True)
        if holder.last_error:
            raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, holder.last_error)
        return {"reloaded": True, **holder.scorer.info()}

    @app.get("/monitoring/summary", tags=["monitoring"])
    def monitoring_summary(role: str = Depends(require("monitor"))):
        """Latest monitoring result and pipeline activity (details are in MLflow)."""
        engine = app.state.engine
        with engine.connect() as conn:
            n_predictions = conn.execute(select(func.count()).select_from(db.predictions)).scalar_one()
            n_labels = conn.execute(select(func.count()).select_from(db.labels)).scalar_one()
        latest = {kind: db.last_event(engine, kind) for kind in ("monitoring", "training", "deployment", "incident")}
        return {
            "model_version": holder.scorer.version if holder.scorer else None,
            "predictions_stored": n_predictions,
            "labels_stored": n_labels,
            "latest_events": latest,
        }

    return app
