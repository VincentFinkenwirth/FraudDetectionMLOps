"""Database tables for predictions, delayed labels and pipeline events.

I use SQLAlchemy Core so the same code works with SQLite (local runs and
tests) and PostgreSQL (Docker Compose). Only the DATABASE_URL changes.

Tables:
- predictions:     every served score together with the exact input features
- labels:          fraud outcomes that arrive about one month later
- pipeline_events: monitoring results, trainings, deployments, incidents
"""

import os
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import pandas as pd
from sqlalchemy import (
    JSON, Boolean, Column, DateTime, Float, Integer, MetaData, String, Table,
    UniqueConstraint, create_engine, func, select,
)

from preprocessing import CATEGORICAL_FEATURES, FEATURE_COLUMNS


PROJECT_DIR = Path(__file__).resolve().parents[1]
metadata = MetaData()


def _feature_columns():
    # One column per model feature instead of one JSON blob: loading 300,000
    # rows for retraining is much faster and the database checks the types.
    return [
        Column(name, String(32) if name in CATEGORICAL_FEATURES else Float, nullable=True)
        for name in FEATURE_COLUMNS
    ]


predictions = Table(
    "predictions", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("prediction_id", String(36), nullable=False, unique=True),
    Column("application_id", String(64), nullable=False, index=True),
    Column("payload_hash", String(64), nullable=False),
    Column("event_time", DateTime, nullable=False, index=True),
    Column("scored_at", DateTime, nullable=False),
    Column("model_version", String(32), nullable=False, index=True),
    Column("fraud_probability", Float, nullable=False),
    Column("flagged", Boolean, nullable=False),
    Column("threshold", Float, nullable=False),
    *_feature_columns(),
    # A retry with the same payload must not create a second prediction.
    UniqueConstraint("application_id", "payload_hash", name="uq_prediction_payload"),
)

labels = Table(
    "labels", metadata,
    Column("application_id", String(64), primary_key=True),
    Column("fraud_bool", Integer, nullable=False),
    Column("label_available_at", DateTime, nullable=False, index=True),
    Column("label_source", String(64), nullable=True),
    # How many applications this label stands for: 1 for a reviewed (flagged)
    # case, 1 / audit share for a randomly audited unflagged case.
    Column("weight", Float, nullable=False, default=1.0),
    Column("received_at", DateTime, nullable=False),
)

events = Table(
    "pipeline_events", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("as_of", DateTime, nullable=False, index=True),
    Column("kind", String(32), nullable=False, index=True),
    Column("status", String(48), nullable=False),
    Column("model_version", String(32), nullable=True),
    Column("details", JSON, nullable=True),
    Column("created_at", DateTime, nullable=False),
)


def database_url():
    return os.environ.get("DATABASE_URL", f"sqlite:///{(PROJECT_DIR / 'work' / 'fraud.db').as_posix()}")


@lru_cache(maxsize=4)
def get_engine(url=None):
    url = url or database_url()
    if url.startswith("sqlite:///"):
        Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    # pool_pre_ping reconnects after a database restart instead of failing once.
    engine = create_engine(url, pool_pre_ping=True)
    metadata.create_all(engine)
    return engine


def utc_naive(value):
    """Store every timestamp as naive UTC so SQLite and PostgreSQL behave the same."""
    if value is None:
        return None
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if value.tzinfo is not None:
        value = value.astimezone(timezone.utc).replace(tzinfo=None)
    return value


def now_utc():
    return datetime.now(timezone.utc).replace(tzinfo=None)


def add_event(engine, kind, status, as_of, model_version=None, details=None):
    with engine.begin() as conn:
        conn.execute(events.insert().values(
            as_of=utc_naive(as_of), kind=kind, status=status, model_version=model_version,
            details=details, created_at=now_utc(),
        ))


def last_event(engine, kind, statuses=None, before=None):
    """Latest event of one kind (optionally only with certain statuses) as a dict."""
    query = select(events).where(events.c.kind == kind)
    if statuses:
        query = query.where(events.c.status.in_(statuses))
    if before is not None:
        query = query.where(events.c.as_of <= utc_naive(before))
    query = query.order_by(events.c.as_of.desc(), events.c.id.desc()).limit(1)
    with engine.connect() as conn:
        row = conn.execute(query).mappings().first()
    return dict(row) if row else None


def read_events(engine, kind=None):
    query = select(events).order_by(events.c.as_of, events.c.id)
    if kind:
        query = query.where(events.c.kind == kind)
    with engine.connect() as conn:
        return pd.DataFrame(conn.execute(query).mappings().all())


def read_predictions(engine, start, end):
    """All predictions with start <= event_time < end, including the features."""
    query = (select(predictions)
             .where(predictions.c.event_time >= utc_naive(start))
             .where(predictions.c.event_time < utc_naive(end))
             .order_by(predictions.c.id))
    with engine.connect() as conn:
        return pd.read_sql(query, conn)


def read_labelled(engine, labels_until, labels_after=None):
    """Predictions joined with the labels that were known at `labels_until`.

    If an application was scored more than once I keep the first prediction,
    because that is the score the reviewers actually saw.
    """
    query = (select(predictions, labels.c.fraud_bool, labels.c.label_available_at, labels.c.weight,
                    labels.c.label_source)
             .join(labels, labels.c.application_id == predictions.c.application_id)
             .where(labels.c.label_available_at <= utc_naive(labels_until))
             .order_by(predictions.c.id))
    if labels_after is not None:
        query = query.where(labels.c.label_available_at > utc_naive(labels_after))
    with engine.connect() as conn:
        data = pd.read_sql(query, conn)
    return data.drop_duplicates("application_id", keep="first").reset_index(drop=True)


def count_labelled_since(engine, event_after, labels_after, labels_until, event_until=None):
    """Labelled applications sent after event_after whose label arrived in (labels_after, labels_until].

    Counted with their weights, i.e. how many applications these labels stand for.
    event_until (inclusive) leaves out newer applications, e.g. the ones a
    retraining would put into the holdout instead of the fit data.
    """
    first_event = (select(predictions.c.application_id, func.min(predictions.c.event_time).label("event_time"))
                   .group_by(predictions.c.application_id).subquery())
    query = (select(func.coalesce(func.sum(labels.c.weight), 0))
             .join(first_event, labels.c.application_id == first_event.c.application_id)
             .where(first_event.c.event_time >= utc_naive(event_after))
             .where(labels.c.label_available_at > utc_naive(labels_after))
             .where(labels.c.label_available_at <= utc_naive(labels_until)))
    if event_until is not None:
        query = query.where(first_event.c.event_time <= utc_naive(event_until))
    with engine.connect() as conn:
        return int(round(float(conn.execute(query).scalar_one())))


def newest_labelled_event(engine, labels_until):
    """Event time of the newest application whose label is known at labels_until (None if there is none)."""
    query = (select(func.max(predictions.c.event_time))
             .join(labels, labels.c.application_id == predictions.c.application_id)
             .where(labels.c.label_available_at <= utc_naive(labels_until)))
    with engine.connect() as conn:
        return conn.execute(query).scalar_one()


def count_labels(engine, after, until):
    """Weighted number of labels that arrived in (after, until]."""
    query = (select(func.coalesce(func.sum(labels.c.weight), 0)).select_from(labels)
             .where(labels.c.label_available_at > utc_naive(after))
             .where(labels.c.label_available_at <= utc_naive(until)))
    with engine.connect() as conn:
        return int(round(float(conn.execute(query).scalar_one())))
