"""Small helpers shared by the monitoring and the retraining job."""

import json
import os
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import requests

import model_registry
from service import db


PROJECT_DIR = Path(__file__).resolve().parents[1]

# With an MLflow *server* MLflow prints "View run ..." with an emoji. On Windows
# the console often uses cp1252 and the job crashed with UnicodeEncodeError, so
# characters that cannot be shown are replaced instead.
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(errors="replace")


def load_json(path):
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_DIR / path
    return json.loads(path.read_text(encoding="utf-8"))


def parse_time(value):
    """Accept an ISO date string (from Jenkins) or a datetime; return naive UTC."""
    if value in (None, ""):
        return db.now_utc()
    return db.utc_naive(value)


def week_number(as_of, start):
    """Simulated week (1, 2, ...) that ends at `as_of`. Used as the MLflow step."""
    return int(round((parse_time(as_of) - parse_time(start)).total_seconds() / (7 * 24 * 3600)))


def models_dir():
    return model_registry.default_models_dir()


def to_jsonable(value):
    """Convert numpy numbers, NaN and datetimes so json.dumps and the database accept them."""
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def effective_count(weights):
    """Kish's effective sample size: how many equally weighted cases the rows are worth.

    100 reviewed fraud cases (weight 1 each) are worth 100. Two audited fraud
    cases with weight 50 also "stand for" 100 cases, but they are only worth
    about 2 - a recall estimate from them is almost pure noise.
    """
    weights = np.asarray(weights, dtype=float)
    return float(weights.sum() ** 2 / (weights ** 2).sum()) if len(weights) else 0.0


def workload_threshold(probabilities, review_share):
    """Threshold that sends `review_share` of these applications to review.

    It only needs the scores, no labels. After the new credit-bureau scale the
    threshold that was chosen on (old) validation data sent up to 19 % of the
    applications to review, and it took 8 weeks until labelled data of the new
    scale could fix it. The review team has a fixed capacity anyway, so I set
    the threshold from the newest applications instead: the model still decides
    WHO is reviewed, the capacity decides HOW MANY.
    """
    probabilities = np.asarray(probabilities, dtype=float)
    if len(probabilities) == 0:
        raise ValueError("A workload threshold needs at least one scored application.")
    if not 0 < review_share < 1:
        raise ValueError("review_share must be between 0 and 1.")
    return float(np.quantile(probabilities, 1 - review_share))


def write_json(path, values):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(to_jsonable(values), indent=2) + "\n", encoding="utf-8")


def api_settings():
    """URL and admin key of the REST API (only needed for deployments)."""
    return os.environ.get("API_URL", ""), os.environ.get("API_KEY_ADMIN", "")


def reload_api(api_url, admin_key, expected_version=None, expected_threshold=None):
    """Tell the API to read models/current.json again and check what it serves now.

    The API also checks current.json by itself every few seconds, but in my fast
    local runs a simulated week took only ~8 seconds, and most weekly threshold
    changes arrived after the next applications were already scored. So the
    jobs do not rely on that: they reload the API and check the answer.
    """
    headers = {"X-API-Key": admin_key}
    response = requests.post(f"{api_url}/admin/reload", headers=headers, timeout=60)
    response.raise_for_status()
    info = requests.get(f"{api_url}/model-info", headers=headers, timeout=10).json()
    if expected_version is not None and info["model_version"] != expected_version:
        raise RuntimeError(f"The API serves {info['model_version']} instead of {expected_version}.")
    if expected_threshold is not None and abs(info["threshold"] - expected_threshold) > 1e-12:
        raise RuntimeError(f"The API uses threshold {info['threshold']} instead of {expected_threshold}.")
    return info
