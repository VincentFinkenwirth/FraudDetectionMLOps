"""Synthetic applications and a tiny model bundle for the tests.

The real data is 213 MB and not in git, so the tests make their own small
data with the same columns. Fraud depends on a few features, so a model can
actually learn something.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from fraud.preprocessing import CATEGORICAL_FEATURES, FEATURE_COLUMNS, TARGET_COLUMN, prepare_dataset
from training.train import calibrated_config, evaluate_model, write_bundle

CATEGORIES = {
    "payment_type": ["AA", "AB", "AC", "AD"],
    "employment_status": ["CA", "CB", "CC", "CD"],
    "housing_status": ["BA", "BB", "BC", "BE"],
    "source": ["INTERNET", "TELEAPP"],
    "device_os": ["windows", "linux", "macintosh", "other", "x11"],
}


def make_applications(n=4000, seed=0, fraud_rate=0.05):
    rng = np.random.default_rng(seed)
    data = {}
    for column in FEATURE_COLUMNS:
        if column in CATEGORICAL_FEATURES:
            data[column] = rng.choice(CATEGORIES[column], size=n)
        elif column in ("email_is_free", "phone_home_valid", "phone_mobile_valid", "has_other_cards",
                        "foreign_request", "keep_alive_session"):
            data[column] = rng.integers(0, 2, size=n)
        elif column == "income":
            data[column] = rng.choice(np.round(np.arange(0.1, 1.0, 0.1), 1), size=n)
        elif column == "customer_age":
            data[column] = rng.choice([20, 30, 40, 50, 60], size=n)
        else:
            data[column] = np.round(rng.gamma(2.0, 20.0, size=n), 3)
    frame = pd.DataFrame(data)
    # A few sentinel values like in BAF (-1 = missing)
    frame.loc[rng.random(n) < 0.3, "prev_address_months_count"] = -1
    frame.loc[rng.random(n) < 0.5, "intended_balcon_amount"] = -5.0

    # Fraud is more likely on Windows, without other cards and with a high risk score.
    logit = (np.log(fraud_rate / (1 - fraud_rate)) + 1.2 * (frame["device_os"] == "windows")
             - 1.0 * frame["has_other_cards"] + 0.02 * (frame["credit_risk_score"] - 40))
    frame[TARGET_COLUMN] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    frame.insert(0, "source_id", np.arange(n) + seed * 1_000_000)
    return frame


def make_bundle(folder, seed=0, class_weights="SqrtBalanced", version="v1"):
    """Train a very small CatBoost model and save it like train.py does."""
    train = make_applications(3000, seed=seed)
    val = make_applications(2000, seed=seed + 100)
    X, y = prepare_dataset(train)
    X_val, y_val = prepare_dataset(val)
    model = CatBoostClassifier(iterations=30, depth=3, auto_class_weights=class_weights, verbose=0,
                               random_seed=seed, cat_features=CATEGORICAL_FEATURES, allow_writing_files=False)
    model.fit(X, y, eval_set=(X_val, y_val))
    config = {"model_params": {"iterations": 30}, "fit_params": {}, "threshold": 0.5,
              "threshold_selection": {"method": "recall_at_fpr", "max_fpr": 0.05}}
    metrics, _, scores = evaluate_model(model, val, 0.5, config["threshold_selection"], return_predictions=True)
    write_bundle(folder, model, calibrated_config(config, model, metrics["decision_threshold"]),
                 metrics, val, scores, {"model_version": version, "run_id": f"test-{seed}"})
    return Path(folder)


def temp_dir():
    return Path(tempfile.mkdtemp(prefix="fraud_test_"))
