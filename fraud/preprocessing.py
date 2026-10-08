"""Prepare model inputs for training, evaluation and the monthly simulation.

The API can use prepare_features() without needing a fraud label.
These functions work on copies and do not change the saved datasets.
"""

from math import nan

import pandas as pd


TARGET_COLUMN = "fraud_bool"

# CatBoost uses these column names to recognise the categorical features.
CATEGORICAL_FEATURES = [
    "payment_type",
    "employment_status",
    "housing_status",
    "source",
    "device_os",
]

# I keep one feature list so training and prediction use the same column order.
# source_id and month stay in the original data for reference.
# device_fraud_count is left out because it is constant in the training set.
FEATURE_COLUMNS = [
    "income",
    "name_email_similarity",
    "prev_address_months_count",
    "current_address_months_count",
    "customer_age",
    "days_since_request",
    "intended_balcon_amount",
    "payment_type",
    "zip_count_4w",
    "velocity_6h",
    "velocity_24h",
    "velocity_4w",
    "bank_branch_count_8w",
    "date_of_birth_distinct_emails_4w",
    "employment_status",
    "credit_risk_score",
    "email_is_free",
    "housing_status",
    "phone_home_valid",
    "phone_mobile_valid",
    "bank_months_count",
    "has_other_cards",
    "proposed_credit_limit",
    "foreign_request",
    "source",
    "session_length_in_minutes",
    "device_os",
    "keep_alive_session",
    "device_distinct_emails_8w",
]

# Only these columns use exactly -1 as a missing-value placeholder.
MISSING_VALUE_COLUMNS = [
    "prev_address_months_count",
    "current_address_months_count",
    "bank_months_count",
    "session_length_in_minutes",
    "device_distinct_emails_8w",
]


def prepare_features(data):
    """Return the cleaned input features without changing the original data."""
    missing_columns = [column for column in FEATURE_COLUMNS if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing input features: {', '.join(missing_columns)}")

    # Selecting this list also leaves out the label and any extra metadata.
    X = data.loc[:, FEATURE_COLUMNS].copy()

    for column in FEATURE_COLUMNS:
        if column in CATEGORICAL_FEATURES:
            # Missing categories need a string value instead of numeric NaN.
            X[column] = X[column].astype("string").fillna("__MISSING__").astype(object)
        else:
            # Floats let the numeric columns hold NaN as well as normal values.
            X[column] = pd.to_numeric(X[column], errors="raise").astype("float64")

    X[MISSING_VALUE_COLUMNS] = X[MISSING_VALUE_COLUMNS].replace(-1, nan)

    # In this column, every negative value means missing, not just -1.
    balance = X["intended_balcon_amount"]
    X["intended_balcon_amount"] = balance.mask(balance < 0, nan)

    # Other negative values, such as credit risk scores, are kept as they are.
    return X


def prepare_dataset(data):
    """Return X and y for a labelled training, evaluation or simulation dataset."""
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"The dataset must contain {TARGET_COLUMN}.")

    # I check the labels before converting them so invalid values are not rounded.
    y = data[TARGET_COLUMN].copy()
    if not y.isin([0, 1]).all():
        raise ValueError("Every fraud_bool value must be either 0 or 1.")
    y = y.astype("int64")

    X = prepare_features(data.drop(columns=[TARGET_COLUMN]))
    return X, y
