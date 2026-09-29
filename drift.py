"""Measure how much new applications differ from the data a model was accepted on.

I use the Population Stability Index (PSI) per feature:

    PSI = sum over bins (current% - reference%) * ln(current% / reference%)

Rule of thumb from credit scoring: < 0.1 small change, 0.1-0.25 moderate,
> 0.25 large. With 30,000 rows a normal statistical test (KS, chi-squared) calls
almost every tiny difference "significant", so an effect size like PSI is more
useful here.

A problem I found in the data: velocity_4w has a PSI of about 4.5 between the
real months, but the model hardly uses it (rank 16, 2 % importance). So I do not
count drifting features, I add up the *model importance* of the drifting
features ("drift share"). 0.43 would mean that features which make up 43 % of
the model's importance changed noticeably.
"""

import numpy as np
import pandas as pd

from preprocessing import CATEGORICAL_FEATURES, FEATURE_COLUMNS


PSI_THRESHOLD = 0.10
N_BINS = 10
EPS = 1e-4          # avoids log(0) for empty bins
NORMALLY_COMPLETE = 0.05   # features with less than 5 % missing values in the reference
IMPORTANT_FEATURE = 0.05   # features with at least 5 % of the model importance
MISSING_CATEGORY = "__MISSING__"   # same placeholder as in preprocessing.py


def psi(reference, current):
    reference = np.clip(np.asarray(reference, dtype=float), EPS, None)
    current = np.clip(np.asarray(current, dtype=float), EPS, None)
    return float(np.sum((current - reference) * np.log(current / reference)))


def _numeric_distribution(values, edges):
    """Shares per quantile bin plus one extra bin for missing values."""
    values = np.asarray(values, dtype=float)
    missing = np.isnan(values)
    bins = np.searchsorted(edges, values[~missing], side="right")
    counts = np.bincount(bins, minlength=len(edges) + 1).astype(float)
    counts = np.append(counts, missing.sum())
    return counts / max(len(values), 1)


def _categorical_distribution(values, categories):
    """Shares per known category plus one bin for categories never seen before."""
    values = pd.Series(values).astype(str)
    counts = [float((values == category).sum()) for category in categories]
    counts.append(float((~values.isin(categories)).sum()))
    return np.asarray(counts) / max(len(values), 1)


def build_reference_profile(X, importance, probabilities, threshold, n_bins=N_BINS):
    """Save the "normal" state for one model version as a JSON-friendly dict.

    X must be prepared features (output of prepare_features). importance is a
    dict feature -> CatBoost importance. The flag rate is saved too, because a
    change in the number of alerts is the cheapest signal we get without labels.
    """
    features = {}
    for column in FEATURE_COLUMNS:
        if column in CATEGORICAL_FEATURES:
            values = X[column].astype(str)
            categories = sorted(values.unique().tolist())
            features[column] = {
                "type": "categorical",
                "categories": categories,
                "proportions": _categorical_distribution(values, categories).tolist(),
                "missing_rate": float((values == MISSING_CATEGORY).mean()),
            }
        else:
            values = X[column].astype(float)
            present = values.dropna()
            quantiles = np.linspace(0, 1, n_bins + 1)[1:-1]
            edges = np.unique(np.quantile(present, quantiles)) if len(present) else np.array([])
            features[column] = {
                "type": "numeric",
                "edges": edges.tolist(),
                "proportions": _numeric_distribution(values, edges).tolist(),
                "missing_rate": float(values.isna().mean()),
            }

    total = sum(importance.values()) or 1.0
    probabilities = np.asarray(probabilities, dtype=float)
    return {
        "rows": int(len(X)),
        "psi_threshold": PSI_THRESHOLD,
        "features": features,
        "importance": {name: float(value) / total for name, value in importance.items()},
        "threshold": float(threshold),
        "flag_rate": float(np.mean(probabilities >= threshold)),
        "mean_probability": float(probabilities.mean()),
    }


def compare_to_reference(profile, X, probabilities=None):
    """Compare a window of prepared features with the saved reference profile."""
    rows = len(X)
    if rows == 0:
        raise ValueError("The comparison window is empty.")

    feature_psi, missing_increase, unseen_share = {}, {}, {}
    for column, reference in profile["features"].items():
        if reference["type"] == "categorical":
            values = X[column].astype(str)
            current = _categorical_distribution(values, reference["categories"])
            unseen_share[column] = float(current[-1])
            current_missing = float((values == MISSING_CATEGORY).mean())
        else:
            values = X[column].astype(float)
            current = _numeric_distribution(values, np.asarray(reference["edges"]))
            current_missing = float(values.isna().mean())
        feature_psi[column] = psi(reference["proportions"], current)
        missing_increase[column] = current_missing - reference["missing_rate"]

    # In BAF a missing value is often normal behaviour (e.g. no previous address)
    # and its share moves by up to 18 percentage points between real months.
    # A data-quality problem is when a feature that is normally almost always
    # filled in suddenly goes missing, so only those features count here.
    complete = [c for c, ref in profile["features"].items() if ref["missing_rate"] < NORMALLY_COMPLETE]
    max_missing_complete = max((missing_increase[c] for c in complete), default=0.0)

    importance = profile["importance"]
    threshold = profile["psi_threshold"]
    drift_share = sum(importance.get(c, 0.0) for c, value in feature_psi.items() if value >= threshold)
    weighted_psi = sum(importance.get(c, 0.0) * value for c, value in feature_psi.items())
    top = sorted(feature_psi.items(), key=lambda item: item[1], reverse=True)[:5]
    # One strongly changed feature that the model really uses is also a problem,
    # even if the drift share is still small (e.g. a new scale for one score).
    important = {c: value for c, value in feature_psi.items() if importance.get(c, 0.0) >= IMPORTANT_FEATURE}
    worst_important = max(important, key=important.get) if important else None

    result = {
        "rows": int(rows),
        "feature_psi": feature_psi,
        "drift_share": float(drift_share),
        "weighted_psi": float(weighted_psi),
        "features_over_threshold": int(sum(value >= threshold for value in feature_psi.values())),
        "max_psi_important_feature": float(important[worst_important]) if worst_important else 0.0,
        "most_drifted_important_feature": worst_important,
        "top_drifted_features": [{"feature": name, "psi": value} for name, value in top],
        "max_missing_increase": float(max(missing_increase.values())),
        "max_missing_increase_complete_features": float(max_missing_complete),
        "missing_increase": missing_increase,
        "max_unseen_category_share": float(max(unseen_share.values(), default=0.0)),
        "unseen_category_share": unseen_share,
    }

    if probabilities is not None and len(probabilities):
        # Binomial z-score for the alert rate. With ~7,500 rows per week one
        # standard error is about 0.26 percentage points. The scores can come
        # from fewer rows than X (only the rows scored by the current model).
        probabilities = np.asarray(probabilities, dtype=float)
        p0 = profile["flag_rate"]
        flag_rate = float(np.mean(probabilities >= profile["threshold"]))
        standard_error = np.sqrt(p0 * (1 - p0) / len(probabilities)) if 0 < p0 < 1 else np.nan
        result.update({
            "flag_rate": flag_rate,
            "reference_flag_rate": p0,
            "flag_rate_z": float((flag_rate - p0) / standard_error) if np.isfinite(standard_error) else None,
            "mean_probability": float(np.mean(probabilities)),
        })
    return result
