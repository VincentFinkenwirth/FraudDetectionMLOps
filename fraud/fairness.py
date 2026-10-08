"""Small fairness check for the age groups.

The BAF dataset was published for fairness research and the authors compare
false-positive rates between applicants aged 50+ and younger applicants
("predictive equality"). For a government agency this matters a lot: a false
positive means a person in need has to wait for a manual review.
"""

import numpy as np


AGE_LIMIT = 50


def _fpr(labels, flagged, weights):
    negatives = labels == 0
    if not negatives.any():
        return None
    return float(np.average(flagged[negatives], weights=weights[negatives]))


def fpr_by_age(labels, flagged, ages, age_limit=AGE_LIMIT, weights=None):
    """FPR for both age groups and their ratio (lower / higher, 1.0 = equal).

    Values are None if a group has no legitimate applications in the batch.
    weights: see calculate_metrics (only used with realistic labels).
    """
    labels = np.asarray(labels).astype(int)
    flagged = np.asarray(flagged).astype(bool)
    ages = np.asarray(ages, dtype=float)
    weights = np.ones(len(labels)) if weights is None else np.asarray(weights, dtype=float)
    if not len(labels) == len(flagged) == len(ages) == len(weights):
        raise ValueError("Labels, flags, ages and weights must have the same length.")

    older = ages >= age_limit
    fpr_older = _fpr(labels[older], flagged[older], weights[older])
    fpr_younger = _fpr(labels[~older], flagged[~older], weights[~older])

    ratio = None
    if fpr_older is not None and fpr_younger is not None:
        higher = max(fpr_older, fpr_younger)
        # If nobody is flagged in both groups they are treated equally.
        ratio = min(fpr_older, fpr_younger) / higher if higher > 0 else 1.0

    return {
        f"fpr_age_{age_limit}_plus": fpr_older,
        f"fpr_age_under_{age_limit}": fpr_younger,
        "fpr_ratio_age": ratio,
    }
