"""Turn CatBoost scores into fraud probabilities and load deployable model bundles.

Why this file exists: the best model from tuning uses SqrtBalanced class
weights. During training every fraud row therefore counts about 10 times as
much as a normal row, and predict_proba() gives far too high numbers (mean
8.5 % on validation although only 0.9 % of the applications are fraud). The
Brier score was even worse than always predicting the base rate.

The task asks the API for a *probability* of fraud, so I undo the weighting:

    odds_real = odds_weighted / k,   k = weight(fraud) / weight(non-fraud)

This is the usual "prior correction" for re-weighted training data. The
function is strictly increasing, so the ranking of the applications (and with
the matching threshold also recall and FPR) stays exactly the same.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import catboost
from catboost import CatBoostClassifier

from fraud.preprocessing import CATEGORICAL_FEATURES, FEATURE_COLUMNS, prepare_features


PROJECT_DIR = Path(__file__).resolve().parents[1]
CALIBRATION_METHOD = "class_weight_prior_correction"


def class_weight_ratio(model):
    """Read k = weight(fraud) / weight(non-fraud) from a fitted CatBoost model.

    CatBoost saves the class weights it actually used inside the model file,
    also for auto_class_weights, so I do not need the training labels here.
    Without class weights the ratio is simply 1.
    """
    weights = model.get_all_params().get("class_weights")
    if not weights:
        return 1.0
    if len(weights) != 2 or weights[0] <= 0:
        raise ValueError(f"Expected two positive class weights, got {weights}.")
    return float(weights[1]) / float(weights[0])


def correct_for_class_weights(raw_scores, odds_divisor):
    """Map scores from a class-weighted model back to probabilities.

    p = s / (s + k * (1 - s)) is the same as dividing the odds by k.
    """
    scores = np.asarray(raw_scores, dtype=float)
    if odds_divisor <= 0:
        raise ValueError("The odds divisor must be positive.")
    if odds_divisor == 1:
        return scores
    return scores / (scores + odds_divisor * (1.0 - scores))


def fraud_probabilities(model, X, odds_divisor=None):
    """predict_proba() for already prepared features, with the correction applied."""
    if odds_divisor is None:
        odds_divisor = class_weight_ratio(model)
    return correct_for_class_weights(model.predict_proba(X)[:, 1], odds_divisor)


def fit_prior_adjustment(probabilities, labels, weights=None, low=0.25, high=4.0):
    """Extra odds divisor c, so that the mean probability equals the observed fraud rate.

    The class-weight correction gives probabilities for the fraud rate of the
    training data. A retrained model is fitted on old history + recent data, and
    the fraud rate keeps changing (prior shift), so in my v3 run the served
    models predicted between 0.7x and 1.5x the real rate. One more odds factor
    fixes the overall level ("calibration in the large") without changing the
    order of the applications. Mean probability falls when c grows, so a
    simple bisection finds c. I clamp it, so a strange holdout cannot produce
    absurd probabilities.
    """
    p = np.asarray(probabilities, dtype=float)
    w = np.ones(len(p)) if weights is None else np.asarray(weights, dtype=float)
    target = np.average(np.asarray(labels, dtype=float), weights=w)

    def mean_with(c):
        return np.average(correct_for_class_weights(p, c), weights=w)

    if mean_with(high) > target:
        return high
    if mean_with(low) < target:
        return low
    log_low, log_high = np.log(low), np.log(high)
    for _ in range(100):
        middle = (log_low + log_high) / 2
        if mean_with(np.exp(middle)) > target:
            log_low = middle
        else:
            log_high = middle
    return float(np.exp((log_low + log_high) / 2))


def calibration_summary(labels, probabilities, weights=None, bins=10):
    """How well the probabilities match the real fraud rate (the API promises probabilities).

    - brier: mean squared error of the probabilities; brier_base_rate is the
      same for always predicting the observed rate (a useful model is lower)
    - calibration_ratio: mean probability / observed rate (1 = right level)
    - ece: average gap between predicted and observed rate in `bins`
      groups of equal size (sorted by probability)
    With realistic labels the rows are weighted (an audited case counts
    1 / audit share times), so every number is a weighted mean.
    """
    y = np.asarray(labels, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    w = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=float)
    observed = float(np.average(y, weights=w))
    predicted = float(np.average(p, weights=w))

    order = np.argsort(p, kind="stable")
    ece = 0.0
    for group in np.array_split(order, min(bins, len(order))):
        group_weight = w[group].sum()
        if group_weight > 0:
            gap = np.average(p[group], weights=w[group]) - np.average(y[group], weights=w[group])
            ece += group_weight / w.sum() * abs(gap)
    return {
        "brier": float(np.average((p - y) ** 2, weights=w)),
        "brier_base_rate": float(np.average((observed - y) ** 2, weights=w)),
        "mean_predicted": predicted,
        "observed_rate": observed,
        "calibration_ratio": predicted / observed if observed > 0 else None,
        "ece": float(ece),
    }


def text_hash(path):
    """Hash a text file independent of Windows/Linux line endings.

    The API runs in a Linux container while I develop on Windows. A git checkout
    can change \\r\\n to \\n, which would change a normal byte hash.
    """
    text = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(text).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


class FraudScorer:
    """Everything the API needs from one model version.

    A bundle folder contains model.cbm, training_config.json (threshold on the
    probability scale + calibration), feature_schema.json, the preprocessing
    code used in training, metadata.json and reference_profile.json (drift).
    """

    def __init__(self, bundle_dir, *, check_preprocessing=True):
        self.bundle_dir = Path(bundle_dir).resolve()
        self.config = read_json(self.bundle_dir / "training_config.json")
        self.metadata = read_json(self.bundle_dir / "metadata.json")
        schema = read_json(self.bundle_dir / "feature_schema.json")

        # The model has to expect exactly the features that preprocessing.py creates.
        if schema["feature_columns"] != FEATURE_COLUMNS or schema["categorical_features"] != CATEGORICAL_FEATURES:
            raise ValueError("The bundle was trained with a different feature list.")
        if check_preprocessing and text_hash(self.bundle_dir / "preprocessing.py") != text_hash(PROJECT_DIR / "fraud" / "preprocessing.py"):
            raise ValueError("preprocessing.py differs from the copy saved with this model bundle.")

        calibration = self.config.get("probability_calibration")
        if not calibration or calibration.get("method") != CALIBRATION_METHOD:
            raise ValueError("This bundle has no probability calibration. Run python -m training.package_model first.")
        if self.config.get("threshold_scale") != "probability":
            raise ValueError("The saved threshold must be on the probability scale.")

        # Training (Jenkins) and serving (API image) install the same lock file. If
        # they ever differ, I would rather keep the old model than serve a model
        # file that another CatBoost version wrote (the hot reload keeps the old one).
        trained_with = self.metadata.get("library_versions", {}).get("catboost")
        if trained_with and trained_with != catboost.__version__:
            raise ValueError(f"The bundle was trained with CatBoost {trained_with}, "
                             f"but CatBoost {catboost.__version__} is installed.")

        self.model = CatBoostClassifier()
        self.model.load_model(str(self.bundle_dir / "model.cbm"))
        self.odds_divisor = float(calibration["odds_divisor"])
        # Small sanity check so a copied config cannot silently belong to another model.
        # A retrained model also has a prior adjustment (see fit_prior_adjustment),
        # then the divisor is class weight ratio * adjustment.
        expected = class_weight_ratio(self.model) * float(calibration.get("prior_adjustment", 1.0))
        if not np.isclose(self.odds_divisor, expected, rtol=1e-4):
            raise ValueError("The saved odds divisor does not match the model's class weights.")

        self.threshold = float(self.config["threshold"])
        # The threshold in the bundle was chosen on the validation data (FPR < 5 %).
        # The weekly monitoring job can replace it with a "workload" threshold
        # (see model_registry.set_threshold); the bundle itself never changes.
        self.bundle_threshold = self.threshold
        self.threshold_source = "bundle (validation, FPR < 5 %)"
        self.version = str(self.metadata["model_version"])
        profile_path = self.bundle_dir / "reference_profile.json"
        self.reference_profile = read_json(profile_path) if profile_path.exists() else None

    def predict_proba(self, data):
        """Raw application data (DataFrame) -> fraud probabilities."""
        return fraud_probabilities(self.model, prepare_features(data), self.odds_divisor)

    def score(self, data):
        """Return (probabilities, flagged_for_review)."""
        probabilities = self.predict_proba(data)
        return probabilities, probabilities >= self.threshold

    def use_threshold(self, threshold, source):
        """Serve with another threshold than the one saved in the bundle."""
        self.threshold = float(threshold)
        self.threshold_source = source

    def info(self):
        return {
            "model_version": self.version,
            "threshold": self.threshold,
            "threshold_source": self.threshold_source,
            "bundle_threshold": self.bundle_threshold,
            "odds_divisor": self.odds_divisor,
            "trained_at": self.metadata.get("trained_at", self.metadata.get("created_at")),
            "training_as_of": self.metadata.get("training_as_of"),
            "mlflow_run_id": self.metadata.get("run_id"),
            "tree_count": self.metadata.get("tree_count"),
            "prior_adjustment": self.config["probability_calibration"].get("prior_adjustment", 1.0),
            # Which code trained this model (older bundles have no lineage yet).
            "git_commit": self.metadata.get("lineage", {}).get("git_commit"),
        }
