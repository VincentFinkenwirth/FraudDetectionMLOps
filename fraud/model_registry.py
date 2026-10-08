"""A very small file-based model registry for the deployed versions.

models/
    current.json         <- which version the API should serve (plus the previous one)
                            and, if set, the threshold it should use
    deployments.jsonl    <- history of every switch, useful for the presentation
    thresholds.jsonl     <- history of every threshold change (weekly workload rule)
    versions/v1/ ...     <- one complete bundle per accepted version

MLflow also has a model registry, and the pipeline registers every candidate
there. But the API should not depend on the MLflow server being up, so the
API only reads this folder. Switching current.json is the actual deployment.
"""

import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]


def default_models_dir():
    return Path(os.environ.get("MODEL_DIR", PROJECT_DIR / "models"))


def _version_number(name):
    match = re.fullmatch(r"v(\d+)", name)
    return int(match.group(1)) if match else None


def list_versions(models_dir):
    folder = Path(models_dir) / "versions"
    if not folder.exists():
        return []
    names = [p.name for p in folder.iterdir() if p.is_dir() and _version_number(p.name) is not None]
    return sorted(names, key=_version_number)


def next_version(models_dir):
    versions = list_versions(models_dir)
    return f"v{_version_number(versions[-1]) + 1}" if versions else "v1"


def version_dir(models_dir, version):
    return Path(models_dir) / "versions" / version


def _write_json_atomic(path, values):
    """Write to a temporary file first and then rename it.

    os.replace is atomic, so the API never reads a half-written current.json.
    """
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def publish(bundle_dir, models_dir, version=None):
    """Copy a finished bundle into versions/<version> and return the version name."""
    version = version or next_version(models_dir)
    target = version_dir(models_dir, version)
    if target.exists():
        raise FileExistsError(f"Model version {version} already exists.")
    shutil.copytree(bundle_dir, target)

    # The version name is also written into the bundle, so /model-info shows it.
    metadata_path = target / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["model_version"] = version
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return version


def read_current(models_dir):
    path = Path(models_dir) / "current.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def current_bundle_dir(models_dir):
    current = read_current(models_dir)
    if current is None:
        raise FileNotFoundError(f"No deployed model in {models_dir}. Run python -m training.package_model first.")
    return version_dir(models_dir, current["version"])


def set_current(models_dir, version, *, threshold=None, threshold_reason="", reason="", as_of=None):
    """Point the API to another version and keep the old one for a rollback.

    The model and its threshold are ONE deployment state, so both go into the
    same (atomic) write of current.json. Without a threshold the API uses the
    one saved in the bundle. I also keep the threshold the old version was
    really serving: before, a rollback brought the old model back with its
    bundle threshold, which after the credit-scale change would have sent
    17 % of the applications to review instead of 5.5 %.
    """
    models_dir = Path(models_dir)
    if not (version_dir(models_dir, version) / "model.cbm").exists():
        raise FileNotFoundError(f"Model version {version} does not exist.")
    old = read_current(models_dir)
    entry = {
        "version": version,
        "previous": old["version"] if old else None,
        "deployed_at": datetime.now(timezone.utc).isoformat(),
        "simulated_time": _as_text(as_of),
        "reason": reason,
    }
    if old and old.get("threshold") is not None:
        entry["previous_threshold"] = old["threshold"]
        entry["previous_threshold_reason"] = old.get("threshold_reason", "")
    if threshold is not None:
        entry.update(threshold=float(threshold), threshold_reason=threshold_reason,
                     threshold_simulated_time=_as_text(as_of))
    _write_json_atomic(models_dir / "current.json", entry)
    with (models_dir / "deployments.jsonl").open("a", encoding="utf-8") as log:
        log.write(json.dumps(entry) + "\n")
    if threshold is not None:
        _log_threshold(models_dir, version, threshold, threshold_reason, as_of)
    return entry


def _as_text(as_of):
    return as_of.isoformat() if hasattr(as_of, "isoformat") else as_of


def _log_threshold(models_dir, version, threshold, reason, as_of):
    with (Path(models_dir) / "thresholds.jsonl").open("a", encoding="utf-8") as log:
        log.write(json.dumps({"version": version, "threshold": float(threshold), "reason": reason,
                              "simulated_time": _as_text(as_of),
                              "changed_at": datetime.now(timezone.utc).isoformat()}) + "\n")


def set_threshold(models_dir, version, threshold, *, reason="", as_of=None):
    """Change only the decision threshold of the served version (no new model).

    The bundles stay unchanged, the threshold goes into current.json and the
    API picks it up with its normal hot reload. A new deployment (set_current)
    starts again with the threshold saved in the new bundle.
    """
    models_dir = Path(models_dir)
    current = read_current(models_dir)
    if current is None or current["version"] != version:
        raise RuntimeError(f"Version {version} is not the served version, so its threshold is not changed.")
    current = {**current, "threshold": float(threshold), "threshold_reason": reason,
               "threshold_simulated_time": _as_text(as_of)}
    _write_json_atomic(models_dir / "current.json", current)
    _log_threshold(models_dir, version, threshold, reason, as_of)
    return current


def rollback(models_dir, *, as_of=None):
    """Go back to the version that was active before the last switch.

    It comes back with the threshold it was serving then (or with its bundle
    threshold if it never had another one).
    """
    current = read_current(models_dir)
    if not current or not current.get("previous"):
        raise RuntimeError("There is no previous model version to roll back to.")
    return set_current(models_dir, current["previous"], threshold=current.get("previous_threshold"),
                       threshold_reason="restored by rollback", reason=f"rollback from {current['version']}",
                       as_of=as_of)
