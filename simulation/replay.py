"""Replay the simulated year through the real REST API, week by week.

    python -m simulation.replay --pipeline local     # monitoring + retraining run in this process
    python -m simulation.replay --pipeline jenkins   # Jenkins decides and retrains (Docker setup)

For every simulated week:
  1. send this week's applications to POST /predict/batch (predict key)
  2. release the labels that became known this week (4 weeks after the
     application) through POST /labels (labels key)
  3. run the weekly monitoring job; it may start the retraining job
  4. ask GET /model-info which model version is served now

After week 48 the script releases the remaining labels for 4 more weeks, so
month 12 can also be evaluated and the last training job can run.

API keys and URLs are read from environment variables or from the .env file
that Docker Compose uses (see .env.example).
"""

import argparse
import csv
import math
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from preprocessing import FEATURE_COLUMNS
from simulation.build_months import DEFAULT_OUTPUT, DEFAULT_SCENARIO, load_scenario


PROJECT_DIR = Path(__file__).resolve().parents[1]
CHUNK_SIZE = 500
LABEL_CHUNK_SIZE = 5000


def load_env_file(path=PROJECT_DIR / ".env"):
    """Tiny .env reader so I do not need python-dotenv. Existing variables win."""
    if not Path(path).exists():
        return
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def load_simulation(data_dir):
    applications = pd.concat([pd.read_csv(p, float_precision="round_trip")
                              for p in sorted((data_dir / "applications").glob("month_*.csv"))], ignore_index=True)
    labels = pd.concat([pd.read_csv(p) for p in sorted((data_dir / "labels").glob("month_*.csv"))], ignore_index=True)
    if applications.empty:
        raise FileNotFoundError(f"No simulated months in {data_dir}. Run python -m simulation.build_months first.")
    return applications, labels


def redraw_audit(labels, seed, share, weeks_per_month):
    """Draw the random audit sample again with another seed.

    The applications and their true labels stay exactly the same; only WHICH
    unflagged applications are checked by hand changes. With a 2 % audit only
    a handful of missed frauds are found per month, so one realistic run could
    be luck. Repeating the year with other audit seeds shows how much the
    results depend on this random sample.
    """
    labels = labels.copy()
    month = (labels["sim_week"] - 1) // weeks_per_month + 1
    for m in sorted(month.unique()):
        rows = month == m
        # Same kind of random stream as in build_months.py, just another seed.
        labels.loc[rows, "audited"] = np.random.default_rng([seed, int(m), 99]).random(int(rows.sum())) < share
    return labels


def post(session, url, key, payload, retries=3):
    """POST with a few retries; the API is idempotent, so a retry is safe."""
    for attempt in range(retries):
        try:
            response = session.post(url, json=payload, headers={"X-API-Key": key}, timeout=120)
            if response.status_code < 500 and response.status_code != 409:
                response.raise_for_status()
                return response.json()
        except requests.ConnectionError:
            if attempt == retries - 1:
                raise
        time.sleep(2 * (attempt + 1))
    response.raise_for_status()


def to_json_value(value):
    """NaN -> None and numpy numbers -> normal Python numbers.

    I first used DataFrame.to_json(), but it rounds floats to 10 digits
    (0.6000000000000001 became 0.6). That moved values into another drift bin
    and created fake drift, so the values are converted one by one.
    """
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def send_applications(session, api_url, key, week_rows):
    features = [{k: to_json_value(v) for k, v in row.items()}
                for row in week_rows[FEATURE_COLUMNS].to_dict(orient="records")]
    items = [{"application_id": app_id, "event_time": event_time, "features": row}
             for app_id, event_time, row in zip(week_rows["application_id"], week_rows["event_time"], features)]
    flagged = []
    for start in range(0, len(items), CHUNK_SIZE):
        result = post(session, f"{api_url}/predict/batch", key, {"applications": items[start:start + CHUNK_SIZE]})
        flagged += [p["application_id"] for p in result["predictions"] if p["flagged_for_review"]]
    return flagged


def select_labels(week_labels, mode, flagged_ids, audit_share):
    """Which labels the investigation team can report this week.

    full:      every application gets its true label (optimistic, used so far)
    realistic: only applications that were flagged (and therefore reviewed) or
               picked for the random audit get a label. The missed fraud of
               unflagged, unaudited applications stays unknown - like in reality.
               An audited case stands for 1 / audit_share unflagged ones, so it
               gets that weight (inverse probability weighting).
    """
    labels = week_labels.copy()
    labels["was_flagged"] = labels["application_id"].isin(flagged_ids)
    if mode == "full":
        labels["label_source"], labels["weight"] = "simulation", 1.0
        return labels
    labels = labels[labels["was_flagged"] | labels["audited"]].copy()
    labels["label_source"] = np.where(labels["was_flagged"], "review", "audit")
    labels["weight"] = np.where(labels["was_flagged"], 1.0, 1.0 / audit_share)
    return labels


def release_labels(session, api_url, key, week_labels):
    items = [{"application_id": a, "fraud_bool": int(y), "label_available_at": t, "label_source": s,
              "weight": float(w)}
             for a, y, t, s, w in zip(week_labels["application_id"], week_labels["fraud_bool"],
                                      week_labels["label_available_at"], week_labels["label_source"],
                                      week_labels["weight"])]
    for start in range(0, len(items), LABEL_CHUNK_SIZE):
        post(session, f"{api_url}/labels", key, {"labels": items[start:start + LABEL_CHUNK_SIZE]})
    return len(items)


class JenkinsRunner:
    """Start the Jenkins monitoring job and wait until it (and a triggered retraining) is finished.

    Jenkins protects POST requests with a "crumb" (CSRF token). When logging in
    with a password, the crumb belongs to the web session, so I keep one
    requests.Session for the crumb and all following calls.
    """

    def __init__(self, url, user, password):
        self.url = url.rstrip("/")
        self.session = requests.Session()
        self.session.auth = (user, password)
        crumb = self.session.get(f"{self.url}/crumbIssuer/api/json", timeout=30)
        crumb.raise_for_status()
        self.session.headers[crumb.json()["crumbRequestField"]] = crumb.json()["crumb"]

    def run(self, as_of):
        response = self.session.post(f"{self.url}/job/fraud-monitor/buildWithParameters",
                                     params={"AS_OF": as_of}, timeout=30)
        response.raise_for_status()
        queue_url = response.headers["Location"].rstrip("/")
        build_url = None
        while build_url is None:            # wait until Jenkins starts the build
            time.sleep(1)
            item = self.session.get(f"{queue_url}/api/json", timeout=30).json()
            build_url = (item.get("executable") or {}).get("url")
        # Jenkins reports its own URL (localhost:8080 inside the container); I keep only the path.
        build_path = "/" + build_url.split("/", 3)[3].rstrip("/")
        while True:                          # wait until the build (and the retraining it started) is finished
            build = self.session.get(f"{self.url}{build_path}/api/json", timeout=30).json()
            if not build["building"]:
                return {"decision": build.get("description") or "", "jenkins_result": build["result"]}
            time.sleep(3)


class LocalRunner:
    """The same Python jobs that Jenkins calls, but directly in this process."""

    def __init__(self, api_url, admin_key):
        # The deploy stage reads these to reload and check the API.
        os.environ["API_URL"], os.environ["API_KEY_ADMIN"] = api_url, admin_key
        from pipeline.common import load_json, models_dir
        from service import db
        self.monitoring_config = load_json("configs/monitoring.json")
        self.retraining_config = load_json("configs/retraining.json")
        self.engine, self.models = db.get_engine(), models_dir()

    def run(self, as_of):
        from pipeline.monitor import run_monitoring
        from pipeline.retrain import run_all
        report = run_monitoring(as_of, self.monitoring_config, self.engine, self.models)
        result = {"decision": report["decision"], "reasons": ",".join(report["reasons"])}
        if report["decision"] == "retrain":
            runs = Path(os.environ.get("PIPELINE_RUNS_DIR", PROJECT_DIR / "work" / "pipeline_runs"))
            run_dir = runs / as_of[:10]
            retrain = run_all(run_dir, as_of, result["reasons"], self.retraining_config, self.engine, self.models)
            result["retrain"] = f"{retrain['status']}: {retrain['message']}"
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pipeline", choices=["local", "jenkins", "none"], default="local")
    parser.add_argument("--data", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scenario", type=Path, default=DEFAULT_SCENARIO)
    parser.add_argument("--start-week", type=int, default=1, help="Continue an interrupted run.")
    parser.add_argument("--last-week", type=int, default=None, help="Stop early (default: all weeks + label weeks).")
    parser.add_argument("--extra-label-weeks", type=int, default=4)
    parser.add_argument("--log", type=Path, default=PROJECT_DIR / "work" / "simulation" / "progress.csv")
    parser.add_argument("--label-mode", choices=["full", "realistic"], default="full",
                        help="full: every application gets its label; realistic: only flagged + audited ones.")
    parser.add_argument("--audit-seed", type=int, default=None,
                        help="Draw the random audit sample with another seed (realistic mode, default: the "
                             "sample saved by build_months.py).")
    args = parser.parse_args()

    load_env_file()
    api_url = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
    keys = {role: os.environ.get(f"API_KEY_{role.upper()}", "") for role in ("predict", "labels", "admin")}
    if not all(keys.values()):
        parser.error("Set API_KEY_PREDICT, API_KEY_LABELS and API_KEY_ADMIN (or create .env from .env.example).")

    scenario = load_scenario(args.scenario)
    start = datetime.fromisoformat(scenario["start_date"])
    applications, labels = load_simulation(args.data)
    if args.audit_seed is not None:
        labels = redraw_audit(labels, args.audit_seed, scenario.get("audit_share", 0.02), scenario["weeks_per_month"])
        print(f"Audit sample drawn with seed {args.audit_seed}: {labels['audited'].sum():,} applications")
    last_week = args.last_week or int(applications["sim_week"].max()) + args.extra_label_weeks

    if args.pipeline == "jenkins":
        runner = JenkinsRunner(os.environ.get("JENKINS_URL", "http://localhost:8080"),
                               os.environ["JENKINS_ADMIN_ID"], os.environ["JENKINS_ADMIN_PASSWORD"])
    elif args.pipeline == "local":
        runner = LocalRunner(api_url, keys["admin"])
    else:
        runner = None

    session = requests.Session()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    new_log = not args.log.exists() or args.start_week == 1
    # Which applications were flagged is only known from the API answers, so I
    # keep them in a file (needed 4 weeks later, also after a restart).
    flags_file = args.log.with_name(args.log.stem + "_flagged.csv")
    if new_log or not flags_file.exists():
        flags_file.write_text("application_id\n", encoding="utf-8")
    flagged_ids = set(pd.read_csv(flags_file)["application_id"])
    with args.log.open("w" if new_log else "a", newline="", encoding="utf-8") as log_file:
        writer = csv.writer(log_file)
        if new_log:
            writer.writerow(["week", "as_of", "applications", "flagged", "labels_released", "decision",
                             "details", "serving_version"])
        for week in range(args.start_week, last_week + 1):
            as_of = (start + timedelta(weeks=week)).isoformat()
            week_rows = applications[applications["sim_week"] == week]
            new_flags = send_applications(session, api_url, keys["predict"], week_rows) if len(week_rows) else []
            flagged_ids.update(new_flags)
            with flags_file.open("a", encoding="utf-8") as file:
                file.writelines(f"{app_id}\n" for app_id in new_flags)
            flagged = len(new_flags)
            week_labels = select_labels(labels[labels["label_week"] == week], args.label_mode, flagged_ids,
                                        scenario.get("audit_share", 0.02))
            released = release_labels(session, api_url, keys["labels"], week_labels) if len(week_labels) else 0

            result = runner.run(as_of) if runner else {"decision": "not run"}
            info = session.get(f"{api_url}/model-info", headers={"X-API-Key": keys["admin"]}, timeout=30).json()
            details = result.get("retrain") or result.get("reasons") or result.get("jenkins_result") or ""
            writer.writerow([week, as_of, len(week_rows), flagged, released, result["decision"], details,
                             info["model_version"]])
            log_file.flush()
            print(f"Week {week:2d} ({as_of[:10]}) | sent {len(week_rows):5,} | flagged {flagged:4,} | "
                  f"labels {released:5,} | {result['decision']:<28} {details} | serving {info['model_version']}")

    print(f"\nDone. Progress log: {args.log}")
    print("Create the result charts with: python -m simulation.summarize")


if __name__ == "__main__":
    main()
