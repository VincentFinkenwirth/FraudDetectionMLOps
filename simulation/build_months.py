"""Create twelve simulated months of applications with planned drift.

    python -m simulation.build_months

Important idea: the fraud labels in BAF are real and belong to one row. If I
just added random noise to the features, the labels would not fit the data any
more and it would not be a realistic kind of drift. So every drift type uses a
mechanism that keeps the rows consistent:

- Covariate shift ("downturn"): rows are *dropped* with a probability that
  depends on the features (rejection sampling). The link between features and
  fraud stays the same, only the mix of applicants changes.
- Prior shift ("drop_legit"): some legitimate rows are dropped, so the share of
  fraud goes up. This matches the story in the task (more fraud suspicions).

Volume: every month has a planned number of applications ("rows" in the
scenario). The script walks through the pool in its original order and keeps
rows (with the probabilities above) until the month is full; the next month
continues where the last one stopped. In my first version every month started
from a fixed block of 30,000 rows, so the dropped rows were simply missing and
the months shrank to ~18,000 during the downturn - although in a real downturn
MORE people apply for benefits. Now the downturn months are the biggest ones:
16,000 applications per month before, up to 20,000 during the downturn. They
just use up more of the pool, because more rows are rejected.
- Upstream change ("score_offset"): from month 5 a new credit bureau reports
  credit_risk_score on another scale. The applicants are the same, but the
  model's learned split points no longer fit. Data monitoring sees this at
  once, but retraining can only fix it after the new data has labels.
- Concept drift ("camouflage"): for a share of the fraud rows, the most useful
  fraud signals are copied from random legitimate applications of the same
  month. The fraudsters now look normal in exactly the features the model
  relies on. Because fraud is only ~1 % of the rows, the overall feature
  distributions hardly change -> only the delayed labels can show this.
- Data-quality incident ("incident"): a new mobile form sends no session length
  and an unknown device type. This should be caught by the quality checks and
  should NOT lead to retraining.

Dropping rows instead of sampling with replacement means that no application
appears twice, so training and evaluation rows can never overlap.

Outputs (in data/simulation/, which is not in git):
- applications/month_XX.csv  features + application_id + event_time, NO label
- labels/month_XX.csv        the private labels, only read by the label release
- summary.json               rows, fraud rate and injected drift per month
"""

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing import FEATURE_COLUMNS, TARGET_COLUMN


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SCENARIO = PROJECT_DIR / "configs" / "drift_scenario.json"
DEFAULT_POOL = PROJECT_DIR / "data" / "splits" / "simulation_pool.csv"
DEFAULT_OUTPUT = PROJECT_DIR / "data" / "simulation"


def load_scenario(path=DEFAULT_SCENARIO):
    scenario = json.loads(Path(path).read_text(encoding="utf-8"))
    months = [spec["month"] for spec in scenario["months"]]
    if months != list(range(1, len(months) + 1)):
        raise ValueError("Months in the scenario must be numbered 1, 2, 3, ... in order.")
    return scenario


def keep_probability(batch, spec):
    """Probability that a row stays in the month (covariate + prior shift).

    The ranks (0 = lowest income / shortest time at the address) are computed
    once for the whole pool in build_all, so a row gets the same probability no
    matter which month reaches it.
    """
    keep = np.ones(len(batch))

    downturn = spec.get("downturn", 0.0)
    if downturn:
        # High income and a long time at the current address -> more likely to be dropped.
        keep *= 1 - downturn * (0.5 * batch["income_rank"].to_numpy() + 0.5 * batch["address_rank"].to_numpy())

    drop_legit = spec.get("drop_legit", 0.0)
    if drop_legit:
        keep *= np.where(batch[TARGET_COLUMN].to_numpy() == 0, 1 - drop_legit, 1.0)
    return keep


def take_month(pool, start, rows, spec, rng):
    """Walk through the pool from `start` and keep rows until the month has `rows` applications.

    Returns the part of the pool that was looked at, the kept rows and the
    position where the next month starts.
    """
    rest = pool.iloc[start:]
    kept = rng.random(len(rest)) < keep_probability(rest, spec)
    positions = np.flatnonzero(kept)
    if len(positions) < rows:
        raise ValueError(f"The simulation pool is too small: month needs {rows:,} rows, "
                         f"only {len(positions):,} are left. Plan fewer applications per month.")
    last = positions[rows - 1]          # the row that fills the month
    looked_at = rest.iloc[:last + 1]
    batch = looked_at[kept[:last + 1]].copy()
    return looked_at, batch, start + last + 1


def camouflage(batch, share, features, rng):
    """Copy the chosen features from random legitimate rows into some fraud rows."""
    fraud_index = batch.index[batch[TARGET_COLUMN] == 1]
    count = int(round(share * len(fraud_index)))
    if count == 0:
        return batch, []
    chosen = rng.choice(fraud_index, size=count, replace=False)
    legit_index = batch.index[batch[TARGET_COLUMN] == 0]
    donors = rng.choice(legit_index, size=count, replace=True)
    # Column by column so every column keeps its own data type.
    for feature in features:
        batch.loc[chosen, feature] = batch.loc[donors, feature].to_numpy()
    return batch, list(chosen)


def data_quality_incident(batch, share, rng):
    """New mobile channel: session length is not recorded and the OS is unknown."""
    hit = batch.index[rng.random(len(batch)) < share]
    batch.loc[hit, "session_length_in_minutes"] = -1
    batch.loc[hit, "device_os"] = "android"
    return batch, list(hit)


def build_month(batch, spec, camouflage_features, rng):
    """Apply the feature changes of one month to its (already selected) rows."""
    batch = batch.copy()
    batch["camouflaged"] = False
    batch["incident"] = False

    if spec.get("score_offset"):
        # A new credit bureau reports the same risk on another scale. The old
        # range is -191..389, so +600 does not overlap with it.
        batch["credit_risk_score"] = batch["credit_risk_score"] + spec["score_offset"]
    if spec.get("camouflage"):
        batch, rows = camouflage(batch, spec["camouflage"], camouflage_features, rng)
        batch.loc[rows, "camouflaged"] = True
    if spec.get("incident"):
        batch, rows = data_quality_incident(batch, spec["incident"], rng)
        batch.loc[rows, "incident"] = True
    return batch


def add_times(batch, month, scenario):
    """Spread the rows evenly over the four weeks of the month (original order kept)."""
    start = datetime.fromisoformat(scenario["start_date"])
    weeks = scenario["weeks_per_month"]
    month_start = start + timedelta(weeks=weeks * (month - 1))
    position = np.arange(len(batch)) / len(batch)          # 0 <= position < 1
    seconds = position * weeks * 7 * 24 * 3600
    batch = batch.copy()
    batch["event_time"] = [(month_start + timedelta(seconds=float(s))).isoformat() for s in seconds]
    batch["sim_week"] = weeks * (month - 1) + 1 + np.floor(position * weeks).astype(int)
    batch["label_week"] = batch["sim_week"] + scenario["label_delay_weeks"]
    # Labels of week w become known at the END of week w + delay.
    batch["label_available_at"] = [
        (start + timedelta(weeks=int(w))).isoformat() for w in batch["label_week"]
    ]
    return batch


def build_all(pool, scenario):
    pool = pool.reset_index(drop=True).copy()
    # Ranks over the whole pool (see keep_probability).
    pool["income_rank"] = pool["income"].rank(pct=True)
    pool["address_rank"] = pool["current_address_months_count"].rank(pct=True)

    months, summary = {}, []
    start = 0
    for spec in scenario["months"]:
        month = spec["month"]
        rows = spec.get("rows", scenario["rows_per_month"])
        # A separate random generator per month: changing month 7 does not change month 3.
        rng = np.random.default_rng([scenario["seed"], month])
        chunk, selected, start = take_month(pool, start, rows, spec, rng)
        batch = add_times(build_month(selected, spec, scenario["camouflage_features"], rng), month, scenario)
        batch.insert(0, "application_id", [f"APP-{int(i):07d}" for i in batch["source_id"]])
        # Realistic label mode: a random audit sample of applications is checked
        # by hand even if the model did not flag it. Own random stream, so the
        # applications above stay exactly the same.
        audit_rng = np.random.default_rng([scenario["seed"], month, 99])
        batch["audited"] = audit_rng.random(len(batch)) < scenario.get("audit_share", 0.0)
        months[month] = batch
        summary.append({
            "month": month,
            "story": spec.get("story", ""),
            "injected": {k: v for k, v in spec.items() if k not in ("month", "story", "rows")},
            "source_months": sorted(int(m) for m in chunk["month"].unique()),
            "pool_rows_used": int(len(chunk)),
            "rows": int(len(batch)),
            "fraud_cases": int(batch[TARGET_COLUMN].sum()),
            "fraud_rate": float(batch[TARGET_COLUMN].mean()),
            "camouflaged_fraud_rows": int(batch["camouflaged"].sum()),
            "incident_rows": int(batch["incident"].sum()),
            "first_week": int(batch["sim_week"].min()),
            "last_week": int(batch["sim_week"].max()),
        })
    print(f"Used {start:,} of {len(pool):,} pool rows ({len(pool) - start:,} left over).")
    return months, summary


def write_outputs(months, summary, scenario, output_dir):
    output_dir = Path(output_dir)
    (output_dir / "applications").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels").mkdir(parents=True, exist_ok=True)
    application_columns = ["application_id", "event_time", "sim_week", "source_id", *FEATURE_COLUMNS]
    label_columns = ["application_id", TARGET_COLUMN, "sim_week", "label_week", "label_available_at",
                     "audited", "camouflaged", "incident"]
    for month, batch in months.items():
        # The label is kept in a separate file so it can never be sent to the API by mistake.
        batch[application_columns].to_csv(output_dir / "applications" / f"month_{month:02d}.csv",
                                          index=False, lineterminator="\n")
        batch[label_columns].to_csv(output_dir / "labels" / f"month_{month:02d}.csv",
                                    index=False, lineterminator="\n")
    (output_dir / "summary.json").write_text(
        json.dumps({"scenario": scenario, "months": summary}, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", type=Path, default=DEFAULT_SCENARIO)
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    scenario = load_scenario(args.scenario)
    pool = pd.read_csv(args.pool, float_precision="round_trip")
    months, summary = build_all(pool, scenario)
    write_outputs(months, summary, scenario, args.output)
    for item in summary:
        print(f"Month {item['month']:2d} | {item['rows']:6,} rows (from {item['pool_rows_used']:6,}) | "
              f"fraud {item['fraud_rate']:.2%} | "
              f"camouflaged {item['camouflaged_fraud_rows']:4d} | incident {item['incident_rows']:5d} | {item['story']}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
