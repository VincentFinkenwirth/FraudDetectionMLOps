"""Compare several simulated years (outputs of simulation.summarize) in one table.

    python -m simulation.compare_runs --run all_labels=work/docker_reports \
        --run realistic_a=work/runs/realistic_a/report --run realistic_b=work/runs/realistic_b/report

I use it for the realistic label mode: one run with a 2 % audit could be luck,
so I repeated the year with three different audit samples and look at the
range of the results instead of one number.
"""

import argparse
from pathlib import Path

import json
import pandas as pd

from pipeline.common import PROJECT_DIR


def describe_run(folder):
    folder = Path(folder)
    overall = json.loads((folder / "overall.json").read_text(encoding="utf-8"))
    weekly = pd.read_csv(folder / "weekly_monitoring.csv")
    trainings = pd.read_csv(folder / "trainings.csv") if (folder / "trainings.csv").stat().st_size > 1 else pd.DataFrame()

    # Recall counts as "measurable" with the same rule as the monitor (configs/monitoring.json).
    measurable = (weekly["effective_fraud_labels"] >= 100) & (weekly["missed_fraud_labels"] >= 20)
    estimated = weekly.dropna(subset=["served_recall_last_28_days_of_labels", "true_recall_last_28_days"])
    recall_error = (estimated["served_recall_last_28_days_of_labels"] - estimated["true_recall_last_28_days"]).abs()
    fpr_error = (estimated["served_fpr_last_28_days_of_labels"] - estimated["true_fpr_last_28_days"]).abs()
    retrain_weeks = int((weekly["decision"] == "retrain").sum())
    trained = len(trainings)

    row = {
        "labelled_share": overall["labelled_share"],
        "audit_labels": overall["audit_labels"],
        "trainings": trained,
        "promoted": overall["promoted"],
        "skipped_not_enough_data": retrain_weeks - trained,
        "weeks_recall_measurable": int(measurable.sum()),
        "recall_estimate_error_mean": recall_error.mean(),
        "recall_estimate_error_max": recall_error.max(),
        "fpr_estimate_error_mean": fpr_error.mean(),
        "fpr_estimate_error_max": fpr_error.max(),
    }
    for name in ("adaptive", "frozen_v1_workload", "frozen_v1"):
        for key in ("recall", "false_positive_rate", "flagged_rate", "recall_at_5pct_review", "average_precision"):
            row[f"{name}_{key}"] = overall[name][key]
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", required=True, help="name=folder with the summarize output")
    parser.add_argument("--output", type=Path, default=PROJECT_DIR / "docs" / "results" / "run_comparison.csv")
    args = parser.parse_args()

    rows = {}
    for item in args.run:
        name, folder = item.split("=", 1)
        rows[name] = describe_run(folder)
    table = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output)
    pd.set_option("display.width", 200)
    print(table.round(4).to_string())
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
