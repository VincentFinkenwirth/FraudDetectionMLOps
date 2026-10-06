"""Tables and charts of the simulated year for the presentation.

    python -m simulation.summarize --output work/reports/simulation

It compares three things on exactly the same applications:
- "adaptive": the scores that the API really served (the model changed over time)
- "frozen v1": the first model, never retrained, with its fixed threshold
  (= no MLOps at all, scored offline for comparison)
- "frozen v1 + workload threshold": still never retrained, but with the weekly
  label-free threshold (= only the fast loop). The difference to "adaptive"
  shows what retraining adds on top of the threshold rule.

The TRUE labels of all applications are used here (from the simulation's
private label files), also the ones that arrived after the last week and, in
the realistic label mode, the ones the system never saw. This is the final
evaluation after the year, not a decision inside it. The weekly table also
keeps what the system itself believed, so both can be compared.
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sqlalchemy import select

import model_registry
from fairness import fpr_by_age
from pipeline.common import PROJECT_DIR, load_json, parse_time
from scoring import FraudScorer
from service import db
from visualization import ChartTheme


THEME = ChartTheme()
EVENTS = {3: "downturn", 5: "new credit\nbureau scale", 7: "fraudsters\ncamouflage", 10: "mobile form\nincident"}


def metrics(y, probabilities, flagged, ages):
    y, flagged = np.asarray(y).astype(int), np.asarray(flagged).astype(bool)
    frauds, legit = y == 1, y == 0
    return {
        "rows": int(len(y)),
        "fraud_rate": float(y.mean()),
        "recall": float(flagged[frauds].mean()) if frauds.any() else None,
        "false_positive_rate": float(flagged[legit].mean()) if legit.any() else None,
        "precision": float(y[flagged].mean()) if flagged.any() else None,
        "flagged_rate": float(flagged.mean()),
        "average_precision": float(average_precision_score(y, probabilities)) if frauds.any() else None,
        # Fair comparison at the SAME workload: which share of fraud is in the 5 %
        # highest scores? (The fixed thresholds flag different amounts.)
        "recall_at_5pct_review": top_share_recall(y, probabilities, 0.05),
        "fpr_ratio_age": fpr_by_age(y, flagged, ages)["fpr_ratio_age"],
    }


def top_share_recall(y, probabilities, share):
    if not y.any():
        return None
    count = max(1, int(np.ceil(len(y) * share)))
    top = np.argsort(-np.asarray(probabilities), kind="stable")[:count]
    return float(y[top].sum() / y.sum())


def frozen_workload_flags(served, frozen_probabilities, frozen_threshold, review_share, incident_weeks):
    """Frozen v1 with the weekly workload threshold (the fast loop without retraining).

    Same rule as pipeline/monitor.py: the threshold for week w flags `review_share`
    of the applications of week w-1, and it is not changed after a week with a
    data-quality incident. Week 1 uses the threshold saved with v1.
    Returns the flags and the threshold of every row (for the export).
    """
    flags = np.zeros(len(served), dtype=bool)
    thresholds = np.zeros(len(served))
    threshold = frozen_threshold
    for week in sorted(served["sim_week"].unique()):
        previous = served["sim_week"] == week - 1
        if previous.any() and (week - 1) not in incident_weeks:
            threshold = np.quantile(frozen_probabilities[previous.to_numpy()], 1 - review_share)
        rows = (served["sim_week"] == week).to_numpy()
        flags[rows] = frozen_probabilities[rows] >= threshold
        thresholds[rows] = threshold
    return flags, thresholds


EXPORT_COLUMNS = [
    "application_id", "event_time", "sim_week", "month", "fraud_bool", "label_source", "label_available_at",
    "customer_age", "served_model_version", "served_probability", "served_threshold", "served_flagged",
    "frozen_v1_probability", "frozen_v1_threshold", "frozen_v1_flagged",
    "frozen_v1_workload_threshold", "frozen_v1_workload_flagged",
    "camouflaged", "incident",
]


def applications_table(served, frozen_probabilities, frozen_threshold, workload_flags, workload_thresholds,
                       labels_seen):
    """One row per application: what the API served and what frozen v1 would have done.

    I save this next to the tables so later charts (e.g. weekly comparisons at
    the same review budget) do not need a new run of the whole year.
    label_source is empty for applications that never got a label (realistic
    mode), and then label_available_at is empty too.
    """
    table = served[["application_id", "event_time", "sim_week", "month", "fraud_bool", "label_available_at",
                    "customer_age"]].copy()
    table["label_source"] = served["application_id"].map(labels_seen.set_index("application_id")["label_source"])
    table.loc[table["label_source"].isna(), "label_available_at"] = pd.NaT
    table["served_model_version"] = served["model_version"]
    table["served_probability"] = served["fraud_probability"]
    table["served_threshold"] = served["threshold"]
    table["served_flagged"] = served["flagged"].astype(bool)
    table["frozen_v1_probability"] = frozen_probabilities
    table["frozen_v1_threshold"] = frozen_threshold
    table["frozen_v1_flagged"] = np.asarray(frozen_probabilities) >= frozen_threshold
    table["frozen_v1_workload_threshold"] = workload_thresholds
    table["frozen_v1_workload_flagged"] = np.asarray(workload_flags, dtype=bool)
    # Which rows the scenario changed (camouflaged fraud, mobile-form incident).
    # The system never sees these flags; I only keep them for the charts.
    for column in ("camouflaged", "incident"):
        table[column] = served[column].astype(bool) if column in served else False
    return table[EXPORT_COLUMNS].reset_index(drop=True)


def monthly_comparison(served, frozen_probabilities, frozen_threshold, workload_flags=None):
    rows = []
    for month, part in served.groupby("month"):
        index = part.index
        adaptive = metrics(part["fraud_bool"], part["fraud_probability"], part["flagged"], part["customer_age"])
        frozen = metrics(part["fraud_bool"], frozen_probabilities[index],
                         frozen_probabilities[index] >= frozen_threshold, part["customer_age"])
        row = {"month": month, "model_versions": ", ".join(sorted(part["model_version"].unique(),
                                                                  key=lambda v: int(v[1:]))),
               **{f"adaptive_{k}": v for k, v in adaptive.items()},
               **{f"frozen_{k}": v for k, v in frozen.items() if k not in ("rows", "fraud_rate")}}
        if workload_flags is not None:
            fast_loop = metrics(part["fraud_bool"], frozen_probabilities[index], workload_flags[index],
                                part["customer_age"])
            row.update({f"frozen_workload_{k}": v for k, v in fast_loop.items() if k not in ("rows", "fraud_rate")})
        rows.append(row)
    return pd.DataFrame(rows)


def load_truth():
    """True labels of all simulated applications (private files of the simulation)."""
    files = sorted((PROJECT_DIR / "data" / "simulation" / "labels").glob("month_*.csv"))
    truth = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    truth["label_available_at"] = pd.to_datetime(truth["label_available_at"])
    # Label files from before the flags were saved do not have these columns.
    for column in ("camouflaged", "incident"):
        if column not in truth:
            truth[column] = False
    return truth[["application_id", "fraud_bool", "label_available_at", "camouflaged", "incident"]]


def true_performance(served, as_of, days=28):
    """What the monitoring would have measured with ALL labels (the truth)."""
    window = served[(served["label_available_at"] > as_of - pd.Timedelta(days=days))
                    & (served["label_available_at"] <= as_of)]
    y, flagged = window["fraud_bool"].to_numpy(), window["flagged"].to_numpy(dtype=bool)
    recall = float(flagged[y == 1].mean()) if (y == 1).any() else None
    fpr = float(flagged[y == 0].mean()) if (y == 0).any() else None
    return recall, fpr


def weekly_table(engine, start, served):
    events = db.read_events(engine, "monitoring")
    rows = []
    for _, event in events.iterrows():
        details = event["details"] or {}
        drift = details.get("drift") or {}
        served_perf = details.get("performance_all_served") or {}
        true_recall, true_fpr = true_performance(served, event["as_of"])
        rows.append({
            "week": details.get("week"),
            "model_version": event["model_version"],
            "decision": event["status"],
            "reasons": ",".join(details.get("reasons", [])),
            "drift_share": drift.get("drift_share"),
            "max_psi_important_feature": drift.get("max_psi_important_feature"),
            "most_drifted_important_feature": drift.get("most_drifted_important_feature"),
            "flag_rate": drift.get("flag_rate"),
            "flag_rate_z": drift.get("flag_rate_z"),
            "unseen_category_share": drift.get("max_unseen_category_share"),
            # What the system believed (weighted estimate from the labels it had) ...
            "served_recall_last_28_days_of_labels": served_perf.get("recall"),
            "served_fpr_last_28_days_of_labels": served_perf.get("false_positive_rate"),
            "effective_fraud_labels": served_perf.get("effective_fraud_cases", served_perf.get("fraud_cases")),
            "missed_fraud_labels": served_perf.get("missed_fraud_labels"),
            "labelled_rows_last_28_days": served_perf.get("rows"),
            # Fast loop: threshold for the next week and the review share of this week
            "served_threshold": (details.get("threshold") or {}).get("threshold_after"),
            "review_share_last_week": (details.get("threshold") or {}).get("review_share_last_week"),
            # ... and the truth for the same applications.
            "true_recall_last_28_days": true_recall,
            "true_fpr_last_28_days": true_fpr,
            "warnings": " | ".join(details.get("warnings", [])),
        })
    return pd.DataFrame(rows)


def weekly_feature_psi(engine):
    """PSI of every feature per week (against the model that was deployed that week)."""
    rows = []
    for _, event in db.read_events(engine, "monitoring").iterrows():
        details = event["details"] or {}
        drift = details.get("drift") or {}
        if drift.get("feature_psi"):
            rows.append({"week": details.get("week"), "model_version": event["model_version"],
                         **drift["feature_psi"]})
    return pd.DataFrame(rows)


def training_table(engine, start):
    events = db.read_events(engine, "training")
    rows = []
    for _, event in events.iterrows():
        details = event["details"] or {}
        rows.append({"week": round((event["as_of"] - start).days / 7), "status": event["status"],
                     "reason": details.get("reason"), "explanation": details.get("explanation"),
                     "candidate_recall": details.get("candidate_recall"),
                     "champion_recall": details.get("champion_recall"),
                     "candidate_fpr": details.get("candidate_fpr"), "champion_fpr": details.get("champion_fpr")})
    deployments = db.read_events(engine, "deployment")
    deployed = {round((e["as_of"] - start).days / 7): e["model_version"]
                for _, e in deployments.iterrows() if e["status"] == "deployed"}
    table = pd.DataFrame(rows)
    if not table.empty:
        table["deployed_version"] = table["week"].map(deployed)
    return table


def mark_events(ax, x_per_month=1.0, offset=0.0):
    for month, label in EVENTS.items():
        x = offset + (month - 1) * x_per_month
        ax.axvline(x, color=THEME.muted, lw=0.8, ls=":")
        ax.text(x, 1.0, label, transform=ax.get_xaxis_transform(), fontsize=7, color=THEME.muted,
                va="bottom", ha="left")


def plot_monthly(table, output):
    fig, axes = plt.subplots(2, 2, figsize=(15, 8.5))
    panels = [("recall", "Recall with each model's own threshold", None),
              ("false_positive_rate", "False-positive rate (limit 5 %)", 0.05),
              ("flagged_rate", "Applications sent to review (workload)", None),
              ("recall_at_5pct_review", "Recall at equal workload (top 5 % reviewed)", None)]
    axes = axes.ravel()
    for ax, (key, title, limit) in zip(axes, panels):
        ax.plot(table["month"], table[f"adaptive_{key}"], "o-", color=THEME.accent, lw=2, label="adaptive (retrained)")
        ax.plot(table["month"], table[f"frozen_{key}"], "s--", color=THEME.warning, lw=1.5, label="frozen v1")
        if f"frozen_workload_{key}" in table:
            ax.plot(table["month"], table[f"frozen_workload_{key}"], "^:", color=THEME.muted, lw=1.5,
                    label="frozen v1 + workload threshold")
        if limit:
            ax.axhline(limit, color=THEME.ink, lw=0.8, ls="--", label="5 % limit")
        ax.set_title(title, loc="left", fontsize=11, pad=28)
        ax.set_xlabel("simulated month")
        ax.set_xticks(table["month"])
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
        ax.grid(alpha=0.3)
        mark_events(ax, offset=1)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(output / "monthly_adaptive_vs_frozen.png", dpi=160)
    plt.close(fig)


def plot_weekly(weekly, trainings, config, output):
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(16, 7), sharex=True)
    top.plot(weekly["week"], weekly["drift_share"], "o-", color=THEME.accent, ms=3, label="drift share")
    top.axhline(config["drift_share_threshold"], color=THEME.accent, ls="--", lw=0.8)
    psi = weekly["max_psi_important_feature"].clip(upper=3)
    top.plot(weekly["week"], psi, "s-", color=THEME.secondary, ms=3, label="max PSI of an important feature (cut at 3)")
    top.axhline(config["important_feature_psi_threshold"], color=THEME.secondary, ls="--", lw=0.8)
    top.set_title("Weekly data monitoring (label-free)", loc="left", fontsize=11)
    top.legend(frameon=False, fontsize=8, loc="upper left")
    top.grid(alpha=0.3)

    bottom.plot(weekly["week"], weekly["served_recall_last_28_days_of_labels"], "o-", color=THEME.accent, ms=3,
                label="recall of served scores (labels of the last 28 days)")
    bottom.plot(weekly["week"], weekly["served_fpr_last_28_days_of_labels"], "o-", color=THEME.warning, ms=3,
                label="FPR of served scores")
    bottom.set_title("Delayed-label performance and retraining", loc="left", fontsize=11)
    bottom.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    bottom.grid(alpha=0.3)
    bottom.set_xlabel("simulated week")

    colors = {"promoted": THEME.accent, "rejected": THEME.warning}
    for _, run in trainings.iterrows():
        for ax in (top, bottom):
            ax.axvline(run["week"], color=colors.get(run["status"], THEME.muted), lw=1.2, alpha=0.6)
        version = run["deployed_version"] if pd.notna(run.get("deployed_version")) else "rejected"
        # Vertical labels inside the plot, so neighbouring runs do not overlap.
        bottom.text(run["week"] + 0.25, 0.97, f"{version}: {run['reason']}", rotation=90,
                    transform=bottom.get_xaxis_transform(), fontsize=6.5, ha="left", va="top",
                    color=colors.get(run["status"], THEME.muted))
    blocked = weekly[weekly["decision"].isin(["blocked_quality", "quality_incident"])]
    top.scatter(blocked["week"], np.zeros(len(blocked)), marker="x", color=THEME.warning, zorder=5,
                label="data-quality incident")
    top.legend(frameon=False, fontsize=8, loc="upper left")
    for month in EVENTS:
        top.axvline(4 * (month - 1) + 0.5, color=THEME.muted, lw=0.8, ls=":")
    bottom.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(output / "weekly_monitoring.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=PROJECT_DIR / "work" / "reports" / "simulation")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    config = load_json("configs/monitoring.json")
    # I keep the scenario and the monitoring settings of THIS run next to the
    # results. simulation.plot_results reads them from here, so the charts still
    # show the right events, audit share and review plan after I change the
    # configs for the next run.
    for name, path in (("scenario.json", "configs/drift_scenario.json"), ("monitoring.json", "configs/monitoring.json")):
        (args.output / name).write_text(json.dumps(load_json(path), indent=2) + "\n", encoding="utf-8")
    start = parse_time(config["simulation_start"])
    engine = db.get_engine()
    # Every served prediction (first one per application) joined with the TRUE label.
    predictions = db.read_predictions(engine, parse_time("2000-01-01"), parse_time("2100-01-01"))
    predictions = predictions.drop_duplicates("application_id", keep="first")
    served = predictions.merge(load_truth(), on="application_id", how="inner").reset_index(drop=True)
    served["month"] = 1 + ((served["event_time"] - start).dt.days // 28)
    with engine.connect() as conn:
        labels_seen = pd.read_sql(select(db.labels.c.application_id, db.labels.c.label_source), conn)

    served["sim_week"] = 1 + ((served["event_time"] - start).dt.days // 7)
    frozen = FraudScorer(model_registry.version_dir(model_registry.default_models_dir(), "v1"))
    frozen_probabilities = frozen.predict_proba(served)
    # Weeks in which the monitoring found a data-quality incident (no threshold change after them)
    monitoring = db.read_events(engine, "monitoring")
    incident_weeks = {round((e["as_of"] - start).days / 7) for _, e in monitoring.iterrows()
                      if e["status"] in ("blocked_quality", "quality_incident")}
    workload_flags, workload_thresholds = frozen_workload_flags(served, frozen_probabilities, frozen.threshold,
                                                                config["review_share"], incident_weeks)

    monthly = monthly_comparison(served, frozen_probabilities, frozen.threshold, workload_flags)
    weekly = weekly_table(engine, start, served)
    # How many applications got a label at all (100 % with full labels, ~7 % realistic).
    labelled = served["application_id"].isin(labels_seen["application_id"])
    monthly["labelled_share"] = served.assign(labelled=labelled).groupby("month")["labelled"].mean().to_numpy()
    trainings = training_table(engine, start)
    monthly.to_csv(args.output / "monthly_adaptive_vs_frozen.csv", index=False)
    weekly.to_csv(args.output / "weekly_monitoring.csv", index=False)
    trainings.to_csv(args.output / "trainings.csv", index=False)
    weekly_feature_psi(engine).to_csv(args.output / "weekly_feature_psi.csv", index=False)
    applications_table(served, frozen_probabilities, frozen.threshold, workload_flags, workload_thresholds,
                       labels_seen).to_parquet(args.output / "applications.parquet", index=False)
    plot_monthly(monthly, args.output)
    if not weekly.empty:
        plot_weekly(weekly, trainings, config, args.output)

    overall = {
        "adaptive": metrics(served["fraud_bool"], served["fraud_probability"], served["flagged"], served["customer_age"]),
        "frozen_v1": metrics(served["fraud_bool"], frozen_probabilities, frozen_probabilities >= frozen.threshold,
                             served["customer_age"]),
        "frozen_v1_workload": metrics(served["fraud_bool"], frozen_probabilities, workload_flags,
                                      served["customer_age"]),
        "trainings": int(len(trainings)),
        "promoted": int((trainings["status"] == "promoted").sum()) if len(trainings) else 0,
        "labelled_share": float(labelled.mean()),
        "audit_labels": int((labels_seen["label_source"] == "audit").sum()),
        "review_labels": int((labels_seen["label_source"] == "review").sum()),
    }
    (args.output / "overall.json").write_text(json.dumps(overall, indent=2) + "\n", encoding="utf-8")

    pd.set_option("display.width", 200)
    columns = ["month", "model_versions", "adaptive_fraud_rate", "adaptive_recall", "frozen_recall",
               "adaptive_false_positive_rate", "frozen_false_positive_rate", "adaptive_flagged_rate", "frozen_flagged_rate"]
    print(monthly[columns].round(3).to_string(index=False))
    print(f"\nWhole year: adaptive recall {overall['adaptive']['recall']:.1%} / FPR {overall['adaptive']['false_positive_rate']:.1%}"
          f" | frozen v1 recall {overall['frozen_v1']['recall']:.1%} / FPR {overall['frozen_v1']['false_positive_rate']:.1%}"
          f" | frozen v1 + workload threshold recall {overall['frozen_v1_workload']['recall']:.1%}"
          f" / FPR {overall['frozen_v1_workload']['false_positive_rate']:.1%}")
    print(f"{overall['trainings']} training runs, {overall['promoted']} promoted. Files in {args.output}")


if __name__ == "__main__":
    main()
