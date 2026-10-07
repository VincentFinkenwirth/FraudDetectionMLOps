"""All result charts of the simulated year, made only from saved result folders.

    python -m simulation.plot_results
    python -m simulation.plot_results --run all_labels=docs/results/all_labels \
        --run audit_a=docs/results/realistic_labels/audit_sample_a \
        --compare all_labels:system,audit_a:system --output charts
    python -m simulation.plot_results --slides --no-monthly-matrices   (deck look, charts/slides)

A result folder is what simulation.summarize writes: applications.parquet,
weekly_monitoring.csv, trainings.csv, weekly_feature_psi.csv and (since the
fourth version) scenario.json + monitoring.json. Nothing here needs the API,
MLflow or a model, so I can redo every chart after all runs are finished, also
when the runs were made in Docker or on another computer. Without --run it
takes docs/results/all_labels and every folder in docs/results/realistic_labels.

The "models" are three policies, all scored on the same applications:
    system     = the scores the API really served (fast loop + retraining)
    fast_loop  = frozen v1 with the weekly workload threshold
    no_mlops   = frozen v1 with its fixed threshold
A series is written as run:model, e.g. all_labels:system. In a comparison the
first series is orange and the second one black. The black one also gets
another marker and line style, so the charts still work in black and white.

All numbers use the TRUE labels of every application (the simulation knows
them), also in the realistic label mode. Only the "seen by the system" charts
use the labels the system really got.
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, PercentFormatter
from scipy.stats import binomtest

from fairness import fpr_by_age
from pipeline.common import PROJECT_DIR, load_json
from simulation.summarize import top_share_recall

RESULTS = PROJECT_DIR / "docs" / "results"

# Same colours and sizes as my slide charts (presentation_charts.py). I copy the
# values instead of importing them, because the slide scripts stay local and this
# pipeline should also run from a fresh clone of the repository.
ORANGE = "#D9650B"        # first series / the key result
ORANGE_WASH = "#F7E3D3"   # light orange areas (confidence band, missed fraud)
INK = "#1F1D1A"           # second series ("black")
TEXT = "#52514E"
MUTED = "#898781"
GRID = "#E6E2DB"
BAND = "#F1EDE7"
SURFACE = "#FAF8F5"       # slide background
PX = 1 / 100              # 1 inch = 100 slide pixels

DPI = 200
SLIDES = False            # True after slide_theme()

FIRST = dict(color=ORANGE, lw=3, marker="o", ms=7)
SECOND = dict(color=INK, lw=2.5, ls="--", marker="s", ms=6)

MODELS = {
    "system": {"flag": "served_flagged", "probability": "served_probability", "label": "full system"},
    "fast_loop": {"flag": "frozen_v1_workload_flagged", "probability": "frozen_v1_probability",
                  "label": "fast loop only"},
    "no_mlops": {"flag": "frozen_v1_flagged", "probability": "frozen_v1_probability", "label": "no MLOps"},
}

# The task allows at most 5 % false positives (also used in summarize.py).
FPR_LIMIT = 0.05

EVENT_NAMES = {"downturn": "downturn", "score_offset": "new credit-\nbureau scale",
               "camouflage": "fraudsters\ncamouflage"}
INCIDENT_NAME = "mobile-form\nincident"


# ---------------------------------------------------------------- style

def style():
    return plt.rc_context({
        "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 18,
        "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT, "axes.labelsize": 18,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 1,
        "xtick.color": TEXT, "ytick.color": TEXT, "xtick.labelsize": 17, "ytick.labelsize": 17,
        "legend.frameon": False, "legend.fontsize": 16,
        "lines.linewidth": 2.5, "lines.solid_capstyle": "round",
    })


def slide_theme():
    """Look of the final slide deck: white background, the deck's colours and
    bigger text. The slide already has a heading, so the charts get no title.

    PX is what makes the text bigger: a figure of 1180 slide pixels is now
    1180/140 inch instead of 1180/100, so every font size (in points) takes
    1.4 times more of the chart. DPI goes up by the same factor, so the PNG
    keeps the same number of pixels."""
    global ORANGE, ORANGE_WASH, INK, TEXT, MUTED, GRID, BAND, SURFACE, PX, DPI, SLIDES, FIRST, SECOND
    global EVENT_NAMES, INCIDENT_NAME
    ORANGE = "#EA6A12"
    ORANGE_WASH = "#FBE1CD"
    INK = "#1F2328"
    TEXT = "#5B6472"
    MUTED = "#9AA1AB"
    GRID = "#E3E6EA"
    BAND = "#F2F3F5"
    SURFACE = "#FFFFFF"
    PX = 1 / 140
    DPI = 280
    SLIDES = True
    # with the bigger text the two-line names overlap (the changes are only 2 months apart)
    EVENT_NAMES = {"downturn": "downturn", "score_offset": "new scale", "camouflage": "camouflage"}
    INCIDENT_NAME = "form bug"
    FIRST = dict(color=ORANGE, lw=4, marker="o", ms=8)
    SECOND = dict(color=INK, lw=3, ls="--", marker="s", ms=7)


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"saved {path.relative_to(PROJECT_DIR) if path.is_relative_to(PROJECT_DIR) else path}")


def pct(decimals=0):
    return PercentFormatter(1, decimals=decimals)


def title(ax, text, subtitle=None):
    if SLIDES:
        return
    ax.set_title(text, loc="left", color=INK, fontsize=20, pad=34 if subtitle else 12)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, color=TEXT, fontsize=15, va="bottom")


# ---------------------------------------------------------------- loading

def read_csv(path):
    """A CSV of the result folder, or an empty table (e.g. no training in the year)."""
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def saved_config(folder, name, fallback):
    """The config that was saved with the run; older folders do not have it yet."""
    if (folder / name).exists():
        return json.loads((folder / name).read_text(encoding="utf-8"))
    print(f"note: {folder} has no {name}, I use the current {fallback} instead")
    return load_json(fallback)


def load_run(name, folder):
    folder = Path(folder)
    if not folder.is_absolute():
        folder = PROJECT_DIR / folder
    apps = pd.read_parquet(folder / "applications.parquet")
    return {
        "name": name,
        "apps": apps,
        "weekly": read_csv(folder / "weekly_monitoring.csv"),
        "trainings": read_csv(folder / "trainings.csv"),
        "psi": read_csv(folder / "weekly_feature_psi.csv"),
        "scenario": saved_config(folder, "scenario.json", "configs/drift_scenario.json"),
        "monitoring": saved_config(folder, "monitoring.json", "configs/monitoring.json"),
        # With all labels every application has label_source "simulation".
        "realistic": bool(apps["label_source"].isna().any()),
    }


def default_runs():
    runs = []
    if (RESULTS / "all_labels" / "applications.parquet").exists():
        runs.append(("all_labels", RESULTS / "all_labels"))
    for folder in sorted((RESULTS / "realistic_labels").glob("*")):
        if (folder / "applications.parquet").exists():
            runs.append((folder.name, folder))
    return runs


def default_comparisons(runs):
    """Which pairs I compare if nothing is given on the command line."""
    first = next(iter(runs))
    pairs = [(f"{first}:system", f"{first}:fast_loop"),    # what retraining adds (same workload)
             (f"{first}:system", f"{first}:no_mlops")]     # the whole system vs doing nothing
    full = [name for name, run in runs.items() if not run["realistic"]]
    realistic = [name for name, run in runs.items() if run["realistic"]]
    if full and realistic:
        pairs.append((f"{full[0]}:system", f"{realistic[0]}:system"))   # what the labels are worth
    return pairs


def series(runs, spec):
    """One model of one run as a plain table: month, week, true label, flag and score."""
    run_name, model = spec.split(":")
    if run_name not in runs or model not in MODELS:
        raise SystemExit(f"unknown series {spec!r}: runs {list(runs)}, models {list(MODELS)}")
    apps = runs[run_name]["apps"]
    frame = apps[["application_id", "month", "sim_week", "fraud_bool", "customer_age"]].copy()
    frame["flagged"] = apps[MODELS[model]["flag"]].astype(bool)
    frame["probability"] = apps[MODELS[model]["probability"]]
    # Result folders from before the fourth version do not have this flag.
    if "camouflaged" in apps:
        frame["camouflaged"] = apps["camouflaged"].astype(bool)
    return frame


def series_label(spec, other):
    """'full system' if both series come from the same run, else with the run name."""
    run_name, model = spec.split(":")
    label = MODELS[model]["label"]
    if run_name != other.split(":")[0]:
        label += f", {run_name.replace('_', ' ')}"
    return label


# ---------------------------------------------------------------- numbers

def wilson(hits, n, z=1.96):
    """95 % interval of a share. With ~250 frauds a month the recall alone moves by
    about +-6 pp, so I show this band instead of reading every bump as a change."""
    if n == 0:
        return math.nan, math.nan
    p = hits / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def counts(y, flagged):
    y, flagged = np.asarray(y).astype(bool), np.asarray(flagged).astype(bool)
    return {"tp": int((flagged & y).sum()), "fn": int((~flagged & y).sum()),
            "fp": int((flagged & ~y).sum()), "tn": int((~flagged & ~y).sum())}


def share(part, whole):
    return part / whole if whole else math.nan


def monthly_metrics(frame):
    rows = []
    for month, part in frame.groupby("month"):
        c = counts(part["fraud_bool"], part["flagged"])
        low, high = wilson(c["tp"], c["tp"] + c["fn"])
        y = part["fraud_bool"].to_numpy()
        rows.append({
            "month": month, "applications": len(part), "frauds": c["tp"] + c["fn"], **c,
            "recall": share(c["tp"], c["tp"] + c["fn"]), "recall_low": low, "recall_high": high,
            "false_positive_rate": share(c["fp"], c["fp"] + c["tn"]),
            "precision": share(c["tp"], c["tp"] + c["fp"]),
            "review_share": part["flagged"].mean(),
            "recall_at_5pct_review": top_share_recall(y, part["probability"].to_numpy(), 0.05),
            "fpr_ratio_age": fpr_by_age(y, part["flagged"], part["customer_age"])["fpr_ratio_age"],
        })
    return pd.DataFrame(rows)


def paired_frauds(a, b):
    """Frauds that only A or only B caught, per month, with an exact McNemar test.

    Both series score the same applications, so the frauds both catch (or both
    miss) say nothing about the difference. Only these disagreements do.
    """
    both = a.merge(b[["application_id", "flagged"]], on="application_id", suffixes=("_a", "_b"))
    frauds = both[both["fraud_bool"] == 1]
    rows = []
    for month, part in frauds.groupby("month"):
        only_a = int((part["flagged_a"] & ~part["flagged_b"]).sum())
        only_b = int((~part["flagged_a"] & part["flagged_b"]).sum())
        p_value = binomtest(only_a, only_a + only_b).pvalue if only_a + only_b else 1.0
        rows.append({"month": month, "only_a": only_a, "only_b": only_b, "p_value": p_value})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- scenario events

def scenario_events(scenario):
    """First month of every injected change, and the months of the form incident."""
    starts = []
    for key, label in EVENT_NAMES.items():
        months = [m["month"] for m in scenario["months"] if m.get(key)]
        if months:
            starts.append((months[0], label))
    incident = [m["month"] for m in scenario["months"] if m.get("incident")]
    return starts, incident


def event_marks(ax, scenario, unit="month", labels=True, band=True):
    """Thin lines where a change starts and a band for the incident."""
    starts, incident = scenario_events(scenario)
    per = scenario["weeks_per_month"] if unit == "week" else 1

    def edge(month):   # left edge of a month on this axis
        return (month - 1) * per + 0.5

    for i, (month, label) in enumerate(starts):
        ax.axvline(edge(month), color=MUTED, lw=1.2, zorder=1)
        # on slides every second name sits a bit lower, so neighbours don't touch
        height = 0.98 - 0.09 * (i % 2) if SLIDES else 0.98
        if labels:
            ax.text(edge(month) + 0.08 * per, height, label, transform=ax.get_xaxis_transform(), color=TEXT,
                    fontsize=13, va="top", ha="left")
    if incident and band:
        ax.axvspan(edge(incident[0]), edge(incident[-1] + 1), color=BAND, lw=0, zorder=0)
        if labels:
            ax.text(edge(incident[0]) + 0.08 * per, 0.98, INCIDENT_NAME,
                    transform=ax.get_xaxis_transform(), color=TEXT, fontsize=13, va="top", ha="left")


def month_axis(ax, months):
    ax.set_xticks(list(months))
    ax.set_xlim(min(months) - 0.6, max(months) + 0.6)
    ax.set_xlabel("Simulated month")


# ---------------------------------------------------------------- confusion matrices

def draw_matrix(ax, c, compact=False):
    """2 x 2 matrix. The fraud row is orange (caught = the key result), the
    legitimate row stays neutral, so the eye goes to the fraud first."""
    frauds, legit = c["tp"] + c["fn"], c["fp"] + c["tn"]
    cells = [  # (column, row, count, share, colour, text colour, share text)
        (0, 1, c["tp"], share(c["tp"], frauds), ORANGE, "white", "of fraud"),
        (1, 1, c["fn"], share(c["fn"], frauds), ORANGE_WASH, INK, "of fraud"),
        (0, 0, c["fp"], share(c["fp"], legit), BAND, INK, "of legit."),
        (1, 0, c["tn"], share(c["tn"], legit), "#FFFFFF", INK, "of legit."),
    ]
    big, small = (15, 11) if compact else (30, 16)
    for col, row, count, part, colour, text_colour, what in cells:
        ax.add_patch(Rectangle((col, row), 1, 1, facecolor=colour, edgecolor=SURFACE, lw=4 if compact else 8))
        ax.text(col + 0.5, row + 0.58, f"{count:,}", ha="center", va="center", fontsize=big, color=text_colour,
                weight="bold")
        decimals = 1 if part < 0.1 else 0
        note = f"{part * 100:.{decimals}f} %" if compact else f"{part * 100:.{decimals}f} % {what}"
        ax.text(col + 0.5, row + 0.28, note, ha="center", va="center", fontsize=small, color=text_colour)
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_aspect("equal" if compact else "auto")
    ax.axis("off")


def matrix_labels(ax, fontsize=16, rows=True, columns=True):
    if columns:
        ax.text(0.5, 2.06, "flagged", ha="center", va="bottom", fontsize=fontsize, color=TEXT)
        ax.text(1.5, 2.06, "not flagged", ha="center", va="bottom", fontsize=fontsize, color=TEXT)
    if rows:
        ax.text(-0.06, 1.5, "fraud", ha="right", va="center", fontsize=fontsize, color=TEXT)
        ax.text(-0.06, 0.5, "legit.", ha="right", va="center", fontsize=fontsize, color=TEXT)


def versions_text(part):
    versions = sorted(part["served_model_version"].unique(), key=lambda v: int(v[1:]))
    return "served by " + ", ".join(versions)


def confusion_month(c, heading, subtitle, path):
    with style():
        fig, ax = plt.subplots(figsize=(1000 * PX, 640 * PX))
        draw_matrix(ax, c)
        matrix_labels(ax, fontsize=18)
        ax.set_xlim(-0.35, 2)
        ax.set_ylim(0, 2.18)
        fig.text(0.04, 0.95, heading, color=INK, fontsize=22, va="top")
        fig.text(0.04, 0.885, subtitle, color=TEXT, fontsize=16, va="top")
        fig.subplots_adjust(left=0.04, right=0.98, top=0.82, bottom=0.03)
        save(fig, path)


def confusion_year(cells, heading, path):
    """All months of one model next to each other (two rows of six)."""
    months = list(cells)
    with style():
        fig, axes = plt.subplots(2, 6, figsize=(1664 * PX, 760 * PX))
        for i, ax in enumerate(axes.ravel()):
            if i >= len(months):
                ax.axis("off")
                continue
            month, (c, note) = months[i], cells[months[i]]
            draw_matrix(ax, c, compact=True)
            matrix_labels(ax, fontsize=12, rows=i % 6 == 0, columns=i < 6)
            ax.set_ylim(0, 2.25 if i < 6 else 2)
            ax.text(0, -0.08, f"M{month}  {note}", transform=ax.transData, fontsize=13, color=INK, va="top")
        fig.text(0.02, 0.97, heading, color=INK, fontsize=22, va="top")
        fig.subplots_adjust(left=0.05, right=0.99, top=0.86, bottom=0.06, wspace=0.12, hspace=0.32)
        save(fig, path)


def confusion_charts(run, output, single_months=True):
    """One overview per model, and (if wanted) one big matrix per month for a slide."""
    apps = run["apps"]
    for model, info in MODELS.items():
        frame = series({run["name"]: run}, f"{run['name']}:{model}")
        year = {}
        for month, part in frame.groupby("month"):
            c = counts(part["fraud_bool"], part["flagged"])
            recall = share(c["tp"], c["tp"] + c["fn"])
            year[month] = (c, f"recall {recall:.0%}")
            if not single_months:
                continue
            fpr = share(c["fp"], c["fp"] + c["tn"])
            precision = share(c["tp"], c["tp"] + c["fp"])
            subtitle = f"recall {recall:.0%}  ·  FPR {fpr:.1%}  ·  precision {precision:.0%}"
            if model == "system":
                subtitle += "  ·  " + versions_text(apps.loc[part.index])
            confusion_month(c, f"Month {month} · {info['label']}", subtitle,
                            output / "confusion" / model / f"month_{month:02d}.png")
        confusion_year(year, f"Fraud caught and missed per month · {info['label']}",
                       output / "confusion" / f"{model}_year.png")


def seen_by_system(run, output):
    """The same matrix, but only with the labels the system really got.

    With realistic labels the 'missed' cell comes only from the random audit,
    so it holds a handful of frauds per month. This is why recall cannot be
    measured from the system's own data.
    """
    apps = run["apps"]
    labelled = apps[apps["label_source"].notna()]
    year = {}
    for month, part in labelled.groupby("month"):
        c = counts(part["fraud_bool"], part["served_flagged"])
        unlabelled = int((apps["month"] == month).sum() - len(part))
        year[month] = (c, f"{unlabelled:,} unlabelled")
    audit = run["scenario"].get("audit_share")
    note = f" (reviews + {audit:.0%} audit)" if audit else ""
    confusion_year(year, f"What the system saw{note} · full system", output / "confusion" / "seen_by_system_year.png")


# ---------------------------------------------------------------- one run

def outcomes_per_month(run, output):
    """Frauds caught and missed by the full system, as stacked bars."""
    table = monthly_metrics(series({run["name"]: run}, f"{run['name']}:system"))
    with style():
        fig, ax = plt.subplots(figsize=(1180 * PX, 620 * PX))
        ax.bar(table["month"], table["tp"], width=0.7, color=ORANGE, label="caught", zorder=3)
        ax.bar(table["month"], table["fn"], bottom=table["tp"], width=0.7, color=GRID, label="missed", zorder=3,
               edgecolor=SURFACE, lw=2)
        ax.set_ylim(0, table["frauds"].max() * 1.3)
        event_marks(ax, run["scenario"])
        month_axis(ax, table["month"])
        ax.set_ylabel("Fraud cases")
        total = table["tp"].sum() / table["frauds"].sum()
        title(ax, "Fraud cases per month · full system", f"{table['tp'].sum():,} of {table['frauds'].sum():,} "
                                                         f"caught in the year ({total:.0%})")
        ax.legend(loc="upper left", bbox_to_anchor=(0, 0.82))
        fig.tight_layout()
        save(fig, output / "outcomes_per_month.png")


def believed_vs_true_recall(run, output):
    weekly = run["weekly"]
    if weekly.empty or "served_recall_last_28_days_of_labels" not in weekly:
        return
    weekly = weekly.dropna(subset=["true_recall_last_28_days"]).sort_values("week")
    monitoring = run["monitoring"]
    measurable = ((weekly["effective_fraud_labels"] >= monitoring["min_fraud_labels_for_performance"])
                  & (weekly["missed_fraud_labels"] >= monitoring["min_missed_fraud_labels"]))
    with style():
        fig, ax = plt.subplots(figsize=(1180 * PX, 620 * PX))
        ax.plot(weekly["week"], weekly["served_recall_last_28_days_of_labels"], **{**FIRST, "ms": 5},
                label="what the system believed")
        ax.plot(weekly["week"], weekly["true_recall_last_28_days"], **{**SECOND, "ms": 4}, label="true recall")
        if "min_served_recall" in monitoring:   # older runs had a relative rule instead
            ax.axhline(monitoring["min_served_recall"], color=MUTED, lw=1.2, ls=":")
            ax.text(weekly["week"].max() + 0.6, monitoring["min_served_recall"], "retrain\nbelow", color=TEXT,
                    fontsize=13, va="center")
        event_marks(ax, run["scenario"], unit="week")
        ax.set_ylim(0, 1.15)
        ax.yaxis.set_major_formatter(pct())
        ax.set_xlabel("Simulated week")
        ax.set_ylabel("Recall (labels of 28 days)")
        title(ax, "Recall: belief vs truth",
              f"recall measurable by the monitor in {int(measurable.sum())} of {len(weekly)} weeks")
        ax.legend(loc="lower left")
        fig.tight_layout()
        save(fig, output / "believed_vs_true_recall.png")


def labels_per_month(run, output):
    """How many of the frauds the system ever learned about, and from where."""
    apps = run["apps"]
    frauds = apps[apps["fraud_bool"] == 1]
    table = pd.DataFrame({
        "all": frauds.groupby("month").size(),
        "review": frauds[frauds["label_source"] == "review"].groupby("month").size(),
        "audit": frauds[frauds["label_source"] == "audit"].groupby("month").size(),
    }).fillna(0)
    with style():
        fig, ax = plt.subplots(figsize=(1180 * PX, 620 * PX))
        ax.bar(table.index, table["all"], width=0.75, color=BAND, edgecolor=MUTED, lw=1, label="all fraud cases",
               zorder=2)
        ax.bar(table.index, table["review"], width=0.5, color=ORANGE, label="known from reviews", zorder=3)
        ax.bar(table.index, table["audit"], bottom=table["review"], width=0.5, color=INK,
               label="known from the audit", zorder=3)
        ax.set_ylim(0, table["all"].max() * 1.45)
        event_marks(ax, run["scenario"])
        month_axis(ax, table.index)
        ax.set_ylabel("Fraud cases")
        known = (table["review"].sum() + table["audit"].sum()) / table["all"].sum()
        title(ax, "Fraud labels the system received",
              f"{known:.0%} of all fraud cases get a label · {int(table['audit'].sum())} through the audit")
        ax.legend(loc="upper left", bbox_to_anchor=(0, 0.82), ncols=3)
        fig.tight_layout()
        save(fig, output / "labels_per_month.png")


def psi_heatmap(run, output, top=12):
    """PSI of the most drifted features per week (log colour scale)."""
    psi = run["psi"]
    if psi.empty:
        return
    features = psi.drop(columns=["week", "model_version"]).max().sort_values(ascending=False).index[:top]
    data = psi.set_index("week")[features].T.clip(lower=0.01, upper=10)
    # White (not the slide colour) for "no drift", so the quiet weeks after a
    # retraining do not look like missing data.
    cmap = LinearSegmentedColormap.from_list("orange", ["#FFFFFF", ORANGE_WASH, ORANGE, "#7A3906"])
    with style():
        fig, ax = plt.subplots(figsize=(1664 * PX, 720 * PX))
        image = ax.imshow(data.to_numpy(), aspect="auto", cmap=cmap, norm=LogNorm(0.01, 10),
                          extent=(data.columns.min() - 0.5, data.columns.max() + 0.5, len(features) - 0.5, -0.5))
        ax.set_yticks(range(len(features)), features, fontsize=14)
        ax.grid(False)
        event_marks(ax, run["scenario"], unit="week", labels=False, band=False)
        ax.set_xlabel("Simulated week")
        bar =fig.colorbar(image, ax=ax, ticks=[0.01, 0.1, 0.25, 1, 10], pad=0.01)
        bar.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        bar.outline.set_visible(False)
        bar.set_label("PSI (alarm at 0.25)", color=TEXT)
        title(ax, "Feature drift per week", f"{top} most drifted features, PSI against the deployed model")
        fig.tight_layout()
        save(fig, output / "feature_psi_heatmap.png")


def model_timeline(run, output):
    """Which version served each week, and every training (promoted or rejected)."""
    apps = run["apps"]
    weekly = apps.groupby("sim_week")["served_model_version"].agg(lambda v: v.mode()[0])
    number = weekly.str[1:].astype(int)
    trainings = run["trainings"]
    with style():
        fig, ax = plt.subplots(figsize=(1180 * PX, 620 * PX))
        ax.step(number.index, number, where="mid", color=MUTED, lw=2, zorder=2)
        if not trainings.empty:
            promoted = trainings[trainings["status"] == "promoted"]
            rejected = trainings[trainings["status"] != "promoted"]
            y = promoted["deployed_version"].str[1:].astype(int)
            ax.scatter(promoted["week"], y, s=110, color=ORANGE, zorder=4, label="retrained and promoted")
            current = [number.get(int(w), number.iloc[0]) for w in rejected["week"]]
            ax.scatter(rejected["week"], current, s=110, marker="X", color=INK, zorder=4, label="candidate rejected")
            # Only the unplanned trainings get a label, the schedule ones are the normal case.
            # (left above the dot, where the step line leaves free space)
            for _, row in promoted[promoted["reason"] != "schedule"].iterrows():
                ax.text(row["week"] - 0.6, int(row["deployed_version"][1:]) + 0.35,
                        str(row["reason"]).replace(",", " + "), fontsize=12, color=TEXT, ha="right", va="bottom")
        event_marks(ax, run["scenario"], unit="week")
        ax.set_ylim(0, number.max() + 3)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"v{int(v)}" if v >= 1 else ""))
        ax.set_xlabel("Simulated week")
        ax.set_ylabel("Served model")
        promoted_count = 0 if trainings.empty else int((trainings["status"] == "promoted").sum())
        title(ax, "Model versions over the year", f"{len(trainings)} trainings, {promoted_count} promoted")
        ax.legend(loc="lower right")
        fig.tight_layout()
        save(fig, output / "model_timeline.png")


def review_share_and_threshold(run, output):
    """Fast loop: weekly review share (system vs fixed threshold) and the threshold."""
    apps = run["apps"]
    weekly = apps.groupby("sim_week").agg(system=("served_flagged", "mean"), fixed=("frozen_v1_flagged", "mean"),
                                          threshold=("served_threshold", "median"))
    plan = run["monitoring"]["review_share"]
    with style():
        fig, (top, bottom) = plt.subplots(2, 1, figsize=(1180 * PX, 720 * PX), sharex=True,
                                          gridspec_kw={"height_ratios": [1.6, 1]})
        # short legend on slides, the long one does not fit with the bigger text
        names = ("full system", "no MLOps") if SLIDES else \
            ("full system (weekly threshold)", "no MLOps (fixed threshold)")
        top.plot(weekly.index, weekly["system"], **{**FIRST, "ms": 5}, label=names[0])
        top.plot(weekly.index, weekly["fixed"], **{**SECOND, "ms": 4}, label=names[1])
        top.axhline(plan, color=MUTED, lw=1.2, ls=":")
        top.text(weekly.index.max() + 0.6, plan, f"plan\n{plan:.1%}", color=TEXT, fontsize=13, va="center")
        # extra room at the top, so the legend sits above the lines
        top.set_ylim(0, max(weekly["fixed"].max(), weekly["system"].max()) * 1.8)
        top.yaxis.set_major_formatter(pct())
        top.set_ylabel("Sent to review")
        top.legend(loc="upper left", bbox_to_anchor=(0, 0.86), ncols=2)
        event_marks(top, run["scenario"], unit="week")
        bottom.plot(weekly.index, weekly["threshold"], color=ORANGE, lw=2.5, drawstyle="steps-mid")
        bottom.set_ylabel("Threshold")
        bottom.set_xlabel("Simulated week")
        event_marks(bottom, run["scenario"], unit="week", labels=False)
        title(top, "Review workload per week", "the fast loop resets the threshold every week, without labels")
        fig.tight_layout()
        save(fig, output / "review_share_and_threshold.png")


# ---------------------------------------------------------------- two series

def compare_lines(ax, table_a, table_b, key, label_a, label_b, scenario, band=False):
    if band:
        ax.fill_between(table_a["month"], table_a[f"{key}_low"], table_a[f"{key}_high"], color=ORANGE_WASH,
                        lw=0, zorder=1)
        ax.fill_between(table_b["month"], table_b[f"{key}_low"], table_b[f"{key}_high"], color=INK, alpha=0.08,
                        lw=0, zorder=1)
    ax.plot(table_a["month"], table_a[key], **FIRST, label=label_a, zorder=4)
    ax.plot(table_b["month"], table_b[key], **SECOND, label=label_b, zorder=3)
    event_marks(ax, scenario)
    month_axis(ax, table_a["month"])


def single_metric(table_a, table_b, labels, scenario, key, heading, subtitle, path, limit=None, band=False,
                  percent=True, decimals=0):
    with style():
        fig, ax = plt.subplots(figsize=(1180 * PX, 620 * PX))
        compare_lines(ax, table_a, table_b, key, *labels, scenario, band=band)
        if limit is not None:
            value, text = limit
            ax.axhline(value, color=MUTED, lw=1.2, ls=":")
            ax.text(table_a["month"].max() + 0.7, value, text, color=TEXT, fontsize=13, va="center")
        values = pd.concat([table_a[key], table_b[key]] + ([table_a[f"{key}_high"]] if band else []))
        top = max(values.max(), limit[0] if limit else 0)
        ax.set_ylim(0, top * 1.35)
        if percent:
            ax.yaxis.set_major_formatter(pct(decimals))
        title(ax, heading, subtitle)
        ax.legend(loc="lower left")
        fig.tight_layout()
        save(fig, path)


def detected_fraud(table_a, table_b, labels, scenario, path):
    with style():
        fig, ax = plt.subplots(figsize=(1180 * PX, 620 * PX))
        x = table_a["month"]
        ax.bar(x, table_a["frauds"], width=0.8, color=BAND, edgecolor=MUTED, lw=1, label="all fraud cases", zorder=2)
        ax.bar(x - 0.18, table_a["tp"], width=0.34, color=ORANGE, label=labels[0], zorder=3)
        ax.bar(x + 0.18, table_b["tp"], width=0.34, color=INK, label=labels[1], zorder=3)
        ax.set_ylim(0, table_a["frauds"].max() * 1.35)
        event_marks(ax, scenario)
        month_axis(ax, x)
        ax.set_ylabel("Fraud cases")
        title(ax, "Fraud cases caught per month",
              f"year: {labels[0]} {table_a['tp'].sum():,}, {labels[1]} {table_b['tp'].sum():,} "
              f"of {table_a['frauds'].sum():,}")
        ax.legend(loc="upper left", bbox_to_anchor=(0, 0.82), ncols=3)
        fig.tight_layout()
        save(fig, path)


def paired_difference(paired, labels, scenario, path):
    """Frauds only one of the two caught. Months with p < 0.05 get a star."""
    with style():
        fig, ax = plt.subplots(figsize=(1180 * PX, 620 * PX))
        x = paired["month"]
        ax.bar(x, paired["only_a"], width=0.6, color=ORANGE, label=labels[0], zorder=3)
        ax.bar(x, -paired["only_b"], width=0.6, color=INK, label=labels[1], zorder=3)
        ax.axhline(0, color=MUTED, lw=1)
        biggest = max(paired["only_a"].max(), paired["only_b"].max(), 1)
        for _, row in paired[paired["p_value"] < 0.05].iterrows():
            # the star goes to the side that caught more
            if row["only_a"] >= row["only_b"]:
                ax.text(row["month"], row["only_a"] + biggest * 0.04, "*", ha="center", va="bottom", fontsize=24,
                        color=INK)
            else:
                ax.text(row["month"], -row["only_b"] - biggest * 0.04, "*", ha="center", va="top", fontsize=24,
                        color=INK)
        ax.set_ylim(-biggest * 1.5, biggest * 1.5)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{abs(int(v))}"))
        event_marks(ax, scenario)
        month_axis(ax, x)
        ax.set_ylabel("Fraud cases")
        significant = int((paired["p_value"] < 0.05).sum())
        title(ax, "Frauds caught by only one of the two",
              f"* = difference unlikely to be chance (McNemar p < 0.05): {significant} of {len(paired)} months")
        ax.legend(loc="lower left", ncols=2, title="caught only by", title_fontsize=15, alignment="left")
        fig.tight_layout()
        save(fig, path)


def recall_by_fraud_type(a, b, labels, scenario, path):
    """Recall on normal and on camouflaged fraud, per month.

    The camouflage is invisible for the data monitoring, so this shows directly
    whether retraining learned the new pattern - and whether the overall drop in
    recall comes from the camouflaged cases or from all fraud.
    """
    if "camouflaged" not in a or "camouflaged" not in b or not (a["camouflaged"] & (a["fraud_bool"] == 1)).any():
        return
    with style():
        fig, axes = plt.subplots(1, 2, figsize=(1664 * PX, 680 * PX), sharey=True)
        for ax, camouflaged, heading in ((axes[0], False, "Normal fraud"), (axes[1], True, "Camouflaged fraud")):
            tables = [monthly_metrics(frame[frame["camouflaged"] == camouflaged]) for frame in (a, b)]
            # A month with very few camouflaged frauds is only noise, so I need at least 20.
            tables = [table[table["frauds"] >= 20] for table in tables]
            for table, line, label in ((tables[0], FIRST, labels[0]), (tables[1], SECOND, labels[1])):
                ax.plot(table["month"], table["recall"], **line, label=label, zorder=3)
            event_marks(ax, scenario, labels=camouflaged)
            month_axis(ax, a["month"].unique())
            ax.set_ylim(0, 1.05)
            ax.yaxis.set_major_formatter(pct())
            frauds = int(((a["fraud_bool"] == 1) & (a["camouflaged"] == camouflaged)).sum())
            ax.set_title(f"{heading} ({frauds:,} cases)", loc="left", color=INK, fontsize=19, pad=12)
        axes[0].set_ylabel("True recall")
        axes[0].legend(loc="lower left")
        fig.tight_layout(w_pad=3)
        save(fig, path)


def cumulative_caught(a, b, labels, scenario, path):
    """Caught frauds added up week by week; small monthly gaps become visible."""
    weekly = pd.DataFrame({
        "a": (a["flagged"] & (a["fraud_bool"] == 1)).groupby(a["sim_week"]).sum(),
        "b": (b["flagged"] & (b["fraud_bool"] == 1)).groupby(b["sim_week"]).sum(),
        "all": a.groupby("sim_week")["fraud_bool"].sum(),
    }).cumsum()
    with style():
        fig, ax = plt.subplots(figsize=(1180 * PX, 620 * PX))
        ax.plot(weekly.index, weekly["all"], color=MUTED, lw=2, ls=":", label="all fraud cases")
        ax.plot(weekly.index, weekly["a"], color=ORANGE, lw=3, label=labels[0])
        ax.plot(weekly.index, weekly["b"], color=INK, lw=2.5, ls="--", label=labels[1])
        end = weekly.index.max()
        # End values as labels; if A and B end close together I push them apart.
        gap = weekly["all"].max() * 0.045
        a_end, b_end = weekly["a"].iloc[-1], weekly["b"].iloc[-1]
        a_y, b_y = a_end, b_end
        if abs(a_end - b_end) < gap:
            middle, sign = (a_end + b_end) / 2, 1 if a_end >= b_end else -1
            a_y, b_y = middle + sign * gap / 2, middle - sign * gap / 2
        for value, y in ((weekly["all"].iloc[-1], weekly["all"].iloc[-1]), (a_end, a_y), (b_end, b_y)):
            ax.text(end + 0.6, y, f"{int(value):,}", color=INK, fontsize=14, va="center")
        event_marks(ax, scenario, unit="week")
        ax.set_ylim(0, weekly["all"].max() * 1.15)
        ax.set_xlim(weekly.index.min() - 0.5, end + 4)
        ax.set_xlabel("Simulated week")
        ax.set_ylabel("Fraud cases (cumulative)")
        gap = int(weekly["a"].iloc[-1] - weekly["b"].iloc[-1])
        title(ax, "Fraud caught over the year", f"difference after the year: {gap:+,} cases")
        ax.legend(loc="upper left", bbox_to_anchor=(0, 0.82))
        fig.tight_layout()
        save(fig, path)


def overview(table_a, table_b, labels, scenario, review_plan, path):
    panels = [("recall", "Recall (95 % band)", None, True),
              ("tp", "Fraud cases caught", None, False),
              ("false_positive_rate", "False-positive rate", (FPR_LIMIT, "limit"), False),
              ("review_share", "Sent to review", (review_plan, "plan"), False)]
    with style():
        fig, axes = plt.subplots(2, 2, figsize=(1664 * PX, 900 * PX))
        for ax, (key, heading, limit, band) in zip(axes.ravel(), panels):
            compare_lines(ax, table_a, table_b, key, *labels, scenario, band=band)
            for text in ax.texts:   # the event names would crowd the small panels
                text.set_visible(False)
            if limit:
                ax.axhline(limit[0], color=MUTED, lw=1.2, ls=":")
            ax.set_ylim(0, max(table_a[key].max(), table_b[key].max(), limit[0] if limit else 0) * 1.25)
            if key != "tp":
                ax.yaxis.set_major_formatter(pct(1 if key != "recall" else 0))
            ax.set_title(heading, loc="left", color=INK, fontsize=18)
            ax.set_xlabel("")
        axes[0, 0].legend(loc="lower left", fontsize=14)
        fig.tight_layout(h_pad=1.5, w_pad=2.5)
        save(fig, path)


def compare(runs, spec_a, spec_b, output):
    a, b = series(runs, spec_a), series(runs, spec_b)
    if set(a["application_id"]) != set(b["application_id"]):
        # Runs with another scenario have other applications; then I only use the shared ones.
        shared = set(a["application_id"]) & set(b["application_id"])
        print(f"note: {spec_a} and {spec_b} share only {len(shared):,} applications")
        a, b = a[a["application_id"].isin(shared)], b[b["application_id"].isin(shared)]
    labels = (series_label(spec_a, spec_b), series_label(spec_b, spec_a))
    scenario = runs[spec_a.split(":")[0]]["scenario"]
    review_plan = runs[spec_a.split(":")[0]]["monitoring"]["review_share"]
    folder = output / "compare" / f"{spec_a.replace(':', '-')}_vs_{spec_b.replace(':', '-')}"
    table_a, table_b = monthly_metrics(a), monthly_metrics(b)
    paired = paired_frauds(a, b)

    numbers = table_a.merge(table_b, on="month", suffixes=("_a", "_b")).merge(paired, on="month")
    numbers.insert(1, "series_a", spec_a)
    numbers.insert(2, "series_b", spec_b)
    folder.mkdir(parents=True, exist_ok=True)
    numbers.to_csv(folder / "monthly_numbers.csv", index=False)

    single_metric(table_a, table_b, labels, scenario, "recall", "True recall per month",
                  "shaded: 95 % interval (only ~200–400 frauds per month)", folder / "recall.png", band=True)
    single_metric(table_a, table_b, labels, scenario, "false_positive_rate", "False-positive rate per month",
                  "legitimate applicants sent to review", folder / "false_positive_rate.png",
                  limit=(FPR_LIMIT, "limit\n5 %"), decimals=1)
    single_metric(table_a, table_b, labels, scenario, "precision", "Precision per month",
                  "share of reviewed applications that are fraud", folder / "precision.png")
    single_metric(table_a, table_b, labels, scenario, "review_share", "Review workload per month",
                  "share of all applications sent to review", folder / "review_share.png",
                  limit=(review_plan, f"plan\n{review_plan:.1%}"), decimals=1)
    single_metric(table_a, table_b, labels, scenario, "recall_at_5pct_review", "Recall at equal workload",
                  "fraud found in the 5 % highest scores (fair ranking comparison)",
                  folder / "recall_at_equal_workload.png")
    fairness_limit = runs[spec_a.split(":")[0]]["monitoring"].get("min_fpr_ratio_age_warning")
    single_metric(table_a, table_b, labels, scenario, "fpr_ratio_age", "Fairness: FPR ratio by age",
                  "lower / higher FPR of the two age groups (1 = equal)", folder / "fairness_fpr_ratio.png",
                  limit=(fairness_limit, "warning") if fairness_limit else None, percent=False)
    detected_fraud(table_a, table_b, labels, scenario, folder / "detected_fraud.png")
    paired_difference(paired, labels, scenario, folder / "paired_difference.png")
    cumulative_caught(a, b, labels, scenario, folder / "cumulative_caught.png")
    recall_by_fraud_type(a, b, labels, scenario, folder / "recall_by_fraud_type.png")
    overview(table_a, table_b, labels, scenario, review_plan, folder / "overview.png")


def label_modes_spread(runs, output):
    """All labels (orange) vs the realistic runs (black = mean, grey = range of the
    audit samples). One realistic run alone could be luck, the range shows how much."""
    full = [name for name, run in runs.items() if not run["realistic"]]
    realistic = [name for name, run in runs.items() if run["realistic"]]
    if not full or len(realistic) < 2:
        return
    first = monthly_metrics(series(runs, f"{full[0]}:system"))
    others = pd.concat([monthly_metrics(series(runs, f"{name}:system")).assign(run=name) for name in realistic])
    spread = others.groupby("month")["recall"].agg(["mean", "min", "max"])
    scenario = runs[full[0]]["scenario"]
    with style():
        fig, ax = plt.subplots(figsize=(1180 * PX, 620 * PX))
        ax.fill_between(spread.index, spread["min"], spread["max"], color=INK, alpha=0.1, lw=0,
                        label=f"range of {len(realistic)} audit samples")
        ax.plot(first["month"], first["recall"], **FIRST, label="all labels")
        ax.plot(spread.index, spread["mean"], **SECOND, label="realistic labels (mean)")
        event_marks(ax, scenario)
        month_axis(ax, first["month"])
        ax.set_ylim(0, max(first["recall"].max(), spread["max"].max()) * 1.35)
        ax.yaxis.set_major_formatter(pct())
        title(ax, "True recall with all vs realistic labels", "full system in both cases, same applications")
        ax.legend(loc="lower left")
        fig.tight_layout()
        save(fig, output / "compare" / "label_modes_recall.png")


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", help="name=folder with the summarize output (repeatable)")
    parser.add_argument("--compare", action="append",
                        help="two series run:model,run:model (repeatable); models: " + ", ".join(MODELS))
    parser.add_argument("--output", type=Path, default=None,
                        help="default: charts (or charts/slides with --slides)")
    parser.add_argument("--no-monthly-matrices", action="store_true",
                        help="skip the single confusion matrix per month (36 files per run)")
    parser.add_argument("--slides", action="store_true",
                        help="white background, bigger text and no chart titles, for the slide deck")
    args = parser.parse_args()
    if args.slides:
        slide_theme()
    if args.output is None:
        args.output = PROJECT_DIR / "charts" / ("slides" if args.slides else "")

    items = [item.split("=", 1) for item in args.run] if args.run else default_runs()
    if not items:
        raise SystemExit("no result folders found, pass them with --run name=folder")
    runs = {name: load_run(name, folder) for name, folder in items}

    for name, run in runs.items():
        output = args.output / name
        confusion_charts(run, output, single_months=not args.no_monthly_matrices)
        if run["realistic"]:
            seen_by_system(run, output)
            labels_per_month(run, output)
        outcomes_per_month(run, output)
        believed_vs_true_recall(run, output)
        psi_heatmap(run, output)
        model_timeline(run, output)
        review_share_and_threshold(run, output)

    pairs = [tuple(item.split(",")) for item in args.compare] if args.compare else default_comparisons(runs)
    for spec_a, spec_b in pairs:
        compare(runs, spec_a, spec_b, args.output)
    label_modes_spread(runs, args.output)
    print(f"\nAll charts in {args.output}")


if __name__ == "__main__":
    main()
