"""Presentation-ready binary fraud evaluation from labels and saved scores.

No fitting or threshold selection happens here. Pass the threshold chosen on
validation unchanged when reporting later months. Figures use a headless canvas
and local styling, so importing this module does not change notebook settings.
"""

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
import textwrap

import matplotlib as mpl
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix, log_loss,
    precision_recall_curve, roc_auc_score, roc_curve,
)


NO_ALERT_THRESHOLD = float(np.nextafter(1.0, np.inf))


@dataclass(frozen=True)
class ChartTheme:
    """Override these colors to match a presentation template."""

    background: str = "#F5F7FA"
    panel: str = "#FFFFFF"
    ink: str = "#172B46"
    muted: str = "#617187"
    grid: str = "#DFE6EE"
    accent: str = "#087F8C"
    warning: str = "#C65C42"
    secondary: str = "#6874B7"
    font: str = "DejaVu Sans"


@dataclass
class EvaluationReport:
    dataset_name: str
    model_name: str
    metrics: dict
    output_dir: Path
    files: dict = field(default_factory=dict)
    fpr_limit: float | None = None

    @classmethod
    def load(cls, output_dir):
        """Reload a saved report's metrics for a later monthly comparison."""
        directory = Path(output_dir).resolve()
        saved = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
        return cls(saved["dataset_name"], saved["model_name"], saved["metrics"],
                   directory, fpr_limit=saved.get("fpr_limit"))


def _validate(y_true, y_score, threshold):
    y, scores = np.asarray(y_true), np.asarray(y_score, dtype=float)
    if y.ndim != 1 or scores.ndim != 1 or len(y) != len(scores) or not len(y):
        raise ValueError("Labels and scores must be nonempty matching one-dimensional arrays.")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Labels must be 0 (non-fraud) or 1 (fraud), without missing labels.")
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Scores must be finite fraud probabilities between 0 and 1.")
    if not np.isfinite(threshold) or not 0 <= threshold <= NO_ALERT_THRESHOLD:
        raise ValueError("Use a threshold in [0, 1] or np.nextafter(1.0, np.inf) for no alerts.")
    # Array-like inputs are paired positionally; Series also need matching indices.
    if isinstance(y_true, pd.Series) and isinstance(y_score, pd.Series):
        if not y_true.index.equals(y_score.index):
            raise ValueError("Label and score Series must have matching indices and order.")
    return y.astype(int), scores


def calculate_metrics(y_true, y_score, threshold, sample_weight=None):
    """Metrics shared by training and reporting; undefined rates are None.

    sample_weight is only used in the realistic label mode: an audited
    application stands for 1 / audit share unreviewed ones, so it counts that
    many times. Without weights every row counts once (the normal case).
    """
    y, scores = _validate(y_true, y_score, threshold)
    predicted = scores >= threshold
    if sample_weight is None:
        tn, fp, fn, tp = (int(v) for v in confusion_matrix(y, predicted, labels=[0, 1]).ravel())
        w = np.ones(len(y))
    else:
        w = np.asarray(sample_weight, dtype=float)
        if w.shape != y.shape or not np.all(w > 0):
            raise ValueError("sample_weight must be positive and match the labels.")
        tp, fp = float(w[predicted & (y == 1)].sum()), float(w[predicted & (y == 0)].sum())
        fn, tn = float(w[~predicted & (y == 1)].sum()), float(w[~predicted & (y == 0)].sum())
    positives, negatives = tp + fn, tn + fp
    both_classes = bool(positives and negatives)
    weights = None if sample_weight is None else w
    metrics = {
        "decision_threshold": float(threshold),
        "rows": len(y),
        "average_precision": float(average_precision_score(y, scores, sample_weight=weights)) if positives else None,
        "roc_auc": float(roc_auc_score(y, scores, sample_weight=weights)) if both_classes else None,
        "log_loss": float(log_loss(y, scores, labels=[0, 1], sample_weight=weights)),
        "brier_score": float(brier_score_loss(y, scores, sample_weight=weights)),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / positives if positives else None,
        "false_positive_rate": fp / negatives if negatives else None,
        "flagged_rate": float(np.average(predicted, weights=w)),
        "fraud_rate": float(np.average(y, weights=w)),
        "true_negatives": tn, "false_positives": fp,
        "false_negatives": fn, "true_positives": tp,
    }
    # Top-k review budgets: go down the ranking until k % of all (weighted) applications are reviewed.
    ranking = np.argsort(-scores, kind="stable")
    cumulative = np.cumsum(w[ranking])
    for percent in [1, 2, 5]:
        count = max(1, int(np.searchsorted(cumulative, cumulative[-1] * percent / 100) + 1))
        reviewed = w[ranking[:count]]
        found = float((y[ranking[:count]] * reviewed).sum())
        metrics[f"precision_at_top_{percent}pct"] = found / reviewed.sum()
        metrics[f"recall_at_top_{percent}pct"] = found / positives if positives else None
    return metrics


def _slug(name):
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(name)).strip("_")
    if not slug:
        raise ValueError("Dataset names need at least one letter or number.")
    # Avoid Windows device filenames, including names with arbitrary suffixes.
    if slug.upper() in {"CON", "PRN", "AUX", "NUL", *[f"{s}{i}" for s in ("COM", "LPT") for i in range(1, 10)]}:
        slug = "dataset_" + slug
    return slug.lower()


def _options(formats, dpi, fpr_limit=None):
    if isinstance(formats, str) or not formats or not set(formats) <= {"png", "svg", "pdf"}:
        raise ValueError("formats must contain png, svg and/or pdf, e.g. ('png', 'svg').")
    if not isinstance(dpi, int) or dpi < 72:
        raise ValueError("dpi must be an integer of at least 72.")
    if fpr_limit is not None and (not np.isfinite(fpr_limit) or not 0 < fpr_limit < 1):
        raise ValueError("fpr_limit must be strictly between 0 and 1.")


def _tracking(enabled):
    if not enabled:
        return None
    import mlflow

    if mlflow.active_run() is None:
        raise RuntimeError("Start or resume an MLflow run before logging charts.")
    return mlflow


def _style(theme):
    return mpl.rc_context({
        "font.family": theme.font, "font.size": 12,
        "text.color": theme.ink, "axes.labelcolor": theme.muted,
        "xtick.color": theme.muted, "ytick.color": theme.muted,
        "axes.edgecolor": theme.grid, "axes.titleweight": "bold",
        "svg.fonttype": "none", "pdf.fonttype": 42,
        "text.usetex": False, "text.parse_math": False,
    })


def _value(value, percent=False):
    if value is None:
        return "N/A"
    return f"{value:.1%}" if percent else f"{value:.3f}"


def _threshold_label(threshold):
    return "No alerts" if threshold > 1 else f"{threshold:.6g}"


def _canvas(title, dataset, model, subtitle, theme, footer):
    fig = Figure(figsize=(12.8, 7.2), facecolor=theme.background)
    FigureCanvasAgg(fig)
    fig.text(.06, .935, "FRAUD MODEL  /  " + textwrap.shorten(str(dataset).upper(), 74),
             color=theme.accent, size=10, weight="bold")
    fig.text(.06, .865, title, size=27, weight="bold")
    fig.text(.06, .814, subtitle, size=11, color=theme.muted)
    fig.text(.06, .035, footer, size=9, color=theme.muted)
    fig.text(.94, .035, textwrap.shorten(str(model), 36), ha="right", size=9, color=theme.muted)
    return fig


def _axes(fig, bounds, theme, xlabel="", ylabel=""):
    ax = fig.add_axes(bounds, facecolor=theme.panel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color=theme.grid, linewidth=.7, alpha=.75)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, pad=8, labelsize=10)
    ax.set_xlabel(xlabel, labelpad=12)
    ax.set_ylabel(ylabel, labelpad=12)
    return ax


def _percent(ax, x=True, y=True):
    if x:
        ax.xaxis.set_major_formatter(PercentFormatter(1))
    if y:
        ax.yaxis.set_major_formatter(PercentFormatter(1))


def _note(ax, message, theme):
    ax.text(.5, .5, message, transform=ax.transAxes, ha="center", va="center",
            color=theme.muted, fontsize=14)


def _legend(ax):
    ax.legend(frameon=False, fontsize=10, loc="best")


def _save(fig, directory, name, formats, dpi):
    paths = []
    try:
        for extension in dict.fromkeys(formats):
            path = directory / f"{name}.{extension}"
            # Fixed 16:9 canvas: no tight bounding box changing slide dimensions.
            fig.savefig(path, dpi=dpi, facecolor=fig.get_facecolor(), format=extension)
            paths.append(path)
    finally:
        fig.clear()
    return paths


def _confusion(report, theme):
    m = report.metrics
    fig = _canvas("Every decision, accounted for", report.dataset_name, report.model_name,
                  "Counts and percentages within each actual class", theme,
                  f"{m['rows']:,} labelled applications  |  Threshold: {_threshold_label(m['decision_threshold'])}")
    ax = fig.add_axes([.12, .15, .46, .59])
    ax.set(xlim=(-.5, 1.5), ylim=(1.5, -.5), xticks=[0, 1], yticks=[0, 1],
           xticklabels=["Not flagged", "Flagged"], yticklabels=["Actual\nnon-fraud", "Actual\nfraud"])
    ax.set_xlabel("Model decision", labelpad=12)
    ax.tick_params(length=0, pad=12)
    for spine in ax.spines.values():
        spine.set_visible(False)
    counts = np.array([[m['true_negatives'], m['false_positives']],
                       [m['false_negatives'], m['true_positives']]])
    labels = [["Correctly cleared", "False alarm"], ["Missed fraud", "Fraud caught"]]
    for row in range(2):
        total = counts[row].sum()
        for col in range(2):
            fraction = counts[row, col] / total if total else None
            base = theme.accent if row == col else theme.warning
            # Color intensity encodes row percentage, not the majority class count.
            amount = .10 + .70 * (fraction or 0)
            color = np.array(to_rgb(theme.panel)) * (1 - amount) + np.array(to_rgb(base)) * amount
            ax.add_patch(FancyBboxPatch((col - .47, row - .46), .94, .92,
                                       boxstyle="round,pad=0,rounding_size=.035",
                                       linewidth=0, facecolor=color))
            ink = "white" if amount > .52 else theme.ink
            ax.text(col, row - .15, f"{counts[row, col]:,}", ha="center", va="center", size=29, weight="bold", color=ink)
            ax.text(col, row + .10, labels[row][col], ha="center", size=11, color=ink)
            ax.text(col, row + .30, _value(fraction, True) + " of actual class", ha="center", size=10, color=ink)
    for y, key, label, color in [( .68, "recall", "Fraud caught · recall", theme.accent),
                                 (.48, "precision", "Useful alerts · precision", theme.ink),
                                 (.28, "false_positive_rate", "False alarms · FPR", theme.warning)]:
        fig.text(.67, y, _value(m[key], True), size=32, weight="bold", color=color)
        fig.text(.67, y - .055, label, size=11, color=theme.muted)
    return fig


def _curve(report, y, scores, theme, kind):
    m = report.metrics
    is_roc = kind == "roc"
    title = "Separation across score thresholds" if is_roc else "How useful are the fraud alerts?"
    subtitle = (f"ROC AUC  {_value(m['roc_auc'])}" if is_roc else
                f"Average precision  {_value(m['average_precision'])}  |  Fraud prevalence  {_value(m['fraud_rate'], True)}")
    fig = _canvas(title, report.dataset_name, report.model_name, subtitle, theme,
                  "The marked point uses the saved decision threshold.")
    ax = _axes(fig, [.10, .17, .83, .56], theme,
               "False-positive rate" if is_roc else "Recall · fraction of fraud caught",
               "Recall · true-positive rate" if is_roc else "Precision · fraction of alerts that are fraud")
    ax.set(xlim=(0, 1), ylim=(0, 1.02))
    _percent(ax)
    if is_roc and m['roc_auc'] is not None:
        fpr, tpr, _ = roc_curve(y, scores)
        ax.fill_between(fpr, tpr, color=theme.accent, alpha=.07)
        ax.plot(fpr, tpr, color=theme.accent, lw=2.5, label="Model")
        ax.plot([0, 1], [0, 1], color=theme.muted, lw=1.2, ls="--", label="Random ranking")
        if report.fpr_limit is not None:
            ax.axvline(report.fpr_limit, color=theme.warning, ls=":", lw=1.5,
                       label=f"FPR limit {report.fpr_limit:.1%}")
        ax.scatter(m['false_positive_rate'], m['recall'], s=80, zorder=5,
                   color=theme.warning, edgecolors=theme.panel, label="Saved threshold")
    elif not is_roc and m['average_precision'] is not None:
        precision, recall, _ = precision_recall_curve(y, scores)
        ax.step(recall, precision, where="post", color=theme.accent, lw=2.5, label="Model")
        ax.axhline(m['fraud_rate'], color=theme.muted, ls="--", lw=1.2, label="Prevalence baseline")
        if m['flagged_rate']:
            ax.scatter(m['recall'], m['precision'], s=80, color=theme.warning,
                       edgecolors=theme.panel, zorder=5, label="Saved threshold")
        else:
            ax.text(.98, .93, "No alerts at saved threshold", ha="right", transform=ax.transAxes, color=theme.muted)
    else:
        _note(ax, "ROC needs both classes." if is_roc else "No fraud labels in this batch.\nPrecision–recall is undefined.", theme)
        return fig
    _legend(ax)
    return fig


def _distribution(report, y, scores, theme):
    fig = _canvas("Where the model places its scores", report.dataset_name, report.model_name,
                  "Each class is normalized separately so rare fraud remains visible", theme,
                  "Scores are model outputs; they have not been calibrated.")
    ax = _axes(fig, [.10, .17, .83, .56], theme, "Predicted fraud score", "Share within actual class")
    for label, name, color in [(0, "Non-fraud", theme.accent), (1, "Fraud", theme.warning)]:
        subset = scores[y == label]
        if len(subset):
            ax.hist(subset, bins=np.linspace(0, 1, 31), weights=np.ones(len(subset)) / len(subset),
                    histtype="stepfilled", alpha=.18, color=color)
            ax.hist(subset, bins=np.linspace(0, 1, 31), weights=np.ones(len(subset)) / len(subset),
                    histtype="step", lw=2, color=color, label=f"{name} · n={len(subset):,}")
    threshold = report.metrics['decision_threshold']
    if threshold <= 1:
        ax.axvline(threshold, color=theme.ink, ls="--", lw=1.5, label=f"Threshold {threshold:.4g}")
    else:
        ax.text(.98, .92, "No-alert threshold (>1)", ha="right", transform=ax.transAxes)
    ax.set_xlim(0, 1)
    _percent(ax, x=False)
    _legend(ax)
    return fig


def _calibration(report, y, scores, theme):
    fig = _canvas("Do scores match observed fraud rates?", report.dataset_name, report.model_name,
                  f"Brier score  {_value(report.metrics['brier_score'])}  ·  Lower is better", theme,
                  "Up to 10 equal-frequency bins; empty bins are omitted. Diagnostic only.")
    ax = _axes(fig, [.10, .17, .83, .56], theme, "Mean predicted fraud score", "Observed fraud rate")
    observed, predicted = calibration_curve(y, scores, n_bins=10, strategy="quantile")
    ax.plot([0, 1], [0, 1], color=theme.muted, ls="--", lw=1.2, label="Perfect calibration")
    ax.plot(predicted, observed, "o-", color=theme.accent, lw=2.5, ms=7, label="Observed by score bin")
    ax.set(xlim=(0, 1), ylim=(0, 1))
    _percent(ax)
    _legend(ax)
    return fig


def _gains(report, y, scores, theme):
    fig = _canvas("Fraud captured within a review budget", report.dataset_name, report.model_name,
                  "Review applications from highest to lowest model score", theme,
                  "Equal scores are ordered by input position; this is a ranking budget, not a score threshold.")
    ax = _axes(fig, [.10, .17, .83, .56], theme, "Share of applications reviewed", "Share of all fraud caught")
    ax.set(xlim=(0, 1), ylim=(0, 1.02))
    _percent(ax)
    if not y.sum():
        _note(ax, "No fraud labels in this batch.\nCumulative recall is undefined.", theme)
        return fig
    # Downsample only for rendering; retain the exact 1%, 2%, 5% budget points.
    order = np.argsort(-scores, kind="stable")
    cumulative = np.r_[0, np.cumsum(y[order])] / y.sum()
    budgets = [max(1, int(np.ceil(len(y) * p / 100))) for p in (1, 2, 5)]
    positions = np.unique(np.r_[np.linspace(0, len(y), min(2000, len(y) + 1), dtype=int), budgets])
    ax.plot(positions / len(y), cumulative[positions], color=theme.accent, lw=2.5, label="Model ranking")
    ax.plot([0, 1], [0, 1], color=theme.muted, ls="--", lw=1.2, label="Random ranking")
    ax.scatter(np.array(budgets) / len(y), cumulative[budgets], color=theme.warning, zorder=5,
               label="1%, 2%, 5% review budgets")
    _legend(ax)
    return fig


def create_evaluation_report(y_true, y_score, *, threshold, output_dir,
                             dataset_name="Validation", model_name="CatBoost",
                             fpr_limit=None, theme=None, formats=("png", "svg"),
                             dpi=200, log_to_mlflow=False):
    """Save six 16:9 charts and metrics.json; optionally log to the active run.

    Inputs are aligned labels (0/1) and P(fraud), not predicted classes or a
    two-column predict_proba array. output_dir is the exact destination directory.
    Reusing it replaces this report's filenames. Missing labels must be joined
    with their original scores before calling. One-class batches are supported.
    MLflow artifacts go under charts/<dataset_name slug>; no run is auto-created.
    """
    y, scores = _validate(y_true, y_score, threshold)
    _options(formats, dpi, fpr_limit)
    slug = _slug(dataset_name)
    tracking = _tracking(log_to_mlflow)
    theme = theme or ChartTheme()
    directory = Path(output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    report = EvaluationReport(str(dataset_name), str(model_name), calculate_metrics(y, scores, threshold),
                              directory, fpr_limit=fpr_limit)
    with _style(theme):
        builders = {
            "confusion_matrix": lambda: _confusion(report, theme),
            "roc_curve": lambda: _curve(report, y, scores, theme, "roc"),
            "precision_recall_curve": lambda: _curve(report, y, scores, theme, "pr"),
            "score_distribution": lambda: _distribution(report, y, scores, theme),
            "calibration": lambda: _calibration(report, y, scores, theme),
            "cumulative_gains": lambda: _gains(report, y, scores, theme),
        }
        for name, builder in builders.items():
            report.files[name] = _save(builder(), directory, name, formats, dpi)
    metadata = directory / "metrics.json"
    metadata.write_text(json.dumps({
        "dataset_name": report.dataset_name, "model_name": report.model_name,
        "fpr_limit": fpr_limit, "metrics": report.metrics,
        "notes": ["Predictions use score >= threshold; no threshold is selected by this report.",
                  "Undefined metrics are null; precision is 0 when no applications are flagged.",
                  "Scores are used as supplied (bundles from train.py correct them for class "
                  "weights). All metrics are unweighted.",
                  "Top-budget metrics break equal-score ties in input order."],
    }, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    report.files["metrics"] = [metadata]
    if tracking is not None:
        # Log only generated files, never unrelated files in the output directory.
        for paths in report.files.values():
            for path in paths:
                tracking.log_artifact(str(path), artifact_path=f"charts/{slug}")
    return report


def create_monthly_summary(reports, *, output_dir, baseline=None, theme=None,
                           formats=("png", "svg"), dpi=200, log_to_mlflow=False):
    """Compare EvaluationReports in supplied chronological order, without rescoring.

    Reports may come from separate monthly jobs. Models/thresholds may vary;
    the CSV records both and the chart discloses changing thresholds. baseline
    is an optional validation EvaluationReport shown as dashed horizontal lines.
    MLflow stores month_* metrics with step 1..N plus charts/monthly_summary.
    """
    reports = list(reports)
    if not reports:
        raise ValueError("Provide at least one monthly report in chronological order.")
    if len({r.dataset_name for r in reports}) != len(reports):
        raise ValueError("Monthly dataset names must be unique.")
    _options(formats, dpi)
    tracking = _tracking(log_to_mlflow)
    theme = theme or ChartTheme()
    directory = Path(output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame([{"dataset_name": r.dataset_name, "model_name": r.model_name,
                           "fpr_limit": r.fpr_limit, **r.metrics} for r in reports])
    table_path = directory / "monthly_metrics.csv"
    table.to_csv(table_path, index=False)
    fixed = table.decision_threshold.nunique() == 1
    subtitle = (f"Fixed threshold  {_threshold_label(table.decision_threshold.iloc[0])}" if fixed else
                "Threshold varies by month; see the exported table")
    model_label = reports[0].model_name if table.model_name.nunique() == 1 else "Multiple model versions · see table"
    with _style(theme):
        fig = _canvas("Performance over time", f"{len(reports)} monthly batches", model_label,
                      subtitle + ("  |  Dashed lines: validation baseline" if baseline else ""), theme,
                      "Months follow the supplied order. Gaps indicate undefined metrics.")
        panels = [
            ("Ranking quality", [("roc_auc", "ROC AUC", theme.accent), ("average_precision", "Average precision", theme.secondary)]),
            ("Fraud detection", [("recall", "Recall", theme.accent), ("precision", "Precision", theme.secondary)]),
            ("Operational rates", [("false_positive_rate", "False-positive rate", theme.warning), ("flagged_rate", "Flagged rate", theme.accent)]),
            ("Batch composition", [("fraud_rate", "Fraud prevalence", theme.warning)]),
        ]
        x = np.arange(1, len(reports) + 1)
        for i, (title, series) in enumerate(panels):
            row, col = divmod(i, 2)
            ax = _axes(fig, [.09 + col * .47, .49 - row * .33, .37, .21], theme)
            ax.set_title(title, loc="left", size=12, y=1.27, pad=0)
            for key, label, color in series:
                values = [r.metrics[key] if r.metrics[key] is not None else np.nan for r in reports]
                ax.plot(x, values, "o-", color=color, ms=4, lw=2, label=label)
                if baseline and baseline.metrics.get(key) is not None:
                    ax.axhline(baseline.metrics[key], color=color, ls="--", lw=1, alpha=.7)
            if i == 2:
                limits = [r.fpr_limit if r.fpr_limit is not None else np.nan for r in reports]
                if np.isfinite(limits).any():
                    ax.plot(x, limits, color=theme.warning, ls=":", lw=1.5, label="FPR limit")
            ax.set_ylim(bottom=0, top=1.03 if i < 2 else None)
            ax.set_xlim(.7, len(reports) + .3)
            ax.set_xticks(x, [textwrap.shorten(r.dataset_name, 16) for r in reports], rotation=35, ha="right")
            ax.tick_params(labelsize=8, pad=3, labelbottom=(row == 1))
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.03),
                      ncol=3, borderaxespad=0, handlelength=1.6, columnspacing=1.0)
        chart_paths = _save(fig, directory, "monthly_comparison", formats, dpi)
    if tracking is not None:
        for path in [table_path, *chart_paths]:
            tracking.log_artifact(str(path), artifact_path="charts/monthly_summary")
        for step, report in enumerate(reports, 1):
            tracking.log_metrics({f"month_{key}": value for key, value in report.metrics.items()
                                  if value is not None}, step=step)
    return table


def create_monthly_reports(monthly_sets, *, threshold, output_dir, model_name="CatBoost",
                           fpr_limit=None, baseline=None, theme=None,
                           formats=("png", "svg"), dpi=200, log_to_mlflow=False):
    """Report any number of batches (e.g. 12), using one saved threshold.

    monthly_sets maps chronological names to (labels, fraud_scores). Prefer
    original served scores once delayed labels arrive. Does not infer dates,
    invent monthly splits, retrain models, or select a threshold on monthly data.
    Returns (list_of_reports, summary_dataframe).
    """
    items = list(monthly_sets.items())
    if not items:
        raise ValueError("Provide at least one monthly dataset.")
    slugs = [_slug(name) for name, _ in items]
    if len(set(slugs)) != len(slugs):
        raise ValueError("Monthly names must remain unique after filename normalization.")
    # Validate all batches before producing a partially written report set.
    for _, (labels, scores) in items:
        _validate(labels, scores, threshold)
    reports = []
    for (name, (labels, scores)), slug in zip(items, slugs):
        reports.append(create_evaluation_report(
            labels, scores, threshold=threshold, output_dir=Path(output_dir) / slug,
            dataset_name=name, model_name=model_name, fpr_limit=fpr_limit,
            theme=theme, formats=formats, dpi=dpi, log_to_mlflow=log_to_mlflow,
        ))
    summary = create_monthly_summary(reports, output_dir=Path(output_dir) / "summary",
                                    baseline=baseline, theme=theme, formats=formats,
                                    dpi=dpi, log_to_mlflow=log_to_mlflow)
    return reports, summary
