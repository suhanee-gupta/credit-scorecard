"""Static evaluation charts."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
REFERENCE = "#9b9a95"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT_SECONDARY,
        "axes.titlecolor": TEXT_PRIMARY,
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "legend.frameon": False,
        "legend.labelcolor": TEXT_PRIMARY,
        "font.size": 10,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    }
)


def roc_plot(y_true, scores: dict[str, np.ndarray], path: Path, title: str) -> None:
    """ROC curves for several score vectors where higher means riskier."""
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot([0, 1], [0, 1], color=REFERENCE, lw=1, ls="--", label="Random (AUC 0.50)")
    for (name, score), color in zip(scores.items(), SERIES):
        fpr, tpr, _ = roc_curve(y_true, score)
        auc = roc_auc_score(y_true, score)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{name}  AUC {auc:.3f} · Gini {2 * auc - 1:.3f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.savefig(path)
    plt.close(fig)


def feature_importance_plot(importance: pd.DataFrame, path: Path) -> None:
    imp = importance.sort_values("share")
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    bars = ax.barh(imp["feature"], imp["share"], color=SERIES[0], height=0.55)
    for bar, share in zip(bars, imp["share"]):
        ax.text(bar.get_width() + 0.008, bar.get_y() + bar.get_height() / 2,
                f"{share:.0%}", va="center", color=TEXT_SECONDARY, fontsize=9)
    ax.set_xlim(0, imp["share"].max() * 1.15)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Share of total gain")
    ax.set_title("XGBoost feature importance (gain)")
    fig.savefig(path)
    plt.close(fig)


def z_score_plot(df: pd.DataFrame, path: Path) -> None:
    """Z-score distribution by default status, split by Altman variant."""
    variants = ["Z (1968)", "Z'' (1995)"]
    zones = {"Z (1968)": (1.81, 2.99), "Z'' (1995)": (1.10, 2.60)}
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharey=False)
    rng = np.random.default_rng(0)
    for ax, variant in zip(axes, variants):
        sub = df[(df["z_model"] == variant) & df["z_score"].notna()]
        for i, (label, color) in enumerate([(0, SERIES[0]), (1, SERIES[1])]):
            z = sub.loc[sub["default"] == label, "z_score"].clip(-10, 15)
            x = i + rng.uniform(-0.18, 0.18, len(z))
            ax.scatter(x, z, s=14, color=color, alpha=0.7, edgecolors=SURFACE, linewidths=0.5)
            if len(z):
                ax.hlines(z.median(), i - 0.28, i + 0.28, color=TEXT_PRIMARY, lw=1.5)
            else:
                ax.text(i, 0.7, "No defaulted firm-years", transform=ax.get_xaxis_transform(),
                        ha="center", va="center", color=TEXT_SECONDARY, fontsize=9)
        lo, hi = zones[variant]
        for cut in (lo, hi):
            ax.axhline(cut, color=REFERENCE, lw=1, ls="--")
        ax.text(1.42, hi, "Safe", va="bottom", ha="right", color=TEXT_SECONDARY, fontsize=8)
        ax.text(1.42, lo, "Distress", va="top", ha="right", color=TEXT_SECONDARY, fontsize=8)
        ax.set_xticks([0, 1], ["Performing", "Defaulted"])
        ax.set_xlim(-0.5, 1.5)
        ax.grid(axis="x", visible=False)
        ax.set_title(variant)
    axes[0].set_ylabel("Z-score (clipped to [-10, 15])")
    fig.suptitle("Altman Z-score by default status", x=0.07, ha="left",
                 fontweight="bold", color=TEXT_PRIMARY)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def rating_distribution_plot(counts: pd.DataFrame, path: Path) -> None:
    """Grouped bars of company counts per rating grade; ``counts`` is indexed by grade."""
    grades = counts.index.tolist()
    x = np.arange(len(grades))
    width = 0.38
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for i, (col, color) in enumerate(zip(counts.columns, SERIES)):
        ax.bar(x + (i - 0.5) * width, counts[col], width=width - 0.03, color=color, label=col)
    ax.set_xticks(x, grades)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Companies")
    ax.set_title("Rating distribution, latest fiscal year")
    ax.legend(loc="upper right")
    fig.savefig(path)
    plt.close(fig)
