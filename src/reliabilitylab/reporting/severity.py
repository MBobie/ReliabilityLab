"""Plots for perturbation-severity experiments."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

LABELS = {
    "typo": "Typo",
    "char_delete": "Character deletion",
    "word_delete": "Word deletion",
}


def plot_severity_accuracy(
    summary_df: pd.DataFrame,
    clean_accuracy: float,
    save_path=None,
):
    """Plot accuracy as perturbation severity increases."""

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    for perturbation in [
        "char_delete",
        "typo",
        "word_delete",
    ]:

        subset = (
            summary_df[
                summary_df["perturbation"]
                == perturbation
            ]
            .sort_values("severity_percent")
        )

        x = subset[
            "mean_realized_severity"
        ].to_numpy() * 100

        y = subset[
            "mean_accuracy"
        ].to_numpy() * 100

        error = subset[
            "accuracy_std"
        ].to_numpy() * 100

        # Add clean starting point
        x = [0, *x]
        y = [
            clean_accuracy * 100,
            *y,
        ]
        error = [
            0,
            *error,
        ]

        ax.errorbar(
            x,
            y,
            yerr=error,
            marker="o",
            capsize=4,
            linewidth=2,
            label=LABELS[perturbation],
        )

    ax.set_xlabel(
        "Realized Corruption Severity (%)"
    )

    ax.set_ylabel(
        "Accuracy (%)"
    )

    ax.set_title(
        "ReliabilityLab — Accuracy Degradation with Perturbation Severity"
    )

    ax.set_xticks(
        [0, 5, 10, 20, 30, 40]
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend()

    fig.tight_layout()

    if save_path is not None:

        save_path = Path(save_path)

        save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

        print(
            f"Figure saved to: {save_path}"
        )

    return fig, ax


def plot_severity_drop(
    summary_df: pd.DataFrame,
    save_path=None,
):
    """Plot accuracy loss as perturbation severity increases."""

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    for perturbation in [
        "char_delete",
        "typo",
        "word_delete",
    ]:

        subset = (
            summary_df[
                summary_df["perturbation"]
                == perturbation
            ]
            .sort_values("severity_percent")
        )

        x = subset[
            "mean_realized_severity"
        ].to_numpy() * 100

        y = subset[
            "mean_drop"
        ].to_numpy() * 100

        error = subset[
            "drop_std"
        ].to_numpy() * 100

        # Add clean reference
        x = [0, *x]
        y = [0, *y]
        error = [0, *error]

        ax.errorbar(
            x,
            y,
            yerr=error,
            marker="o",
            capsize=4,
            linewidth=2,
            label=LABELS[perturbation],
        )

    ax.set_xlabel(
        "Realized Corruption Severity (%)"
    )

    ax.set_ylabel(
        "Accuracy Drop (percentage points)"
    )

    ax.set_title(
        "ReliabilityLab — Robustness Failure Curves"
    )

    ax.set_xticks(
        [0, 5, 10, 20, 30, 40]
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend()

    fig.tight_layout()

    if save_path is not None:

        save_path = Path(save_path)

        save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

        print(
            f"Figure saved to: {save_path}"
        )

    return fig, ax