"""Visualisations for ReliabilityLab robustness experiments."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_robustness_accuracy(
    repeated_summary: pd.DataFrame,
    clean_accuracy: float,
    case_accuracy: float,
    punctuation_accuracy: float,
    save_path=None,
):
    """Plot clean and perturbed accuracy.

    Repeated perturbations are shown with standard-deviation
    error bars. Deterministic control conditions are shown
    without uncertainty bars.
    """

    conditions = [
        "Clean",
        "Case",
        "Punctuation",
        "Word deletion",
        "Typo",
        "Character deletion",
    ]

    word_row = repeated_summary[
        repeated_summary["perturbation"] == "word_delete"
    ].iloc[0]

    typo_row = repeated_summary[
        repeated_summary["perturbation"] == "typo"
    ].iloc[0]

    char_row = repeated_summary[
        repeated_summary["perturbation"] == "char_delete"
    ].iloc[0]

    accuracies = np.array(
        [
            clean_accuracy,
            case_accuracy,
            punctuation_accuracy,
            word_row["mean_accuracy"],
            typo_row["mean_accuracy"],
            char_row["mean_accuracy"],
        ]
    ) * 100

    errors = np.array(
        [
            0.0,
            0.0,
            0.0,
            word_row["accuracy_std"],
            typo_row["accuracy_std"],
            char_row["accuracy_std"],
        ]
    ) * 100

    x = np.arange(len(conditions))

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    ax.errorbar(
        x,
        accuracies,
        yerr=errors,
        fmt="o",
        capsize=6,
        markersize=8,
        linewidth=2,
    )

    ax.plot(
        x,
        accuracies,
        linewidth=1.5,
        alpha=0.6,
    )

    for x_value, accuracy in zip(
        x,
        accuracies,
    ):
        ax.annotate(
            f"{accuracy:.2f}%",
            (x_value, accuracy),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        conditions,
        rotation=20,
        ha="right",
    )

    ax.set_ylabel("Accuracy (%)")

    ax.set_title(
        "ReliabilityLab — Accuracy Under Text Perturbations"
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

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


def plot_robustness_drop(
    repeated_summary: pd.DataFrame,
    save_path=None,
):
    """Plot mean accuracy degradation with 95% confidence intervals."""

    plot_data = repeated_summary.copy()

    order = [
        "word_delete",
        "char_delete",
        "typo",
    ]

    plot_data = (
        plot_data
        .set_index("perturbation")
        .loc[order]
        .reset_index()
    )

    labels = [
        "Word deletion",
        "Character deletion",
        "Typo",
    ]

    means = (
        plot_data["mean_drop"].to_numpy()
        * 100
    )

    lower = (
        plot_data["drop_ci_lower"].to_numpy()
        * 100
    )

    upper = (
        plot_data["drop_ci_upper"].to_numpy()
        * 100
    )

    lower_errors = means - lower
    upper_errors = upper - means

    asymmetric_error = np.vstack(
        [
            lower_errors,
            upper_errors,
        ]
    )

    x = np.arange(len(labels))

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.errorbar(
        x,
        means,
        yerr=asymmetric_error,
        fmt="o",
        markersize=9,
        capsize=7,
        linewidth=2,
    )

    for x_value, value in zip(
        x,
        means,
    ):
        ax.annotate(
            f"{value:.2f} pp",
            (x_value, value),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    ax.set_ylabel(
        "Accuracy Drop (percentage points)"
    )

    ax.set_title(
        "ReliabilityLab — Robustness Degradation"
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

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