"""Plots for training-data stability experiments."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_data_stability_curve(
    summary_df: pd.DataFrame,
    save_path=None,
):
    """Plot mean accuracy against training-data availability.

    Error bars represent one standard deviation across repeated
    training subsets. The 100% point is shown separately because
    it is a single full-data reference experiment.
    """

    repeated = summary_df[
        summary_df["n_runs"] > 1
    ].sort_values("train_percent")

    reference = summary_df[
        summary_df["train_percent"] == 100
    ]

    x = repeated["train_percent"].to_numpy()

    means = (
        repeated["accuracy_mean"].to_numpy()
        * 100
    )

    stds = (
        repeated["accuracy_std"].to_numpy()
        * 100
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.errorbar(
        x,
        means,
        yerr=stds,
        marker="o",
        capsize=5,
        linewidth=2,
        label="Repeated subset experiments",
    )

    if not reference.empty:

        reference_accuracy = (
            reference["accuracy_mean"].iloc[0]
            * 100
        )

        ax.scatter(
            [100],
            [reference_accuracy],
            marker="D",
            s=80,
            label="100% full-data reference",
        )

    ax.set_xlabel(
        "Training Data Used (%)"
    )

    ax.set_ylabel(
        "Accuracy (%)"
    )

    ax.set_title(
        "ReliabilityLab — Performance vs. Training Data"
    )

    ax.set_xticks(
        [20, 40, 60, 80, 100]
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


def plot_subset_instability(
    summary_df: pd.DataFrame,
    save_path=None,
):
    """Plot subset-to-subset accuracy variability.

    Only repeated experiments are included. The 100% deterministic
    reference is intentionally excluded.
    """

    repeated = summary_df[
        summary_df["n_runs"] > 1
    ].sort_values("train_percent")

    x = repeated[
        "train_percent"
    ].to_numpy()

    instability = (
        repeated["accuracy_std"].to_numpy()
        * 100
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.plot(
        x,
        instability,
        marker="o",
        linewidth=2,
    )

    for x_value, y_value in zip(
        x,
        instability,
    ):
        ax.annotate(
            f"{y_value:.2f} pp",
            (x_value, y_value),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
        )

    ax.set_xlabel(
        "Training Data Used (%)"
    )

    ax.set_ylabel(
        "Accuracy Standard Deviation (percentage points)"
    )

    ax.set_title(
        "ReliabilityLab — Training-Subset Sensitivity"
    )

    ax.set_xticks(
        [20, 40, 60, 80]
    )

    ax.grid(
        alpha=0.25
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