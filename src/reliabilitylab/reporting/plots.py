"""Visualisation utilities for ReliabilityLab."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_run_stability(
    values,
    baseline=None,
    metric_name="Accuracy",
    save_path=None,
):
    """Plot repeated-run performance and its mean."""

    values = np.asarray(values, dtype=float)

    runs = np.arange(
        1,
        len(values) + 1,
    )

    mean_value = np.mean(values)

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.plot(
        runs,
        values * 100,
        marker="o",
        linewidth=1.8,
        label="Repeated runs",
    )

    ax.axhline(
        mean_value * 100,
        linestyle="--",
        linewidth=1.8,
        label=f"80% mean = {mean_value * 100:.2f}%",
    )

    if baseline is not None:
        ax.axhline(
            baseline * 100,
            linestyle=":",
            linewidth=2,
            label=f"100% baseline = {baseline * 100:.2f}%",
        )

    ax.set_xlabel("Experiment Run")
    ax.set_ylabel(f"{metric_name} (%)")

    ax.set_title(
        f"ReliabilityLab — {metric_name} Across "
        "Repeated Training Subsets"
    )

    ax.set_xticks(runs)

    ax.grid(
        alpha=0.25,
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