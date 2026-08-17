"""Cross-dataset reliability visualizations."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATASET_LABELS = {
    "banking77": "BANKING77",
    "clinc150": "CLINC150",
    "hwu64": "HWU64",
}


PERTURBATION_LABELS = {
    "typo": "Typo",
    "char_delete": "Character deletion",
    "word_delete": "Word deletion",
}


def plot_cross_dataset_retention(
    comparison: pd.DataFrame,
    save_path=None,
):
    """Plot accuracy retention across datasets and perturbations."""

    datasets = [
        "banking77",
        "clinc150",
        "hwu64",
    ]

    perturbations = [
        "typo",
        "char_delete",
        "word_delete",
    ]

    x = np.arange(
        len(perturbations)
    )

    width = 0.24

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    for index, dataset in enumerate(
        datasets
    ):

        subset = (
            comparison[
                comparison["dataset"]
                == dataset
            ]
            .set_index(
                "perturbation"
            )
            .loc[
                perturbations
            ]
        )

        values = (
            subset[
                "accuracy_retention"
            ].to_numpy()
            * 100
        )

        offset = (
            index
            - (len(datasets) - 1) / 2
        ) * width

        bars = ax.bar(
            x + offset,
            values,
            width,
            label=DATASET_LABELS[
                dataset
            ],
        )

        ax.bar_label(
            bars,
            fmt="%.2f",
            padding=3,
            fontsize=9,
        )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        [
            PERTURBATION_LABELS[
                perturbation
            ]
            for perturbation
            in perturbations
        ]
    )

    ax.set_ylabel(
        "Accuracy Retention (%)"
    )

    ax.set_title(
        "TF-IDF Reliability Across Intent-Classification Datasets"
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    ax.set_ylim(
        84,
        94,
    )

    fig.tight_layout()

    if save_path is not None:

        save_path = Path(
            save_path
        )

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
            f"Figure saved to: "
            f"{save_path}"
        )

    return fig, ax


def plot_clean_vs_retention(
    comparison: pd.DataFrame,
    save_path=None,
):
    """Compare clean accuracy with average corruption retention."""

    summary = (
        comparison.groupby(
            "dataset",
            as_index=False,
        )
        .agg(
            clean_accuracy=(
                "clean_accuracy",
                "first",
            ),

            mean_retention=(
                "accuracy_retention",
                "mean",
            ),
        )
    )

    summary[
        "clean_accuracy"
    ] *= 100

    summary[
        "mean_retention"
    ] *= 100

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    ax.scatter(
        summary[
            "clean_accuracy"
        ],
        summary[
            "mean_retention"
        ],
        s=90,
    )

    for _, row in summary.iterrows():

        label = DATASET_LABELS.get(
            row["dataset"],
            row["dataset"],
        )

        ax.annotate(
            label,
            (
                row[
                    "clean_accuracy"
                ],
                row[
                    "mean_retention"
                ],
            ),
            xytext=(6, 6),
            textcoords="offset points",
        )

    ax.set_xlabel(
        "Clean Accuracy (%)"
    )

    ax.set_ylabel(
        "Mean Accuracy Retention (%)"
    )

    ax.set_title(
        "Clean Performance vs Corruption Retention"
    )

    ax.grid(
        alpha=0.25,
    )

    fig.tight_layout()

    if save_path is not None:

        save_path = Path(
            save_path
        )

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
            f"Figure saved to: "
            f"{save_path}"
        )

    return fig, ax