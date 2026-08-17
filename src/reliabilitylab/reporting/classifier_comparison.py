"""Visualizations for TF-IDF classifier comparisons."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATASET_LABELS = {
    "banking77": "BANKING77",
    "clinc150": "CLINC150",
    "hwu64": "HWU64",
}


MODEL_LABELS = {
    "tfidf_logreg": "Logistic Regression",
    "tfidf_svm": "Linear SVM",
}


def plot_absolute_classifier_performance(
    summary: pd.DataFrame,
    save_path=None,
):
    """Plot clean and mean perturbed accuracy by model and dataset."""

    datasets = [
        "banking77",
        "clinc150",
        "hwu64",
    ]

    models = [
        "tfidf_logreg",
        "tfidf_svm",
    ]

    x = np.arange(
        len(datasets)
    )

    width = 0.18

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    offsets = {
        ("tfidf_logreg", "clean"):
            -1.5 * width,

        ("tfidf_logreg", "perturbed"):
            -0.5 * width,

        ("tfidf_svm", "clean"):
            0.5 * width,

        ("tfidf_svm", "perturbed"):
            1.5 * width,
    }

    for model in models:

        model_subset = (
            summary[
                summary["model"]
                == model
            ]
            .set_index(
                "dataset"
            )
            .loc[
                datasets
            ]
        )

        clean_values = (
            model_subset[
                "clean_accuracy"
            ].to_numpy()
            * 100
        )

        perturbed_values = (
            model_subset[
                "mean_perturbed_accuracy"
            ].to_numpy()
            * 100
        )

        clean_bars = ax.bar(
            x
            + offsets[
                (
                    model,
                    "clean",
                )
            ],
            clean_values,
            width,
            label=(
                f"{MODEL_LABELS[model]} "
                "— Clean"
            ),
        )

        perturbed_bars = ax.bar(
            x
            + offsets[
                (
                    model,
                    "perturbed",
                )
            ],
            perturbed_values,
            width,
            label=(
                f"{MODEL_LABELS[model]} "
                "— Perturbed"
            ),
        )

        ax.bar_label(
            clean_bars,
            fmt="%.1f",
            padding=2,
            fontsize=8,
        )

        ax.bar_label(
            perturbed_bars,
            fmt="%.1f",
            padding=2,
            fontsize=8,
        )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        [
            DATASET_LABELS[
                dataset
            ]
            for dataset in datasets
        ]
    )

    ax.set_ylabel(
        "Accuracy (%)"
    )

    ax.set_title(
        "Absolute Performance: "
        "Logistic Regression vs Linear SVM"
    )

    ax.set_ylim(
        65,
        95,
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    ax.legend(
        ncol=2,
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


def plot_classifier_retention(
    summary: pd.DataFrame,
    save_path=None,
):
    """Plot mean corruption retention by classifier and dataset."""

    datasets = [
        "banking77",
        "clinc150",
        "hwu64",
    ]

    models = [
        "tfidf_logreg",
        "tfidf_svm",
    ]

    x = np.arange(
        len(datasets)
    )

    width = 0.32

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    for index, model in enumerate(
        models
    ):

        subset = (
            summary[
                summary["model"]
                == model
            ]
            .set_index(
                "dataset"
            )
            .loc[
                datasets
            ]
        )

        values = (
            subset[
                "mean_accuracy_retention"
            ].to_numpy()
            * 100
        )

        offset = (
            index
            - 0.5
        ) * width

        bars = ax.bar(
            x + offset,
            values,
            width,
            label=MODEL_LABELS[
                model
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
            DATASET_LABELS[
                dataset
            ]
            for dataset in datasets
        ]
    )

    ax.set_ylabel(
        "Mean Accuracy Retention (%)"
    )

    ax.set_title(
        "Normalized Robustness Remains Stable "
        "Across TF-IDF Classifiers"
    )

    ax.set_ylim(
        87,
        92,
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    ax.legend()

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