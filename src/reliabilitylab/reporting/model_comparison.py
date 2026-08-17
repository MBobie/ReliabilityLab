"""Cross-model comparison plots for ReliabilityLab."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LABELS = {
    "typo": "Typo",
    "char_delete": "Character deletion",
    "word_delete": "Word deletion",
}


def plot_model_accuracy_comparison(
    tfidf_summary: pd.DataFrame,
    distilbert_summary: pd.DataFrame,
    tfidf_clean: float,
    distilbert_clean: float,
    save_path=None,
):
    """Compare clean and perturbed model accuracy."""

    perturbations = [
        "typo",
        "char_delete",
        "word_delete",
    ]

    conditions = [
        "Clean",
        "Typo\n20%",
        "Character deletion\n20%",
        "Word deletion\n20%",
    ]

    tfidf_values = [
        tfidf_clean * 100
    ]

    distilbert_values = [
        distilbert_clean * 100
    ]

    tfidf_errors = [0.0]
    distilbert_errors = [0.0]

    for perturbation in perturbations:

        tf_row = tfidf_summary[
            tfidf_summary["perturbation"]
            == perturbation
        ].iloc[0]

        db_row = distilbert_summary[
            distilbert_summary["perturbation"]
            == perturbation
        ].iloc[0]

        tfidf_values.append(
            tf_row["mean_accuracy"] * 100
        )

        distilbert_values.append(
            db_row["mean_accuracy"] * 100
        )

        tfidf_errors.append(
            tf_row["accuracy_std"] * 100
        )

        distilbert_errors.append(
            db_row["accuracy_std"] * 100
        )

    x = np.arange(
        len(conditions)
    )

    width = 0.36

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    bars_tf = ax.bar(
        x - width / 2,
        tfidf_values,
        width,
        yerr=tfidf_errors,
        capsize=5,
        label="TF-IDF + Logistic Regression",
    )

    bars_db = ax.bar(
        x + width / 2,
        distilbert_values,
        width,
        yerr=distilbert_errors,
        capsize=5,
        label="DistilBERT",
    )

    ax.bar_label(
        bars_tf,
        fmt="%.2f",
        padding=3,
    )

    ax.bar_label(
        bars_db,
        fmt="%.2f",
        padding=3,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        conditions
    )

    ax.set_ylabel(
        "Accuracy (%)"
    )

    ax.set_title(
        "ReliabilityLab — Clean and Perturbed Model Performance"
    )

    ax.grid(
        axis="y",
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


def plot_model_degradation_comparison(
    tfidf_summary: pd.DataFrame,
    distilbert_summary: pd.DataFrame,
    save_path=None,
):
    """Compare robustness degradation between models."""

    perturbations = [
        "typo",
        "char_delete",
        "word_delete",
    ]

    labels = [
        "Typo",
        "Character deletion",
        "Word deletion",
    ]

    tfidf_values = []
    distilbert_values = []

    tfidf_errors = []
    distilbert_errors = []

    for perturbation in perturbations:

        tf_row = tfidf_summary[
            tfidf_summary["perturbation"]
            == perturbation
        ].iloc[0]

        db_row = distilbert_summary[
            distilbert_summary["perturbation"]
            == perturbation
        ].iloc[0]

        tfidf_values.append(
            tf_row["mean_drop"] * 100
        )

        distilbert_values.append(
            db_row["mean_drop"] * 100
        )

        tfidf_errors.append(
            tf_row["drop_std"] * 100
        )

        distilbert_errors.append(
            db_row["drop_std"] * 100
        )

    x = np.arange(
        len(labels)
    )

    width = 0.36

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    bars_tf = ax.bar(
        x - width / 2,
        tfidf_values,
        width,
        yerr=tfidf_errors,
        capsize=5,
        label="TF-IDF + Logistic Regression",
    )

    bars_db = ax.bar(
        x + width / 2,
        distilbert_values,
        width,
        yerr=distilbert_errors,
        capsize=5,
        label="DistilBERT",
    )

    ax.bar_label(
        bars_tf,
        fmt="%.2f pp",
        padding=3,
    )

    ax.bar_label(
        bars_db,
        fmt="%.2f pp",
        padding=3,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        labels
    )

    ax.set_ylabel(
        "Accuracy Drop (percentage points)"
    )

    ax.set_title(
        "ReliabilityLab — Robustness Degradation by Model"
    )

    ax.grid(
        axis="y",
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