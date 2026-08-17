"""Plot controlled word-vs-character TF-IDF representation effects."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COMPARISON_DIR = (
    Path("results")
    / "comparison"
)

FIGURE_DIR = (
    Path("results")
    / "figures"
)

CLEAN_PATH = (
    COMPARISON_DIR
    / "representation_clean_comparison.csv"
)

PAIRED_PATH = (
    COMPARISON_DIR
    / "representation_paired_summary.csv"
)


DATASET_ORDER = [
    "banking77",
    "clinc150",
    "hwu64",
]

DATASET_LABELS = {
    "banking77": "BANKING77",
    "clinc150": "CLINC150",
    "hwu64": "HWU64",
}

PERTURBATION_ORDER = [
    "typo",
    "char_delete",
    "word_delete",
]

PERTURBATION_LABELS = {
    "typo": "Typo",
    "char_delete": "Character deletion",
    "word_delete": "Word deletion",
}


def load_results():
    """Load representation comparison outputs."""

    if not CLEAN_PATH.exists():
        raise FileNotFoundError(
            f"Missing file: {CLEAN_PATH}"
        )

    if not PAIRED_PATH.exists():
        raise FileNotFoundError(
            f"Missing file: {PAIRED_PATH}"
        )

    clean = pd.read_csv(
        CLEAN_PATH
    )

    paired = pd.read_csv(
        PAIRED_PATH
    )

    return clean, paired


def plot_clean_accuracy(
    clean: pd.DataFrame,
):
    """Plot clean accuracy for word and character TF-IDF."""

    clean = (
        clean.set_index("dataset")
        .loc[DATASET_ORDER]
        .reset_index()
    )

    word_values = (
        clean["word_clean_accuracy"]
        .to_numpy()
        * 100
    )

    char_values = (
        clean["char_clean_accuracy"]
        .to_numpy()
        * 100
    )

    x = np.arange(
        len(DATASET_ORDER)
    )

    width = 0.34

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    word_bars = ax.bar(
        x - width / 2,
        word_values,
        width,
        label="Word TF-IDF + Linear SVM",
    )

    char_bars = ax.bar(
        x + width / 2,
        char_values,
        width,
        label="Character TF-IDF + Linear SVM",
    )

    ax.bar_label(
        word_bars,
        fmt="%.2f",
        padding=3,
        fontsize=9,
    )

    ax.bar_label(
        char_bars,
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
            for dataset in DATASET_ORDER
        ]
    )

    ax.set_ylabel(
        "Clean Accuracy (%)"
    )

    ax.set_title(
        "Clean Performance by TF-IDF Representation"
    )

    ax.set_ylim(
        82,
        94,
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    ax.legend()

    fig.tight_layout()

    output_path = (
        FIGURE_DIR
        / "representation_clean_accuracy.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    print(
        f"Saved: {output_path}"
    )


def plot_retention_effect(
    paired: pd.DataFrame,
):
    """Plot paired retention differences with 95% CIs."""

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    x = np.arange(
        len(PERTURBATION_ORDER)
    )

    offsets = {
        "banking77": -0.18,
        "clinc150": 0.0,
        "hwu64": 0.18,
    }

    markers = {
        "banking77": "o",
        "clinc150": "s",
        "hwu64": "^",
    }

    for dataset in DATASET_ORDER:

        subset = paired[
            paired["dataset"]
            == dataset
        ].copy()

        subset["perturbation"] = (
            pd.Categorical(
                subset["perturbation"],
                categories=PERTURBATION_ORDER,
                ordered=True,
            )
        )

        subset = subset.sort_values(
            "perturbation"
        )

        means = (
            subset[
                "mean_retention_difference"
            ].to_numpy()
            * 100
        )

        lower = (
            subset[
                "retention_difference_ci_lower"
            ].to_numpy()
            * 100
        )

        upper = (
            subset[
                "retention_difference_ci_upper"
            ].to_numpy()
            * 100
        )

        lower_error = (
            means
            - lower
        )

        upper_error = (
            upper
            - means
        )

        yerr = np.vstack(
            [
                lower_error,
                upper_error,
            ]
        )

        ax.errorbar(
            x
            + offsets[
                dataset
            ],
            means,
            yerr=yerr,
            fmt=markers[
                dataset
            ],
            capsize=4,
            markersize=7,
            label=DATASET_LABELS[
                dataset
            ],
        )

    ax.axhline(
        0,
        linewidth=1,
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
            in PERTURBATION_ORDER
        ]
    )

    ax.set_ylabel(
        "Retention Difference (percentage points)\n"
        "Character TF-IDF − Word TF-IDF"
    )

    ax.set_title(
        "Representation Effect on Robustness Retention"
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    ax.legend(
        title="Dataset"
    )

    fig.tight_layout()

    output_path = (
        FIGURE_DIR
        / "representation_retention_effect.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    print(
        f"Saved: {output_path}"
    )


def main():
    """Generate representation-effect figures."""

    print("=" * 80)
    print("ReliabilityLab")
    print("Representation Effect Figures")
    print("=" * 80)

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    clean, paired = (
        load_results()
    )

    print(
        "\nGenerating clean-accuracy figure..."
    )

    plot_clean_accuracy(
        clean
    )

    print(
        "\nGenerating paired robustness figure..."
    )

    plot_retention_effect(
        paired
    )

    print(
        "\nFigures generated successfully."
    )


if __name__ == "__main__":
    main()