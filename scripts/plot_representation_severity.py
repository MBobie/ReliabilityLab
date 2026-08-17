"""Plot TF-IDF representation robustness across corruption severity."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

INPUT_PATH = (
    Path("results")
    / "comparison"
    / "representation_severity_cross_dataset.csv"
)

OUTPUT_DIR = (
    Path("results")
    / "figures"
)


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


def load_data() -> pd.DataFrame:
    """Load cross-dataset representation severity results."""

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing input file: {INPUT_PATH}"
        )

    data = pd.read_csv(
        INPUT_PATH
    )

    data[
        "severity_percent"
    ] = (
        data[
            "requested_severity"
        ]
        * 100
    )

    return data


def plot_retention_difference(
    data: pd.DataFrame,
) -> None:
    """Plot character-minus-word TF-IDF retention advantage."""

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    for perturbation in PERTURBATION_ORDER:

        subset = (
            data[
                data["perturbation"]
                == perturbation
            ]
            .sort_values(
                "requested_severity"
            )
        )

        ax.plot(
            subset[
                "severity_percent"
            ],
            subset[
                "mean_retention_difference"
            ]
            * 100,
            marker="o",
            linewidth=2,
            label=PERTURBATION_LABELS[
                perturbation
            ],
        )

    ax.axhline(
        0,
        linewidth=1,
    )

    ax.set_xlabel(
        "Requested Corruption Severity (%)"
    )

    ax.set_ylabel(
        "Mean Retention Difference "
        "(percentage points)\n"
        "Character TF-IDF − Word TF-IDF"
    )

    ax.set_title(
        "Representation Advantage Grows "
        "with Character-Level Corruption Severity"
    )

    ax.set_xticks(
        [
            5,
            10,
            20,
            30,
            40,
        ]
    )

    ax.grid(
        alpha=0.25,
    )

    ax.legend(
        title="Perturbation"
    )

    fig.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "representation_severity_retention_difference.png"
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


def plot_retention_curves(
    data: pd.DataFrame,
    perturbation: str,
) -> None:
    """Plot absolute retention curves for one perturbation."""

    subset = (
        data[
            data["perturbation"]
            == perturbation
        ]
        .sort_values(
            "requested_severity"
        )
    )

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.plot(
        subset[
            "severity_percent"
        ],
        subset[
            "mean_word_retention"
        ]
        * 100,
        marker="o",
        linewidth=2,
        label="Word TF-IDF + Linear SVM",
    )

    ax.plot(
        subset[
            "severity_percent"
        ],
        subset[
            "mean_char_retention"
        ]
        * 100,
        marker="s",
        linewidth=2,
        label="Character TF-IDF + Linear SVM",
    )

    ax.set_xlabel(
        "Requested Corruption Severity (%)"
    )

    ax.set_ylabel(
        "Mean Accuracy Retention (%)"
    )

    ax.set_title(
        f"{PERTURBATION_LABELS[perturbation]}: "
        "Representation Severity Response"
    )

    ax.set_xticks(
        [
            5,
            10,
            20,
            30,
            40,
        ]
    )

    ax.set_ylim(
        65,
        101,
    )

    ax.grid(
        alpha=0.25,
    )

    ax.legend()

    fig.tight_layout()

    output_path = (
        OUTPUT_DIR
        / (
            "representation_severity_"
            f"{perturbation}.png"
        )
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


def main() -> None:
    """Generate representation severity figures."""

    print("=" * 80)
    print("ReliabilityLab")
    print("Representation Severity Figures")
    print("=" * 80)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = load_data()

    print(
        "\nGenerating flagship retention-difference figure..."
    )

    plot_retention_difference(
        data
    )

    for perturbation in PERTURBATION_ORDER:

        print(
            "\nGenerating "
            f"{PERTURBATION_LABELS[perturbation]} "
            "severity curve..."
        )

        plot_retention_curves(
            data,
            perturbation,
        )

    print(
        "\nAll severity figures generated successfully."
    )


if __name__ == "__main__":
    main()