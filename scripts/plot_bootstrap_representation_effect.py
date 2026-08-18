"""Plot example-level bootstrap representation effects."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INPUT_PATH = (
    Path("results")
    / "bootstrap"
    / "representation_example_bootstrap_summary.csv"
)

OUTPUT_DIR = (
    Path("results")
    / "figures"
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


def load_data() -> pd.DataFrame:
    """Load example-level bootstrap summary."""

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing bootstrap summary: {INPUT_PATH}"
        )

    return pd.read_csv(
        INPUT_PATH
    )


def build_labels(
    data: pd.DataFrame,
) -> pd.Series:
    """Build readable condition labels."""

    return data.apply(
        lambda row: (
            f"{DATASET_LABELS[row['dataset']]} — "
            f"{PERTURBATION_LABELS[row['perturbation']]}"
        ),
        axis=1,
    )


def plot_severity_forest(
    data: pd.DataFrame,
    severity: float,
) -> None:
    """Plot bootstrap retention effects for one severity."""

    subset = data[
        np.isclose(
            data["requested_severity"],
            severity,
        )
    ].copy()

    dataset_rank = {
        dataset: index
        for index, dataset in enumerate(
            DATASET_ORDER
        )
    }

    perturbation_rank = {
        perturbation: index
        for index, perturbation in enumerate(
            PERTURBATION_ORDER
        )
    }

    subset[
        "dataset_rank"
    ] = subset[
        "dataset"
    ].map(
        dataset_rank
    )

    subset[
        "perturbation_rank"
    ] = subset[
        "perturbation"
    ].map(
        perturbation_rank
    )

    subset = subset.sort_values(
        [
            "perturbation_rank",
            "dataset_rank",
        ],
        ascending=[
            True,
            True,
        ],
    ).reset_index(
        drop=True
    )

    subset[
        "label"
    ] = build_labels(
        subset
    )

    means = (
        subset[
            "retention_difference"
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

    y = np.arange(
        len(subset)
    )

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    ax.errorbar(
        means,
        y,
        xerr=np.vstack(
            [
                lower_error,
                upper_error,
            ]
        ),
        fmt="o",
        capsize=4,
        markersize=7,
    )

    ax.axvline(
        0,
        linewidth=1,
    )

    ax.set_yticks(
        y
    )

    ax.set_yticklabels(
        subset[
            "label"
        ]
    )

    ax.invert_yaxis()

    ax.set_xlabel(
        "Accuracy Retention Difference "
        "(percentage points)\n"
        "Character TF-IDF − Word TF-IDF"
    )

    ax.set_title(
        "Example-Level Bootstrap Representation Effect "
        f"at {severity * 100:.0f}% Corruption"
    )

    ax.grid(
        axis="x",
        alpha=0.25,
    )

    fig.tight_layout()

    severity_tag = (
        f"{round(severity * 100)}pct"
    )

    output_path = (
        OUTPUT_DIR
        / (
            "representation_bootstrap_"
            f"{severity_tag}.png"
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

def plot_combined_bootstrap(
    data: pd.DataFrame,
) -> None:
    """Plot 20% and 40% bootstrap effects together."""

    fig, ax = plt.subplots(
        figsize=(11, 7)
    )

    x_positions = {
        0.20: 0,
        0.40: 1,
    }

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

    default_colors = (
        plt.rcParams[
            "axes.prop_cycle"
        ]
        .by_key()[
            "color"
        ]
    )

    dataset_colors = {
        dataset: default_colors[index]
        for index, dataset in enumerate(
            DATASET_ORDER
        )
    }

    perturbation_panels = {
        "typo": 0,
        "char_delete": 3,
        "word_delete": 6,
    }

    tick_positions = []
    tick_labels = []

    for perturbation in PERTURBATION_ORDER:

        base = perturbation_panels[
            perturbation
        ]

        for severity in [
            0.20,
            0.40,
        ]:

            x_base = (
                base
                + x_positions[
                    severity
                ]
            )

            tick_positions.append(
                x_base
            )

            tick_labels.append(
                
                    f"{PERTURBATION_LABELS[perturbation]}\n"
                    f"{severity * 100:.0f}%"
                
            )

            for dataset in DATASET_ORDER:

                row = data[
                    (
                        data[
                            "dataset"
                        ]
                        == dataset
                    )
                    & (
                        data[
                            "perturbation"
                        ]
                        == perturbation
                    )
                    & np.isclose(
                        data[
                            "requested_severity"
                        ],
                        severity,
                    )
                ]

                if len(row) != 1:
                    raise ValueError(
                        "Expected exactly one bootstrap row for "
                        f"{dataset}/{perturbation}/{severity}"
                    )

                row = row.iloc[
                    0
                ]

                mean = (
                    row[
                        "retention_difference"
                    ]
                    * 100
                )

                lower = (
                    row[
                        "retention_difference_ci_lower"
                    ]
                    * 100
                )

                upper = (
                    row[
                        "retention_difference_ci_upper"
                    ]
                    * 100
                )

                ax.errorbar(
                    x_base
                    + offsets[
                        dataset
                    ],
                    mean,
                    yerr=[
                        [
                            mean
                            - lower
                        ],
                        [
                            upper
                            - mean
                        ],
                    ],
                    fmt=markers[
                        dataset
                    ],
                    color=dataset_colors[
                        dataset
                    ],
                    capsize=4,
                    markersize=7,
                    label=(
                        DATASET_LABELS[
                            dataset
                        ]
                        if (
                            perturbation
                            == PERTURBATION_ORDER[0]
                            and severity
                            == 0.20
                        )
                        else None
                    ),
                )

    ax.axhline(
        0,
        linewidth=1,
    )

    ax.set_xticks(
        tick_positions
    )

    ax.set_xticklabels(
        tick_labels
    )

    ax.set_ylabel(
        "Accuracy Retention Difference "
        "(percentage points)\n"
        "Character TF-IDF − Word TF-IDF"
    )

    ax.set_title(
        "Paired Example-Level Bootstrap Estimates "
        "of Representation Effect"
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
        OUTPUT_DIR
        / "representation_bootstrap_combined.png"
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
    """Generate example-level bootstrap figures."""

    print("=" * 80)
    print("ReliabilityLab")
    print("Example-Level Bootstrap Figures")
    print("=" * 80)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = load_data()

    plot_severity_forest(
        data,
        0.20,
    )

    plot_severity_forest(
        data,
        0.40,
    )

    plot_combined_bootstrap(
        data
    )

    print(
        "\nBootstrap figures generated successfully."
    )


if __name__ == "__main__":
    main()