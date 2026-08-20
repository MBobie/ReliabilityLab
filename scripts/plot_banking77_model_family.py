"""Plot BANKING77 model-family reliability comparison."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INPUT_PATH = (
    Path("results")
    / "comparison"
    / "banking77_model_family_20pct.csv"
)

OUTPUT_DIR = (
    Path("results")
    / "figures"
)

MODEL_ORDER = [
    "Word TF-IDF + Linear SVM",
    "Character TF-IDF + Linear SVM",
    "DistilBERT",
]

MODEL_LABELS = {
    "Word TF-IDF + Linear SVM":
        "Word TF-IDF + SVM",

    "Character TF-IDF + Linear SVM":
        "Character TF-IDF + SVM",

    "DistilBERT":
        "DistilBERT",
}

PERTURBATION_ORDER = [
    "typo",
    "char_delete",
    "word_delete",
]

PERTURBATION_LABELS = {
    "typo":
        "Typo",

    "char_delete":
        "Character deletion",

    "word_delete":
        "Word deletion",
}


def load_data() -> pd.DataFrame:
    """Load model-family comparison results."""

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing comparison file: {INPUT_PATH}"
        )

    return pd.read_csv(
        INPUT_PATH
    )


def plot_retention(
    data: pd.DataFrame,
) -> None:
    """Plot normalized retention for all model families."""

    fig, ax = plt.subplots(
        figsize=(9.5, 6.5)
    )

    x = np.arange(
        len(
            PERTURBATION_ORDER
        )
    )

    offsets = {
        MODEL_ORDER[0]:
            -0.18,

        MODEL_ORDER[1]:
            0.0,

        MODEL_ORDER[2]:
            0.18,
    }

    markers = {
        MODEL_ORDER[0]:
            "o",

        MODEL_ORDER[1]:
            "s",

        MODEL_ORDER[2]:
            "^",
    }

    default_colors = (
        plt.rcParams[
            "axes.prop_cycle"
        ]
        .by_key()[
            "color"
        ]
    )

    model_colors = {
        model:
            default_colors[index]
        for index, model in enumerate(
            MODEL_ORDER
        )
    }

    for model in MODEL_ORDER:

        subset = data[
            (
                data[
                    "model"
                ]
                == model
            )
            & (
                data[
                    "perturbation"
                ].isin(
                    PERTURBATION_ORDER
                )
            )
        ].copy()

        subset[
            "perturbation_rank"
        ] = subset[
            "perturbation"
        ].map(
            {
                perturbation:
                    index
                for index, perturbation
                in enumerate(
                    PERTURBATION_ORDER
                )
            }
        )

        subset = subset.sort_values(
            "perturbation_rank"
        )

        retention = (
            subset[
                "accuracy_retention"
            ].to_numpy()
            * 100
        )

        positions = (
            x
            + offsets[
                model
            ]
        )

        if model == "DistilBERT":

            errors = (
                subset[
                    "training_seed_sd_retention"
                ].to_numpy()
                * 100
            )

            ax.errorbar(
                positions,
                retention,
                yerr=errors,
                fmt=markers[
                    model
                ],
                markersize=8,
                capsize=4,
                linewidth=1.5,
                color=model_colors[
                    model
                ],
                label=(
                    MODEL_LABELS[
                        model
                    ]
                    + " (mean ± training-seed SD)"
                ),
            )

        else:

            ax.plot(
                positions,
                retention,
                linestyle="none",
                marker=markers[
                    model
                ],
                markersize=8,
                color=model_colors[
                    model
                ],
                label=MODEL_LABELS[
                    model
                ],
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
        "Accuracy Retention (%)"
    )

    ax.set_title(
        "BANKING77 Robustness at 20% Corruption"
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    ax.legend()

    fig.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "banking77_model_family_retention.png"
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


def plot_accuracy(
    data: pd.DataFrame,
) -> None:
    """Plot clean and corrupted accuracy for all model families."""

    conditions = [
        "clean",
        "typo",
        "char_delete",
        "word_delete",
    ]

    condition_labels = {
        "clean":
            "Clean",

        "typo":
            "Typo",

        "char_delete":
            "Character deletion",

        "word_delete":
            "Word deletion",
    }

    x = np.arange(
        len(
            conditions
        )
    )

    offsets = {
        MODEL_ORDER[0]:
            -0.18,

        MODEL_ORDER[1]:
            0.0,

        MODEL_ORDER[2]:
            0.18,
    }

    markers = {
        MODEL_ORDER[0]:
            "o",

        MODEL_ORDER[1]:
            "s",

        MODEL_ORDER[2]:
            "^",
    }

    default_colors = (
        plt.rcParams[
            "axes.prop_cycle"
        ]
        .by_key()[
            "color"
        ]
    )

    model_colors = {
        model:
            default_colors[index]
        for index, model in enumerate(
            MODEL_ORDER
        )
    }

    fig, ax = plt.subplots(
        figsize=(10, 6.5)
    )

    for model in MODEL_ORDER:

        model_rows = data[
            data[
                "model"
            ]
            == model
        ]

        values = []

        errors = []

        for condition in conditions:

            row = model_rows[
                model_rows[
                    "perturbation"
                ]
                == condition
            ]

            if len(
                row
            ) != 1:
                raise ValueError(
                    "Expected one result for "
                    f"{model}/{condition}, "
                    f"found {len(row)}."
                )

            row = row.iloc[
                0
            ]

            values.append(
                float(
                    row[
                        "accuracy"
                    ]
                )
                * 100
            )

            if model == "DistilBERT":

                errors.append(
                    float(
                        row[
                            "training_seed_sd_accuracy"
                        ]
                    )
                    * 100
                )

            else:

                errors.append(
                    np.nan
                )

        positions = (
            x
            + offsets[
                model
            ]
        )

        values = np.asarray(
            values
        )

        if model == "DistilBERT":

            ax.errorbar(
                positions,
                values,
                yerr=np.asarray(
                    errors
                ),
                fmt=markers[
                    model
                ],
                markersize=8,
                capsize=4,
                linewidth=1.5,
                color=model_colors[
                    model
                ],
                label=(
                    MODEL_LABELS[
                        model
                    ]
                    + " (mean ± training-seed SD)"
                ),
            )

        else:

            ax.plot(
                positions,
                values,
                linestyle="none",
                marker=markers[
                    model
                ],
                markersize=8,
                color=model_colors[
                    model
                ],
                label=MODEL_LABELS[
                    model
                ],
            )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        [
            condition_labels[
                condition
            ]
            for condition
            in conditions
        ]
    )

    ax.set_ylabel(
        "Accuracy (%)"
    )

    ax.set_title(
        "BANKING77 Clean and Corrupted Accuracy"
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    ax.legend()

    fig.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "banking77_model_family_accuracy.png"
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
    """Generate BANKING77 model-family figures."""

    print(
        "=" * 80
    )
    print(
        "ReliabilityLab"
    )
    print(
        "BANKING77 Model-Family Figures"
    )
    print(
        "=" * 80
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = load_data()

    plot_retention(
        data
    )

    plot_accuracy(
        data
    )

    print(
        "\nFigures generated successfully."
    )


if __name__ == "__main__":
    main()