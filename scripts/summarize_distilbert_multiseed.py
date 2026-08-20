"""Summarize DistilBERT variability across training and perturbation seeds."""

from pathlib import Path

import numpy as np
import pandas as pd

TRAINING_SEEDS = [
    42,
    123,
    2026,
]

PERTURBATIONS = [
    "typo",
    "char_delete",
    "word_delete",
]

BASE_DIR = (
    Path("results")
    / "distilbert"
    / "multiseed"
)

OUTPUT_DIR = (
    Path("results")
    / "comparison"
)


def runs_path(
    seed: int,
) -> Path:
    """Return evaluation-runs path for one training seed."""

    return (
        BASE_DIR
        / f"seed_{seed}"
        / "evaluation_runs.csv"
    )


def load_seed_runs(
    seed: int,
) -> pd.DataFrame:
    """Load results for one DistilBERT training seed."""

    path = runs_path(
        seed
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing DistilBERT results for seed {seed}: "
            f"{path}"
        )

    return pd.read_csv(
        path
    )


def sample_std(
    values,
) -> float:
    """Return sample standard deviation."""

    array = np.asarray(
        values,
        dtype=float,
    )

    if len(array) < 2:
        return float(
            "nan"
        )

    return float(
        np.std(
            array,
            ddof=1,
        )
    )


def build_seed_level_summary() -> pd.DataFrame:
    """Build one summary row per training seed and condition."""

    rows = []

    for seed in TRAINING_SEEDS:

        data = load_seed_runs(
            seed
        )

        clean_rows = data[
            data[
                "condition"
            ]
            == "clean"
        ]

        if len(
            clean_rows
        ) != 1:
            raise ValueError(
                f"Expected one clean row for training seed {seed}, "
                f"found {len(clean_rows)}."
            )

        clean = clean_rows.iloc[
            0
        ]

        clean_accuracy = float(
            clean[
                "accuracy"
            ]
        )

        clean_macro_f1 = float(
            clean[
                "macro_f1"
            ]
        )

        rows.append(
            {
                "training_seed":
                    seed,

                "perturbation":
                    "clean",

                "clean_accuracy":
                    clean_accuracy,

                "clean_macro_f1":
                    clean_macro_f1,

                "mean_accuracy":
                    clean_accuracy,

                "within_seed_accuracy_std":
                    0.0,

                "mean_macro_f1":
                    clean_macro_f1,

                "within_seed_macro_f1_std":
                    0.0,

                "mean_accuracy_drop":
                    0.0,

                "mean_accuracy_retention":
                    1.0,

                "within_seed_retention_std":
                    0.0,

                "n_perturbation_seeds":
                    0,
            }
        )

        perturbed = data[
            data[
                "condition"
            ]
            == "perturbed"
        ]

        for perturbation in PERTURBATIONS:

            subset = perturbed[
                perturbed[
                    "perturbation"
                ]
                == perturbation
            ]

            if len(
                subset
            ) != 10:
                raise ValueError(
                    "Expected 10 perturbation runs for "
                    f"seed {seed}/{perturbation}, "
                    f"found {len(subset)}."
                )

            rows.append(
                {
                    "training_seed":
                        seed,

                    "perturbation":
                        perturbation,

                    "clean_accuracy":
                        clean_accuracy,

                    "clean_macro_f1":
                        clean_macro_f1,

                    "mean_accuracy":
                        subset[
                            "accuracy"
                        ].mean(),

                    "within_seed_accuracy_std":
                        subset[
                            "accuracy"
                        ].std(
                            ddof=1
                        ),

                    "mean_macro_f1":
                        subset[
                            "macro_f1"
                        ].mean(),

                    "within_seed_macro_f1_std":
                        subset[
                            "macro_f1"
                        ].std(
                            ddof=1
                        ),

                    "mean_accuracy_drop":
                        subset[
                            "accuracy_drop"
                        ].mean(),

                    "mean_accuracy_retention":
                        subset[
                            "accuracy_retention"
                        ].mean(),

                    "within_seed_retention_std":
                        subset[
                            "accuracy_retention"
                        ].std(
                            ddof=1
                        ),

                    "n_perturbation_seeds":
                        len(
                            subset
                        ),
                }
            )

    return pd.DataFrame(
        rows
    )


def build_training_seed_summary(
    seed_level: pd.DataFrame,
) -> pd.DataFrame:
    """Separate training-seed variation from perturbation-seed variation."""

    rows = []

    for perturbation in [
        "clean",
        *PERTURBATIONS,
    ]:

        subset = seed_level[
            seed_level[
                "perturbation"
            ]
            == perturbation
        ]

        if len(
            subset
        ) != len(
            TRAINING_SEEDS
        ):
            raise ValueError(
                f"Expected {len(TRAINING_SEEDS)} training seeds "
                f"for {perturbation}, "
                f"found {len(subset)}."
            )

        rows.append(
            {
                "perturbation":
                    perturbation,

                "n_training_seeds":
                    len(
                        subset
                    ),

                "mean_clean_accuracy":
                    subset[
                        "clean_accuracy"
                    ].mean(),

                "training_seed_sd_clean_accuracy":
                    sample_std(
                        subset[
                            "clean_accuracy"
                        ]
                    ),

                "mean_clean_macro_f1":
                    subset[
                        "clean_macro_f1"
                    ].mean(),

                "training_seed_sd_clean_macro_f1":
                    sample_std(
                        subset[
                            "clean_macro_f1"
                        ]
                    ),

                "mean_accuracy":
                    subset[
                        "mean_accuracy"
                    ].mean(),

                "training_seed_sd_accuracy":
                    sample_std(
                        subset[
                            "mean_accuracy"
                        ]
                    ),

                "mean_within_seed_accuracy_std":
                    subset[
                        "within_seed_accuracy_std"
                    ].mean(),

                "mean_macro_f1":
                    subset[
                        "mean_macro_f1"
                    ].mean(),

                "training_seed_sd_macro_f1":
                    sample_std(
                        subset[
                            "mean_macro_f1"
                        ]
                    ),

                "mean_accuracy_drop":
                    subset[
                        "mean_accuracy_drop"
                    ].mean(),

                "training_seed_sd_accuracy_drop":
                    sample_std(
                        subset[
                            "mean_accuracy_drop"
                        ]
                    ),

                "mean_accuracy_retention":
                    subset[
                        "mean_accuracy_retention"
                    ].mean(),

                "training_seed_sd_retention":
                    sample_std(
                        subset[
                            "mean_accuracy_retention"
                        ]
                    ),

                "mean_within_seed_retention_std":
                    subset[
                        "within_seed_retention_std"
                    ].mean(),
            }
        )

    return pd.DataFrame(
        rows
    )


def load_classical_summary(
    model: str,
) -> pd.DataFrame:
    """Load BANKING77 20% robustness summary for a classical model."""

    path = (
        Path("results")
        / "robustness"
        / "banking77"
        / f"{model}_20pct_summary.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing classical summary: {path}"
        )

    return pd.read_csv(
        path
    )


def build_model_family_comparison(
    distilbert_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Compare word TF-IDF, character TF-IDF, and DistilBERT."""

    rows = []

    classical_models = {
        "tfidf_svm":
            "Word TF-IDF + Linear SVM",

        "char_tfidf_svm":
            "Character TF-IDF + Linear SVM",
    }

    for model, label in classical_models.items():

        data = load_classical_summary(
            model
        )

        clean_accuracy = float(
            data[
                "clean_accuracy"
            ].iloc[
                0
            ]
        )

        rows.append(
            {
                "model":
                    label,

                "perturbation":
                    "clean",

                "accuracy":
                    clean_accuracy,

                "training_seed_sd_accuracy":
                    np.nan,

                "accuracy_retention":
                    1.0,

                "training_seed_sd_retention":
                    np.nan,

                "accuracy_drop":
                    0.0,
            }
        )

        for perturbation in PERTURBATIONS:

            row = data[
                data[
                    "perturbation"
                ]
                == perturbation
            ]

            if len(
                row
            ) != 1:
                raise ValueError(
                    "Expected one row for "
                    f"{model}/{perturbation}, "
                    f"found {len(row)}."
                )

            row = row.iloc[
                0
            ]

            rows.append(
                {
                    "model":
                        label,

                    "perturbation":
                        perturbation,

                    "accuracy":
                        float(
                            row[
                                "mean_accuracy"
                            ]
                        ),

                    "training_seed_sd_accuracy":
                        0.0,

                    "accuracy_retention":
                        float(
                            row[
                                "accuracy_retention"
                            ]
                        ),

                    "training_seed_sd_retention":
                        0.0,

                    "accuracy_drop":
                        float(
                            row[
                                "mean_drop"
                            ]
                        ),
                }
            )

    for _, row in distilbert_summary.iterrows():

        rows.append(
            {
                "model":
                    "DistilBERT",

                "perturbation":
                    row[
                        "perturbation"
                    ],

                "accuracy":
                    float(
                        row[
                            "mean_accuracy"
                        ]
                    ),

                "training_seed_sd_accuracy":
                    float(
                        row[
                            "training_seed_sd_accuracy"
                        ]
                    ),

                "accuracy_retention":
                    float(
                        row[
                            "mean_accuracy_retention"
                        ]
                    ),

                "training_seed_sd_retention":
                    float(
                        row[
                            "training_seed_sd_retention"
                        ]
                    ),

                "accuracy_drop":
                    float(
                        row[
                            "mean_accuracy_drop"
                        ]
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def print_seed_table(
    seed_level: pd.DataFrame,
) -> None:
    """Print individual training-seed results."""

    display = seed_level.copy()

    for column in [
        "clean_accuracy",
        "clean_macro_f1",
        "mean_accuracy",
        "within_seed_accuracy_std",
        "mean_macro_f1",
        "within_seed_macro_f1_std",
        "mean_accuracy_drop",
        "mean_accuracy_retention",
        "within_seed_retention_std",
    ]:

        display[
            column
        ] *= 100

    print("\n")
    print(
        "=" * 110
    )
    print(
        "DISTILBERT — INDIVIDUAL TRAINING SEEDS"
    )
    print(
        "=" * 110
    )

    print(
        display[
            [
                "training_seed",
                "perturbation",
                "clean_accuracy",
                "mean_accuracy",
                "within_seed_accuracy_std",
                "mean_accuracy_drop",
                "mean_accuracy_retention",
            ]
        ].to_string(
            index=False,
            formatters={
                "clean_accuracy":
                    "{:.2f}%".format,

                "mean_accuracy":
                    "{:.2f}%".format,

                "within_seed_accuracy_std":
                    "{:.2f} pp".format,

                "mean_accuracy_drop":
                    "{:.2f} pp".format,

                "mean_accuracy_retention":
                    "{:.2f}%".format,
            },
        )
    )


def print_training_seed_table(
    summary: pd.DataFrame,
) -> None:
    """Print aggregate variability across training seeds."""

    display = summary.copy()

    percentage_columns = [
        "mean_clean_accuracy",
        "training_seed_sd_clean_accuracy",
        "mean_clean_macro_f1",
        "training_seed_sd_clean_macro_f1",
        "mean_accuracy",
        "training_seed_sd_accuracy",
        "mean_within_seed_accuracy_std",
        "mean_macro_f1",
        "training_seed_sd_macro_f1",
        "mean_accuracy_drop",
        "training_seed_sd_accuracy_drop",
        "mean_accuracy_retention",
        "training_seed_sd_retention",
        "mean_within_seed_retention_std",
    ]

    for column in percentage_columns:

        display[
            column
        ] *= 100

    print("\n")
    print(
        "=" * 120
    )
    print(
        "DISTILBERT — TRAINING-SEED VARIABILITY"
    )
    print(
        "=" * 120
    )

    print(
        display[
            [
                "perturbation",
                "mean_accuracy",
                "training_seed_sd_accuracy",
                "mean_within_seed_accuracy_std",
                "mean_accuracy_drop",
                "training_seed_sd_accuracy_drop",
                "mean_accuracy_retention",
                "training_seed_sd_retention",
            ]
        ].to_string(
            index=False,
            formatters={
                "mean_accuracy":
                    "{:.2f}%".format,

                "training_seed_sd_accuracy":
                    "{:.2f} pp".format,

                "mean_within_seed_accuracy_std":
                    "{:.2f} pp".format,

                "mean_accuracy_drop":
                    "{:.2f} pp".format,

                "training_seed_sd_accuracy_drop":
                    "{:.2f} pp".format,

                "mean_accuracy_retention":
                    "{:.2f}%".format,

                "training_seed_sd_retention":
                    "{:.2f} pp".format,
            },
        )
    )


def print_model_comparison(
    comparison: pd.DataFrame,
) -> None:
    """Print BANKING77 model-family comparison."""

    display = comparison.copy()

    for column in [
        "accuracy",
        "training_seed_sd_accuracy",
        "accuracy_retention",
        "training_seed_sd_retention",
        "accuracy_drop",
    ]:

        display[
            column
        ] *= 100

    print("\n")
    print(
        "=" * 118
    )
    print(
        "BANKING77 — MODEL FAMILY COMPARISON "
        "AT 20% CORRUPTION"
    )
    print(
        "=" * 118
    )

    print(
        display.to_string(
            index=False,
            formatters={
                "accuracy":
                    "{:.2f}%".format,

                "training_seed_sd_accuracy":
                    "{:.2f} pp".format,

                "accuracy_retention":
                    "{:.2f}%".format,

                "training_seed_sd_retention":
                    "{:.2f} pp".format,

                "accuracy_drop":
                    "{:.2f} pp".format,
            },
        )
    )


def main() -> None:
    """Run DistilBERT multi-training-seed analysis."""

    print(
        "=" * 92
    )
    print(
        "ReliabilityLab"
    )
    print(
        "DistilBERT Multi-Training-Seed Analysis"
    )
    print(
        "=" * 92
    )

    seed_level = (
        build_seed_level_summary()
    )

    training_seed_summary = (
        build_training_seed_summary(
            seed_level
        )
    )

    comparison = (
        build_model_family_comparison(
            training_seed_summary
        )
    )

    print_seed_table(
        seed_level
    )

    print_training_seed_table(
        training_seed_summary
    )

    print_model_comparison(
        comparison
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    seed_level_path = (
        OUTPUT_DIR
        / "distilbert_training_seed_results.csv"
    )

    aggregate_path = (
        OUTPUT_DIR
        / "distilbert_multiseed_summary.csv"
    )

    comparison_path = (
        OUTPUT_DIR
        / "banking77_model_family_20pct.csv"
    )

    seed_level.to_csv(
        seed_level_path,
        index=False,
    )

    training_seed_summary.to_csv(
        aggregate_path,
        index=False,
    )

    comparison.to_csv(
        comparison_path,
        index=False,
    )

    print("\n")
    print(
        "Saved:"
    )

    print(
        seed_level_path
    )

    print(
        aggregate_path
    )

    print(
        comparison_path
    )


if __name__ == "__main__":
    main()