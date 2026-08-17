"""Compare word and character TF-IDF representations with Linear SVM."""

from math import sqrt
from pathlib import Path

import pandas as pd
from scipy.stats import t as student_t

DATASETS = [
    "banking77",
    "clinc150",
    "hwu64",
]

PERTURBATIONS = [
    "typo",
    "char_delete",
    "word_delete",
]

WORD_MODEL = "tfidf_svm"
CHAR_MODEL = "char_tfidf_svm"

SEVERITY_TAG = "20pct"


def summary_path(
    dataset: str,
    model: str,
) -> Path:
    """Return robustness-summary path."""

    return (
        Path("results")
        / "robustness"
        / dataset
        / f"{model}_{SEVERITY_TAG}_summary.csv"
    )


def runs_path(
    dataset: str,
    model: str,
) -> Path:
    """Return seed-level robustness-runs path."""

    return (
        Path("results")
        / "robustness"
        / dataset
        / f"{model}_{SEVERITY_TAG}_runs.csv"
    )


def load_summary(
    dataset: str,
    model: str,
) -> pd.DataFrame:
    """Load one robustness summary."""

    path = summary_path(
        dataset,
        model,
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing summary file: {path}"
        )

    return pd.read_csv(
        path
    )


def load_runs(
    dataset: str,
    model: str,
) -> pd.DataFrame:
    """Load one seed-level robustness file."""

    path = runs_path(
        dataset,
        model,
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing run file: {path}"
        )

    return pd.read_csv(
        path
    )


def paired_ci(
    values: pd.Series,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Calculate a t-based confidence interval for paired differences."""

    values = values.dropna()

    n = len(values)

    if n < 2:
        return float("nan"), float("nan")

    mean = values.mean()

    std = values.std(
        ddof=1
    )

    standard_error = (
        std
        / sqrt(n)
    )

    alpha = (
        1.0
        - confidence
    )

    critical = student_t.ppf(
        1.0
        - alpha / 2.0,
        df=n - 1,
    )

    margin = (
        critical
        * standard_error
    )

    return (
        mean - margin,
        mean + margin,
    )


def build_summary_comparison() -> pd.DataFrame:
    """Build perturbation-level word-vs-character representation comparison."""

    rows = []

    for dataset in DATASETS:

        word = load_summary(
            dataset,
            WORD_MODEL,
        )

        char = load_summary(
            dataset,
            CHAR_MODEL,
        )

        for perturbation in PERTURBATIONS:

            word_row = word[
                word["perturbation"]
                == perturbation
            ]

            char_row = char[
                char["perturbation"]
                == perturbation
            ]

            if len(word_row) != 1:
                raise ValueError(
                    f"Expected one word-TFIDF row for "
                    f"{dataset}/{perturbation}"
                )

            if len(char_row) != 1:
                raise ValueError(
                    f"Expected one char-TFIDF row for "
                    f"{dataset}/{perturbation}"
                )

            word_row = word_row.iloc[0]
            char_row = char_row.iloc[0]

            rows.append(
                {
                    "dataset":
                        dataset,

                    "perturbation":
                        perturbation,

                    "word_clean_accuracy":
                        word_row[
                            "clean_accuracy"
                        ],

                    "char_clean_accuracy":
                        char_row[
                            "clean_accuracy"
                        ],

                    "clean_accuracy_difference":
                        (
                            char_row[
                                "clean_accuracy"
                            ]
                            - word_row[
                                "clean_accuracy"
                            ]
                        ),

                    "word_mean_accuracy":
                        word_row[
                            "mean_accuracy"
                        ],

                    "char_mean_accuracy":
                        char_row[
                            "mean_accuracy"
                        ],

                    "perturbed_accuracy_difference":
                        (
                            char_row[
                                "mean_accuracy"
                            ]
                            - word_row[
                                "mean_accuracy"
                            ]
                        ),

                    "word_retention":
                        word_row[
                            "accuracy_retention"
                        ],

                    "char_retention":
                        char_row[
                            "accuracy_retention"
                        ],

                    "retention_difference":
                        (
                            char_row[
                                "accuracy_retention"
                            ]
                            - word_row[
                                "accuracy_retention"
                            ]
                        ),

                    "word_relative_drop":
                        word_row[
                            "relative_accuracy_drop"
                        ],

                    "char_relative_drop":
                        char_row[
                            "relative_accuracy_drop"
                        ],

                    "relative_drop_difference":
                        (
                            char_row[
                                "relative_accuracy_drop"
                            ]
                            - word_row[
                                "relative_accuracy_drop"
                            ]
                        ),
                }
            )

    return pd.DataFrame(
        rows
    )


def build_paired_runs() -> pd.DataFrame:
    """Pair word and character TF-IDF results by perturbation seed."""

    rows = []

    for dataset in DATASETS:

        word_runs = load_runs(
            dataset,
            WORD_MODEL,
        )

        char_runs = load_runs(
            dataset,
            CHAR_MODEL,
        )

        for perturbation in PERTURBATIONS:

            word_subset = (
                word_runs[
                    word_runs[
                        "perturbation"
                    ]
                    == perturbation
                ]
                .sort_values(
                    "seed"
                )
                .reset_index(
                    drop=True
                )
            )

            char_subset = (
                char_runs[
                    char_runs[
                        "perturbation"
                    ]
                    == perturbation
                ]
                .sort_values(
                    "seed"
                )
                .reset_index(
                    drop=True
                )
            )

            word_seeds = (
                word_subset[
                    "seed"
                ].tolist()
            )

            char_seeds = (
                char_subset[
                    "seed"
                ].tolist()
            )

            if word_seeds != char_seeds:
                raise ValueError(
                    f"Seed mismatch for "
                    f"{dataset}/{perturbation}"
                )

            for index in range(
                len(word_subset)
            ):

                word_row = (
                    word_subset.iloc[
                        index
                    ]
                )

                char_row = (
                    char_subset.iloc[
                        index
                    ]
                )

                rows.append(
                    {
                        "dataset":
                            dataset,

                        "perturbation":
                            perturbation,

                        "seed":
                            int(
                                word_row[
                                    "seed"
                                ]
                            ),

                        "word_accuracy":
                            word_row[
                                "accuracy"
                            ],

                        "char_accuracy":
                            char_row[
                                "accuracy"
                            ],

                        "accuracy_difference":
                            (
                                char_row[
                                    "accuracy"
                                ]
                                - word_row[
                                    "accuracy"
                                ]
                            ),

                        "word_retention":
                            word_row[
                                "accuracy_retention"
                            ],

                        "char_retention":
                            char_row[
                                "accuracy_retention"
                            ],

                        "retention_difference":
                            (
                                char_row[
                                    "accuracy_retention"
                                ]
                                - word_row[
                                    "accuracy_retention"
                                ]
                            ),

                        "word_relative_drop":
                            word_row[
                                "relative_accuracy_drop"
                            ],

                        "char_relative_drop":
                            char_row[
                                "relative_accuracy_drop"
                            ],

                        "relative_drop_difference":
                            (
                                char_row[
                                    "relative_accuracy_drop"
                                ]
                                - word_row[
                                    "relative_accuracy_drop"
                                ]
                            ),
                    }
                )

    return pd.DataFrame(
        rows
    )


def build_paired_summary(
    paired: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize paired seed-level representation differences."""

    rows = []

    for (
        dataset,
        perturbation,
    ), group in paired.groupby(
        [
            "dataset",
            "perturbation",
        ]
    ):

        accuracy_lower, accuracy_upper = (
            paired_ci(
                group[
                    "accuracy_difference"
                ]
            )
        )

        retention_lower, retention_upper = (
            paired_ci(
                group[
                    "retention_difference"
                ]
            )
        )

        rows.append(
            {
                "dataset":
                    dataset,

                "perturbation":
                    perturbation,

                "n":
                    len(group),

                "mean_accuracy_difference":
                    group[
                        "accuracy_difference"
                    ].mean(),

                "accuracy_difference_std":
                    group[
                        "accuracy_difference"
                    ].std(
                        ddof=1
                    ),

                "accuracy_difference_ci_lower":
                    accuracy_lower,

                "accuracy_difference_ci_upper":
                    accuracy_upper,

                "mean_retention_difference":
                    group[
                        "retention_difference"
                    ].mean(),

                "retention_difference_std":
                    group[
                        "retention_difference"
                    ].std(
                        ddof=1
                    ),

                "retention_difference_ci_lower":
                    retention_lower,

                "retention_difference_ci_upper":
                    retention_upper,
            }
        )

    return pd.DataFrame(
        rows
    )


def build_dataset_clean_summary(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    """Create one clean-accuracy comparison row per dataset."""

    rows = []

    for dataset in DATASETS:

        subset = summary[
            summary["dataset"]
            == dataset
        ]

        word_clean = subset[
            "word_clean_accuracy"
        ].iloc[0]

        char_clean = subset[
            "char_clean_accuracy"
        ].iloc[0]

        rows.append(
            {
                "dataset":
                    dataset,

                "word_clean_accuracy":
                    word_clean,

                "char_clean_accuracy":
                    char_clean,

                "clean_accuracy_difference":
                    char_clean
                    - word_clean,
            }
        )

    return pd.DataFrame(
        rows
    )


def print_clean_table(
    clean: pd.DataFrame,
) -> None:
    """Print clean accuracy comparison."""

    display = clean.copy()

    for column in [
        "word_clean_accuracy",
        "char_clean_accuracy",
        "clean_accuracy_difference",
    ]:
        display[column] *= 100

    print("\n")
    print("=" * 92)
    print("CLEAN PERFORMANCE — REPRESENTATION EFFECT")
    print("=" * 92)

    print(
        display.to_string(
            index=False,
            formatters={
                "word_clean_accuracy":
                    "{:.2f}%".format,

                "char_clean_accuracy":
                    "{:.2f}%".format,

                "clean_accuracy_difference":
                    "{:+.2f} pp".format,
            },
        )
    )


def print_robustness_table(
    summary: pd.DataFrame,
) -> None:
    """Print perturbation-specific representation comparison."""

    display = summary.copy()

    columns = [
        "word_mean_accuracy",
        "char_mean_accuracy",
        "perturbed_accuracy_difference",
        "word_retention",
        "char_retention",
        "retention_difference",
    ]

    for column in columns:
        display[column] *= 100

    print("\n")
    print("=" * 110)
    print("PERTURBATION-SPECIFIC REPRESENTATION EFFECT")
    print("=" * 110)

    print(
        display[
            [
                "dataset",
                "perturbation",
                "word_mean_accuracy",
                "char_mean_accuracy",
                "perturbed_accuracy_difference",
                "word_retention",
                "char_retention",
                "retention_difference",
            ]
        ].to_string(
            index=False,
            formatters={
                "word_mean_accuracy":
                    "{:.2f}%".format,

                "char_mean_accuracy":
                    "{:.2f}%".format,

                "perturbed_accuracy_difference":
                    "{:+.2f} pp".format,

                "word_retention":
                    "{:.2f}%".format,

                "char_retention":
                    "{:.2f}%".format,

                "retention_difference":
                    "{:+.2f} pp".format,
            },
        )
    )


def print_paired_table(
    paired_summary: pd.DataFrame,
) -> None:
    """Print paired corruption-seed differences."""

    display = paired_summary.copy()

    percentage_columns = [
        "mean_accuracy_difference",
        "accuracy_difference_std",
        "accuracy_difference_ci_lower",
        "accuracy_difference_ci_upper",
        "mean_retention_difference",
        "retention_difference_std",
        "retention_difference_ci_lower",
        "retention_difference_ci_upper",
    ]

    for column in percentage_columns:
        display[column] *= 100

    print("\n")
    print("=" * 120)
    print("MATCHED PERTURBATION-SEED REPRESENTATION DIFFERENCES")
    print("Character TF-IDF minus Word TF-IDF")
    print("=" * 120)

    print(
        display.to_string(
            index=False,
            formatters={
                "mean_accuracy_difference":
                    "{:+.2f} pp".format,

                "accuracy_difference_std":
                    "{:.2f} pp".format,

                "accuracy_difference_ci_lower":
                    "{:+.2f} pp".format,

                "accuracy_difference_ci_upper":
                    "{:+.2f} pp".format,

                "mean_retention_difference":
                    "{:+.2f} pp".format,

                "retention_difference_std":
                    "{:.2f} pp".format,

                "retention_difference_ci_lower":
                    "{:+.2f} pp".format,

                "retention_difference_ci_upper":
                    "{:+.2f} pp".format,
            },
        )
    )


def main() -> None:
    """Run controlled TF-IDF representation comparison."""

    print("=" * 92)
    print("ReliabilityLab")
    print("Controlled Representation Comparison")
    print("Word TF-IDF + Linear SVM vs Character TF-IDF + Linear SVM")
    print("=" * 92)

    summary = (
        build_summary_comparison()
    )

    clean = (
        build_dataset_clean_summary(
            summary
        )
    )

    paired = (
        build_paired_runs()
    )

    paired_summary = (
        build_paired_summary(
            paired
        )
    )

    print_clean_table(
        clean
    )

    print_robustness_table(
        summary
    )

    print_paired_table(
        paired_summary
    )

    print("\n")
    print("=" * 92)
    print("DESCRIPTIVE OVERALL EFFECT")
    print("=" * 92)

    print(
        "Mean clean accuracy change "
        "(character - word): "
        f"{clean['clean_accuracy_difference'].mean() * 100:+.2f} pp"
    )

    character_level = summary[
        summary[
            "perturbation"
        ].isin(
            [
                "typo",
                "char_delete",
            ]
        )
    ]

    word_level = summary[
        summary[
            "perturbation"
        ]
        == "word_delete"
    ]

    print(
        "Mean retention change for "
        "character-level perturbations: "
        f"{character_level['retention_difference'].mean() * 100:+.2f} pp"
    )

    print(
        "Mean retention change for "
        "word deletion: "
        f"{word_level['retention_difference'].mean() * 100:+.2f} pp"
    )

    output_dir = (
        Path("results")
        / "comparison"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    clean.to_csv(
        output_dir
        / "representation_clean_comparison.csv",
        index=False,
    )

    summary.to_csv(
        output_dir
        / "representation_robustness_comparison.csv",
        index=False,
    )

    paired.to_csv(
        output_dir
        / "representation_paired_runs.csv",
        index=False,
    )

    paired_summary.to_csv(
        output_dir
        / "representation_paired_summary.csv",
        index=False,
    )

    print("\nSaved:")
    print(
        output_dir
        / "representation_clean_comparison.csv"
    )
    print(
        output_dir
        / "representation_robustness_comparison.csv"
    )
    print(
        output_dir
        / "representation_paired_runs.csv"
    )
    print(
        output_dir
        / "representation_paired_summary.csv"
    )


if __name__ == "__main__":
    main()