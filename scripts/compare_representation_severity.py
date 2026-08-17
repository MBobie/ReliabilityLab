"""Compare word and character TF-IDF robustness across corruption severity."""

from math import sqrt
from pathlib import Path

import pandas as pd
from scipy.stats import t as student_t

DATASETS = [
    "banking77",
    "clinc150",
    "hwu64",
]

MODELS = [
    "tfidf_svm",
    "char_tfidf_svm",
]

PERTURBATIONS = [
    "typo",
    "char_delete",
    "word_delete",
]

SEVERITIES = [
    0.05,
    0.10,
    0.20,
    0.30,
    0.40,
]


def severity_tag(
    severity: float,
) -> str:
    """Convert severity to filename tag."""

    return (
        f"{round(severity * 100)}pct"
    )


def summary_path(
    dataset: str,
    model: str,
    severity: float,
) -> Path:
    """Return robustness-summary path."""

    return (
        Path("results")
        / "robustness"
        / dataset
        / (
            f"{model}_"
            f"{severity_tag(severity)}"
            "_summary.csv"
        )
    )


def runs_path(
    dataset: str,
    model: str,
    severity: float,
) -> Path:
    """Return seed-level robustness path."""

    return (
        Path("results")
        / "robustness"
        / dataset
        / (
            f"{model}_"
            f"{severity_tag(severity)}"
            "_runs.csv"
        )
    )


def load_summary(
    dataset: str,
    model: str,
    severity: float,
) -> pd.DataFrame:
    """Load one robustness summary."""

    path = summary_path(
        dataset,
        model,
        severity,
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
    severity: float,
) -> pd.DataFrame:
    """Load seed-level robustness runs."""

    path = runs_path(
        dataset,
        model,
        severity,
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing run file: {path}"
        )

    return pd.read_csv(
        path
    )


def confidence_interval(
    values: pd.Series,
) -> tuple[float, float]:
    """Calculate t-based 95% CI."""

    values = (
        values.dropna()
    )

    n = len(
        values
    )

    if n < 2:
        return (
            float("nan"),
            float("nan"),
        )

    mean = values.mean()

    std = values.std(
        ddof=1
    )

    standard_error = (
        std
        / sqrt(n)
    )

    critical = student_t.ppf(
        0.975,
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


def build_summary_table() -> pd.DataFrame:
    """Create summary-level severity comparison."""

    rows = []

    for dataset in DATASETS:

        for severity in SEVERITIES:

            word = load_summary(
                dataset,
                "tfidf_svm",
                severity,
            )

            char = load_summary(
                dataset,
                "char_tfidf_svm",
                severity,
            )

            for perturbation in PERTURBATIONS:

                word_row = word[
                    word["perturbation"]
                    == perturbation
                ].iloc[0]

                char_row = char[
                    char["perturbation"]
                    == perturbation
                ].iloc[0]

                rows.append(
                    {
                        "dataset":
                            dataset,

                        "perturbation":
                            perturbation,

                        "requested_severity":
                            severity,

                        "word_realized_severity":
                            word_row[
                                "mean_realized_severity"
                            ],

                        "char_realized_severity":
                            char_row[
                                "mean_realized_severity"
                            ],

                        "word_clean_accuracy":
                            word_row[
                                "clean_accuracy"
                            ],

                        "char_clean_accuracy":
                            char_row[
                                "clean_accuracy"
                            ],

                        "word_mean_accuracy":
                            word_row[
                                "mean_accuracy"
                            ],

                        "char_mean_accuracy":
                            char_row[
                                "mean_accuracy"
                            ],

                        "accuracy_difference":
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
                    }
                )

    return pd.DataFrame(
        rows
    )


def build_paired_runs() -> pd.DataFrame:
    """Pair word and character representations by corruption seed."""

    rows = []

    for dataset in DATASETS:

        for severity in SEVERITIES:

            word_runs = load_runs(
                dataset,
                "tfidf_svm",
                severity,
            )

            char_runs = load_runs(
                dataset,
                "char_tfidf_svm",
                severity,
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

                if (
                    word_subset[
                        "seed"
                    ].tolist()
                    != char_subset[
                        "seed"
                    ].tolist()
                ):
                    raise ValueError(
                        "Seed mismatch for "
                        f"{dataset}, "
                        f"{severity}, "
                        f"{perturbation}"
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

                            "requested_severity":
                                severity,

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
                        }
                    )

    return pd.DataFrame(
        rows
    )


def build_paired_summary(
    paired: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize paired representation differences."""

    rows = []

    groups = paired.groupby(
        [
            "dataset",
            "perturbation",
            "requested_severity",
        ]
    )

    for (
        dataset,
        perturbation,
        severity,
    ), group in groups:

        accuracy_lower, accuracy_upper = (
            confidence_interval(
                group[
                    "accuracy_difference"
                ]
            )
        )

        retention_lower, retention_upper = (
            confidence_interval(
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

                "requested_severity":
                    severity,

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


def build_cross_dataset_summary(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    """Average representation effect across datasets."""

    return (
        summary.groupby(
            [
                "perturbation",
                "requested_severity",
            ],
            as_index=False,
        )
        .agg(
            mean_word_retention=(
                "word_retention",
                "mean",
            ),

            mean_char_retention=(
                "char_retention",
                "mean",
            ),

            mean_retention_difference=(
                "retention_difference",
                "mean",
            ),

            mean_accuracy_difference=(
                "accuracy_difference",
                "mean",
            ),
        )
    )


def print_cross_dataset_table(
    aggregate: pd.DataFrame,
) -> None:
    """Print cross-dataset severity effects."""

    display = (
        aggregate.copy()
    )

    display[
        "severity_percent"
    ] = (
        display[
            "requested_severity"
        ]
        * 100
    )

    percentage_columns = [
        "mean_word_retention",
        "mean_char_retention",
        "mean_retention_difference",
        "mean_accuracy_difference",
    ]

    for column in percentage_columns:

        display[
            column
        ] *= 100

    display = display[
        [
            "perturbation",
            "severity_percent",
            "mean_word_retention",
            "mean_char_retention",
            "mean_retention_difference",
            "mean_accuracy_difference",
        ]
    ]

    print("\n")
    print("=" * 108)
    print(
        "CROSS-DATASET REPRESENTATION "
        "SEVERITY EFFECT"
    )
    print("=" * 108)

    print(
        display.to_string(
            index=False,
            formatters={
                "severity_percent":
                    "{:.0f}%".format,

                "mean_word_retention":
                    "{:.2f}%".format,

                "mean_char_retention":
                    "{:.2f}%".format,

                "mean_retention_difference":
                    "{:+.2f} pp".format,

                "mean_accuracy_difference":
                    "{:+.2f} pp".format,
            },
        )
    )


def main() -> None:
    """Run representation severity analysis."""

    print("=" * 92)
    print("ReliabilityLab")
    print("Representation Severity Analysis")
    print(
        "Word TF-IDF + Linear SVM "
        "vs Character TF-IDF + Linear SVM"
    )
    print("=" * 92)

    summary = (
        build_summary_table()
    )

    paired = (
        build_paired_runs()
    )

    paired_summary = (
        build_paired_summary(
            paired
        )
    )

    aggregate = (
        build_cross_dataset_summary(
            summary
        )
    )

    print_cross_dataset_table(
        aggregate
    )

    print("\n")
    print("=" * 92)
    print("CHARACTER-LEVEL EFFECT BY SEVERITY")
    print("=" * 92)

    character_level = (
        aggregate[
            aggregate[
                "perturbation"
            ].isin(
                [
                    "typo",
                    "char_delete",
                ]
            )
        ]
        .groupby(
            "requested_severity",
            as_index=False,
        )
        .agg(
            mean_retention_difference=(
                "mean_retention_difference",
                "mean",
            )
        )
    )

    for _, row in (
        character_level.iterrows()
    ):

        print(
            f"{row['requested_severity'] * 100:>4.0f}% "
            f"severity: "
            f"{row['mean_retention_difference'] * 100:+.2f} pp"
        )

    print("\n")
    print(
        "Word deletion effect by severity:"
    )

    word_delete = aggregate[
        aggregate[
            "perturbation"
        ]
        == "word_delete"
    ]

    for _, row in (
        word_delete.iterrows()
    ):

        print(
            f"{row['requested_severity'] * 100:>4.0f}% "
            f"severity: "
            f"{row['mean_retention_difference'] * 100:+.2f} pp"
        )

    output_dir = (
        Path("results")
        / "comparison"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        output_dir
        / "representation_severity_summary.csv",
        index=False,
    )

    paired.to_csv(
        output_dir
        / "representation_severity_paired_runs.csv",
        index=False,
    )

    paired_summary.to_csv(
        output_dir
        / "representation_severity_paired_summary.csv",
        index=False,
    )

    aggregate.to_csv(
        output_dir
        / "representation_severity_cross_dataset.csv",
        index=False,
    )

    print("\nSaved:")
    print(
        output_dir
        / "representation_severity_summary.csv"
    )
    print(
        output_dir
        / "representation_severity_paired_runs.csv"
    )
    print(
        output_dir
        / "representation_severity_paired_summary.csv"
    )
    print(
        output_dir
        / "representation_severity_cross_dataset.csv"
    )


if __name__ == "__main__":
    main()