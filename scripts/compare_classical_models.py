"""Compare TF-IDF Logistic Regression and Linear SVM across datasets."""

from pathlib import Path

import pandas as pd

DATASETS = [
    "banking77",
    "clinc150",
    "hwu64",
]

MODELS = [
    "tfidf_logreg",
    "tfidf_svm",
]

PERTURBATIONS = [
    "typo",
    "char_delete",
    "word_delete",
]

SEVERITY_TAG = "20pct"


MODEL_LABELS = {
    "tfidf_logreg": "Logistic Regression",
    "tfidf_svm": "Linear SVM",
}


def load_summary(
    dataset: str,
    model: str,
) -> pd.DataFrame:
    """Load one model robustness summary."""

    path = (
        Path("results")
        / "robustness"
        / dataset
        / f"{model}_{SEVERITY_TAG}_summary.csv"
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
    """Load one model's seed-level robustness runs."""

    path = (
        Path("results")
        / "robustness"
        / dataset
        / f"{model}_{SEVERITY_TAG}_runs.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing run file: {path}"
        )

    return pd.read_csv(
        path
    )


def build_summary_table() -> pd.DataFrame:
    """Combine all model and dataset summaries."""

    frames = []

    for dataset in DATASETS:

        for model in MODELS:

            frame = load_summary(
                dataset,
                model,
            )

            frames.append(
                frame
            )

    return pd.concat(
        frames,
        ignore_index=True,
    )


def build_paired_seed_table() -> pd.DataFrame:
    """Build matched LR-vs-SVM comparisons using identical perturbation seeds."""

    rows = []

    for dataset in DATASETS:

        lr_runs = load_runs(
            dataset,
            "tfidf_logreg",
        )

        svm_runs = load_runs(
            dataset,
            "tfidf_svm",
        )

        for perturbation in PERTURBATIONS:

            lr_subset = (
                lr_runs[
                    lr_runs["perturbation"]
                    == perturbation
                ]
                .sort_values(
                    "seed"
                )
                .reset_index(
                    drop=True
                )
            )

            svm_subset = (
                svm_runs[
                    svm_runs["perturbation"]
                    == perturbation
                ]
                .sort_values(
                    "seed"
                )
                .reset_index(
                    drop=True
                )
            )

            if not (
                lr_subset["seed"].tolist()
                == svm_subset["seed"].tolist()
            ):
                raise ValueError(
                    f"Seed mismatch for "
                    f"{dataset} / {perturbation}"
                )

            for index in range(
                len(lr_subset)
            ):

                lr_row = lr_subset.iloc[
                    index
                ]

                svm_row = svm_subset.iloc[
                    index
                ]

                rows.append(
                    {
                        "dataset":
                            dataset,

                        "perturbation":
                            perturbation,

                        "seed":
                            int(
                                lr_row["seed"]
                            ),

                        "lr_accuracy":
                            lr_row[
                                "accuracy"
                            ],

                        "svm_accuracy":
                            svm_row[
                                "accuracy"
                            ],

                        "accuracy_difference":
                            (
                                svm_row[
                                    "accuracy"
                                ]
                                - lr_row[
                                    "accuracy"
                                ]
                            ),

                        "lr_retention":
                            lr_row[
                                "accuracy_retention"
                            ],

                        "svm_retention":
                            svm_row[
                                "accuracy_retention"
                            ],

                        "retention_difference":
                            (
                                svm_row[
                                    "accuracy_retention"
                                ]
                                - lr_row[
                                    "accuracy_retention"
                                ]
                            ),

                        "lr_relative_drop":
                            lr_row[
                                "relative_accuracy_drop"
                            ],

                        "svm_relative_drop":
                            svm_row[
                                "relative_accuracy_drop"
                            ],
                    }
                )

    return pd.DataFrame(
        rows
    )


def main():
    """Generate LR-vs-SVM comparison tables."""

    print("=" * 90)
    print("ReliabilityLab")
    print("TF-IDF Classifier Comparison")
    print("Logistic Regression vs Linear SVM")
    print("=" * 90)

    # ---------------------------------------------------------
    # Load summaries
    # ---------------------------------------------------------
    summary = build_summary_table()

    # ---------------------------------------------------------
    # Per-model average robustness
    # ---------------------------------------------------------
    model_dataset_summary = (
        summary.groupby(
            [
                "dataset",
                "model",
            ],
            as_index=False,
        )
        .agg(
            clean_accuracy=(
                "clean_accuracy",
                "first",
            ),

            clean_macro_f1=(
                "clean_macro_f1",
                "first",
            ),

            mean_perturbed_accuracy=(
                "mean_accuracy",
                "mean",
            ),

            mean_accuracy_retention=(
                "accuracy_retention",
                "mean",
            ),

            mean_relative_drop=(
                "relative_accuracy_drop",
                "mean",
            ),

            mean_absolute_drop=(
                "mean_drop",
                "mean",
            ),
        )
    )

    # ---------------------------------------------------------
    # Model labels
    # ---------------------------------------------------------
    model_dataset_summary[
        "model_label"
    ] = model_dataset_summary[
        "model"
    ].map(
        MODEL_LABELS
    )

    # ---------------------------------------------------------
    # Display model-dataset summary
    # ---------------------------------------------------------
    display = (
        model_dataset_summary.copy()
    )

    percentage_columns = [
        "clean_accuracy",
        "clean_macro_f1",
        "mean_perturbed_accuracy",
        "mean_accuracy_retention",
        "mean_relative_drop",
        "mean_absolute_drop",
    ]

    for column in percentage_columns:
        display[column] *= 100

    print("\n")
    print("=" * 90)
    print("MODEL × DATASET SUMMARY")
    print("=" * 90)

    print(
        display[
            [
                "dataset",
                "model_label",
                "clean_accuracy",
                "clean_macro_f1",
                "mean_perturbed_accuracy",
                "mean_absolute_drop",
                "mean_accuracy_retention",
                "mean_relative_drop",
            ]
        ].to_string(
            index=False,
            formatters={
                "clean_accuracy":
                    "{:.2f}%".format,

                "clean_macro_f1":
                    "{:.2f}%".format,

                "mean_perturbed_accuracy":
                    "{:.2f}%".format,

                "mean_absolute_drop":
                    "{:.2f} pp".format,

                "mean_accuracy_retention":
                    "{:.2f}%".format,

                "mean_relative_drop":
                    "{:.2f}%".format,
            },
        )
    )

    # ---------------------------------------------------------
    # LR-vs-SVM dataset-level comparison
    # ---------------------------------------------------------
    wide = model_dataset_summary.pivot(
        index="dataset",
        columns="model",
        values=[
            "clean_accuracy",
            "mean_perturbed_accuracy",
            "mean_accuracy_retention",
            "mean_relative_drop",
        ],
    )

    comparison_rows = []

    for dataset in DATASETS:

        lr_clean = wide.loc[
            dataset,
            (
                "clean_accuracy",
                "tfidf_logreg",
            ),
        ]

        svm_clean = wide.loc[
            dataset,
            (
                "clean_accuracy",
                "tfidf_svm",
            ),
        ]

        lr_perturbed = wide.loc[
            dataset,
            (
                "mean_perturbed_accuracy",
                "tfidf_logreg",
            ),
        ]

        svm_perturbed = wide.loc[
            dataset,
            (
                "mean_perturbed_accuracy",
                "tfidf_svm",
            ),
        ]

        lr_retention = wide.loc[
            dataset,
            (
                "mean_accuracy_retention",
                "tfidf_logreg",
            ),
        ]

        svm_retention = wide.loc[
            dataset,
            (
                "mean_accuracy_retention",
                "tfidf_svm",
            ),
        ]

        lr_relative_drop = wide.loc[
            dataset,
            (
                "mean_relative_drop",
                "tfidf_logreg",
            ),
        ]

        svm_relative_drop = wide.loc[
            dataset,
            (
                "mean_relative_drop",
                "tfidf_svm",
            ),
        ]

        comparison_rows.append(
            {
                "dataset":
                    dataset,

                "lr_clean_accuracy":
                    lr_clean,

                "svm_clean_accuracy":
                    svm_clean,

                "clean_accuracy_gain":
                    svm_clean
                    - lr_clean,

                "lr_mean_perturbed_accuracy":
                    lr_perturbed,

                "svm_mean_perturbed_accuracy":
                    svm_perturbed,

                "perturbed_accuracy_gain":
                    svm_perturbed
                    - lr_perturbed,

                "lr_mean_retention":
                    lr_retention,

                "svm_mean_retention":
                    svm_retention,

                "retention_difference":
                    svm_retention
                    - lr_retention,

                "lr_mean_relative_drop":
                    lr_relative_drop,

                "svm_mean_relative_drop":
                    svm_relative_drop,

                "relative_drop_difference":
                    svm_relative_drop
                    - lr_relative_drop,
            }
        )

    classifier_comparison = pd.DataFrame(
        comparison_rows
    )

    classifier_display = (
        classifier_comparison.copy()
    )

    for column in classifier_display.columns:

        if column != "dataset":
            classifier_display[
                column
            ] *= 100

    print("\n")
    print("=" * 90)
    print("CLASSIFIER EFFECT")
    print("=" * 90)

    print(
        classifier_display.to_string(
            index=False,
            formatters={
                "lr_clean_accuracy":
                    "{:.2f}%".format,

                "svm_clean_accuracy":
                    "{:.2f}%".format,

                "clean_accuracy_gain":
                    "{:+.2f} pp".format,

                "lr_mean_perturbed_accuracy":
                    "{:.2f}%".format,

                "svm_mean_perturbed_accuracy":
                    "{:.2f}%".format,

                "perturbed_accuracy_gain":
                    "{:+.2f} pp".format,

                "lr_mean_retention":
                    "{:.2f}%".format,

                "svm_mean_retention":
                    "{:.2f}%".format,

                "retention_difference":
                    "{:+.2f} pp".format,

                "lr_mean_relative_drop":
                    "{:.2f}%".format,

                "svm_mean_relative_drop":
                    "{:.2f}%".format,

                "relative_drop_difference":
                    "{:+.2f} pp".format,
            },
        )
    )

    # ---------------------------------------------------------
    # Perturbation-specific retention comparison
    # ---------------------------------------------------------
    print("\n")
    print("=" * 90)
    print("PERTURBATION-SPECIFIC RETENTION")
    print("=" * 90)

    retention_table = summary.pivot_table(
        index=[
            "dataset",
            "perturbation",
        ],
        columns="model",
        values="accuracy_retention",
    ).reset_index()

    retention_table[
        "retention_difference"
    ] = (
        retention_table[
            "tfidf_svm"
        ]
        - retention_table[
            "tfidf_logreg"
        ]
    )

    retention_display = (
        retention_table.copy()
    )

    for column in [
        "tfidf_logreg",
        "tfidf_svm",
        "retention_difference",
    ]:
        retention_display[
            column
        ] *= 100

    print(
        retention_display.to_string(
            index=False,
            formatters={
                "tfidf_logreg":
                    "{:.2f}%".format,

                "tfidf_svm":
                    "{:.2f}%".format,

                "retention_difference":
                    "{:+.2f} pp".format,
            },
        )
    )

    # ---------------------------------------------------------
    # Seed-level paired comparison
    # ---------------------------------------------------------
    paired = (
        build_paired_seed_table()
    )

    paired_summary = (
        paired.groupby(
            [
                "dataset",
                "perturbation",
            ],
            as_index=False,
        )
        .agg(
            mean_accuracy_difference=(
                "accuracy_difference",
                "mean",
            ),

            accuracy_difference_std=(
                "accuracy_difference",
                "std",
            ),

            mean_retention_difference=(
                "retention_difference",
                "mean",
            ),

            retention_difference_std=(
                "retention_difference",
                "std",
            ),
        )
    )

    paired_display = (
        paired_summary.copy()
    )

    for column in [
        "mean_accuracy_difference",
        "accuracy_difference_std",
        "mean_retention_difference",
        "retention_difference_std",
    ]:
        paired_display[
            column
        ] *= 100

    print("\n")
    print("=" * 90)
    print("MATCHED PERTURBATION-SEED DIFFERENCES")
    print("=" * 90)

    print(
        paired_display.to_string(
            index=False,
            formatters={
                "mean_accuracy_difference":
                    "{:+.2f} pp".format,

                "accuracy_difference_std":
                    "{:.2f} pp".format,

                "mean_retention_difference":
                    "{:+.2f} pp".format,

                "retention_difference_std":
                    "{:.2f} pp".format,
            },
        )
    )

    # ---------------------------------------------------------
    # Overall classifier effect
    # ---------------------------------------------------------
    print("\n")
    print("=" * 90)
    print("AVERAGE CLASSIFIER EFFECT ACROSS DATASETS")
    print("=" * 90)

    print(
        "Mean clean accuracy gain "
        "(SVM - LR): "
        f"{classifier_comparison['clean_accuracy_gain'].mean() * 100:+.2f} pp"
    )

    print(
        "Mean perturbed accuracy gain "
        "(SVM - LR): "
        f"{classifier_comparison['perturbed_accuracy_gain'].mean() * 100:+.2f} pp"
    )

    print(
        "Mean retention difference "
        "(SVM - LR): "
        f"{classifier_comparison['retention_difference'].mean() * 100:+.2f} pp"
    )

    print(
        "Mean relative-drop difference "
        "(SVM - LR): "
        f"{classifier_comparison['relative_drop_difference'].mean() * 100:+.2f} pp"
    )

    # ---------------------------------------------------------
    # Save outputs
    # ---------------------------------------------------------
    output_dir = (
        Path("results")
        / "comparison"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_dataset_summary.to_csv(
        output_dir
        / "tfidf_classifier_model_dataset_summary.csv",
        index=False,
    )

    classifier_comparison.to_csv(
        output_dir
        / "tfidf_classifier_effect.csv",
        index=False,
    )

    retention_table.to_csv(
        output_dir
        / "tfidf_classifier_retention_by_perturbation.csv",
        index=False,
    )

    paired.to_csv(
        output_dir
        / "tfidf_classifier_paired_runs.csv",
        index=False,
    )

    paired_summary.to_csv(
        output_dir
        / "tfidf_classifier_paired_summary.csv",
        index=False,
    )

    print("\n")
    print(
        "Comparison files saved to:"
    )

    print(
        output_dir
    )


if __name__ == "__main__":
    main()