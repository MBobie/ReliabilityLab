"""Run repeated TF-IDF robustness evaluation on any registered dataset."""

import argparse
import time
from pathlib import Path

import pandas as pd

from reliabilitylab.data import (
    available_datasets,
    load_intent_dataset,
)
from reliabilitylab.metrics import (
    classification_metrics,
    summarize_repeated_runs,
)
from reliabilitylab.models import (
    build_tfidf_logreg,
)
from reliabilitylab.perturbations import (
    perturb_texts_probabilistic,
)

DEFAULT_SEEDS = [
    1,
    7,
    21,
    42,
    84,
    123,
    256,
    512,
    1024,
    2026,
]


DEFAULT_PERTURBATIONS = [
    "typo",
    "char_delete",
    "word_delete",
]


def parse_args():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Run repeated probabilistic robustness evaluation "
            "for TF-IDF + Logistic Regression."
        )
    )

    parser.add_argument(
        "--dataset",
        required=True,
        choices=available_datasets(),
        help="Dataset to evaluate.",
    )

    parser.add_argument(
        "--severity",
        type=float,
        default=0.20,
        help=(
            "Requested probabilistic corruption severity. "
            "Default: 0.20"
        ),
    )

    return parser.parse_args()


def main():
    """Run the robustness benchmark."""

    args = parse_args()

    # ---------------------------------------------------------
    # Validate arguments
    # ---------------------------------------------------------
    if not 0.0 <= args.severity <= 1.0:
        raise ValueError(
            "severity must be between 0 and 1."
        )

    print("=" * 78)
    print("ReliabilityLab")
    print("Reusable TF-IDF Robustness Benchmark")
    print("=" * 78)

    print(
        f"\nDataset            : "
        f"{args.dataset}"
    )

    print(
        f"Requested severity : "
        f"{args.severity * 100:.1f}%"
    )

    print(
        f"Perturbation seeds : "
        f"{len(DEFAULT_SEEDS)}"
    )

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------
    print(
        "\nLoading dataset..."
    )

    dataset = load_intent_dataset(
        args.dataset
    )

    print(
        f"Training samples : "
        f"{dataset.num_train:,}"
    )

    print(
        f"Validation       : "
        f"{dataset.num_validation:,}"
    )

    print(
        f"Test samples     : "
        f"{dataset.num_test:,}"
    )

    print(
        f"Classes          : "
        f"{dataset.num_labels}"
    )

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------
    print(
        "\nTraining TF-IDF "
        "+ Logistic Regression..."
    )

    model = build_tfidf_logreg()

    train_start = time.time()

    model.fit(
        dataset.train_texts,
        dataset.train_labels,
    )

    training_time = (
        time.time()
        - train_start
    )

    print(
        f"Training time: "
        f"{training_time:.2f} seconds"
    )

    # ---------------------------------------------------------
    # Clean reference
    # ---------------------------------------------------------
    print(
        "\nEvaluating clean test set..."
    )

    clean_predictions = model.predict(
        dataset.test_texts
    )

    clean_metrics = classification_metrics(
        dataset.test_labels,
        clean_predictions,
    )

    clean_accuracy = (
        clean_metrics["accuracy"]
    )

    clean_macro_f1 = (
        clean_metrics["macro_f1"]
    )

    print(
        f"Clean accuracy : "
        f"{clean_accuracy * 100:.2f}%"
    )

    print(
        f"Clean Macro F1 : "
        f"{clean_macro_f1 * 100:.2f}%"
    )

    # ---------------------------------------------------------
    # Repeated perturbation evaluation
    # ---------------------------------------------------------
    rows = []

    experiment_start = time.time()

    for perturbation in DEFAULT_PERTURBATIONS:

        print("\n" + "#" * 78)

        print(
            f"PERTURBATION: "
            f"{perturbation.upper()}"
        )

        print("#" * 78)

        for run_number, seed in enumerate(
            DEFAULT_SEEDS,
            start=1,
        ):

            (
                perturbed_texts,
                stats,
            ) = perturb_texts_probabilistic(
                texts=dataset.test_texts,
                perturbation=perturbation,
                severity=args.severity,
                seed=seed,
                return_stats=True,
            )

            predictions = model.predict(
                perturbed_texts
            )

            metrics = classification_metrics(
                dataset.test_labels,
                predictions,
            )

            accuracy_drop = (
                clean_accuracy
                - metrics["accuracy"]
            )

            macro_f1_drop = (
                clean_macro_f1
                - metrics["macro_f1"]
            )

            accuracy_retention = (
                metrics["accuracy"]
                / clean_accuracy
            )

            relative_accuracy_drop = (
                accuracy_drop
                / clean_accuracy
            )

            rows.append(
                {
                    "dataset":
                        args.dataset,

                    "model":
                        "tfidf_logreg",

                    "perturbation":
                        perturbation,

                    "run":
                        run_number,

                    "seed":
                        seed,

                    "requested_severity":
                        args.severity,

                    "realized_severity":
                        stats[
                            "realized_severity"
                        ],

                    "accuracy":
                        metrics[
                            "accuracy"
                        ],

                    "macro_f1":
                        metrics[
                            "macro_f1"
                        ],

                    "accuracy_drop":
                        accuracy_drop,

                    "macro_f1_drop":
                        macro_f1_drop,

                    "accuracy_retention":
                        accuracy_retention,

                    "relative_accuracy_drop":
                        relative_accuracy_drop,
                }
            )

            print(
                f"Run {run_number:02d}/"
                f"{len(DEFAULT_SEEDS)} "
                f"| seed={seed:<4} "
                f"| realized="
                f"{stats['realized_severity'] * 100:.2f}% "
                f"| accuracy="
                f"{metrics['accuracy'] * 100:.2f}% "
                f"| retention="
                f"{accuracy_retention * 100:.2f}% "
                f"| drop="
                f"{accuracy_drop * 100:.2f} pp"
            )

    # ---------------------------------------------------------
    # Raw results
    # ---------------------------------------------------------
    results = pd.DataFrame(
        rows
    )

    severity_tag = (
        f"{round(args.severity * 100)}pct"
    )

    output_dir = (
        Path("results")
        / "robustness"
        / args.dataset
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_path = (
        output_dir
        / f"tfidf_{severity_tag}_runs.csv"
    )

    results.to_csv(
        raw_path,
        index=False,
    )

    # ---------------------------------------------------------
    # Summaries
    # ---------------------------------------------------------
    summary_rows = []

    for perturbation in DEFAULT_PERTURBATIONS:

        subset = results[
            results["perturbation"]
            == perturbation
        ]

        accuracy_summary = (
            summarize_repeated_runs(
                subset["accuracy"]
            )
        )

        macro_f1_summary = (
            summarize_repeated_runs(
                subset["macro_f1"]
            )
        )

        drop_summary = (
            summarize_repeated_runs(
                subset[
                    "accuracy_drop"
                ]
            )
        )

        retention_summary = (
            summarize_repeated_runs(
                subset[
                    "accuracy_retention"
                ]
            )
        )

        relative_drop_summary = (
            summarize_repeated_runs(
                subset[
                    "relative_accuracy_drop"
                ]
            )
        )

        summary_rows.append(
            {
                "dataset":
                    args.dataset,

                "model":
                    "tfidf_logreg",

                "perturbation":
                    perturbation,

                "requested_severity":
                    args.severity,

                "mean_realized_severity":
                    subset[
                        "realized_severity"
                    ].mean(),

                "clean_accuracy":
                    clean_accuracy,

                "clean_macro_f1":
                    clean_macro_f1,

                "mean_accuracy":
                    accuracy_summary[
                        "mean"
                    ],

                "accuracy_std":
                    accuracy_summary[
                        "std"
                    ],

                "mean_macro_f1":
                    macro_f1_summary[
                        "mean"
                    ],

                "macro_f1_std":
                    macro_f1_summary[
                        "std"
                    ],

                "mean_drop":
                    drop_summary[
                        "mean"
                    ],

                "drop_std":
                    drop_summary[
                        "std"
                    ],

                "drop_ci_lower":
                    drop_summary[
                        "ci_95_lower"
                    ],

                "drop_ci_upper":
                    drop_summary[
                        "ci_95_upper"
                    ],

                "accuracy_retention":
                    retention_summary[
                        "mean"
                    ],

                "retention_std":
                    retention_summary[
                        "std"
                    ],

                "relative_accuracy_drop":
                    relative_drop_summary[
                        "mean"
                    ],

                "relative_drop_std":
                    relative_drop_summary[
                        "std"
                    ],
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    summary_path = (
        output_dir
        / f"tfidf_{severity_tag}_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    # ---------------------------------------------------------
    # Display preparation
    # ---------------------------------------------------------
    display = summary.copy()

    percentage_columns = [
        "mean_realized_severity",
        "clean_accuracy",
        "clean_macro_f1",
        "mean_accuracy",
        "accuracy_std",
        "mean_macro_f1",
        "macro_f1_std",
        "mean_drop",
        "drop_std",
        "drop_ci_lower",
        "drop_ci_upper",
        "accuracy_retention",
        "retention_std",
        "relative_accuracy_drop",
        "relative_drop_std",
    ]

    for column in percentage_columns:
        display[column] *= 100

    total_time = (
        time.time()
        - experiment_start
    )

    # ---------------------------------------------------------
    # Display summary
    # ---------------------------------------------------------
    print("\n")
    print("=" * 78)
    print("ROBUSTNESS SUMMARY")
    print("=" * 78)

    print(
        display.to_string(
            index=False,
            formatters={
                "mean_realized_severity":
                    "{:.2f}%".format,

                "clean_accuracy":
                    "{:.2f}%".format,

                "clean_macro_f1":
                    "{:.2f}%".format,

                "mean_accuracy":
                    "{:.2f}%".format,

                "accuracy_std":
                    "{:.2f} pp".format,

                "mean_macro_f1":
                    "{:.2f}%".format,

                "macro_f1_std":
                    "{:.2f} pp".format,

                "mean_drop":
                    "{:.2f} pp".format,

                "drop_std":
                    "{:.2f} pp".format,

                "drop_ci_lower":
                    "{:.2f} pp".format,

                "drop_ci_upper":
                    "{:.2f} pp".format,

                "accuracy_retention":
                    "{:.2f}%".format,

                "retention_std":
                    "{:.2f} pp".format,

                "relative_accuracy_drop":
                    "{:.2f}%".format,

                "relative_drop_std":
                    "{:.2f} pp".format,
            },
        )
    )

    print(
        f"\nTraining time: "
        f"{training_time:.2f} seconds"
    )

    print(
        f"Robustness evaluation time: "
        f"{total_time:.2f} seconds"
    )

    print(
        f"\nRaw results:\n"
        f"{raw_path}"
    )

    print(
        f"\nSummary:\n"
        f"{summary_path}"
    )


if __name__ == "__main__":
    main()