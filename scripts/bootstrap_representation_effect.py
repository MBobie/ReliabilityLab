"""Example-level paired bootstrap for TF-IDF representation robustness."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from reliabilitylab.data import (
    load_intent_dataset,
)
from reliabilitylab.metrics.bootstrap import (
    paired_example_bootstrap,
)
from reliabilitylab.models.registry import (
    build_model,
)
from reliabilitylab.perturbations import (
    perturb_texts_probabilistic,
)

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

PERTURBATION_SEEDS = [
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

WORD_MODEL = "tfidf_svm"
CHAR_MODEL = "char_tfidf_svm"


def parse_args():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Run paired example-level bootstrap "
            "for word vs character TF-IDF robustness."
        )
    )

    parser.add_argument(
        "--severities",
        nargs="+",
        type=float,
        default=[
            0.20,
            0.40,
        ],
        help=(
            "Requested corruption severities. "
            "Default: 0.20 0.40"
        ),
    )

    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=5000,
        help=(
            "Number of paired example bootstrap "
            "resamples. Default: 5000"
        ),
    )

    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=2026,
        help=(
            "Base random seed for bootstrap "
            "resampling. Default: 2026"
        ),
    )

    return parser.parse_args()


def validate_args(
    args,
) -> None:
    """Validate requested experiment settings."""

    for severity in args.severities:

        if not 0.0 <= severity <= 1.0:
            raise ValueError(
                "All severities must be between 0 and 1."
            )

    if args.n_bootstrap < 1:
        raise ValueError(
            "n-bootstrap must be at least 1."
        )


def correctness(
    labels: np.ndarray,
    predictions: np.ndarray,
) -> np.ndarray:
    """Return per-example binary correctness."""

    return (
        np.asarray(
            predictions
        )
        == labels
    ).astype(
        float
    )


def train_models(
    dataset,
):
    """Train word and character TF-IDF Linear SVM models."""

    print(
        "\nTraining Word TF-IDF + Linear SVM..."
    )

    word_model = build_model(
        WORD_MODEL
    )

    word_model.fit(
        dataset.train_texts,
        dataset.train_labels,
    )

    print(
        "Training Character TF-IDF + Linear SVM..."
    )

    char_model = build_model(
        CHAR_MODEL
    )

    char_model.fit(
        dataset.train_texts,
        dataset.train_labels,
    )

    return (
        word_model,
        char_model,
    )


def evaluate_condition(
    *,
    dataset_name: str,
    dataset,
    word_model,
    char_model,
    severity: float,
    perturbation: str,
    word_clean_correct: np.ndarray,
    char_clean_correct: np.ndarray,
    n_bootstrap: int,
    bootstrap_seed: int,
):
    """Evaluate one perturbation/severity condition."""

    labels = np.asarray(
        dataset.test_labels
    )

    word_seed_correctness = []

    char_seed_correctness = []

    realized_severities = []

    print(
        f"\n{dataset_name} | "
        f"{perturbation} | "
        f"{severity * 100:.0f}%"
    )

    for run_number, seed in enumerate(
        PERTURBATION_SEEDS,
        start=1,
    ):

        (
            perturbed_texts,
            stats,
        ) = perturb_texts_probabilistic(
            texts=dataset.test_texts,
            perturbation=perturbation,
            severity=severity,
            seed=seed,
            return_stats=True,
        )

        word_predictions = (
            word_model.predict(
                perturbed_texts
            )
        )

        char_predictions = (
            char_model.predict(
                perturbed_texts
            )
        )

        word_correct = correctness(
            labels,
            word_predictions,
        )

        char_correct = correctness(
            labels,
            char_predictions,
        )

        word_seed_correctness.append(
            word_correct
        )

        char_seed_correctness.append(
            char_correct
        )

        realized_severities.append(
            stats[
                "realized_severity"
            ]
        )

        print(
            f"  Run {run_number:02d}/"
            f"{len(PERTURBATION_SEEDS)} "
            f"| seed={seed:<4} "
            f"| realized="
            f"{stats['realized_severity'] * 100:.2f}%"
        )

    word_matrix = np.vstack(
        word_seed_correctness
    )

    char_matrix = np.vstack(
        char_seed_correctness
    )

    # Each test example becomes one bootstrap unit.
    #
    # Within an example we first average correctness
    # over the 10 matched perturbation realizations.
    #
    # This prevents the 10 corruptions of one query
    # from being treated as 10 independent examples.
    word_example_score = (
        word_matrix.mean(
            axis=0
        )
    )

    char_example_score = (
        char_matrix.mean(
            axis=0
        )
    )

    bootstrap = paired_example_bootstrap(
        clean_a=word_clean_correct,
        clean_b=char_clean_correct,
        perturbed_a=word_example_score,
        perturbed_b=char_example_score,
        n_resamples=n_bootstrap,
        confidence=0.95,
        seed=bootstrap_seed,
    )

    summary_row = {
        "dataset":
            dataset_name,

        "perturbation":
            perturbation,

        "requested_severity":
            severity,

        "mean_realized_severity":
            float(
                np.mean(
                    realized_severities
                )
            ),

        "n_examples":
            bootstrap[
                "n_examples"
            ],

        "n_perturbation_seeds":
            len(
                PERTURBATION_SEEDS
            ),

        "n_bootstrap":
            bootstrap[
                "n_resamples"
            ],

        "word_clean_accuracy":
            bootstrap[
                "clean_a"
            ],

        "char_clean_accuracy":
            bootstrap[
                "clean_b"
            ],

        "clean_accuracy_difference":
            bootstrap[
                "clean_difference"
            ],

        "word_mean_perturbed_accuracy":
            bootstrap[
                "perturbed_a"
            ],

        "char_mean_perturbed_accuracy":
            bootstrap[
                "perturbed_b"
            ],

        "perturbed_accuracy_difference":
            bootstrap[
                "perturbed_difference"
            ],

        "perturbed_accuracy_difference_ci_lower":
            bootstrap[
                "perturbed_difference_ci_lower"
            ],

        "perturbed_accuracy_difference_ci_upper":
            bootstrap[
                "perturbed_difference_ci_upper"
            ],

        "word_retention":
            bootstrap[
                "retention_a"
            ],

        "char_retention":
            bootstrap[
                "retention_b"
            ],

        "retention_difference":
            bootstrap[
                "retention_difference"
            ],

        "retention_difference_ci_lower":
            bootstrap[
                "retention_difference_ci_lower"
            ],

        "retention_difference_ci_upper":
            bootstrap[
                "retention_difference_ci_upper"
            ],

        "bootstrap_fraction_accuracy_difference_gt_zero":
            bootstrap[
                "bootstrap_fraction_accuracy_difference_gt_zero"
            ],

        "bootstrap_fraction_retention_difference_gt_zero":
            bootstrap[
                "bootstrap_fraction_retention_difference_gt_zero"
            ],
    }

    example_frame = pd.DataFrame(
        {
            "dataset":
                dataset_name,

            "perturbation":
                perturbation,

            "requested_severity":
                severity,

            "example_id":
                np.arange(
                    labels.size
                ),

            "label":
                labels,

            "word_clean_correct":
                word_clean_correct,

            "char_clean_correct":
                char_clean_correct,

            "word_seed_averaged_correctness":
                word_example_score,

            "char_seed_averaged_correctness":
                char_example_score,

            "seed_averaged_correctness_difference":
                (
                    char_example_score
                    - word_example_score
                ),
        }
    )

    return (
        summary_row,
        example_frame,
    )


def print_summary(
    summary: pd.DataFrame,
) -> None:
    """Print paper-oriented bootstrap summary."""

    display = summary.copy()

    percentage_columns = [
        "word_mean_perturbed_accuracy",
        "char_mean_perturbed_accuracy",
        "perturbed_accuracy_difference",
        "perturbed_accuracy_difference_ci_lower",
        "perturbed_accuracy_difference_ci_upper",
        "word_retention",
        "char_retention",
        "retention_difference",
        "retention_difference_ci_lower",
        "retention_difference_ci_upper",
    ]

    for column in percentage_columns:

        display[
            column
        ] *= 100

    display[
        "severity"
    ] = (
        display[
            "requested_severity"
        ]
        * 100
    )

    print("\n")
    print("=" * 130)
    print("EXAMPLE-LEVEL PAIRED BOOTSTRAP SUMMARY")
    print("Character TF-IDF minus Word TF-IDF")
    print("=" * 130)

    print(
        display[
            [
                "dataset",
                "perturbation",
                "severity",
                "perturbed_accuracy_difference",
                "perturbed_accuracy_difference_ci_lower",
                "perturbed_accuracy_difference_ci_upper",
                "retention_difference",
                "retention_difference_ci_lower",
                "retention_difference_ci_upper",
            ]
        ].to_string(
            index=False,
            formatters={
                "severity":
                    "{:.0f}%".format,

                "perturbed_accuracy_difference":
                    "{:+.2f} pp".format,

                "perturbed_accuracy_difference_ci_lower":
                    "{:+.2f} pp".format,

                "perturbed_accuracy_difference_ci_upper":
                    "{:+.2f} pp".format,

                "retention_difference":
                    "{:+.2f} pp".format,

                "retention_difference_ci_lower":
                    "{:+.2f} pp".format,

                "retention_difference_ci_upper":
                    "{:+.2f} pp".format,
            },
        )
    )


def main():
    """Run example-level paired bootstrap analysis."""

    args = parse_args()

    validate_args(
        args
    )

    print("=" * 92)
    print("ReliabilityLab")
    print("Example-Level Paired Bootstrap")
    print("Word TF-IDF + SVM vs Character TF-IDF + SVM")
    print("=" * 92)

    print(
        f"\nBootstrap resamples : "
        f"{args.n_bootstrap:,}"
    )

    print(
        "Perturbation seeds : "
        f"{len(PERTURBATION_SEEDS)}"
    )

    print(
        "Severities         : "
        + ", ".join(
            f"{severity * 100:.0f}%"
            for severity in args.severities
        )
    )

    summary_rows = []

    example_frames = []

    condition_index = 0

    for dataset_name in DATASETS:

        print("\n")
        print("#" * 92)
        print(
            f"DATASET: "
            f"{dataset_name.upper()}"
        )
        print("#" * 92)

        dataset = load_intent_dataset(
            dataset_name
        )

        labels = np.asarray(
            dataset.test_labels
        )

        (
            word_model,
            char_model,
        ) = train_models(
            dataset
        )

        word_clean_predictions = (
            word_model.predict(
                dataset.test_texts
            )
        )

        char_clean_predictions = (
            char_model.predict(
                dataset.test_texts
            )
        )

        word_clean_correct = correctness(
            labels,
            word_clean_predictions,
        )

        char_clean_correct = correctness(
            labels,
            char_clean_predictions,
        )

        print(
            "\nClean accuracy:"
        )

        print(
            "  Word TF-IDF : "
            f"{word_clean_correct.mean() * 100:.2f}%"
        )

        print(
            "  Char TF-IDF : "
            f"{char_clean_correct.mean() * 100:.2f}%"
        )

        for severity in args.severities:

            for perturbation in PERTURBATIONS:

                condition_seed = (
                    args.bootstrap_seed
                    + condition_index
                )

                (
                    summary_row,
                    example_frame,
                ) = evaluate_condition(
                    dataset_name=dataset_name,
                    dataset=dataset,
                    word_model=word_model,
                    char_model=char_model,
                    severity=severity,
                    perturbation=perturbation,
                    word_clean_correct=word_clean_correct,
                    char_clean_correct=char_clean_correct,
                    n_bootstrap=args.n_bootstrap,
                    bootstrap_seed=condition_seed,
                )

                summary_rows.append(
                    summary_row
                )

                example_frames.append(
                    example_frame
                )

                condition_index += 1

    summary = pd.DataFrame(
        summary_rows
    )

    example_results = pd.concat(
        example_frames,
        ignore_index=True,
    )

    print_summary(
        summary
    )

    output_dir = (
        Path("results")
        / "bootstrap"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        output_dir
        / "representation_example_bootstrap_summary.csv"
    )

    example_path = (
        output_dir
        / "representation_example_scores.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    example_results.to_csv(
        example_path,
        index=False,
    )

    print("\n")
    print(
        f"Summary saved: "
        f"{summary_path}"
    )

    print(
        f"Example scores saved: "
        f"{example_path}"
    )


if __name__ == "__main__":
    main()