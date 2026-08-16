"""Probabilistic perturbation severity experiment."""

from pathlib import Path

import pandas as pd

from reliabilitylab.data import load_banking77
from reliabilitylab.metrics import classification_metrics
from reliabilitylab.models import build_tfidf_logreg
from reliabilitylab.perturbations import (
    perturb_texts_probabilistic,
)


SEEDS = [
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


SEVERITIES = [
    0.05,
    0.10,
    0.20,
    0.30,
    0.40,
]


PERTURBATIONS = [
    "typo",
    "char_delete",
    "word_delete",
]


def main():

    print("=" * 76)
    print("ReliabilityLab")
    print("Probabilistic Perturbation Severity Experiment")
    print("=" * 76)

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------
    print("\nLoading BANKING77...")

    dataset = load_banking77()

    train = dataset["train"]
    test = dataset["test"]

    X_train = train["utterance"]
    y_train = train["label"]

    X_test = test["utterance"]
    y_test = test["label"]

    # ---------------------------------------------------------
    # Train model ONCE
    # ---------------------------------------------------------
    print("\nTraining clean full-data model...")

    model = build_tfidf_logreg()

    model.fit(
        X_train,
        y_train,
    )

    # ---------------------------------------------------------
    # Clean baseline
    # ---------------------------------------------------------
    clean_predictions = model.predict(
        X_test
    )

    clean_metrics = classification_metrics(
        y_true=y_test,
        y_pred=clean_predictions,
    )

    clean_accuracy = clean_metrics[
        "accuracy"
    ]

    clean_macro_f1 = clean_metrics[
        "macro_f1"
    ]

    print(
        f"Clean accuracy: "
        f"{clean_accuracy * 100:.2f}%"
    )

    rows = []

    # ---------------------------------------------------------
    # Severity sweep
    # ---------------------------------------------------------
    for perturbation in PERTURBATIONS:

        print("\n" + "#" * 76)
        print(
            f"PERTURBATION: "
            f"{perturbation.upper()}"
        )
        print("#" * 76)

        for severity in SEVERITIES:

            print(
                f"\nRequested severity: "
                f"{severity * 100:.0f}%"
            )

            for run_number, seed in enumerate(
                SEEDS,
                start=1,
            ):

                perturbed_texts, stats = (
                    perturb_texts_probabilistic(
                        texts=X_test,
                        perturbation=perturbation,
                        severity=severity,
                        seed=seed,
                        return_stats=True,
                    )
                )

                predictions = model.predict(
                    perturbed_texts
                )

                metrics = classification_metrics(
                    y_true=y_test,
                    y_pred=predictions,
                )

                accuracy_drop = (
                    clean_accuracy
                    - metrics["accuracy"]
                )

                macro_f1_drop = (
                    clean_macro_f1
                    - metrics["macro_f1"]
                )

                rows.append(
                    {
                        "perturbation":
                            perturbation,

                        "severity":
                            severity,

                        "severity_percent":
                            int(severity * 100),

                        "run":
                            run_number,

                        "seed":
                            seed,

                        "realized_severity":
                            stats[
                                "realized_severity"
                            ],

                        "eligible_units":
                            stats[
                                "eligible_units"
                            ],

                        "affected_units":
                            stats[
                                "affected_units"
                            ],

                        "accuracy":
                            metrics["accuracy"],

                        "macro_f1":
                            metrics["macro_f1"],

                        "accuracy_drop":
                            accuracy_drop,

                        "macro_f1_drop":
                            macro_f1_drop,
                    }
                )

                print(
                    f"  Run {run_number:02d}/"
                    f"{len(SEEDS)} "
                    f"| seed={seed:<4} "
                    f"| realized="
                    f"{stats['realized_severity'] * 100:.2f}% "
                    f"| accuracy="
                    f"{metrics['accuracy'] * 100:.2f}% "
                    f"| drop="
                    f"{accuracy_drop * 100:.2f} pp"
                )

    # ---------------------------------------------------------
    # Raw results
    # ---------------------------------------------------------
    results = pd.DataFrame(
        rows
    )

    raw_path = Path(
        "results/robustness/"
        "tfidf_probabilistic_severity_runs.csv"
    )

    raw_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        raw_path,
        index=False,
    )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------
    summary = (
        results
        .groupby(
            [
                "perturbation",
                "severity",
                "severity_percent",
            ],
            as_index=False,
        )
        .agg(
            mean_realized_severity=(
                "realized_severity",
                "mean",
            ),
            realized_severity_std=(
                "realized_severity",
                "std",
            ),
            mean_accuracy=(
                "accuracy",
                "mean",
            ),
            accuracy_std=(
                "accuracy",
                "std",
            ),
            mean_macro_f1=(
                "macro_f1",
                "mean",
            ),
            macro_f1_std=(
                "macro_f1",
                "std",
            ),
            mean_drop=(
                "accuracy_drop",
                "mean",
            ),
            drop_std=(
                "accuracy_drop",
                "std",
            ),
        )
    )

    summary_path = Path(
        "results/robustness/"
        "tfidf_probabilistic_severity_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    # ---------------------------------------------------------
    # Display
    # ---------------------------------------------------------
    display = summary.copy()

    percentage_columns = [
        "mean_realized_severity",
        "realized_severity_std",
        "mean_accuracy",
        "accuracy_std",
        "mean_macro_f1",
        "macro_f1_std",
        "mean_drop",
        "drop_std",
    ]

    for column in percentage_columns:
        display[column] *= 100

    print("\n")
    print("=" * 76)
    print("PROBABILISTIC SEVERITY SUMMARY")
    print("=" * 76)

    print(
        display.to_string(
            index=False,
            formatters={
                "mean_realized_severity":
                    "{:.2f}%".format,

                "realized_severity_std":
                    "{:.2f} pp".format,

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
            },
        )
    )

    print(
        f"\nRaw results saved to:\n"
        f"{raw_path}"
    )

    print(
        f"\nSummary saved to:\n"
        f"{summary_path}"
    )


if __name__ == "__main__":
    main()