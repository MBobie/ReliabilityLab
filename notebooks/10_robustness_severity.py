"""Evaluate robustness as perturbation severity increases."""

from pathlib import Path

import pandas as pd

from reliabilitylab.data import load_banking77
from reliabilitylab.metrics import (
    classification_metrics,
)
from reliabilitylab.models import (
    build_tfidf_logreg,
)
from reliabilitylab.perturbations import (
    perturb_texts_with_severity,
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

    print("=" * 72)
    print("ReliabilityLab")
    print("Perturbation Severity Experiment")
    print("=" * 72)

    # ---------------------------------------------------------
    # Load data
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
    # Train once
    # ---------------------------------------------------------
    print("\nTraining clean full-data model...")

    model = build_tfidf_logreg()

    model.fit(
        X_train,
        y_train,
    )

    # ---------------------------------------------------------
    # Clean reference
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

    print(
        f"Clean accuracy: "
        f"{clean_accuracy * 100:.2f}%"
    )

    rows = []

    # ---------------------------------------------------------
    # Severity sweep
    # ---------------------------------------------------------
    for perturbation in PERTURBATIONS:

        print("\n" + "#" * 72)
        print(
            f"PERTURBATION: {perturbation.upper()}"
        )
        print("#" * 72)

        for severity in SEVERITIES:

            print(
                f"\nSeverity: "
                f"{severity * 100:.0f}%"
            )

            for run_number, seed in enumerate(
                SEEDS,
                start=1,
            ):

                perturbed_texts = (
                    perturb_texts_with_severity(
                        texts=X_test,
                        perturbation=perturbation,
                        severity=severity,
                        seed=seed,
                    )
                )

                predictions = model.predict(
                    perturbed_texts
                )

                metrics = classification_metrics(
                    y_true=y_test,
                    y_pred=predictions,
                )

                drop = (
                    clean_accuracy
                    - metrics["accuracy"]
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

                        "accuracy":
                            metrics["accuracy"],

                        "accuracy_drop":
                            drop,
                    }
                )

                print(
                    f"  Run "
                    f"{run_number:02d}/"
                    f"{len(SEEDS)} "
                    f"| seed={seed:<4} "
                    f"| accuracy="
                    f"{metrics['accuracy'] * 100:.2f}% "
                    f"| drop="
                    f"{drop * 100:.2f} pp"
                )

    # ---------------------------------------------------------
    # Save raw results
    # ---------------------------------------------------------
    results = pd.DataFrame(rows)

    raw_path = Path(
        "results/robustness/"
        "tfidf_severity_runs.csv"
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
    # Summarize
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
            mean_accuracy=(
                "accuracy",
                "mean",
            ),
            accuracy_std=(
                "accuracy",
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
        "tfidf_severity_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    # ---------------------------------------------------------
    # Display compact summary
    # ---------------------------------------------------------
    display = summary.copy()

    display[
        "mean_accuracy"
    ] *= 100

    display[
        "accuracy_std"
    ] *= 100

    display[
        "mean_drop"
    ] *= 100

    display[
        "drop_std"
    ] *= 100

    print("\n")
    print("=" * 72)
    print("SEVERITY SUMMARY")
    print("=" * 72)

    print(
        display.to_string(
            index=False,
            formatters={
                "mean_accuracy":
                    "{:.2f}%".format,

                "accuracy_std":
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