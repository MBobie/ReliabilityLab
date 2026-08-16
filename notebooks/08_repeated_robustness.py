"""Repeated perturbation robustness experiments for ReliabilityLab."""

from pathlib import Path

import pandas as pd

from reliabilitylab.data import load_banking77
from reliabilitylab.metrics import summarize_repeated_runs
from reliabilitylab.models import build_tfidf_logreg
from reliabilitylab.perturbations import perturb_texts
from reliabilitylab.metrics import classification_metrics


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


PERTURBATIONS = [
    "typo",
    "char_delete",
    "word_delete",
]


def main():

    print("=" * 70)
    print("ReliabilityLab")
    print("Repeated Perturbation Robustness Experiment")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load dataset
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
        f"\nClean accuracy: "
        f"{clean_accuracy * 100:.2f}%"
    )

    print(
        f"Clean Macro F1: "
        f"{clean_macro_f1 * 100:.2f}%"
    )

    all_results = []

    # ---------------------------------------------------------
    # Repeated perturbation runs
    # ---------------------------------------------------------
    for perturbation in PERTURBATIONS:

        print("\n" + "#" * 70)
        print(
            f"PERTURBATION: "
            f"{perturbation.upper()}"
        )
        print("#" * 70)

        for run_number, seed in enumerate(
            SEEDS,
            start=1,
        ):

            perturbed_texts = perturb_texts(
                texts=X_test,
                perturbation=perturbation,
                seed=seed,
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

            result = {
                "perturbation": perturbation,
                "run": run_number,
                "seed": seed,
                "accuracy":
                    metrics["accuracy"],
                "macro_f1":
                    metrics["macro_f1"],
                "accuracy_drop":
                    accuracy_drop,
                "macro_f1_drop":
                    macro_f1_drop,
            }

            all_results.append(result)

            print(
                f"Run {run_number:02d}/"
                f"{len(SEEDS)} "
                f"| seed={seed:<4} "
                f"| accuracy="
                f"{metrics['accuracy'] * 100:.2f}% "
                f"| drop="
                f"{accuracy_drop * 100:.2f} pp"
            )

    # ---------------------------------------------------------
    # Save raw results
    # ---------------------------------------------------------
    results_df = pd.DataFrame(
        all_results
    )

    raw_path = Path(
        "results/robustness/"
        "tfidf_repeated_robustness_runs.csv"
    )

    raw_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        raw_path,
        index=False,
    )

    # ---------------------------------------------------------
    # Summarize each perturbation
    # ---------------------------------------------------------
    summary_rows = []

    for perturbation in PERTURBATIONS:

        subset = results_df[
            results_df["perturbation"]
            == perturbation
        ]

        accuracy_summary = (
            summarize_repeated_runs(
                subset["accuracy"]
            )
        )

        drop_summary = (
            summarize_repeated_runs(
                subset["accuracy_drop"]
            )
        )

        summary_rows.append(
            {
                "perturbation":
                    perturbation,

                "mean_accuracy":
                    accuracy_summary["mean"],

                "accuracy_std":
                    accuracy_summary["std"],

                "min_accuracy":
                    accuracy_summary["min"],

                "max_accuracy":
                    accuracy_summary["max"],

                "mean_drop":
                    drop_summary["mean"],

                "drop_std":
                    drop_summary["std"],

                "drop_ci_lower":
                    drop_summary[
                        "ci_95_lower"
                    ],

                "drop_ci_upper":
                    drop_summary[
                        "ci_95_upper"
                    ],
            }
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_path = Path(
        "results/robustness/"
        "tfidf_repeated_robustness_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    # ---------------------------------------------------------
    # Display summary
    # ---------------------------------------------------------
    display = summary_df.copy()

    for column in [
        "mean_accuracy",
        "accuracy_std",
        "min_accuracy",
        "max_accuracy",
        "mean_drop",
        "drop_std",
        "drop_ci_lower",
        "drop_ci_upper",
    ]:
        display[column] *= 100

    print("\n")
    print("=" * 70)
    print("REPEATED ROBUSTNESS SUMMARY")
    print("=" * 70)

    print(
        display.to_string(
            index=False,
            formatters={
                "mean_accuracy":
                    "{:.2f}%".format,

                "accuracy_std":
                    "{:.2f} pp".format,

                "min_accuracy":
                    "{:.2f}%".format,

                "max_accuracy":
                    "{:.2f}%".format,

                "mean_drop":
                    "{:.2f} pp".format,

                "drop_std":
                    "{:.2f} pp".format,

                "drop_ci_lower":
                    "{:.2f} pp".format,

                "drop_ci_upper":
                    "{:.2f} pp".format,
            },
        )
    )

    print(
        "\nRaw results saved to:"
    )
    print(raw_path)

    print(
        "\nSummary saved to:"
    )
    print(summary_path)


if __name__ == "__main__":
    main()