"""Repeated 20% perturbation robustness evaluation for DistilBERT."""

from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from reliabilitylab.data import load_banking77
from reliabilitylab.metrics import (
    classification_metrics,
    summarize_repeated_runs,
)
from reliabilitylab.perturbations import (
    perturb_texts_probabilistic,
)


MODEL_PATH = (
    "results/distilbert/"
    "baseline/final_model"
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

PERTURBATIONS = [
    "typo",
    "char_delete",
    "word_delete",
]

SEVERITY = 0.20

BATCH_SIZE = 32
MAX_LENGTH = 64


def predict_texts(
    model,
    tokenizer,
    texts,
):
    """Predict labels for a sequence of texts."""

    model.eval()

    predictions = []

    for start in range(
        0,
        len(texts),
        BATCH_SIZE,
    ):

        batch = texts[
            start:start + BATCH_SIZE
        ]

        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = model(**encoded)

        batch_predictions = (
            torch.argmax(
                outputs.logits,
                dim=-1,
            )
            .cpu()
            .numpy()
        )

        predictions.extend(
            batch_predictions
        )

    return np.asarray(predictions)


def main():

    print("=" * 76)
    print("ReliabilityLab")
    print("Repeated DistilBERT 20% Robustness Experiment")
    print("=" * 76)

    start_time = time.time()

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------
    print("\nLoading BANKING77...")

    dataset = load_banking77()

    X_test = dataset["test"]["utterance"]
    y_test = dataset["test"]["label"]

    print(
        f"Test samples: {len(X_test):,}"
    )

    # ---------------------------------------------------------
    # Saved DistilBERT
    # ---------------------------------------------------------
    print(
        "\nLoading saved DistilBERT..."
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_PATH
        )
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            MODEL_PATH
        )
    )

    model.to("cpu")

    # ---------------------------------------------------------
    # Clean reference
    # ---------------------------------------------------------
    print(
        "\nEvaluating clean reference..."
    )

    clean_predictions = predict_texts(
        model,
        tokenizer,
        X_test,
    )

    clean_metrics = classification_metrics(
        y_true=y_test,
        y_pred=clean_predictions,
    )

    clean_accuracy = (
        clean_metrics["accuracy"]
    )

    clean_macro_f1 = (
        clean_metrics["macro_f1"]
    )

    print(
        f"Clean accuracy: "
        f"{clean_accuracy * 100:.2f}%"
    )

    print(
        f"Clean Macro F1: "
        f"{clean_macro_f1 * 100:.2f}%"
    )

    rows = []

    # ---------------------------------------------------------
    # Repeated perturbation realizations
    # ---------------------------------------------------------
    for perturbation in PERTURBATIONS:

        print("\n" + "#" * 76)
        print(
            f"PERTURBATION: "
            f"{perturbation.upper()}"
        )
        print("#" * 76)

        for run_number, seed in enumerate(
            SEEDS,
            start=1,
        ):

            perturbed_texts, stats = (
                perturb_texts_probabilistic(
                    texts=X_test,
                    perturbation=perturbation,
                    severity=SEVERITY,
                    seed=seed,
                    return_stats=True,
                )
            )

            predictions = predict_texts(
                model,
                tokenizer,
                perturbed_texts,
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

                    "run":
                        run_number,

                    "seed":
                        seed,

                    "requested_severity":
                        SEVERITY,

                    "realized_severity":
                        stats[
                            "realized_severity"
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
                f"Run {run_number:02d}/"
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
    results = pd.DataFrame(rows)

    raw_path = Path(
        "results/robustness/"
        "distilbert_repeated_20pct_runs.csv"
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
    # Reliability summaries
    # ---------------------------------------------------------
    summaries = []

    for perturbation in PERTURBATIONS:

        subset = results[
            results["perturbation"]
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

        summaries.append(
            {
                "perturbation":
                    perturbation,

                "mean_realized_severity":
                    subset[
                        "realized_severity"
                    ].mean(),

                "mean_accuracy":
                    accuracy_summary[
                        "mean"
                    ],

                "accuracy_std":
                    accuracy_summary[
                        "std"
                    ],

                "min_accuracy":
                    accuracy_summary[
                        "min"
                    ],

                "max_accuracy":
                    accuracy_summary[
                        "max"
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
            }
        )

    summary = pd.DataFrame(
        summaries
    )

    summary_path = Path(
        "results/robustness/"
        "distilbert_repeated_20pct_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    # ---------------------------------------------------------
    # Display
    # ---------------------------------------------------------
    display = summary.copy()

    for column in [
        "mean_realized_severity",
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
    print("=" * 76)
    print("DISTILBERT REPEATED ROBUSTNESS SUMMARY")
    print("=" * 76)

    print(
        display.to_string(
            index=False,
            formatters={
                "mean_realized_severity":
                    "{:.2f}%".format,

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

    elapsed = (
        time.time()
        - start_time
    )

    print(
        f"\nTotal experiment time: "
        f"{elapsed / 60:.2f} minutes"
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