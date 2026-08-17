"""Evaluate the saved DistilBERT baseline under text perturbations."""

import time

import numpy as np
import pandas as pd
import torch

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from reliabilitylab.data import load_banking77
from reliabilitylab.metrics import classification_metrics
from reliabilitylab.perturbations import (
    perturb_texts_probabilistic,
)


MODEL_PATH = (
    "results/distilbert/"
    "baseline/final_model"
)

PERTURBATIONS = [
    "typo",
    "char_delete",
    "word_delete",
]

SEVERITY = 0.20
SEED = 42

BATCH_SIZE = 32
MAX_LENGTH = 64


def predict_texts(
    model,
    tokenizer,
    texts,
):
    """Generate predictions for a collection of texts."""

    model.eval()

    predictions = []

    for start in range(
        0,
        len(texts),
        BATCH_SIZE,
    ):

        batch_texts = texts[
            start:start + BATCH_SIZE
        ]

        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        with torch.no_grad():

            outputs = model(
                **encoded
            )

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
    print("DistilBERT Robustness Experiment")
    print("=" * 76)

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------
    print("\nLoading BANKING77...")

    dataset = load_banking77()

    X_test = dataset["test"][
        "utterance"
    ]

    y_test = dataset["test"][
        "label"
    ]

    print(
        f"Test samples: {len(X_test):,}"
    )

    # ---------------------------------------------------------
    # Saved model
    # ---------------------------------------------------------
    print(
        "\nLoading saved DistilBERT model..."
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
    # Clean evaluation
    # ---------------------------------------------------------
    print(
        "\nEvaluating clean test set..."
    )

    start_time = time.time()

    clean_predictions = predict_texts(
        model=model,
        tokenizer=tokenizer,
        texts=X_test,
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

    rows.append(
        {
            "model": "DistilBERT",
            "condition": "clean",
            "severity": 0.0,
            "realized_severity": 0.0,
            "accuracy":
                clean_accuracy,
            "macro_f1":
                clean_macro_f1,
            "accuracy_drop":
                0.0,
        }
    )

    # ---------------------------------------------------------
    # Perturbation evaluation
    # ---------------------------------------------------------
    for perturbation in PERTURBATIONS:

        print("\n" + "#" * 76)

        print(
            f"PERTURBATION: "
            f"{perturbation.upper()}"
        )

        print(
            f"Requested severity: "
            f"{SEVERITY * 100:.0f}%"
        )

        print("#" * 76)

        perturbed_texts, stats = (
            perturb_texts_probabilistic(
                texts=X_test,
                perturbation=perturbation,
                severity=SEVERITY,
                seed=SEED,
                return_stats=True,
            )
        )

        print(
            f"Realized severity: "
            f"{stats['realized_severity'] * 100:.2f}%"
        )

        predictions = predict_texts(
            model=model,
            tokenizer=tokenizer,
            texts=perturbed_texts,
        )

        metrics = classification_metrics(
            y_true=y_test,
            y_pred=predictions,
        )

        accuracy_drop = (
            clean_accuracy
            - metrics["accuracy"]
        )

        print(
            f"Accuracy: "
            f"{metrics['accuracy'] * 100:.2f}%"
        )

        print(
            f"Macro F1: "
            f"{metrics['macro_f1'] * 100:.2f}%"
        )

        print(
            f"Accuracy drop: "
            f"{accuracy_drop * 100:.2f} pp"
        )

        rows.append(
            {
                "model":
                    "DistilBERT",

                "condition":
                    perturbation,

                "severity":
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
            }
        )

    # ---------------------------------------------------------
    # Results
    # ---------------------------------------------------------
    results = pd.DataFrame(
        rows
    )

    save_path = (
        "results/robustness/"
        "distilbert_20pct_robustness.csv"
    )

    results.to_csv(
        save_path,
        index=False,
    )

    elapsed = (
        time.time()
        - start_time
    )

    display = results.copy()

    display[
        "realized_severity"
    ] *= 100

    display[
        "accuracy"
    ] *= 100

    display[
        "macro_f1"
    ] *= 100

    display[
        "accuracy_drop"
    ] *= 100

    print("\n")
    print("=" * 76)
    print("DISTILBERT ROBUSTNESS SUMMARY")
    print("=" * 76)

    print(
        display.to_string(
            index=False,
            formatters={
                "realized_severity":
                    "{:.2f}%".format,

                "accuracy":
                    "{:.2f}%".format,

                "macro_f1":
                    "{:.2f}%".format,

                "accuracy_drop":
                    "{:.2f} pp".format,
            },
        )
    )

    print(
        f"\nTotal evaluation time: "
        f"{elapsed / 60:.2f} minutes"
    )

    print(
        f"\nResults saved to:\n"
        f"{save_path}"
    )


if __name__ == "__main__":
    main()