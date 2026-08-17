"""Text robustness experiments for ReliabilityLab."""

from pathlib import Path

import pandas as pd

from reliabilitylab.metrics import (
    classification_metrics,
    relative_robustness_drop,
    robustness_drop,
)
from reliabilitylab.perturbations import perturb_texts

PERTURBATIONS = [
    "typo",
    "char_delete",
    "word_delete",
    "case",
    "punctuation",
]


def run_text_robustness_experiment(
    model,
    test_dataset,
    seed: int = 42,
    save_path=None,
):
    """Evaluate a trained model under text perturbations."""

    X_test = test_dataset["utterance"]
    y_test = test_dataset["label"]

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

    results = [
        {
            "condition": "clean",
            "accuracy": clean_metrics["accuracy"],
            "macro_f1": clean_metrics["macro_f1"],
            "accuracy_drop": 0.0,
            "macro_f1_drop": 0.0,
            "accuracy_relative_drop": 0.0,
        }
    ]

    # ---------------------------------------------------------
    # Perturbed conditions
    # ---------------------------------------------------------
    for perturbation in PERTURBATIONS:

        print(
            f"Evaluating perturbation: "
            f"{perturbation}"
        )

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

        accuracy_drop = robustness_drop(
            clean_metrics["accuracy"],
            metrics["accuracy"],
        )

        macro_f1_drop = robustness_drop(
            clean_metrics["macro_f1"],
            metrics["macro_f1"],
        )

        relative_drop = relative_robustness_drop(
            clean_metrics["accuracy"],
            metrics["accuracy"],
        )

        results.append(
            {
                "condition": perturbation,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "accuracy_drop": accuracy_drop,
                "macro_f1_drop": macro_f1_drop,
                "accuracy_relative_drop":
                    relative_drop,
            }
        )

    results_df = pd.DataFrame(results)

    if save_path is not None:

        save_path = Path(save_path)

        save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        results_df.to_csv(
            save_path,
            index=False,
        )

        print(
            f"\nRobustness results saved to: "
            f"{save_path}"
        )

    return results_df