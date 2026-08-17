"""Run a registered model baseline on any registered dataset."""

import argparse
import time

from reliabilitylab.data import (
    available_datasets,
    load_intent_dataset,
)
from reliabilitylab.metrics import (
    classification_metrics,
)
from reliabilitylab.models import (
    available_models,
    build_model,
)


def parse_args():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Run a ReliabilityLab "
            "classification baseline."
        )
    )

    parser.add_argument(
        "--dataset",
        required=True,
        choices=available_datasets(),
        help="Dataset to evaluate.",
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=available_models(),
        help="Model to evaluate.",
    )

    return parser.parse_args()


def main():
    """Run the baseline experiment."""

    args = parse_args()

    print("=" * 72)
    print("ReliabilityLab")
    print("Reusable Model Baseline")
    print("=" * 72)

    print(
        f"\nDataset : "
        f"{args.dataset}"
    )

    print(
        f"Model   : "
        f"{args.model}"
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
        "\nBuilding model..."
    )

    model = build_model(
        args.model
    )

    print(
        "Training model..."
    )

    train_start = time.time()

    model.fit(
        dataset.train_texts,
        dataset.train_labels,
    )

    training_time = (
        time.time()
        - train_start
    )

    # ---------------------------------------------------------
    # Prediction
    # ---------------------------------------------------------
    print(
        "Generating predictions..."
    )

    inference_start = time.time()

    predictions = model.predict(
        dataset.test_texts
    )

    inference_time = (
        time.time()
        - inference_start
    )

    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------
    metrics = classification_metrics(
        dataset.test_labels,
        predictions,
    )

    print("\n" + "=" * 72)
    print("RESULTS")
    print("=" * 72)

    print(
        f"Dataset       : "
        f"{args.dataset}"
    )

    print(
        f"Model         : "
        f"{args.model}"
    )

    print(
        f"Accuracy      : "
        f"{metrics['accuracy'] * 100:.2f}%"
    )

    print(
        f"Macro F1      : "
        f"{metrics['macro_f1'] * 100:.2f}%"
    )

    print(
        f"Training time : "
        f"{training_time:.2f} seconds"
    )

    print(
        f"Inference time: "
        f"{inference_time:.4f} seconds"
    )


if __name__ == "__main__":
    main()