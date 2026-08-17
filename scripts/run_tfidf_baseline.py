"""Run TF-IDF + Logistic Regression on any registered dataset."""

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
    build_tfidf_logreg,
)


def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Run the ReliabilityLab "
            "TF-IDF baseline."
        )
    )

    parser.add_argument(
        "--dataset",
        required=True,
        choices=available_datasets(),
    )

    return parser.parse_args()


def main():

    args = parse_args()

    print("=" * 72)
    print("ReliabilityLab")
    print("Reusable TF-IDF Baseline")
    print("=" * 72)

    print(
        f"\nDataset: "
        f"{args.dataset}"
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

    model = (
        build_tfidf_logreg()
    )

    print(
        "\nTraining model..."
    )

    start = time.time()

    model.fit(
        dataset.train_texts,
        dataset.train_labels,
    )

    training_time = (
        time.time()
        - start
    )

    print(
        "Generating predictions..."
    )

    predictions = model.predict(
        dataset.test_texts
    )

    metrics = (
        classification_metrics(
            dataset.test_labels,
            predictions,
        )
    )

    print("\n" + "=" * 72)
    print("RESULTS")
    print("=" * 72)

    print(
        f"Accuracy : "
        f"{metrics['accuracy'] * 100:.2f}%"
    )

    print(
        f"Macro F1 : "
        f"{metrics['macro_f1'] * 100:.2f}%"
    )

    print(
        f"Training time : "
        f"{training_time:.2f} seconds"
    )


if __name__ == "__main__":
    main()