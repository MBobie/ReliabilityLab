"""Validate the new ReliabilityLab dataset abstraction."""

from reliabilitylab.data import (
    load_banking77,
    load_intent_dataset,
)


def main():

    print("=" * 72)
    print("ReliabilityLab")
    print("Dataset Interface Validation")
    print("=" * 72)

    # ---------------------------------------------------------
    # Old interface
    # ---------------------------------------------------------
    print(
        "\nLoading BANKING77 "
        "through original interface..."
    )

    raw = load_banking77()

    raw_train = len(
        raw["train"]
    )

    raw_test = len(
        raw["test"]
    )

    raw_labels = set(
        raw["train"]["label"]
    ) | set(
        raw["test"]["label"]
    )

    print(
        f"Original train samples : "
        f"{raw_train:,}"
    )

    print(
        f"Original test samples  : "
        f"{raw_test:,}"
    )

    print(
        f"Original labels        : "
        f"{len(raw_labels)}"
    )

    # ---------------------------------------------------------
    # New interface
    # ---------------------------------------------------------
    print(
        "\nLoading BANKING77 "
        "through normalized interface..."
    )

    dataset = load_intent_dataset(
        "banking77"
    )

    print(
        f"Dataset name           : "
        f"{dataset.name}"
    )

    print(
        f"Normalized train       : "
        f"{dataset.num_train:,}"
    )

    print(
        f"Normalized test        : "
        f"{dataset.num_test:,}"
    )

    print(
        f"Normalized labels      : "
        f"{dataset.num_labels}"
    )

    print(
        f"Label names available  : "
        f"{dataset.label_names is not None}"
    )

    print(
        "\nFirst training example:"
    )

    print(
        dataset.train_texts[0]
    )

    print(
        f"Label: "
        f"{dataset.train_labels[0]}"
    )

    # ---------------------------------------------------------
    # Regression checks
    # ---------------------------------------------------------
    assert (
        dataset.num_train
        == raw_train
        == 10003
    )

    assert (
        dataset.num_test
        == raw_test
        == 3080
    )

    assert (
        dataset.num_labels
        == len(raw_labels)
        == 77
    )

    assert (
        dataset.train_texts[0]
        == raw["train"][0]["utterance"]
    )

    assert (
        dataset.train_labels[0]
        == raw["train"][0]["label"]
    )

    print("\n" + "=" * 72)
    print(
        "ALL DATASET REGRESSION "
        "CHECKS PASSED"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()