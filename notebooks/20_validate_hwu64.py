"""Validate HWU64 inside ReliabilityLab."""

from reliabilitylab.data import (
    load_hwu64_raw,
    load_intent_dataset,
)


def main():
    """Validate raw and normalized HWU64 data."""

    print("=" * 72)
    print("ReliabilityLab")
    print("HWU64 Dataset Validation")
    print("=" * 72)

    # ---------------------------------------------------------
    # Raw dataset
    # ---------------------------------------------------------
    print(
        "\nLoading raw HWU64..."
    )

    raw = load_hwu64_raw()

    print(
        "\nAvailable raw splits:"
    )

    for split_name in raw:

        split = raw[
            split_name
        ]

        print(
            f"{split_name:<12}: "
            f"{len(split):,}"
        )

    raw_train = len(
        raw["train"]
    )

    raw_test = len(
        raw["test"]
    )

    raw_labels = (
        set(
            raw["train"]["label"]
        )
        | set(
            raw["test"]["label"]
        )
    )

    print(
        f"\nRaw training samples : "
        f"{raw_train:,}"
    )

    print(
        f"Raw test samples     : "
        f"{raw_test:,}"
    )

    print(
        f"Raw intent classes   : "
        f"{len(raw_labels)}"
    )

    # ---------------------------------------------------------
    # Normalized dataset
    # ---------------------------------------------------------
    print(
        "\nLoading normalized HWU64..."
    )

    dataset = load_intent_dataset(
        "hwu64"
    )

    print(
        f"\nDataset name       : "
        f"{dataset.name}"
    )

    print(
        f"Train samples      : "
        f"{dataset.num_train:,}"
    )

    print(
        f"Validation samples : "
        f"{dataset.num_validation:,}"
    )

    print(
        f"Test samples       : "
        f"{dataset.num_test:,}"
    )

    print(
        f"Intent classes     : "
        f"{dataset.num_labels}"
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
        == 8954
    )

    assert (
        dataset.num_test
        == raw_test
        == 1076
    )

    assert (
        dataset.num_validation
        == 0
    )

    assert (
        dataset.num_labels
        == len(raw_labels)
        == 64
    )

    assert (
        min(
            dataset.train_labels
            + dataset.test_labels
        )
        == 0
    )

    assert (
        max(
            dataset.train_labels
            + dataset.test_labels
        )
        == 63
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
        "HWU64 VALIDATION PASSED"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()