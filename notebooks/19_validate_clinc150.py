"""Validate CLINC150 inside ReliabilityLab."""

from reliabilitylab.data import (
    load_clinc150_raw,
    load_intent_dataset,
)


def count_unlabeled(split):

    return sum(
        label is None
        for label in split["label"]
    )


def main():

    print("=" * 72)
    print("ReliabilityLab")
    print("CLINC150 Dataset Validation")
    print("=" * 72)

    print(
        "\nLoading raw CLINC150..."
    )

    raw = load_clinc150_raw()

    print(
        "\nRaw split sizes:"
    )

    for split_name in [
        "train",
        "validation",
        "test",
    ]:

        split = raw[
            split_name
        ]

        print(
            f"{split_name:<12}: "
            f"{len(split):,} "
            f"| unlabeled/OOS: "
            f"{count_unlabeled(split):,}"
        )

    print(
        "\nLoading normalized "
        "closed-set CLINC150..."
    )

    dataset = (
        load_intent_dataset(
            "clinc150"
        )
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

    assert (
        dataset.num_labels
        == 150
    )

    assert all(
        label is not None
        for label in dataset.train_labels
    )

    assert all(
        label is not None
        for label in dataset.validation_labels
    )

    assert all(
        label is not None
        for label in dataset.test_labels
    )

    print("\n" + "=" * 72)

    print(
        "CLINC150 VALIDATION PASSED"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()