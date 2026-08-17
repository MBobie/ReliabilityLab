"""CLINC150 dataset loader."""

from datasets import Dataset, DatasetDict, load_dataset

from .base import IntentDataset

DATASET_NAME = "DeepPavlov/clinc150"


def load_clinc150_raw() -> DatasetDict:
    """Load the raw DeepPavlov CLINC150 dataset."""

    return load_dataset(
        DATASET_NAME
    )


def _extract_in_domain(
    split: Dataset,
) -> tuple[
    list[str],
    list[int],
]:
    """Extract labeled in-domain examples.

    CLINC150 also contains out-of-scope
    examples whose label is null. Those are
    excluded for the closed-set 150-intent
    classification benchmark.
    """

    texts = []
    labels = []

    for text, label in zip(
        split["utterance"],
        split["label"],
        strict=True,
    ):

        if label is None:
            continue

        texts.append(
            str(text)
        )

        labels.append(
            int(label)
        )

    return (
        texts,
        labels,
    )


def load_clinc150_intent() -> IntentDataset:
    """Load CLINC150 through the normalized ReliabilityLab interface."""

    dataset = load_clinc150_raw()

    (
        train_texts,
        train_labels,
    ) = _extract_in_domain(
        dataset["train"]
    )

    (
        validation_texts,
        validation_labels,
    ) = _extract_in_domain(
        dataset["validation"]
    )

    (
        test_texts,
        test_labels,
    ) = _extract_in_domain(
        dataset["test"]
    )

    return IntentDataset(
        name="clinc150",

        train_texts=
            train_texts,

        train_labels=
            train_labels,

        validation_texts=
            validation_texts,

        validation_labels=
            validation_labels,

        test_texts=
            test_texts,

        test_labels=
            test_labels,

        label_names=None,
    )