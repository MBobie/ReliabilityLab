"""HWU64 dataset loader."""

from datasets import DatasetDict, load_dataset

from .base import IntentDataset

DATASET_NAME = "DeepPavlov/hwu64"


def load_hwu64_raw() -> DatasetDict:
    """Load the raw HWU64 Hugging Face dataset."""

    return load_dataset(
        DATASET_NAME
    )


def load_hwu64_intent() -> IntentDataset:
    """Load HWU64 through the normalized ReliabilityLab interface."""

    dataset = load_hwu64_raw()

    train = dataset["train"]
    test = dataset["test"]

    train_texts = [
        str(text)
        for text in train["utterance"]
    ]

    train_labels = [
        int(label)
        for label in train["label"]
    ]

    test_texts = [
        str(text)
        for text in test["utterance"]
    ]

    test_labels = [
        int(label)
        for label in test["label"]
    ]

    return IntentDataset(
        name="hwu64",

        train_texts=
            train_texts,

        train_labels=
            train_labels,

        test_texts=
            test_texts,

        test_labels=
            test_labels,

        validation_texts=None,

        validation_labels=None,

        label_names=None,
    )