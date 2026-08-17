"""BANKING77 dataset loaders."""

from datasets import DatasetDict, load_dataset

from .base import IntentDataset

DATASET_NAME = "DeepPavlov/banking77"


def load_banking77() -> DatasetDict:
    """Load the raw BANKING77 Hugging Face DatasetDict.

    This function is retained for backward compatibility with the
    original ReliabilityLab experiments.
    """

    return load_dataset(
        DATASET_NAME
    )


def load_banking77_intent() -> IntentDataset:
    """Load BANKING77 using the normalized ReliabilityLab interface."""

    dataset = load_banking77()

    train = dataset["train"]
    test = dataset["test"]

    train_texts = list(
        train["utterance"]
    )

    train_labels = [
        int(label)
        for label in train["label"]
    ]

    test_texts = list(
        test["utterance"]
    )

    test_labels = [
        int(label)
        for label in test["label"]
    ]

    # Hugging Face datasets sometimes expose
    # human-readable class names through ClassLabel.
    label_feature = (
        train.features.get(
            "label"
        )
    )

    label_names = getattr(
        label_feature,
        "names",
        None,
    )

    if label_names is not None:
        label_names = list(
            label_names
        )

    return IntentDataset(
        name="banking77",
        train_texts=train_texts,
        train_labels=train_labels,
        test_texts=test_texts,
        test_labels=test_labels,
        label_names=label_names,
    )