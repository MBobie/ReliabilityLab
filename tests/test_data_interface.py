"""Tests for ReliabilityLab dataset interfaces."""

import pytest

from reliabilitylab.data import (
    IntentDataset,
    available_datasets,
    load_intent_dataset,
)


def test_available_datasets_contains_banking77():

    assert (
        "banking77"
        in available_datasets()
    )


def test_intent_dataset_validation():

    dataset = IntentDataset(
        name="toy",
        train_texts=[
            "hello",
            "goodbye",
        ],
        train_labels=[
            0,
            1,
        ],
        test_texts=[
            "hi",
        ],
        test_labels=[
            0,
        ],
        label_names=[
            "greeting",
            "farewell",
        ],
    )

    assert dataset.num_train == 2
    assert dataset.num_test == 1
    assert dataset.num_labels == 2


def test_invalid_train_lengths_raise():

    with pytest.raises(
        ValueError
    ):

        IntentDataset(
            name="broken",
            train_texts=[
                "hello",
            ],
            train_labels=[
                0,
                1,
            ],
            test_texts=[
                "test",
            ],
            test_labels=[
                0,
            ],
        )

def test_available_datasets_contains_hwu64():

    assert (
        "hwu64"
        in available_datasets()
    )

def test_available_datasets_contains_clinc150():

    assert (
        "clinc150"
        in available_datasets()
    )
        
def test_unknown_dataset_raises():

    with pytest.raises(
        ValueError
    ):

        load_intent_dataset(
            "does-not-exist"
        )

def test_intent_dataset_validation_split():

    dataset = IntentDataset(
        name="toy",

        train_texts=[
            "hello",
            "goodbye",
        ],

        train_labels=[
            0,
            1,
        ],

        validation_texts=[
            "hey",
        ],

        validation_labels=[
            0,
        ],

        test_texts=[
            "bye",
        ],

        test_labels=[
            1,
        ],
    )

    assert (
        dataset.num_validation
        == 1
    )        
def test_invalid_validation_lengths_raise():

    with pytest.raises(
        ValueError
    ):

        IntentDataset(
            name="broken",

            train_texts=[
                "hello",
            ],

            train_labels=[
                0,
            ],

            validation_texts=[
                "a",
                "b",
            ],

            validation_labels=[
                0,
            ],

            test_texts=[
                "test",
            ],

            test_labels=[
                0,
            ],
        )    