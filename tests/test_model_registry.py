"""Tests for the ReliabilityLab model registry."""

import pytest

from reliabilitylab.models import (
    available_models,
    build_model,
)


def test_available_models_contains_logistic_regression():

    assert (
        "tfidf_logreg"
        in available_models()
    )


def test_available_models_contains_linear_svm():

    assert (
        "tfidf_svm"
        in available_models()
    )


def test_build_logistic_regression_model():

    model = build_model(
        "tfidf_logreg"
    )

    assert (
        "tfidf"
        in model.named_steps
    )

    assert (
        "classifier"
        in model.named_steps
    )


def test_build_linear_svm_model():

    model = build_model(
        "tfidf_svm"
    )

    assert (
        "tfidf"
        in model.named_steps
    )

    assert (
        "classifier"
        in model.named_steps
    )


def test_unknown_model_raises():

    with pytest.raises(
        ValueError
    ):

        build_model(
            "does-not-exist"
        )