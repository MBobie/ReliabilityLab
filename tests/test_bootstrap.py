"""Tests for example-level bootstrap utilities."""

import numpy as np
import pytest

from reliabilitylab.metrics.bootstrap import (
    paired_example_bootstrap,
)


def test_identical_models_have_zero_difference():
    """Identical inputs should produce exactly zero differences."""

    clean = np.array(
        [
            1.0,
            1.0,
            1.0,
            0.0,
            1.0,
            1.0,
        ]
    )

    perturbed = np.array(
        [
            1.0,
            0.5,
            1.0,
            0.0,
            0.5,
            1.0,
        ]
    )

    result = paired_example_bootstrap(
        clean,
        clean,
        perturbed,
        perturbed,
        n_resamples=500,
        seed=42,
    )

    assert result[
        "perturbed_difference"
    ] == pytest.approx(
        0.0
    )

    assert result[
        "retention_difference"
    ] == pytest.approx(
        0.0
    )

    assert result[
        "perturbed_difference_ci_lower"
    ] == pytest.approx(
        0.0
    )

    assert result[
        "perturbed_difference_ci_upper"
    ] == pytest.approx(
        0.0
    )


def test_bootstrap_detects_clear_improvement():
    """A uniformly better perturbed model should have positive effect."""

    clean_a = np.ones(
        20
    )

    clean_b = np.ones(
        20
    )

    perturbed_a = np.zeros(
        20
    )

    perturbed_b = np.ones(
        20
    )

    result = paired_example_bootstrap(
        clean_a,
        clean_b,
        perturbed_a,
        perturbed_b,
        n_resamples=500,
        seed=42,
    )

    assert result[
        "perturbed_difference"
    ] == pytest.approx(
        1.0
    )

    assert result[
        "retention_difference"
    ] == pytest.approx(
        1.0
    )

    assert (
        result[
            "perturbed_difference_ci_lower"
        ]
        > 0.0
    )

    assert (
        result[
            "retention_difference_ci_lower"
        ]
        > 0.0
    )


def test_bootstrap_rejects_mismatched_lengths():
    """All paired arrays must refer to the same examples."""

    with pytest.raises(
        ValueError
    ):
        paired_example_bootstrap(
            [1.0, 0.0],
            [1.0],
            [1.0, 0.0],
            [1.0, 0.0],
        )