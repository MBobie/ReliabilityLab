"""Bootstrap utilities for ReliabilityLab."""

from collections.abc import Sequence

import numpy as np


def _as_1d_float_array(
    values: Sequence[float] | np.ndarray,
    name: str,
) -> np.ndarray:
    """Convert values to a validated one-dimensional float array."""

    array = np.asarray(
        values,
        dtype=float,
    )

    if array.ndim != 1:
        raise ValueError(
            f"{name} must be one-dimensional."
        )

    if array.size == 0:
        raise ValueError(
            f"{name} must not be empty."
        )

    if not np.all(
        np.isfinite(array)
    ):
        raise ValueError(
            f"{name} contains non-finite values."
        )

    return array


def paired_example_bootstrap(
    clean_a: Sequence[float] | np.ndarray,
    clean_b: Sequence[float] | np.ndarray,
    perturbed_a: Sequence[float] | np.ndarray,
    perturbed_b: Sequence[float] | np.ndarray,
    *,
    n_resamples: int = 5000,
    confidence: float = 0.95,
    seed: int = 2026,
    batch_size: int = 250,
) -> dict[str, float | int]:
    """Paired bootstrap over examples.

    The four input arrays must refer to the same examples.

    ``clean_a`` and ``clean_b`` contain per-example clean correctness
    scores for two models.

    ``perturbed_a`` and ``perturbed_b`` may contain either binary
    correctness values or seed-averaged correctness rates.

    Examples are resampled jointly, preserving pairing between models
    and between clean and perturbed outcomes.
    """

    clean_a_array = _as_1d_float_array(
        clean_a,
        "clean_a",
    )

    clean_b_array = _as_1d_float_array(
        clean_b,
        "clean_b",
    )

    perturbed_a_array = _as_1d_float_array(
        perturbed_a,
        "perturbed_a",
    )

    perturbed_b_array = _as_1d_float_array(
        perturbed_b,
        "perturbed_b",
    )

    lengths = {
        clean_a_array.size,
        clean_b_array.size,
        perturbed_a_array.size,
        perturbed_b_array.size,
    }

    if len(lengths) != 1:
        raise ValueError(
            "All arrays must contain the same number of examples."
        )

    if n_resamples < 1:
        raise ValueError(
            "n_resamples must be at least 1."
        )

    if batch_size < 1:
        raise ValueError(
            "batch_size must be at least 1."
        )

    if not 0.0 < confidence < 1.0:
        raise ValueError(
            "confidence must be between 0 and 1."
        )

    n_examples = (
        clean_a_array.size
    )

    clean_a_mean = float(
        clean_a_array.mean()
    )

    clean_b_mean = float(
        clean_b_array.mean()
    )

    perturbed_a_mean = float(
        perturbed_a_array.mean()
    )

    perturbed_b_mean = float(
        perturbed_b_array.mean()
    )

    if clean_a_mean <= 0.0:
        raise ValueError(
            "Mean clean performance for model A must be positive."
        )

    if clean_b_mean <= 0.0:
        raise ValueError(
            "Mean clean performance for model B must be positive."
        )

    clean_difference = (
        clean_b_mean
        - clean_a_mean
    )

    perturbed_difference = (
        perturbed_b_mean
        - perturbed_a_mean
    )

    retention_a = (
        perturbed_a_mean
        / clean_a_mean
    )

    retention_b = (
        perturbed_b_mean
        / clean_b_mean
    )

    retention_difference = (
        retention_b
        - retention_a
    )

    rng = np.random.default_rng(
        seed
    )

    accuracy_difference_draws = (
        np.empty(
            n_resamples,
            dtype=float,
        )
    )

    retention_difference_draws = (
        np.empty(
            n_resamples,
            dtype=float,
        )
    )

    position = 0

    while position < n_resamples:

        current_batch = min(
            batch_size,
            n_resamples - position,
        )

        indices = rng.integers(
            low=0,
            high=n_examples,
            size=(
                current_batch,
                n_examples,
            ),
        )

        clean_a_draw = (
            clean_a_array[
                indices
            ].mean(
                axis=1
            )
        )

        clean_b_draw = (
            clean_b_array[
                indices
            ].mean(
                axis=1
            )
        )

        perturbed_a_draw = (
            perturbed_a_array[
                indices
            ].mean(
                axis=1
            )
        )

        perturbed_b_draw = (
            perturbed_b_array[
                indices
            ].mean(
                axis=1
            )
        )

        accuracy_difference_batch = (
            perturbed_b_draw
            - perturbed_a_draw
        )

        retention_a_draw = np.divide(
            perturbed_a_draw,
            clean_a_draw,
            out=np.full(
                current_batch,
                np.nan,
                dtype=float,
            ),
            where=clean_a_draw > 0.0,
        )

        retention_b_draw = np.divide(
            perturbed_b_draw,
            clean_b_draw,
            out=np.full(
                current_batch,
                np.nan,
                dtype=float,
            ),
            where=clean_b_draw > 0.0,
        )

        retention_difference_batch = (
            retention_b_draw
            - retention_a_draw
        )

        end = (
            position
            + current_batch
        )

        accuracy_difference_draws[
            position:end
        ] = accuracy_difference_batch

        retention_difference_draws[
            position:end
        ] = retention_difference_batch

        position = end

    alpha = (
        1.0
        - confidence
    )

    lower_quantile = (
        alpha
        / 2.0
    )

    upper_quantile = (
        1.0
        - alpha / 2.0
    )

    accuracy_lower = float(
        np.quantile(
            accuracy_difference_draws,
            lower_quantile,
        )
    )

    accuracy_upper = float(
        np.quantile(
            accuracy_difference_draws,
            upper_quantile,
        )
    )

    valid_retention_draws = (
        retention_difference_draws[
            np.isfinite(
                retention_difference_draws
            )
        ]
    )

    if valid_retention_draws.size == 0:
        raise RuntimeError(
            "No valid retention bootstrap draws were produced."
        )

    retention_lower = float(
        np.quantile(
            valid_retention_draws,
            lower_quantile,
        )
    )

    retention_upper = float(
        np.quantile(
            valid_retention_draws,
            upper_quantile,
        )
    )

    return {
        "n_examples":
            int(
                n_examples
            ),

        "n_resamples":
            int(
                n_resamples
            ),

        "clean_a":
            clean_a_mean,

        "clean_b":
            clean_b_mean,

        "clean_difference":
            clean_difference,

        "perturbed_a":
            perturbed_a_mean,

        "perturbed_b":
            perturbed_b_mean,

        "perturbed_difference":
            perturbed_difference,

        "perturbed_difference_ci_lower":
            accuracy_lower,

        "perturbed_difference_ci_upper":
            accuracy_upper,

        "retention_a":
            retention_a,

        "retention_b":
            retention_b,

        "retention_difference":
            retention_difference,

        "retention_difference_ci_lower":
            retention_lower,

        "retention_difference_ci_upper":
            retention_upper,

        "bootstrap_fraction_accuracy_difference_gt_zero":
            float(
                np.mean(
                    accuracy_difference_draws
                    > 0.0
                )
            ),

        "bootstrap_fraction_retention_difference_gt_zero":
            float(
                np.mean(
                    valid_retention_draws
                    > 0.0
                )
            ),
    }