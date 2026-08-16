"""Metrics for robustness evaluation."""


def robustness_drop(
    clean_score: float,
    perturbed_score: float,
) -> float:
    """Calculate absolute performance degradation."""

    return clean_score - perturbed_score


def relative_robustness_drop(
    clean_score: float,
    perturbed_score: float,
) -> float:
    """Calculate degradation relative to clean performance."""

    if clean_score == 0:
        raise ValueError(
            "clean_score must be greater than zero."
        )

    return (
        clean_score - perturbed_score
    ) / clean_score