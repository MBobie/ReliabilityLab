"""Model registry for ReliabilityLab."""

from collections.abc import Callable

from sklearn.pipeline import Pipeline

from .tfidf_logreg import (
    build_tfidf_logreg,
)
from .tfidf_svm import (
    build_tfidf_svm,
)

ModelBuilder = Callable[
    [],
    Pipeline,
]


_MODEL_REGISTRY: dict[
    str,
    ModelBuilder,
] = {
    "tfidf_logreg":
        build_tfidf_logreg,

    "tfidf_svm":
        build_tfidf_svm,
}


def available_models() -> list[str]:
    """Return registered model names."""

    return sorted(
        _MODEL_REGISTRY
    )


def build_model(
    name: str,
):
    """Build a registered ReliabilityLab model."""

    normalized_name = (
        name.strip().lower()
    )

    if normalized_name not in _MODEL_REGISTRY:

        available = ", ".join(
            available_models()
        )

        raise ValueError(
            f"Unknown model: {name!r}. "
            f"Available models: {available}"
        )

    builder = _MODEL_REGISTRY[
        normalized_name
    ]

    return builder()