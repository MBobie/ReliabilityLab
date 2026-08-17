"""Dataset registry for ReliabilityLab."""

from collections.abc import Callable

from .banking77 import (
    load_banking77_intent,
)
from .base import (
    IntentDataset,
)
from .clinc150 import (
    load_clinc150_intent,
)
from .hwu64 import (
    load_hwu64_intent,
)

DatasetLoader = Callable[
    [],
    IntentDataset,
]


_DATASET_REGISTRY: dict[
    str,
    DatasetLoader,
] = {
    "banking77":
        load_banking77_intent,

    "clinc150":
        load_clinc150_intent,

    "hwu64":
        load_hwu64_intent,
}


def available_datasets() -> list[str]:
    """Return the names of registered datasets."""

    return sorted(
        _DATASET_REGISTRY
    )


def load_intent_dataset(
    name: str,
) -> IntentDataset:
    """Load a registered intent-classification dataset."""

    normalized_name = (
        name.strip().lower()
    )

    if normalized_name not in _DATASET_REGISTRY:

        available = ", ".join(
            available_datasets()
        )

        raise ValueError(
            f"Unknown dataset: {name!r}. "
            f"Available datasets: {available}"
        )

    loader = _DATASET_REGISTRY[
        normalized_name
    ]

    return loader()