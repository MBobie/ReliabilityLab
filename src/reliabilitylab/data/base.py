"""Common dataset structures used by ReliabilityLab."""

from dataclasses import dataclass


@dataclass
class IntentDataset:
    """Normalized representation of an intent-classification dataset."""

    name: str

    train_texts: list[str]
    train_labels: list[int]

    test_texts: list[str]
    test_labels: list[int]

    validation_texts: list[str] | None = None
    validation_labels: list[int] | None = None

    label_names: list[str] | None = None

    def __post_init__(self):
        """Validate the basic dataset structure."""

        if len(self.train_texts) != len(
            self.train_labels
        ):
            raise ValueError(
                "train_texts and train_labels "
                "must have the same length."
            )

        if len(self.test_texts) != len(
            self.test_labels
        ):
            raise ValueError(
                "test_texts and test_labels "
                "must have the same length."
            )

        if len(self.train_texts) == 0:
            raise ValueError(
                "Training split cannot be empty."
            )

        if len(self.test_texts) == 0:
            raise ValueError(
                "Test split cannot be empty."
            )

        has_validation_texts = (
            self.validation_texts
            is not None
        )

        has_validation_labels = (
            self.validation_labels
            is not None
        )

        if (
            has_validation_texts
            != has_validation_labels
        ):
            raise ValueError(
                "validation_texts and "
                "validation_labels must either "
                "both be provided or both be None."
            )

        if has_validation_texts and len(
            self.validation_texts
        ) != len(
            self.validation_labels
        ):
            raise ValueError(
                "validation_texts and "
                "validation_labels must "
                "have the same length."
            )

    @property
    def num_train(self) -> int:
        """Number of training examples."""

        return len(
            self.train_texts
        )

    @property
    def num_test(self) -> int:
        """Number of test examples."""

        return len(
            self.test_texts
        )

    @property
    def num_validation(self) -> int:
        """Number of validation examples."""

        if self.validation_texts is None:
            return 0

        return len(
            self.validation_texts
        )

    @property
    def num_labels(self) -> int:
        """Number of unique intent labels."""

        labels = set(
            self.train_labels
        ) | set(
            self.test_labels
        )

        if (
            self.validation_labels
            is not None
        ):
            labels |= set(
                self.validation_labels
            )

        return len(
            labels
        )

    def summary(self) -> dict:
        """Return a compact dataset summary."""

        return {
            "name":
                self.name,

            "num_train":
                self.num_train,

            "num_validation":
                self.num_validation,

            "num_test":
                self.num_test,

            "num_labels":
                self.num_labels,
        }