"""DistilBERT utilities for BANKING77 intent classification."""

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

MODEL_NAME = "distilbert-base-uncased"


def load_distilbert_components(
    num_labels: int = 77,
):
    """Load the DistilBERT tokenizer and classification model."""

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
    )

    return tokenizer, model