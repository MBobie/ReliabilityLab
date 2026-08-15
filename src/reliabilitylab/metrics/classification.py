"""Classification metrics used by ReliabilityLab."""

from typing import Dict

from sklearn.metrics import accuracy_score, f1_score


def classification_metrics(
    y_true,
    y_pred,
) -> Dict[str, float]:
    """Calculate core classification metrics."""

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            average="macro",
        ),
    }