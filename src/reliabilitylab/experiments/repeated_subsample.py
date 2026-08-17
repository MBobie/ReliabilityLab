"""Repeated stratified subsampling experiments."""

from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from reliabilitylab.metrics import classification_metrics
from reliabilitylab.models import build_tfidf_logreg


def run_repeated_subsample_experiment(
    train_dataset,
    test_dataset,
    seeds: Sequence[int],
    train_fraction: float = 0.8,
    save_path: str | Path | None = None,
) -> pd.DataFrame:
    """Run repeated training experiments on stratified subsamples.

    Parameters
    ----------
    train_dataset
        Hugging Face training dataset containing
        "utterance" and "label".
    test_dataset
        Fixed Hugging Face test dataset.
    seeds
        Random seeds controlling the training subset selected
        in each experiment.
    train_fraction
        Fraction of the original training set used per run.
    save_path
        Optional CSV path for saving run-level results.

    Returns
    -------
    pandas.DataFrame
        One row per experiment containing seed, sample size,
        accuracy, and macro F1.
    """

    if not 0 < train_fraction <= 1:
        raise ValueError(
            "train_fraction must be greater than 0 and at most 1."
        )

    X_train_full = train_dataset["utterance"]
    y_train_full = train_dataset["label"]

    X_test = test_dataset["utterance"]
    y_test = test_dataset["label"]

    indices = list(range(len(train_dataset)))

    results = []

    for run_number, seed in enumerate(seeds, start=1):

        print(
            f"Run {run_number:02d}/{len(seeds)} "
            f"| seed={seed}"
        )

        if train_fraction < 1.0:
            selected_indices, _ = train_test_split(
                indices,
                train_size=train_fraction,
                stratify=y_train_full,
                random_state=seed,
            )
        else:
            selected_indices = indices

        X_train = [
            X_train_full[i]
            for i in selected_indices
        ]

        y_train = [
            y_train_full[i]
            for i in selected_indices
        ]

        model = build_tfidf_logreg()

        model.fit(
            X_train,
            y_train,
        )

        predictions = model.predict(X_test)

        metrics = classification_metrics(
            y_true=y_test,
            y_pred=predictions,
        )

        result = {
            "run": run_number,
            "seed": seed,
            "train_fraction": train_fraction,
            "train_samples": len(X_train),
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
        }

        results.append(result)

        print(
            f"    samples={len(X_train):,} "
            f"| accuracy={metrics['accuracy']:.4f} "
            f"| macro_f1={metrics['macro_f1']:.4f}"
        )

    results_df = pd.DataFrame(results)

    if save_path is not None:

        save_path = Path(save_path)

        save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        results_df.to_csv(
            save_path,
            index=False,
        )

        print(
            f"\nRun-level results saved to: {save_path}"
        )

    return results_df