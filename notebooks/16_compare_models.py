"""Formal ReliabilityLab comparison of TF-IDF and DistilBERT."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


TFIDF_PATH = (
    "results/robustness/"
    "tfidf_probabilistic_severity_runs.csv"
)

DISTILBERT_PATH = (
    "results/robustness/"
    "distilbert_repeated_20pct_runs.csv"
)


PERTURBATIONS = [
    "typo",
    "char_delete",
    "word_delete",
]


def main():

    print("=" * 80)
    print("ReliabilityLab")
    print("Paired Cross-Model Robustness Comparison")
    print("=" * 80)

    # ---------------------------------------------------------
    # Load data
    # ---------------------------------------------------------
    tfidf = pd.read_csv(
        TFIDF_PATH
    )

    distilbert = pd.read_csv(
        DISTILBERT_PATH
    )

    # ---------------------------------------------------------
    # Keep TF-IDF 20% experiments only
    # ---------------------------------------------------------
    tfidf = tfidf[
        np.isclose(
            tfidf["severity"],
            0.20,
        )
    ].copy()

    summary_rows = []
    paired_rows = []

    # ---------------------------------------------------------
    # Compare each perturbation
    # ---------------------------------------------------------
    for perturbation in PERTURBATIONS:

        tf = tfidf[
            tfidf["perturbation"]
            == perturbation
        ][
            [
                "seed",
                "accuracy",
                "accuracy_drop",
                "realized_severity",
            ]
        ].copy()

        db = distilbert[
            distilbert["perturbation"]
            == perturbation
        ][
            [
                "seed",
                "accuracy",
                "accuracy_drop",
                "realized_severity",
            ]
        ].copy()

        tf = tf.rename(
            columns={
                "accuracy":
                    "tfidf_accuracy",

                "accuracy_drop":
                    "tfidf_drop",

                "realized_severity":
                    "tfidf_realized_severity",
            }
        )

        db = db.rename(
            columns={
                "accuracy":
                    "distilbert_accuracy",

                "accuracy_drop":
                    "distilbert_drop",

                "realized_severity":
                    "distilbert_realized_severity",
            }
        )

        paired = pd.merge(
            tf,
            db,
            on="seed",
            how="inner",
        )

        paired[
            "perturbation"
        ] = perturbation

        # Positive = TF-IDF higher accuracy
        paired[
            "accuracy_difference"
        ] = (
            paired["tfidf_accuracy"]
            - paired["distilbert_accuracy"]
        )

        # Positive = DistilBERT degraded more
        paired[
            "extra_distilbert_drop"
        ] = (
            paired["distilbert_drop"]
            - paired["tfidf_drop"]
        )

        paired_rows.append(
            paired
        )

        differences = paired[
            "accuracy_difference"
        ].to_numpy()

        n = len(differences)

        mean_difference = (
            differences.mean()
        )

        difference_std = (
            differences.std(ddof=1)
        )

        standard_error = (
            difference_std
            / np.sqrt(n)
        )

        t_critical = stats.t.ppf(
            0.975,
            df=n - 1,
        )

        ci_lower = (
            mean_difference
            - t_critical
            * standard_error
        )

        ci_upper = (
            mean_difference
            + t_critical
            * standard_error
        )

        # Paired t-test
        t_statistic, p_value = (
            stats.ttest_rel(
                paired["tfidf_accuracy"],
                paired["distilbert_accuracy"],
            )
        )

        summary_rows.append(
            {
                "perturbation":
                    perturbation,

                "n_pairs":
                    n,

                "tfidf_mean_accuracy":
                    paired[
                        "tfidf_accuracy"
                    ].mean(),

                "distilbert_mean_accuracy":
                    paired[
                        "distilbert_accuracy"
                    ].mean(),

                "mean_accuracy_difference":
                    mean_difference,

                "difference_std":
                    difference_std,

                "ci_95_lower":
                    ci_lower,

                "ci_95_upper":
                    ci_upper,

                "tfidf_mean_drop":
                    paired[
                        "tfidf_drop"
                    ].mean(),

                "distilbert_mean_drop":
                    paired[
                        "distilbert_drop"
                    ].mean(),

                "extra_distilbert_drop":
                    paired[
                        "extra_distilbert_drop"
                    ].mean(),

                "paired_t":
                    t_statistic,

                "p_value":
                    p_value,
            }
        )

    # ---------------------------------------------------------
    # Save paired observations
    # ---------------------------------------------------------
    paired_results = pd.concat(
        paired_rows,
        ignore_index=True,
    )

    paired_path = Path(
        "results/comparison/"
        "tfidf_vs_distilbert_paired_runs.csv"
    )

    paired_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    paired_results.to_csv(
        paired_path,
        index=False,
    )

    # ---------------------------------------------------------
    # Save summary
    # ---------------------------------------------------------
    summary = pd.DataFrame(
        summary_rows
    )

    summary_path = Path(
        "results/comparison/"
        "tfidf_vs_distilbert_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    # ---------------------------------------------------------
    # Display
    # ---------------------------------------------------------
    display = summary.copy()

    percentage_columns = [
        "tfidf_mean_accuracy",
        "distilbert_mean_accuracy",
        "mean_accuracy_difference",
        "difference_std",
        "ci_95_lower",
        "ci_95_upper",
        "tfidf_mean_drop",
        "distilbert_mean_drop",
        "extra_distilbert_drop",
    ]

    for column in percentage_columns:
        display[column] *= 100

    print("\n")
    print("=" * 80)
    print("PAIRED MODEL COMPARISON")
    print("=" * 80)

    print(
        display.to_string(
            index=False,
            formatters={
                "tfidf_mean_accuracy":
                    "{:.2f}%".format,

                "distilbert_mean_accuracy":
                    "{:.2f}%".format,

                "mean_accuracy_difference":
                    "{:+.2f} pp".format,

                "difference_std":
                    "{:.2f} pp".format,

                "ci_95_lower":
                    "{:+.2f} pp".format,

                "ci_95_upper":
                    "{:+.2f} pp".format,

                "tfidf_mean_drop":
                    "{:.2f} pp".format,

                "distilbert_mean_drop":
                    "{:.2f} pp".format,

                "extra_distilbert_drop":
                    "{:+.2f} pp".format,

                "paired_t":
                    "{:.3f}".format,

                "p_value":
                    "{:.3e}".format,
            },
        )
    )

    print(
        f"\nPaired results saved to:\n"
        f"{paired_path}"
    )

    print(
        f"\nComparison summary saved to:\n"
        f"{summary_path}"
    )


if __name__ == "__main__":
    main()