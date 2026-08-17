"""Generate flagship TF-IDF vs DistilBERT comparison figures."""

import numpy as np
import pandas as pd

from reliabilitylab.reporting import (
    plot_model_accuracy_comparison,
    plot_model_degradation_comparison,
)


TFIDF_PATH = (
    "results/robustness/"
    "tfidf_probabilistic_severity_summary.csv"
)

DISTILBERT_PATH = (
    "results/robustness/"
    "distilbert_repeated_20pct_summary.csv"
)


TFIDF_CLEAN = 0.8588
DISTILBERT_CLEAN = 0.8506


def main():

    print("=" * 78)
    print("ReliabilityLab")
    print("TF-IDF vs DistilBERT Visual Comparison")
    print("=" * 78)

    # ---------------------------------------------------------
    # Load summaries
    # ---------------------------------------------------------
    tfidf = pd.read_csv(
        TFIDF_PATH
    )

    distilbert = pd.read_csv(
        DISTILBERT_PATH
    )

    # ---------------------------------------------------------
    # Keep only TF-IDF 20% severity
    # ---------------------------------------------------------
    tfidf = tfidf[
        np.isclose(
            tfidf["severity"],
            0.20,
        )
    ].copy()

    print(
        "\nGenerating model accuracy comparison..."
    )

    plot_model_accuracy_comparison(
        tfidf_summary=tfidf,
        distilbert_summary=distilbert,
        tfidf_clean=TFIDF_CLEAN,
        distilbert_clean=DISTILBERT_CLEAN,
        save_path=(
            "results/figures/"
            "tfidf_vs_distilbert_accuracy.png"
        ),
    )

    print(
        "\nGenerating robustness degradation comparison..."
    )

    plot_model_degradation_comparison(
        tfidf_summary=tfidf,
        distilbert_summary=distilbert,
        save_path=(
            "results/figures/"
            "tfidf_vs_distilbert_degradation.png"
        ),
    )

    print(
        "\nCross-model figures generated successfully."
    )


if __name__ == "__main__":
    main()