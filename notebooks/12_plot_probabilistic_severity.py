"""Generate probabilistic robustness-severity figures."""

import pandas as pd

from reliabilitylab.reporting import (
    plot_severity_accuracy,
    plot_severity_drop,
)


SUMMARY_PATH = (
    "results/robustness/"
    "tfidf_probabilistic_severity_summary.csv"
)

CLEAN_ACCURACY = 0.8588


def main():

    print("=" * 72)
    print("ReliabilityLab")
    print("Probabilistic Severity Visualisation")
    print("=" * 72)

    print("\nLoading severity summary...")

    summary = pd.read_csv(
        SUMMARY_PATH
    )

    print(
        "\nGenerating accuracy degradation curve..."
    )

    plot_severity_accuracy(
        summary_df=summary,
        clean_accuracy=CLEAN_ACCURACY,
        save_path=(
            "results/figures/"
            "tfidf_probabilistic_severity_accuracy.png"
        ),
    )

    print(
        "\nGenerating robustness failure curve..."
    )

    plot_severity_drop(
        summary_df=summary,
        save_path=(
            "results/figures/"
            "tfidf_probabilistic_severity_drop.png"
        ),
    )

    print(
        "\nSeverity figures generated successfully."
    )


if __name__ == "__main__":
    main()