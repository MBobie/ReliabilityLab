"""Generate ReliabilityLab data-stability figures."""

import pandas as pd

from reliabilitylab.reporting import (
    plot_data_stability_curve,
    plot_subset_instability,
)


SUMMARY_PATH = (
    "results/data_stability/"
    "tfidf_data_stability_summary.csv"
)


def main():

    print("=" * 65)
    print("ReliabilityLab — Data Stability Visualisation")
    print("=" * 65)

    print("\nLoading experiment summary...")

    summary = pd.read_csv(
        SUMMARY_PATH
    )

    print(summary)

    print(
        "\nGenerating performance curve..."
    )

    plot_data_stability_curve(
        summary_df=summary,
        save_path=(
            "results/figures/"
            "tfidf_data_stability_curve.png"
        ),
    )

    print(
        "\nGenerating subset-sensitivity curve..."
    )

    plot_subset_instability(
        summary_df=summary,
        save_path=(
            "results/figures/"
            "tfidf_subset_instability.png"
        ),
    )

    print(
        "\nAll figures generated successfully."
    )


if __name__ == "__main__":
    main()