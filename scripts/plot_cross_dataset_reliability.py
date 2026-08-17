"""Generate cross-dataset ReliabilityLab figures."""

from pathlib import Path

import pandas as pd

from reliabilitylab.reporting import (
    plot_clean_vs_retention,
    plot_cross_dataset_retention,
)

INPUT_PATH = (
    Path("results")
    / "comparison"
    / "tfidf_cross_dataset_20pct.csv"
)


def main():
    """Generate cross-dataset comparison figures."""

    print("=" * 78)
    print("ReliabilityLab")
    print("Cross-Dataset Reliability Figures")
    print("=" * 78)

    if not INPUT_PATH.exists():

        raise FileNotFoundError(
            f"Missing comparison file: "
            f"{INPUT_PATH}"
        )

    comparison = pd.read_csv(
        INPUT_PATH
    )

    print(
        "\nGenerating retention figure..."
    )

    plot_cross_dataset_retention(
        comparison=comparison,
        save_path=(
            "results/figures/"
            "tfidf_cross_dataset_retention.png"
        ),
    )

    print(
        "\nGenerating clean-versus-retention figure..."
    )

    plot_clean_vs_retention(
        comparison=comparison,
        save_path=(
            "results/figures/"
            "tfidf_clean_vs_retention.png"
        ),
    )

    print(
        "\nFigures generated successfully."
    )


if __name__ == "__main__":
    main()