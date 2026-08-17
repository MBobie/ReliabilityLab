"""Generate LR-vs-SVM ReliabilityLab comparison figures."""

from pathlib import Path

import pandas as pd

from reliabilitylab.reporting.classifier_comparison import (
    plot_absolute_classifier_performance,
    plot_classifier_retention,
)

INPUT_PATH = (
    Path("results")
    / "comparison"
    / "tfidf_classifier_model_dataset_summary.csv"
)


def main():
    """Generate classifier-comparison figures."""

    print("=" * 80)
    print("ReliabilityLab")
    print("TF-IDF Classifier Comparison Figures")
    print("=" * 80)

    if not INPUT_PATH.exists():

        raise FileNotFoundError(
            f"Missing comparison file: "
            f"{INPUT_PATH}"
        )

    summary = pd.read_csv(
        INPUT_PATH
    )

    output_dir = (
        Path("results")
        / "figures"
    )

    print(
        "\nGenerating absolute-performance figure..."
    )

    plot_absolute_classifier_performance(
        summary=summary,
        save_path=(
            output_dir
            / "tfidf_lr_vs_svm_absolute_performance.png"
        ),
    )

    print(
        "\nGenerating retention figure..."
    )

    plot_classifier_retention(
        summary=summary,
        save_path=(
            output_dir
            / "tfidf_lr_vs_svm_retention.png"
        ),
    )

    print(
        "\nFigures generated successfully."
    )


if __name__ == "__main__":
    main()