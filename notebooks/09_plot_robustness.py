"""Generate ReliabilityLab robustness figures."""

import pandas as pd

from reliabilitylab.reporting import (
    plot_robustness_accuracy,
    plot_robustness_drop,
)


REPEATED_SUMMARY_PATH = (
    "results/robustness/"
    "tfidf_repeated_robustness_summary.csv"
)

CONTROL_RESULTS_PATH = (
    "results/robustness/"
    "tfidf_text_robustness.csv"
)


def main():

    print("=" * 70)
    print("ReliabilityLab — Robustness Visualisation")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load results
    # ---------------------------------------------------------
    repeated = pd.read_csv(
        REPEATED_SUMMARY_PATH
    )

    controls = pd.read_csv(
        CONTROL_RESULTS_PATH
    )

    # ---------------------------------------------------------
    # Extract deterministic reference/control values
    # ---------------------------------------------------------
    clean_accuracy = controls.loc[
        controls["condition"] == "clean",
        "accuracy",
    ].iloc[0]

    case_accuracy = controls.loc[
        controls["condition"] == "case",
        "accuracy",
    ].iloc[0]

    punctuation_accuracy = controls.loc[
        controls["condition"] == "punctuation",
        "accuracy",
    ].iloc[0]

    print(
        f"\nClean accuracy: "
        f"{clean_accuracy * 100:.2f}%"
    )

    print(
        f"Case accuracy: "
        f"{case_accuracy * 100:.2f}%"
    )

    print(
        f"Punctuation accuracy: "
        f"{punctuation_accuracy * 100:.2f}%"
    )

    # ---------------------------------------------------------
    # Figure 1
    # ---------------------------------------------------------
    print(
        "\nGenerating robustness accuracy figure..."
    )

    plot_robustness_accuracy(
        repeated_summary=repeated,
        clean_accuracy=clean_accuracy,
        case_accuracy=case_accuracy,
        punctuation_accuracy=punctuation_accuracy,
        save_path=(
            "results/figures/"
            "tfidf_robustness_accuracy.png"
        ),
    )

    # ---------------------------------------------------------
    # Figure 2
    # ---------------------------------------------------------
    print(
        "\nGenerating degradation figure..."
    )

    plot_robustness_drop(
        repeated_summary=repeated,
        save_path=(
            "results/figures/"
            "tfidf_robustness_drop.png"
        ),
    )

    print(
        "\nRobustness figures generated successfully."
    )


if __name__ == "__main__":
    main()