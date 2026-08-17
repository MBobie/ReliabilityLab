"""Compare TF-IDF reliability across intent-classification datasets."""

from pathlib import Path

import pandas as pd

DATASETS = [
    "banking77",
    "clinc150",
    "hwu64",
]

SEVERITY_TAG = "20pct"


def main():
    """Generate a cross-dataset reliability comparison."""

    print("=" * 82)
    print("ReliabilityLab")
    print("Cross-Dataset Reliability Comparison")
    print("=" * 82)

    frames = []

    # ---------------------------------------------------------
    # Load dataset summaries
    # ---------------------------------------------------------
    for dataset in DATASETS:

        path = (
            Path("results")
            / "robustness"
            / dataset
            / f"tfidf_{SEVERITY_TAG}_summary.csv"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Missing robustness summary: {path}"
            )

        frame = pd.read_csv(
            path
        )

        frames.append(
            frame
        )

    # ---------------------------------------------------------
    # Combine datasets
    # ---------------------------------------------------------
    results = pd.concat(
        frames,
        ignore_index=True,
    )

    # ---------------------------------------------------------
    # Recompute normalized measures
    # ---------------------------------------------------------
    results[
        "accuracy_retention"
    ] = (
        results["mean_accuracy"]
        / results["clean_accuracy"]
    )

    results[
        "relative_accuracy_drop"
    ] = (
        results["mean_drop"]
        / results["clean_accuracy"]
    )

    # ---------------------------------------------------------
    # Select comparison columns
    # ---------------------------------------------------------
    comparison = results[
        [
            "dataset",
            "model",
            "perturbation",
            "requested_severity",
            "mean_realized_severity",
            "clean_accuracy",
            "mean_accuracy",
            "accuracy_std",
            "mean_drop",
            "drop_std",
            "accuracy_retention",
            "relative_accuracy_drop",
        ]
    ].copy()

    # ---------------------------------------------------------
    # Save complete comparison
    # ---------------------------------------------------------
    output_path = (
        Path("results")
        / "comparison"
        / "tfidf_cross_dataset_20pct.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison.to_csv(
        output_path,
        index=False,
    )

    # ---------------------------------------------------------
    # Display preparation
    # ---------------------------------------------------------
    display = comparison.copy()

    percentage_columns = [
        "requested_severity",
        "mean_realized_severity",
        "clean_accuracy",
        "mean_accuracy",
        "accuracy_std",
        "mean_drop",
        "drop_std",
        "accuracy_retention",
        "relative_accuracy_drop",
    ]

    for column in percentage_columns:
        display[column] *= 100

    # ---------------------------------------------------------
    # Full comparison table
    # ---------------------------------------------------------
    print("\n")

    print(
        display.to_string(
            index=False,
            formatters={
                "requested_severity":
                    "{:.2f}%".format,

                "mean_realized_severity":
                    "{:.2f}%".format,

                "clean_accuracy":
                    "{:.2f}%".format,

                "mean_accuracy":
                    "{:.2f}%".format,

                "accuracy_std":
                    "{:.2f} pp".format,

                "mean_drop":
                    "{:.2f} pp".format,

                "drop_std":
                    "{:.2f} pp".format,

                "accuracy_retention":
                    "{:.2f}%".format,

                "relative_accuracy_drop":
                    "{:.2f}%".format,
            },
        )
    )

    # ---------------------------------------------------------
    # Retention comparison
    # ---------------------------------------------------------
    print("\n")
    print("=" * 82)
    print("RETENTION COMPARISON")
    print("=" * 82)

    retention_table = (
        comparison.pivot(
            index="perturbation",
            columns="dataset",
            values="accuracy_retention",
        )
        * 100
    )

    print(
        retention_table.round(2)
    )

    # ---------------------------------------------------------
    # Relative-drop comparison
    # ---------------------------------------------------------
    print("\n")
    print("=" * 82)
    print("RELATIVE DROP COMPARISON")
    print("=" * 82)

    relative_drop_table = (
        comparison.pivot(
            index="perturbation",
            columns="dataset",
            values="relative_accuracy_drop",
        )
        * 100
    )

    print(
        relative_drop_table.round(2)
    )

    # ---------------------------------------------------------
    # Absolute corrupted accuracy
    # ---------------------------------------------------------
    print("\n")
    print("=" * 82)
    print("PERTURBED ACCURACY COMPARISON")
    print("=" * 82)

    accuracy_table = (
        comparison.pivot(
            index="perturbation",
            columns="dataset",
            values="mean_accuracy",
        )
        * 100
    )

    print(
        accuracy_table.round(2)
    )

    # ---------------------------------------------------------
    # Average robustness
    # ---------------------------------------------------------
    print("\n")
    print("=" * 82)
    print("AVERAGE ACROSS PERTURBATIONS")
    print("=" * 82)

    dataset_summary = (
        comparison.groupby(
            "dataset",
            as_index=False,
        )
        .agg(
            clean_accuracy=(
                "clean_accuracy",
                "first",
            ),

            mean_perturbed_accuracy=(
                "mean_accuracy",
                "mean",
            ),

            mean_absolute_drop=(
                "mean_drop",
                "mean",
            ),

            mean_accuracy_retention=(
                "accuracy_retention",
                "mean",
            ),

            mean_relative_drop=(
                "relative_accuracy_drop",
                "mean",
            ),
        )
    )

    dataset_display = (
        dataset_summary.copy()
    )

    for column in [
        "clean_accuracy",
        "mean_perturbed_accuracy",
        "mean_absolute_drop",
        "mean_accuracy_retention",
        "mean_relative_drop",
    ]:
        dataset_display[column] *= 100

    print(
        dataset_display.to_string(
            index=False,
            formatters={
                "clean_accuracy":
                    "{:.2f}%".format,

                "mean_perturbed_accuracy":
                    "{:.2f}%".format,

                "mean_absolute_drop":
                    "{:.2f} pp".format,

                "mean_accuracy_retention":
                    "{:.2f}%".format,

                "mean_relative_drop":
                    "{:.2f}%".format,
            },
        )
    )

    # ---------------------------------------------------------
    # Dataset ranking
    # ---------------------------------------------------------
    print("\n")
    print("=" * 82)
    print("DATASET RANKING BY ROBUSTNESS RETENTION")
    print("=" * 82)

    ranking = (
        dataset_summary.sort_values(
            "mean_accuracy_retention",
            ascending=False,
        )
        [
            [
                "dataset",
                "clean_accuracy",
                "mean_accuracy_retention",
                "mean_relative_drop",
            ]
        ]
        .copy()
    )

    ranking[
        "clean_accuracy"
    ] *= 100

    ranking[
        "mean_accuracy_retention"
    ] *= 100

    ranking[
        "mean_relative_drop"
    ] *= 100

    print(
        ranking.to_string(
            index=False,
            formatters={
                "clean_accuracy":
                    "{:.2f}%".format,

                "mean_accuracy_retention":
                    "{:.2f}%".format,

                "mean_relative_drop":
                    "{:.2f}%".format,
            },
        )
    )

    print(
        f"\nComparison saved to:\n"
        f"{output_path}"
    )


if __name__ == "__main__":
    main()