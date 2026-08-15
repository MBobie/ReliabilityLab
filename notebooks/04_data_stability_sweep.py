"""Run a training-data stability sweep for ReliabilityLab."""

from pathlib import Path

import pandas as pd

from reliabilitylab.data import load_banking77
from reliabilitylab.experiments import (
    run_repeated_subsample_experiment,
)
from reliabilitylab.metrics import summarize_repeated_runs


SEEDS = [
    1,
    7,
    21,
    42,
    84,
    123,
    256,
    512,
    1024,
    2026,
]


TRAIN_FRACTIONS = [
    0.20,
    0.40,
    0.60,
    0.80,
]


# Full-data reference from our original baseline experiment
FULL_DATA_ACCURACY = 0.8588
FULL_DATA_MACRO_F1 = 0.8581


def main():

    print("=" * 70)
    print("ReliabilityLab")
    print("Training-Data Stability Sweep")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load BANKING77
    # ---------------------------------------------------------
    print("\nLoading BANKING77...")

    dataset = load_banking77()

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    print(f"Full training samples: {len(train_dataset):,}")
    print(f"Test samples:          {len(test_dataset):,}")

    # This list will hold one summary row per training fraction
    summary_rows = []

    # ---------------------------------------------------------
    # Run each training fraction
    # ---------------------------------------------------------
    for fraction in TRAIN_FRACTIONS:

        percentage = int(fraction * 100)

        print("\n")
        print("#" * 70)
        print(
            f"RUNNING {percentage}% TRAINING-DATA EXPERIMENT"
        )
        print("#" * 70)

        save_path = (
            f"results/data_stability/"
            f"tfidf_{percentage}pct_runs.csv"
        )

        results = run_repeated_subsample_experiment(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            seeds=SEEDS,
            train_fraction=fraction,
            save_path=save_path,
        )

        # -----------------------------------------------------
        # Calculate reliability statistics
        # -----------------------------------------------------
        accuracy_summary = summarize_repeated_runs(
            results["accuracy"]
        )

        f1_summary = summarize_repeated_runs(
            results["macro_f1"]
        )

        # -----------------------------------------------------
        # Store one summary row
        # -----------------------------------------------------
        summary_rows.append(
            {
                "train_fraction": fraction,
                "train_percent": percentage,
                "train_samples": int(
                    results["train_samples"].iloc[0]
                ),
                "n_runs": len(results),

                "accuracy_mean":
                    accuracy_summary["mean"],

                "accuracy_std":
                    accuracy_summary["std"],

                "accuracy_min":
                    accuracy_summary["min"],

                "accuracy_max":
                    accuracy_summary["max"],

                "accuracy_range":
                    accuracy_summary["range"],

                "accuracy_peak_mean_gap":
                    accuracy_summary["peak_mean_gap"],

                "accuracy_ci_lower":
                    accuracy_summary["ci_95_lower"],

                "accuracy_ci_upper":
                    accuracy_summary["ci_95_upper"],

                "macro_f1_mean":
                    f1_summary["mean"],

                "macro_f1_std":
                    f1_summary["std"],

                "macro_f1_min":
                    f1_summary["min"],

                "macro_f1_max":
                    f1_summary["max"],

                "macro_f1_peak_mean_gap":
                    f1_summary["peak_mean_gap"],
            }
        )

    # ---------------------------------------------------------
    # Add the 100% reference baseline
    # ---------------------------------------------------------
    summary_rows.append(
        {
            "train_fraction": 1.00,
            "train_percent": 100,
            "train_samples": len(train_dataset),
            "n_runs": 1,

            "accuracy_mean": FULL_DATA_ACCURACY,
            "accuracy_std": 0.0,
            "accuracy_min": FULL_DATA_ACCURACY,
            "accuracy_max": FULL_DATA_ACCURACY,
            "accuracy_range": 0.0,
            "accuracy_peak_mean_gap": 0.0,
            "accuracy_ci_lower": FULL_DATA_ACCURACY,
            "accuracy_ci_upper": FULL_DATA_ACCURACY,

            "macro_f1_mean": FULL_DATA_MACRO_F1,
            "macro_f1_std": 0.0,
            "macro_f1_min": FULL_DATA_MACRO_F1,
            "macro_f1_max": FULL_DATA_MACRO_F1,
            "macro_f1_peak_mean_gap": 0.0,
        }
    )

    # ---------------------------------------------------------
    # Convert summaries to DataFrame
    # ---------------------------------------------------------
    summary_df = pd.DataFrame(summary_rows)

    summary_df = summary_df.sort_values(
        "train_percent"
    )

    # ---------------------------------------------------------
    # Save combined summary
    # ---------------------------------------------------------
    summary_path = Path(
        "results/data_stability/"
        "tfidf_data_stability_summary.csv"
    )

    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    # ---------------------------------------------------------
    # Display readable summary
    # ---------------------------------------------------------
    print("\n")
    print("=" * 70)
    print("DATA-STABILITY SUMMARY")
    print("=" * 70)

    display_df = summary_df[
        [
            "train_percent",
            "train_samples",
            "n_runs",
            "accuracy_mean",
            "accuracy_std",
            "macro_f1_mean",
            "macro_f1_std",
        ]
    ].copy()

    display_df["accuracy_mean"] *= 100
    display_df["accuracy_std"] *= 100
    display_df["macro_f1_mean"] *= 100
    display_df["macro_f1_std"] *= 100

    print(
        display_df.to_string(
            index=False,
            formatters={
                "accuracy_mean": "{:.2f}%".format,
                "accuracy_std": "{:.2f} pp".format,
                "macro_f1_mean": "{:.2f}%".format,
                "macro_f1_std": "{:.2f} pp".format,
            },
        )
    )

    print("\nCombined summary saved to:")
    print(summary_path)

    print("\nData-stability sweep complete.")


if __name__ == "__main__":
    main()