"""Run ReliabilityLab's first repeated reliability experiment."""

from reliabilitylab.data import load_banking77
from reliabilitylab.experiments import (
    run_repeated_subsample_experiment,
)
from reliabilitylab.metrics import summarize_repeated_runs
from reliabilitylab.reporting import (
    plot_run_stability,
    save_summary_json,
)

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


def print_summary(name, summary):
    """Pretty-print repeated-run reliability statistics."""

    print("\n" + "=" * 60)
    print(f"{name.upper()} RELIABILITY SUMMARY")
    print("=" * 60)

    print(f"Runs              : {summary['n_runs']}")
    print(f"Mean              : {summary['mean'] * 100:.2f}%")
    print(f"Std deviation     : {summary['std'] * 100:.2f} pp")
    print(f"Median            : {summary['median'] * 100:.2f}%")
    print(f"Minimum           : {summary['min'] * 100:.2f}%")
    print(f"Maximum           : {summary['max'] * 100:.2f}%")
    print(f"Range             : {summary['range'] * 100:.2f} pp")

    print(
        "95% CI            : "
        f"[{summary['ci_95_lower'] * 100:.2f}%, "
        f"{summary['ci_95_upper'] * 100:.2f}%]"
    )

    print(
        "Peak–Mean Gap     : "
        f"{summary['peak_mean_gap'] * 100:.2f} pp"
    )


def main():

    print("=" * 60)
    print("ReliabilityLab")
    print("Repeated Stratified Training-Subset Experiment")
    print("=" * 60)

    dataset = load_banking77()

    results = run_repeated_subsample_experiment(
        train_dataset=dataset["train"],
        test_dataset=dataset["test"],
        seeds=SEEDS,
        train_fraction=0.80,
        save_path="results/tfidf_80pct_repeated_runs.csv",
    )

    accuracy_summary = summarize_repeated_runs(
        results["accuracy"]
    )

    f1_summary = summarize_repeated_runs(
        results["macro_f1"]
    )

    print_summary(
        "Accuracy",
        accuracy_summary,
    )

    print_summary(
        "Macro F1",
        f1_summary,
    )

    print("\n" + "=" * 60)
    print("RUN TABLE")
    print("=" * 60)

    display_results = results.copy()

    display_results["accuracy"] *= 100
    display_results["macro_f1"] *= 100

    print(
        display_results.to_string(
            index=False,
            formatters={
                "accuracy": "{:.2f}%".format,
                "macro_f1": "{:.2f}%".format,
            },
        )
    )

    combined_summary = {
        "accuracy": accuracy_summary,
        "macro_f1": f1_summary,
    }

    metadata = {
        "dataset": "BANKING77",
        "dataset_source": "DeepPavlov/banking77",
        "model": "TF-IDF + Logistic Regression",
        "train_fraction": 0.80,
        "full_training_samples": len(dataset["train"]),
        "samples_per_run": int(results["train_samples"].iloc[0]),
        "test_samples": len(dataset["test"]),
        "seeds": SEEDS,
        "reference_full_data_accuracy": 0.8588,
        "reference_full_data_macro_f1": 0.8581,
    }

    save_summary_json(
        summary=combined_summary,
        metadata=metadata,
        save_path="results/tfidf_80pct_summary.json",
    )

    plot_run_stability(
        values=results["accuracy"],
        baseline=0.8588,
        metric_name="Accuracy",
        save_path="results/figures/tfidf_80pct_accuracy_stability.png",
    )


if __name__ == "__main__":
    main()