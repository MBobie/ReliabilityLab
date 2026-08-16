"""Evaluate TF-IDF Logistic Regression under text perturbations."""

from reliabilitylab.data import load_banking77
from reliabilitylab.experiments import (
    run_text_robustness_experiment,
)
from reliabilitylab.models import (
    build_tfidf_logreg,
)


def main():

    print("=" * 70)
    print("ReliabilityLab")
    print("Text Perturbation Robustness Experiment")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load data
    # ---------------------------------------------------------
    print("\nLoading BANKING77...")

    dataset = load_banking77()

    train = dataset["train"]
    test = dataset["test"]

    X_train = train["utterance"]
    y_train = train["label"]

    print(
        f"Training samples: {len(train):,}"
    )

    print(
        f"Test samples:     {len(test):,}"
    )

    # ---------------------------------------------------------
    # Train ONCE on clean full training data
    # ---------------------------------------------------------
    print(
        "\nTraining clean full-data baseline..."
    )

    model = build_tfidf_logreg()

    model.fit(
        X_train,
        y_train,
    )

    # ---------------------------------------------------------
    # Test robustness
    # ---------------------------------------------------------
    print(
        "\nBeginning robustness evaluation...\n"
    )

    results = run_text_robustness_experiment(
        model=model,
        test_dataset=test,
        seed=42,
        save_path=(
            "results/robustness/"
            "tfidf_text_robustness.csv"
        ),
    )

    # ---------------------------------------------------------
    # Display
    # ---------------------------------------------------------
    display = results.copy()

    for column in [
        "accuracy",
        "macro_f1",
    ]:
        display[column] *= 100

    for column in [
        "accuracy_drop",
        "macro_f1_drop",
    ]:
        display[column] *= 100

    display[
        "accuracy_relative_drop"
    ] *= 100

    print("\n")
    print("=" * 70)
    print("ROBUSTNESS SUMMARY")
    print("=" * 70)

    print(
        display.to_string(
            index=False,
            formatters={
                "accuracy":
                    "{:.2f}%".format,

                "macro_f1":
                    "{:.2f}%".format,

                "accuracy_drop":
                    "{:.2f} pp".format,

                "macro_f1_drop":
                    "{:.2f} pp".format,

                "accuracy_relative_drop":
                    "{:.2f}%".format,
            },
        )
    )

    print("\nExperiment complete.")


if __name__ == "__main__":
    main()