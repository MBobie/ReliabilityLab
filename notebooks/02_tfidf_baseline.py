"""Run the first ReliabilityLab classification baseline."""

from reliabilitylab.data import load_banking77
from reliabilitylab.models import build_tfidf_logreg
from reliabilitylab.metrics import classification_metrics


def main():
    print("=" * 60)
    print("ReliabilityLab — TF-IDF + Logistic Regression Baseline")
    print("=" * 60)

    # ---------------------------------------------------------
    # Load dataset
    # ---------------------------------------------------------
    print("\nLoading BANKING77...")

    dataset = load_banking77()

    train = dataset["train"]
    test = dataset["test"]

    X_train = train["utterance"]
    y_train = train["label"]

    X_test = test["utterance"]
    y_test = test["label"]

    print(f"Training samples: {len(X_train):,}")
    print(f"Test samples:     {len(X_test):,}")

    # ---------------------------------------------------------
    # Build model
    # ---------------------------------------------------------
    print("\nBuilding TF-IDF + Logistic Regression model...")

    model = build_tfidf_logreg()

    # ---------------------------------------------------------
    # Train model
    # ---------------------------------------------------------
    print("Training model...")

    model.fit(X_train, y_train)

    # ---------------------------------------------------------
    # Predict
    # ---------------------------------------------------------
    print("Generating predictions...")

    predictions = model.predict(X_test)

    # ---------------------------------------------------------
    # Evaluate
    # ---------------------------------------------------------
    metrics = classification_metrics(
        y_true=y_test,
        y_pred=predictions,
    )

    # ---------------------------------------------------------
    # Display results
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("BASELINE RESULTS")
    print("=" * 60)

    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Macro F1 : {metrics['macro_f1']:.4f}")

    print("\nPercentage form:")

    print(f"Accuracy : {metrics['accuracy'] * 100:.2f}%")
    print(f"Macro F1 : {metrics['macro_f1'] * 100:.2f}%")

    print("=" * 60)


if __name__ == "__main__":
    main()