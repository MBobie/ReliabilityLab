"""Fine-tune DistilBERT on BANKING77."""

import time

import numpy as np

from transformers import (
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from reliabilitylab.data import load_banking77
from reliabilitylab.metrics import classification_metrics
from reliabilitylab.models import (
    load_distilbert_components,
)


def main():

    print("=" * 72)
    print("ReliabilityLab")
    print("DistilBERT BANKING77 Baseline")
    print("=" * 72)

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------
    print("\nLoading BANKING77...")

    dataset = load_banking77()

    print(
        f"Training samples: "
        f"{len(dataset['train']):,}"
    )

    print(
        f"Test samples: "
        f"{len(dataset['test']):,}"
    )

    # ---------------------------------------------------------
    # Model and tokenizer
    # ---------------------------------------------------------
    print(
        "\nLoading DistilBERT..."
    )

    tokenizer, model = (
        load_distilbert_components(
            num_labels=77
        )
    )

    # ---------------------------------------------------------
    # Tokenization
    # ---------------------------------------------------------
    def tokenize_batch(batch):

        return tokenizer(
            batch["utterance"],
            truncation=True,
            max_length=64,
        )

    print(
        "\nTokenizing dataset..."
    )

    tokenized = dataset.map(
        tokenize_batch,
        batched=True,
    )

    # Trainer expects "labels"
    tokenized = tokenized.rename_column(
        "label",
        "labels",
    )

    tokenized = tokenized.remove_columns(
        ["utterance"]
    )

    data_collator = (
        DataCollatorWithPadding(
            tokenizer=tokenizer
        )
    )

    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------
    def compute_metrics(eval_prediction):

        logits, labels = eval_prediction

        predictions = np.argmax(
            logits,
            axis=-1,
        )

        return classification_metrics(
            y_true=labels,
            y_pred=predictions,
        )

    # ---------------------------------------------------------
    # Training configuration
    # ---------------------------------------------------------
    training_args = TrainingArguments(
        output_dir="results/distilbert/baseline",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=100,
        load_best_model_at_end=False,
        seed=42,
        data_seed=42,
        use_cpu=True,
        report_to="none",
    )

    print("\nTraining configuration:")
    print("Model            : distilbert-base-uncased")
    print("Training samples : 10,003")
    print("Test samples     : 3,080")
    print("Epochs           : 3")
    print("Train batch size : 16")
    print("Eval batch size  : 32")
    print("Learning rate    : 2e-5")
    print("Max length       : 64")
    print("Device           : CPU")
    print("Seed             : 42")

    # ---------------------------------------------------------
    # Trainer
    # ---------------------------------------------------------
    trainer = Trainer(
        model=model,
        args=training_args,

        train_dataset=(
            tokenized["train"]
        ),

        eval_dataset=(
            tokenized["test"]
        ),

        processing_class=tokenizer,

        data_collator=data_collator,

        compute_metrics=compute_metrics,
    )

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------
    print(
        "\nStarting DistilBERT training..."
    )

    start_time = time.time()

    trainer.train()

    elapsed = time.time() - start_time

    print(
        f"\nTotal training time: "
        f"{elapsed / 60:.2f} minutes"
    )

    # ---------------------------------------------------------
    # Final evaluation
    # ---------------------------------------------------------
    print(
        "\nEvaluating final model..."
    )

    results = trainer.evaluate()

    print("\n")
    print("=" * 72)
    print("DISTILBERT BASELINE RESULTS")
    print("=" * 72)

    print(
        f"Accuracy: "
        f"{results['eval_accuracy'] * 100:.2f}%"
    )

    print(
        f"Macro F1: "
        f"{results['eval_macro_f1'] * 100:.2f}%"
    )

    print(
        f"Evaluation loss: "
        f"{results['eval_loss']:.4f}"
    )

    # ---------------------------------------------------------
    # Save model
    # ---------------------------------------------------------
    trainer.save_model(
        "results/distilbert/"
        "baseline/final_model"
    )

    tokenizer.save_pretrained(
        "results/distilbert/"
        "baseline/final_model"
    )

    print(
        "\nModel saved successfully."
    )


if __name__ == "__main__":
    main()