"""Train or reuse one DistilBERT seed and evaluate robustness on BANKING77."""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from reliabilitylab.data import load_intent_dataset
from reliabilitylab.metrics import classification_metrics
from reliabilitylab.perturbations import perturb_texts_probabilistic

MODEL_NAME = "distilbert-base-uncased"

DATASET_NAME = "banking77"

MAX_LENGTH = 64

TRAIN_BATCH_SIZE = 16

EVAL_BATCH_SIZE = 32

LEARNING_RATE = 2e-5

WEIGHT_DECAY = 0.01

NUM_EPOCHS = 3

ROBUSTNESS_SEVERITY = 0.20


PERTURBATION_SEEDS = [
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


PERTURBATIONS = [
    "typo",
    "char_delete",
    "word_delete",
]


LEGACY_SEED42_MODEL = (
    Path("results")
    / "distilbert"
    / "baseline"
    / "final_model"
)


def parse_args():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Train one DistilBERT seed and evaluate "
            "BANKING77 robustness."
        )
    )

    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Training seed.",
    )

    parser.add_argument(
        "--force-train",
        action="store_true",
        help=(
            "Retrain even if a saved final model exists."
        ),
    )

    return parser.parse_args()


def seed_dir(
    seed: int,
) -> Path:
    """Return output directory for one training seed."""

    return (
        Path("results")
        / "distilbert"
        / "multiseed"
        / f"seed_{seed}"
    )


def final_model_dir(
    seed: int,
) -> Path:
    """Return final model directory for one seed."""

    return (
        seed_dir(seed)
        / "final_model"
    )


def results_path(
    seed: int,
) -> Path:
    """Return raw evaluation-results path."""

    return (
        seed_dir(seed)
        / "evaluation_runs.csv"
    )


def summary_path(
    seed: int,
) -> Path:
    """Return per-seed summary path."""

    return (
        seed_dir(seed)
        / "evaluation_summary.csv"
    )


def training_info_path(
    seed: int,
) -> Path:
    """Return training metadata path."""

    return (
        seed_dir(seed)
        / "training_info.csv"
    )


def build_label_maps(
    dataset,
) -> tuple[dict[int, str], dict[str, int]]:
    """Build explicit model label mappings."""

    if dataset.label_names is not None:

        label_names = list(
            dataset.label_names
        )

    else:

        label_names = [
            str(index)
            for index in range(
                dataset.num_labels
            )
        ]

    id2label = {
        index: label
        for index, label in enumerate(
            label_names
        )
    }

    label2id = {
        label: index
        for index, label in id2label.items()
    }

    return (
        id2label,
        label2id,
    )


def tokenize_training_dataset(
    tokenizer,
    texts,
    labels,
) -> Dataset:
    """Tokenize training texts for Hugging Face Trainer."""

    dataset = Dataset.from_dict(
        {
            "text": list(
                texts
            ),
            "labels": list(
                labels
            ),
        }
    )

    def tokenize_batch(
        batch,
    ):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
        )

    dataset = dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=[
            "text",
        ],
    )

    return dataset


def train_model(
    *,
    dataset,
    seed: int,
    output_dir: Path,
):
    """Train DistilBERT with the established ReliabilityLab configuration."""

    print(
        f"\nTraining DistilBERT with seed {seed}..."
    )

    set_seed(
        seed
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_NAME
        )
    )

    train_dataset = (
        tokenize_training_dataset(
            tokenizer,
            dataset.train_texts,
            dataset.train_labels,
        )
    )

    (
        id2label,
        label2id,
    ) = build_label_maps(
        dataset
    )

    def model_init():
        """Create a freshly initialized classification head."""

        return (
            AutoModelForSequenceClassification.from_pretrained(
                MODEL_NAME,
                num_labels=dataset.num_labels,
                id2label=id2label,
                label2id=label2id,
            )
        )

    data_collator = (
        DataCollatorWithPadding(
            tokenizer=tokenizer
        )
    )

    trainer_work_dir = (
        output_dir
        / "checkpoint-work"
    )

    training_args = TrainingArguments(
        output_dir=str(
            trainer_work_dir
        ),
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        weight_decay=WEIGHT_DECAY,
        logging_strategy="epoch",
        save_strategy="no",
        eval_strategy="no",
        report_to="none",
        seed=seed,
        data_seed=seed,
        use_cpu=True,
    )

    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    start_time = time.time()

    train_output = trainer.train()

    training_seconds = (
        time.time()
        - start_time
    )

    model = trainer.model

    final_dir = (
        output_dir
        / "final_model"
    )

    final_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save_pretrained(
        final_dir
    )

    tokenizer.save_pretrained(
        final_dir
    )

    training_info = {
        "seed":
            seed,

        "model_name":
            MODEL_NAME,

        "dataset":
            DATASET_NAME,

        "num_epochs":
            NUM_EPOCHS,

        "learning_rate":
            LEARNING_RATE,

        "weight_decay":
            WEIGHT_DECAY,

        "train_batch_size":
            TRAIN_BATCH_SIZE,

        "eval_batch_size":
            EVAL_BATCH_SIZE,

        "max_length":
            MAX_LENGTH,

        "training_seconds":
            training_seconds,

        "training_loss":
            float(
                train_output.training_loss
            ),

        "reused_existing_model":
            False,
    }

    return (
        model,
        tokenizer,
        training_info,
    )


def load_saved_model(
    model_path: Path,
    seed: int,
):
    """Load a previously trained DistilBERT model."""

    print(
        "\nLoading saved model:"
    )

    print(
        model_path
    )

    tokenizer_source = (
        model_path
        if (
            model_path
            / "tokenizer_config.json"
        ).exists()
        else MODEL_NAME
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            tokenizer_source
        )
    )

    model = (
        AutoModelForSequenceClassification.from_pretrained(
            model_path
        )
    )

    training_info = {
        "seed":
            seed,

        "model_name":
            MODEL_NAME,

        "dataset":
            DATASET_NAME,

        "num_epochs":
            NUM_EPOCHS,

        "learning_rate":
            LEARNING_RATE,

        "weight_decay":
            WEIGHT_DECAY,

        "train_batch_size":
            TRAIN_BATCH_SIZE,

        "eval_batch_size":
            EVAL_BATCH_SIZE,

        "max_length":
            MAX_LENGTH,

        "training_seconds":
            np.nan,

        "training_loss":
            np.nan,

        "reused_existing_model":
            True,
    }

    return (
        model,
        tokenizer,
        training_info,
    )


class PredictionDataset(
    torch.utils.data.Dataset
):
    """Tokenized text dataset used for model prediction."""

    def __init__(
        self,
        encodings,
    ):
        self.encodings = (
            encodings
        )

    def __len__(
        self,
    ):
        return len(
            self.encodings[
                "input_ids"
            ]
        )

    def __getitem__(
        self,
        index,
    ):
        return {
            key: torch.tensor(
                values[index]
            )
            for key, values
            in self.encodings.items()
        }


def predict_texts(
    model,
    tokenizer,
    texts,
) -> np.ndarray:
    """Generate DistilBERT predictions for texts."""

    encodings = tokenizer(
        list(
            texts
        ),
        truncation=True,
        max_length=MAX_LENGTH,
    )

    prediction_dataset = (
        PredictionDataset(
            encodings
        )
    )

    collator = (
        DataCollatorWithPadding(
            tokenizer=tokenizer,
            return_tensors="pt",
        )
    )

    loader = DataLoader(
        prediction_dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        collate_fn=collator,
    )

    device = torch.device(
        "cpu"
    )

    model.to(
        device
    )

    model.eval()

    predictions = []

    with torch.no_grad():

        for batch in loader:

            batch = {
                key: value.to(
                    device
                )
                for key, value
                in batch.items()
            }

            outputs = model(
                **batch
            )

            batch_predictions = (
                outputs.logits.argmax(
                    dim=-1
                )
            )

            predictions.extend(
                batch_predictions.cpu().numpy()
            )

    return np.asarray(
        predictions
    )


def evaluate_model(
    *,
    dataset,
    model,
    tokenizer,
    training_seed: int,
) -> pd.DataFrame:
    """Evaluate clean and repeated robustness performance."""

    rows = []

    print(
        "\nEvaluating clean BANKING77 test set..."
    )

    clean_predictions = (
        predict_texts(
            model,
            tokenizer,
            dataset.test_texts,
        )
    )

    clean_metrics = (
        classification_metrics(
            dataset.test_labels,
            clean_predictions,
        )
    )

    clean_accuracy = (
        clean_metrics[
            "accuracy"
        ]
    )

    clean_macro_f1 = (
        clean_metrics[
            "macro_f1"
        ]
    )

    print(
        f"Clean accuracy : "
        f"{clean_accuracy * 100:.2f}%"
    )

    print(
        f"Clean Macro F1 : "
        f"{clean_macro_f1 * 100:.2f}%"
    )

    rows.append(
        {
            "training_seed":
                training_seed,

            "condition":
                "clean",

            "perturbation":
                "clean",

            "perturbation_seed":
                np.nan,

            "requested_severity":
                0.0,

            "realized_severity":
                0.0,

            "accuracy":
                clean_accuracy,

            "macro_f1":
                clean_macro_f1,

            "accuracy_drop":
                0.0,

            "macro_f1_drop":
                0.0,

            "accuracy_retention":
                1.0,

            "relative_accuracy_drop":
                0.0,
        }
    )

    for perturbation in PERTURBATIONS:

        print("\n")
        print(
            "#" * 78
        )

        print(
            f"PERTURBATION: "
            f"{perturbation.upper()}"
        )

        print(
            "#" * 78
        )

        for run_number, perturbation_seed in enumerate(
            PERTURBATION_SEEDS,
            start=1,
        ):

            (
                perturbed_texts,
                stats,
            ) = perturb_texts_probabilistic(
                texts=dataset.test_texts,
                perturbation=perturbation,
                severity=ROBUSTNESS_SEVERITY,
                seed=perturbation_seed,
                return_stats=True,
            )

            predictions = (
                predict_texts(
                    model,
                    tokenizer,
                    perturbed_texts,
                )
            )

            metrics = (
                classification_metrics(
                    dataset.test_labels,
                    predictions,
                )
            )

            accuracy_drop = (
                clean_accuracy
                - metrics[
                    "accuracy"
                ]
            )

            macro_f1_drop = (
                clean_macro_f1
                - metrics[
                    "macro_f1"
                ]
            )

            accuracy_retention = (
                metrics[
                    "accuracy"
                ]
                / clean_accuracy
            )

            relative_accuracy_drop = (
                accuracy_drop
                / clean_accuracy
            )

            rows.append(
                {
                    "training_seed":
                        training_seed,

                    "condition":
                        "perturbed",

                    "perturbation":
                        perturbation,

                    "perturbation_seed":
                        perturbation_seed,

                    "requested_severity":
                        ROBUSTNESS_SEVERITY,

                    "realized_severity":
                        stats[
                            "realized_severity"
                        ],

                    "accuracy":
                        metrics[
                            "accuracy"
                        ],

                    "macro_f1":
                        metrics[
                            "macro_f1"
                        ],

                    "accuracy_drop":
                        accuracy_drop,

                    "macro_f1_drop":
                        macro_f1_drop,

                    "accuracy_retention":
                        accuracy_retention,

                    "relative_accuracy_drop":
                        relative_accuracy_drop,
                }
            )

            print(
                f"Run {run_number:02d}/"
                f"{len(PERTURBATION_SEEDS)} "
                f"| seed={perturbation_seed:<4} "
                f"| realized="
                f"{stats['realized_severity'] * 100:.2f}% "
                f"| accuracy="
                f"{metrics['accuracy'] * 100:.2f}% "
                f"| retention="
                f"{accuracy_retention * 100:.2f}%"
            )

    return pd.DataFrame(
        rows
    )


def summarize_results(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Build a compact per-training-seed summary."""

    clean = results[
        results[
            "condition"
        ]
        == "clean"
    ].iloc[
        0
    ]

    rows = [
        {
            "training_seed":
                int(
                    clean[
                        "training_seed"
                    ]
                ),

            "perturbation":
                "clean",

            "clean_accuracy":
                clean[
                    "accuracy"
                ],

            "clean_macro_f1":
                clean[
                    "macro_f1"
                ],

            "mean_accuracy":
                clean[
                    "accuracy"
                ],

            "accuracy_std":
                0.0,

            "mean_macro_f1":
                clean[
                    "macro_f1"
                ],

            "macro_f1_std":
                0.0,

            "mean_accuracy_drop":
                0.0,

            "mean_accuracy_retention":
                1.0,
        }
    ]

    perturbed = results[
        results[
            "condition"
        ]
        == "perturbed"
    ]

    for perturbation in PERTURBATIONS:

        subset = perturbed[
            perturbed[
                "perturbation"
            ]
            == perturbation
        ]

        rows.append(
            {
                "training_seed":
                    int(
                        clean[
                            "training_seed"
                        ]
                    ),

                "perturbation":
                    perturbation,

                "clean_accuracy":
                    clean[
                        "accuracy"
                    ],

                "clean_macro_f1":
                    clean[
                        "macro_f1"
                    ],

                "mean_accuracy":
                    subset[
                        "accuracy"
                    ].mean(),

                "accuracy_std":
                    subset[
                        "accuracy"
                    ].std(
                        ddof=1
                    ),

                "mean_macro_f1":
                    subset[
                        "macro_f1"
                    ].mean(),

                "macro_f1_std":
                    subset[
                        "macro_f1"
                    ].std(
                        ddof=1
                    ),

                "mean_accuracy_drop":
                    subset[
                        "accuracy_drop"
                    ].mean(),

                "mean_accuracy_retention":
                    subset[
                        "accuracy_retention"
                    ].mean(),
            }
        )

    return pd.DataFrame(
        rows
    )


def main():
    """Train/reuse one seed and evaluate it."""

    args = parse_args()

    seed = (
        args.seed
    )

    output_dir = (
        seed_dir(
            seed
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("ReliabilityLab")
    print("DistilBERT Multi-Training-Seed Experiment")
    print("=" * 80)

    print(
        f"\nTraining seed      : "
        f"{seed}"
    )

    print(
        f"Robustness severity: "
        f"{ROBUSTNESS_SEVERITY * 100:.0f}%"
    )

    print(
        f"Perturbation seeds : "
        f"{len(PERTURBATION_SEEDS)}"
    )

    print(
        "Device             : CPU"
    )

    dataset = (
        load_intent_dataset(
            DATASET_NAME
        )
    )

    new_model_path = (
        final_model_dir(
            seed
        )
    )

    if (
        seed == 42
        and LEGACY_SEED42_MODEL.exists()
        and not args.force_train
        and not new_model_path.exists()
    ):

        (
            model,
            tokenizer,
            training_info,
        ) = load_saved_model(
            LEGACY_SEED42_MODEL,
            seed,
        )

        training_info[
            "source_model_path"
        ] = str(
            LEGACY_SEED42_MODEL
        )

    elif (
        new_model_path.exists()
        and not args.force_train
    ):

        (
            model,
            tokenizer,
            training_info,
        ) = load_saved_model(
            new_model_path,
            seed,
        )

        training_info[
            "source_model_path"
        ] = str(
            new_model_path
        )

    else:

        (
            model,
            tokenizer,
            training_info,
        ) = train_model(
            dataset=dataset,
            seed=seed,
            output_dir=output_dir,
        )

        training_info[
            "source_model_path"
        ] = str(
            new_model_path
        )

    evaluation_start = (
        time.time()
    )

    results = evaluate_model(
        dataset=dataset,
        model=model,
        tokenizer=tokenizer,
        training_seed=seed,
    )

    evaluation_seconds = (
        time.time()
        - evaluation_start
    )

    training_info[
        "evaluation_seconds"
    ] = evaluation_seconds

    summary = (
        summarize_results(
            results
        )
    )

    results.to_csv(
        results_path(
            seed
        ),
        index=False,
    )

    summary.to_csv(
        summary_path(
            seed
        ),
        index=False,
    )

    pd.DataFrame(
        [
            training_info
        ]
    ).to_csv(
        training_info_path(
            seed
        ),
        index=False,
    )

    print("\n")
    print("=" * 80)
    print("SEED SUMMARY")
    print("=" * 80)

    display = (
        summary.copy()
    )

    for column in [
        "clean_accuracy",
        "clean_macro_f1",
        "mean_accuracy",
        "accuracy_std",
        "mean_macro_f1",
        "macro_f1_std",
        "mean_accuracy_drop",
        "mean_accuracy_retention",
    ]:

        display[
            column
        ] *= 100

    print(
        display.to_string(
            index=False,
            formatters={
                "clean_accuracy":
                    "{:.2f}%".format,

                "clean_macro_f1":
                    "{:.2f}%".format,

                "mean_accuracy":
                    "{:.2f}%".format,

                "accuracy_std":
                    "{:.2f} pp".format,

                "mean_macro_f1":
                    "{:.2f}%".format,

                "macro_f1_std":
                    "{:.2f} pp".format,

                "mean_accuracy_drop":
                    "{:.2f} pp".format,

                "mean_accuracy_retention":
                    "{:.2f}%".format,
            },
        )
    )

    print("\nSaved:")
    print(
        results_path(
            seed
        )
    )
    print(
        summary_path(
            seed
        )
    )
    print(
        training_info_path(
            seed
        )
    )


if __name__ == "__main__":
    main()