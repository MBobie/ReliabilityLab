# ReliabilityLab Multi-Dataset Refactor Plan

## Goal

Convert ReliabilityLab from a BANKING77-oriented experiment collection into a reusable benchmark engine supporting BANKING77, CLINC150, HWU64, and future intent-classification datasets without duplicating experimental logic.

## Core Design Principle

Dataset-specific code should answer only:

> How do I load and normalize this dataset?

Experiment code should answer:

> How do I evaluate any normalized dataset?

Model code should answer:

> How do I train, save, load, and predict with this model?

Reporting code should answer:

> How do I summarize and visualize the resulting experiment records?

## 1. Normalized Dataset Interface

Create `src/reliabilitylab/data/base.py` with a lightweight specification such as:

```python
from dataclasses import dataclass

@dataclass
class IntentDataset:
    name: str
    train_texts: list[str]
    train_labels: list[int]
    test_texts: list[str]
    test_labels: list[int]
    label_names: list[str] | None = None
```

Every dataset loader should return the same structure.

## 2. Dataset Registry

Create `src/reliabilitylab/data/registry.py`.

Concept:

```python
DATASET_LOADERS = {
    "banking77": load_banking77,
    "clinc150": load_clinc150,
    "hwu64": load_hwu64,
}
```

Then experiments call:

```python
dataset = load_dataset_by_name("banking77")
```

rather than importing a dataset-specific loader.

## 3. Refactor BANKING77 First

Before adding another dataset, make BANKING77 conform to the new interface and verify the existing TF-IDF baseline remains approximately 85.88% accuracy.

That regression check protects the current experimental record.

## 4. Add CLINC150

Create `src/reliabilitylab/data/clinc150.py`.

Responsibilities:

- load the dataset;
- choose the intended splits;
- normalize text;
- normalize labels to integer IDs;
- expose label names where available;
- return `IntentDataset`.

Do not put model training or perturbation code in this file.

## 5. Add HWU64

Create `src/reliabilitylab/data/hwu64.py` with the same normalized interface.

## 6. Replace Dataset-Specific Experiment Scripts

Eventually introduce reusable runners:

```text
scripts/run_baseline.py
scripts/run_robustness.py
scripts/run_severity.py
scripts/run_data_stability.py
scripts/compare_models.py
```

Example:

```bash
uv run python scripts/run_baseline.py --dataset banking77 --model tfidf_logreg
uv run python scripts/run_baseline.py --dataset clinc150 --model tfidf_logreg
```

The same code path should execute both experiments.

## 7. Configuration Files

Use YAML under `configs/`:

```text
configs/
├── datasets/
│   ├── banking77.yaml
│   ├── clinc150.yaml
│   └── hwu64.yaml
├── models/
│   ├── tfidf_logreg.yaml
│   └── distilbert.yaml
└── experiments/
    ├── clean.yaml
    ├── robustness_20pct.yaml
    └── severity_sweep.yaml
```

Example robustness configuration:

```yaml
seeds:
  - 1
  - 7
  - 21
  - 42
  - 84
  - 123
  - 256
  - 512
  - 1024
  - 2026

severity: 0.20

perturbations:
  - typo
  - char_delete
  - word_delete
```

## 8. Standard Result Schema

Every experiment should emit a common record format:

```text
experiment_id
dataset
model
condition
perturbation
requested_severity
realized_severity
training_seed
perturbation_seed
train_fraction
accuracy
macro_f1
runtime_seconds
device
timestamp
```

## 9. Experiment Metadata

Create `src/reliabilitylab/experiments/metadata.py` and record automatically:

- Python version
- ReliabilityLab version
- package versions
- operating system
- CPU/GPU device
- timestamp
- Git commit hash when available

## 10. Model Interface

Use a small model-facing protocol such as:

```python
class IntentClassifier:
    def fit(self, texts, labels):
        ...

    def predict(self, texts):
        ...

    def save(self, path):
        ...

    @classmethod
    def load(cls, path):
        ...
```

TF-IDF and transformer models can keep different internals while exposing compatible experiment-facing behavior.

## 11. Preserve Existing Results

Do not delete the current notebooks. Retain them as historical reproducibility scripts until the new engine reproduces the same results.

## 12. Regression Tests

Before adding CLINC150, add checks for:

- BANKING77 train count = 10,003
- BANKING77 test count = 3,080
- 77 labels
- fixed-seed perturbation reproducibility
- zero severity preserves input
- realized severity lies in [0, 1]
- TF-IDF pipeline trains and predicts on a small fixture

## 13. Recommended Implementation Order

1. create `IntentDataset`;
2. create dataset registry;
3. refactor BANKING77;
4. add loader tests;
5. verify the BANKING77 baseline remains unchanged;
6. add CLINC150;
7. run CLINC150 TF-IDF clean baseline;
8. run CLINC150 data sensitivity;
9. run CLINC150 20% robustness;
10. run CLINC150 severity sweep;
11. add HWU64;
12. add Linear SVM;
13. add neural/hybrid baseline;
14. add second transformer;
15. add multi-seed neural training and calibration;
16. add stronger resampling statistics and automated compute logging.

## Immediate Next Coding Step

Create:

```text
src/reliabilitylab/data/base.py
src/reliabilitylab/data/registry.py
```

Then refactor BANKING77 to return the normalized `IntentDataset`.

Only after the current BANKING77 pipeline passes regression checks should CLINC150 be added.
