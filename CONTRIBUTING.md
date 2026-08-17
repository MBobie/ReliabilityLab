# Contributing to ReliabilityLab

Thank you for your interest in contributing to ReliabilityLab.

ReliabilityLab is a research-oriented project for evaluating machine-learning and NLP systems beyond clean benchmark accuracy. Contributions should preserve the project's emphasis on **reproducibility, statistical clarity, and explicit experimental assumptions**.

## Ways to Contribute

Useful contributions include new datasets, model baselines, perturbation operators, robustness and calibration metrics, statistical utilities, experiment tracking, tests, documentation, bug fixes, and reproducibility improvements.

## Development Setup

Clone the repository:

```bash
git clone https://github.com/MBobie/ReliabilityLab.git
cd ReliabilityLab
```

Synchronize the environment:

```bash
uv sync
```

Run tests and linting:

```bash
uv run pytest
uv run ruff check .
```

## Project Structure

```text
src/reliabilitylab/
├── data/
├── models/
├── experiments/
├── metrics/
├── perturbations/
├── reporting/
└── utils/
```

Please place new functionality in the appropriate package rather than embedding reusable logic directly in notebooks.

## Contribution Principles

### Reproducibility first

Every experiment should record enough information to be repeated. Where relevant, include dataset and split, model, seed, hyperparameters, perturbation, requested and realized severity, hardware/device, runtime, and output path.

### Do not overclaim

ReliabilityLab distinguishes between one experimental run, repeated runs, perturbation variability, training-seed variability, and test-set uncertainty. Documentation should make those distinctions explicit.

Avoid:

> Model A is universally more reliable than Model B.

Prefer:

> Under the evaluated configuration, Model A exhibited lower degradation under the tested perturbation conditions.

### Preserve paired designs

When comparing models under stochastic perturbations, use the same random seed and perturbation realization wherever possible.

### Separate requested and realized severity

For probabilistic perturbations, always record both:

```text
requested_severity
realized_severity
```

### Keep deterministic references explicit

A deterministic full-data baseline evaluated once should not be described as having zero experimental variance. Use wording such as:

> single deterministic reference; variability not estimated.

## Adding a Dataset

New datasets should be exposed through `reliabilitylab.data` and normalized so that experiment code does not depend on dataset-specific field names.

Document the source, license or usage conditions, number of classes, train/test sizes, text field, label field, and citation.

## Adding a Model

New models belong in `src/reliabilitylab/models/` and should document architecture, checkpoint if applicable, hyperparameters, seed behavior, training requirements, expected device, and save/load behavior.

Avoid hiding training inside evaluation functions.

## Adding a Perturbation

New perturbations belong in `src/reliabilitylab/perturbations/` and should be deterministic for a fixed seed, expose severity explicitly where meaningful, return realized-severity statistics where applicable, and include tests.

Semantic perturbations should explain how semantic preservation is constrained or checked.

## Tests

At minimum, perturbation tests should check fixed-seed reproducibility, zero-severity identity behavior, valid severity range, and protection against invalid empty examples unless explicitly intended.

## Experiment Outputs

Recommended organization:

```text
results/
├── baselines/
├── robustness/
├── comparison/
├── figures/
└── metadata/
```

Code should create directories automatically when needed.

## Pull Requests

A pull request should explain what changed, why it was needed, how it was tested, whether results changed, any new dependencies, and any limitations. Experiment-related changes should include the exact reproduction command.

## Scientific Changes

If a change alters perturbation semantics, metric definitions, seed handling, preprocessing, statistical tests, or the evaluation protocol, describe the methodological impact explicitly because it may alter comparability with earlier results.
