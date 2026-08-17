# Changelog

All notable changes to ReliabilityLab will be documented in this file.

## [Unreleased]

### Planned

- CLINC150 dataset support
- HWU64 dataset support
- Linear SVM baseline
- additional neural and transformer baselines
- multi-seed neural training
- calibration metrics
- effect-size reporting
- Holm multiple-comparison correction
- paired bootstrap / permutation analysis
- automated compute tracking
- severity-curve summary statistics

## [0.1.0] - 2026-08-17

### Added

- BANKING77 dataset loader
- TF-IDF + Logistic Regression baseline
- accuracy and Macro F1 metrics
- repeated stratified training-subset experiments
- reliability summaries with mean, standard deviation, median, range, 95% confidence interval, and Peak–Mean Gap
- training-data sensitivity analysis at 20%, 40%, 60%, 80%, and a 100% deterministic reference
- text perturbations for adjacent-character typo swap, character deletion, word deletion, case transformation, and punctuation noise
- repeated stochastic robustness experiments
- probabilistic perturbation-severity mechanism
- realized-severity tracking
- severity-response and robustness-degradation curves
- DistilBERT sequence-classification baseline
- saved-model reuse for robustness evaluation
- repeated DistilBERT robustness evaluation across 10 perturbation seeds
- paired TF-IDF versus DistilBERT comparison
- confidence intervals for paired differences
- paired t-tests
- publication-ready result figures
- JSON, CSV, and PNG result artifacts
- initial GitHub README
- initial paper-development roadmap

### Initial BANKING77 Findings

Clean performance:

```text
TF-IDF + Logistic Regression
Accuracy : 85.88%
Macro F1 : 85.81%

DistilBERT
Accuracy : 85.06%
Macro F1 : 84.24%
```

At approximately 20% realized corruption:

```text
                 TF-IDF     DistilBERT
Typo             76.48%      68.51%
Char deletion    78.92%      73.92%
Word deletion    75.51%      69.64%
```

Mean paired TF-IDF accuracy advantage:

```text
Typo             +7.97 pp
Char deletion    +5.00 pp
Word deletion    +5.87 pp
```

These findings are configuration-specific and should not be interpreted as evidence that classical models universally outperform transformers.

### Known Limitations

- one dataset
- two model families
- one DistilBERT training seed
- limited perturbation families
- no calibration analysis
- no cross-dataset study
- no systematic compute metadata pipeline yet
- significance testing should be strengthened with resampling methods before publication

[Unreleased]: https://github.com/MBobie/ReliabilityLab/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/MBobie/ReliabilityLab/releases/tag/v0.1.0
