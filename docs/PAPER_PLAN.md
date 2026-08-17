# ReliabilityLab — Paper Development Plan

## Recommended Working Title

**Beyond Clean Accuracy: Multi-Dimensional Reliability Evaluation of Intent Classification Models**

Strong alternative:

**When Simpler Models Are More Reliable: Stability, Robustness, and Cost in Intent Classification**

The first title is safer until the experiments cover more datasets and model families.

---

## 1. Publication Assessment

### Current state

**Promising short paper / workshop paper / case-study paper.**

The current study already contains a coherent finding:

- TF-IDF + Logistic Regression: 85.88% clean accuracy;
- DistilBERT: 85.06% clean accuracy;
- under matched 20% corruption, TF-IDF retains substantially higher accuracy;
- the difference survives 10 matched perturbation realizations;
- TF-IDF also requires far less computational effort in the current setup.

The main reviewer objection is predictable:

> Is this a general reliability phenomenon, or just one dataset and one transformer configuration?

The paper should be expanded specifically to answer that objection.

### Paper-ready target

**3 datasets + about 5 models + multi-seed neural training + calibration + stronger statistics** would turn this into a credible full empirical paper.

---

## 2. Central Claim

Do **not** frame the paper as:

> TF-IDF is better than DistilBERT.

Frame it as:

> **Clean benchmark accuracy is an incomplete indicator of model quality. Reliability-aware evaluation can produce materially different conclusions when stability, corruption severity, uncertainty, calibration, and computational cost are considered.**

---

## 3. Research Questions

### RQ1 — Ranking disagreement

Can clean-accuracy rankings disagree with reliability-aware rankings?

### RQ2 — Data availability

How does training-data availability affect both predictive performance and result stability?

### RQ3 — Robustness

How do classical, neural, and transformer models differ in their response to progressively stronger corruption?

### RQ4 — Training instability

How much does model ranking depend on training seed rather than architecture?

### RQ5 — Reliability versus cost

Do improvements in complexity and computation correspond to improvements in reliability?

---

## 4. Minimum Full-Paper Experimental Matrix

### Datasets

1. BANKING77
2. CLINC150
3. HWU64

### Models

1. TF-IDF + Logistic Regression
2. Linear SVM
3. one neural/hybrid model
4. DistilBERT
5. one stronger transformer baseline

A good neural/hybrid candidate is the existing Dual-Input CNN line of work, provided the implementation is reproducible and cleanly integrated.

### Training seeds

- deterministic classical models: repeated data subsets where relevant;
- stochastic neural/transformer models: **minimum 5 seeds**, preferably 10 if compute permits.

### Perturbation seeds

Keep the current 10 matched perturbation seeds.

### Severity levels

0%, 5%, 10%, 20%, 30%, 40%.

---

## 5. Perturbations

### Current

- adjacent-character typo swap;
- character deletion;
- word deletion;
- case control;
- punctuation control.

### Add before submission

At least 2–4 additional transformations from:

- keyboard-neighbor substitution;
- repeated characters;
- transposition;
- abbreviation/contraction variation;
- synonym replacement with semantic constraints;
- whitespace variation;
- naturally occurring misspellings/noisy queries where available.

The paper should explicitly distinguish **synthetic perturbation robustness** from **natural distribution shift**.

---

## 6. Calibration

Add:

- Expected Calibration Error (ECE);
- Brier score;
- negative log-likelihood;
- confidence on correct vs incorrect predictions;
- confidence degradation with increasing corruption.

Key question:

> Does a model merely become wrong under corruption, or does it remain confidently wrong?

---

## 7. Severity-Curve Summary

Report area under the accuracy-versus-severity curve as a **descriptive summary**.

Do not claim novelty for the metric until the literature review is complete.

Possible normalized form:

\[
S = \frac{1}{s_{max}}\int_0^{s_{max}} A(s)\,ds
\]

where \(A(s)\) is accuracy at severity \(s\).

Also report performance retention:

\[
R(s)=\frac{A(s)}{A(0)}
\]

This separates robustness from clean-baseline differences.

---

## 8. Statistical Analysis

Keep:

- means;
- standard deviations;
- 95% confidence intervals;
- matched perturbation seeds.

Improve with:

- Holm correction for multiple comparisons;
- effect sizes;
- paired bootstrap confidence intervals or permutation tests;
- resampling over test examples;
- explicit separation of training-seed uncertainty, perturbation-seed uncertainty, and test-sample uncertainty.

The current seed-level paired t-tests are useful development evidence, but the publication version should not rely on them alone.

---

## 9. Compute Reporting

Automatically record for every model:

- CPU/GPU device;
- hardware name;
- trainable parameter count;
- training runtime;
- inference runtime;
- batch size;
- number of epochs;
- peak memory where practical.

Target table:

| Model | Params | Train Time | Clean Acc | Robust Acc | Severity-Area |
|---|---:|---:|---:|---:|---:|

---

## 10. Proposed Contributions

A defensible contribution list could become:

1. **A unified multi-dimensional evaluation protocol** for intent-classification reliability covering clean performance, data sensitivity, perturbation robustness, severity response, calibration, training variability, and compute cost.
2. **A matched stochastic corruption protocol** that distinguishes requested from realized severity and supports paired model comparison.
3. **A cross-dataset empirical study** showing when clean benchmark rankings do and do not agree with reliability-aware rankings.
4. **ReliabilityLab**, an open-source implementation for reproducing the experiments and extending them to new classifiers.
5. **An empirical analysis of complexity versus reliability**, testing whether more expensive architectures consistently provide more reliable behavior.

Avoid claiming novelty for generic perturbation generation itself.

---

## 11. Proposed Paper Structure

### 1. Introduction

Motivation: clean accuracy can hide instability and failure behavior.

### 2. Related Work

- reliability-aware NLP evaluation;
- behavioral testing;
- adversarial/perturbation robustness;
- statistical reliability of benchmarks;
- intent classification.

### 3. ReliabilityLab Framework

- evaluation dimensions;
- probabilistic severity;
- matched stochastic evaluation;
- statistical analysis.

### 4. Experimental Setup

- datasets;
- models;
- hyperparameters;
- compute;
- perturbations.

### 5. Results

- clean performance;
- data-scarcity sensitivity;
- matched robustness;
- severity curves;
- calibration;
- training-seed instability;
- reliability versus compute.

### 6. Discussion

Ask when simple models remain competitive and whether additional compute buys reliability.

### 7. Limitations

Be explicit about synthetic noise, language/task scope, finite seeds, and hardware-dependent runtime.

### 8. Conclusion

End on the evaluation principle, not on TF-IDF versus DistilBERT.

---

## 12. Target Figures and Tables

### Figures

1. ReliabilityLab framework diagram
2. Data-availability performance/stability curves
3. Accuracy versus corruption severity by model
4. Normalized performance retention versus severity
5. Calibration error versus severity
6. Reliability versus training/inference cost

### Tables

1. Dataset statistics
2. Model configurations
3. Clean performance
4. Matched 20% corruption comparison
5. Severity-curve summary
6. Calibration
7. Compute cost

---

## 13. What NOT to Claim

Do not write:

- “TF-IDF is more reliable than transformers.”
- “Transformers are fragile.”
- “ReliabilityLab proves simple models are better.”
- “Our severity metric is novel” before literature verification.
- “The 100% TF-IDF condition has zero variance.”
- “A significant p-value proves practical superiority.”

Prefer:

> “Under the evaluated configurations…”

and always report effect magnitude.

---

## 14. Immediate Development Sequence

### Phase A — stabilize the repository

1. Replace README.
2. Generate flagship cross-model figures.
3. Add tests for perturbation reproducibility.
4. Add experiment metadata logging.
5. Create a versioned results manifest.

### Phase B — make the paper general

6. Add CLINC150.
7. Add HWU64.
8. Add Linear SVM.
9. Add hybrid/neural baseline.
10. Add a second transformer.

### Phase C — deepen reliability

11. Repeat stochastic training seeds.
12. Add calibration.
13. Add paired bootstrap/permutation analysis.
14. Add multiple-comparison correction.
15. Add severity-curve summary.
16. Add automated compute tracking.

### Phase D — manuscript

17. Freeze experiment configuration.
18. Generate final tables from CSV files.
19. Generate final figures from scripts.
20. Write manuscript.
21. Create archived release and CITATION.cff.
22. Publish code/data artifacts needed for reproduction.

---

## 15. Bottom Line

The project is **not too shallow to publish**; it is currently **too narrow for the strongest full-paper claim**.

The current BANKING77 experiment is a strong case study. The next scientific priority is breadth across datasets and model families, followed by training-seed uncertainty, calibration, and stronger statistical analysis.
