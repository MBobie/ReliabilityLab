# ReliabilityLab — Paper Results Architecture

## Working title

**Beyond Clean Accuracy: Classifier, Representation, and Corruption Effects in Reliable Intent Classification**

Alternative:

**When Clean Accuracy Is Not Enough: Representation-Dependent Robustness in Intent Classification**

---

# 1. Central research problem

Machine-learning models are commonly compared using clean-test accuracy.
However, clean accuracy alone does not reveal how model performance changes
when inputs are corrupted.

ReliabilityLab evaluates models beyond clean accuracy by separating:

1. clean predictive performance,
2. corrupted predictive performance,
3. normalized accuracy retention,
4. perturbation-specific robustness,
5. severity-response behavior,
6. variability across stochastic perturbation realizations,
7. computational cost.

The central question is:

> Does a model or design choice that improves clean accuracy necessarily
> improve reliability under input corruption?

The current experiments show that the answer depends strongly on which
component of the learning system is changed.

---

# 2. Research questions

## RQ1 — Classifier effect

When the text representation is held fixed, how does changing the classifier
affect clean performance and corruption robustness?

Controlled comparison:

- Word TF-IDF + Logistic Regression
- Word TF-IDF + Linear SVM

The representation remains fixed.

Only the classifier changes.

---

## RQ2 — Representation effect

When the classifier is held fixed, how does changing the representation affect
clean performance and corruption robustness?

Controlled comparison:

- Word TF-IDF + Linear SVM
- Character TF-IDF + Linear SVM

The classifier remains fixed.

Only the representation changes.

---

## RQ3 — Perturbation interaction

Does the effect of representation depend on the type of corruption?

Perturbations:

- typo corruption,
- character deletion,
- word deletion.

This tests whether robustness is a universal model property or an interaction
between model representation and corruption structure.

---

## RQ4 — Severity interaction

Does the representation effect remain constant as corruption severity
increases?

Requested severities:

- 5%,
- 10%,
- 20%,
- 30%,
- 40%.

---

## RQ5 — Cross-dataset consistency

Are the observed classifier and representation effects reproduced across
multiple intent-classification datasets?

Datasets:

- BANKING77,
- CLINC150,
- HWU64.

---

# 3. Experimental design

## 3.1 Datasets

### BANKING77

- training samples: 10,003
- test samples: 3,080
- classes: 77

### CLINC150

- training samples: 15,000
- validation samples: 3,000
- test samples: 4,500
- closed-set classes: 150

### HWU64

- training samples: 8,954
- test samples: 1,076
- classes: 64

---

## 3.2 Models

### Model A — Word TF-IDF + Logistic Regression

Representation:

- lowercase word TF-IDF,
- unigram and bigram features,
- minimum document frequency = 2,
- sublinear term frequency.

Classifier:

- Logistic Regression.

---

### Model B — Word TF-IDF + Linear SVM

Representation:

- identical word TF-IDF representation to Model A.

Classifier:

- Linear SVM.

This creates a controlled classifier experiment.

---

### Model C — Character TF-IDF + Linear SVM

Representation:

- character-within-word TF-IDF,
- character n-grams of length 3–5,
- minimum document frequency = 2,
- sublinear term frequency.

Classifier:

- same Linear SVM classifier family as Model B.

This creates a controlled representation experiment.

---

# 4. Corruption protocol

Three lexical perturbation families are currently evaluated:

1. typo corruption,
2. character deletion,
3. word deletion.

For each requested severity, eligible corruption units are independently
perturbed with the requested probability.

Primary severity levels:

- 5%,
- 10%,
- 20%,
- 30%,
- 40%.

Each model × dataset × perturbation × severity condition is evaluated using
10 matched perturbation seeds:

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

The same perturbation seeds are used when comparing models within a dataset,
allowing paired comparisons over stochastic corruption realizations.

---

# 5. Metrics

## 5.1 Clean accuracy

Accuracy on the uncorrupted test set.

---

## 5.2 Macro F1

Macro-averaged F1 score on the test set.

---

## 5.3 Perturbed accuracy

Accuracy after applying a specified corruption.

---

## 5.4 Absolute accuracy drop

Absolute decrease from clean accuracy:

\[
D = A_{clean} - A_{perturbed}
\]

---

## 5.5 Accuracy retention

Normalized retention of clean accuracy:

\[
R =
\frac{A_{perturbed}}
     {A_{clean}}
\]

A retention of 0.90 means that the corrupted model retains 90% of its clean
accuracy.

---

# 6. Results

## 6.1 Classifier choice improves absolute performance substantially

The first controlled experiment compared Logistic Regression and Linear SVM
while holding the word TF-IDF representation fixed.

### Clean accuracy

| Dataset | Logistic Regression | Linear SVM | SVM gain |
|---|---:|---:|---:|
| BANKING77 | 85.88% | 89.06% | +3.18 pp |
| CLINC150 | 88.93% | 91.13% | +2.20 pp |
| HWU64 | 83.09% | 85.97% | +2.88 pp |

Mean clean-accuracy improvement:

**+2.75 percentage points**

Thus, classifier choice materially affects absolute predictive performance.

---

## 6.2 Classifier choice has little effect on normalized corruption retention

Despite the clean-accuracy gains, average 20% corruption retention changed
only slightly.

| Dataset | Logistic Regression | Linear SVM |
|---|---:|---:|
| BANKING77 | 89.63% | 89.81% |
| CLINC150 | 89.49% | 89.83% |
| HWU64 | 90.17% | 89.88% |

Across datasets:

- mean clean accuracy gain: approximately +2.75 pp,
- mean perturbed accuracy gain: approximately +2.55 pp,
- mean retention difference: approximately +0.08 pp.

### Interpretation

Changing the classifier primarily shifts absolute performance upward while
leaving the normalized lexical-corruption response comparatively stable.

This does not prove that the TF-IDF representation causally determines
robustness.

It shows that two different classifiers operating on the same representation
exhibit very similar normalized degradation profiles.

---

# 7. Representation effect

## 7.1 Character TF-IDF improves clean performance

With Linear SVM held fixed:

| Dataset | Word TF-IDF | Character TF-IDF | Difference |
|---|---:|---:|---:|
| BANKING77 | 89.06% | 90.39% | +1.33 pp |
| CLINC150 | 91.13% | 91.64% | +0.51 pp |
| HWU64 | 85.97% | 87.73% | +1.77 pp |

Mean clean improvement:

**+1.20 percentage points**

Character TF-IDF therefore provides a modest clean-performance improvement
across all three datasets.

---

## 7.2 Representation strongly changes character-level robustness

At 20% requested corruption severity:

### Typo retention advantage

Character TF-IDF minus Word TF-IDF:

- BANKING77: +5.08 pp
- CLINC150: +4.20 pp
- HWU64: +3.97 pp

### Character-deletion retention advantage

- BANKING77: +5.85 pp
- CLINC150: +5.25 pp
- HWU64: +5.41 pp

Across typo and character deletion:

**mean retention improvement = +4.96 pp**

This is substantially larger than the approximately +0.08 pp change observed
when only the classifier was changed.

---

## 7.3 The representation advantage disappears under word deletion

At the same 20% severity:

- BANKING77: -0.51 pp
- CLINC150: -0.42 pp
- HWU64: +0.37 pp

Mean effect:

**-0.19 pp**

Thus, the character representation does not provide a general robustness
advantage across all corruption types.

Instead, the advantage depends on the perturbation mechanism.

---

# 8. Severity-response result

The representation effect becomes increasingly pronounced as character-level
corruption severity rises.

## 8.1 Mean character-level retention advantage

Averaging typo and character deletion across all three datasets:

| Requested severity | Character TF-IDF advantage |
|---:|---:|
| 5% | +1.09 pp |
| 10% | +2.33 pp |
| 20% | +4.96 pp |
| 30% | +8.19 pp |
| 40% | +12.18 pp |

This shows systematic divergence between the two representations as
character-level corruption becomes stronger.

---

## 8.2 Perturbation-specific severity response

### Character deletion

Cross-dataset mean retention difference:

| Severity | Character − Word TF-IDF |
|---:|---:|
| 5% | +1.26 pp |
| 10% | +2.70 pp |
| 20% | +5.50 pp |
| 30% | +9.08 pp |
| 40% | +13.42 pp |

### Typo corruption

| Severity | Character − Word TF-IDF |
|---:|---:|
| 5% | +0.91 pp |
| 10% | +1.96 pp |
| 20% | +4.42 pp |
| 30% | +7.31 pp |
| 40% | +10.94 pp |

### Word deletion

| Severity | Character − Word TF-IDF |
|---:|---:|
| 5% | -0.09 pp |
| 10% | -0.34 pp |
| 20% | -0.19 pp |
| 30% | -0.14 pp |
| 40% | +0.18 pp |

The word-deletion difference remains close to zero over the entire severity
range.

---

# 9. Mechanistic interpretation

Character n-gram features provide redundancy under character-level
perturbations.

For example, if a word is misspelled or one character is deleted, many local
character n-grams can remain shared between the clean and corrupted forms.

Word-level TF-IDF is more brittle to such changes because a corrupted token may
no longer match the original vocabulary item.

Word deletion produces a different mechanism.

Deleting an entire word removes:

- the complete word feature from word TF-IDF,
- and all character n-grams belonging to that word from character TF-IDF.

Therefore, character representation does not provide the same structural
advantage under complete word removal.

The empirical results are consistent with this explanation.

---

# 10. Main scientific finding

The experiments provide evidence that reliability cannot be summarized by
clean accuracy alone.

Three distinct effects are observed:

### 1. Classifier effect

Changing Logistic Regression to Linear SVM substantially improves clean and
corrupted absolute accuracy while leaving normalized corruption retention
nearly unchanged.

### 2. Representation effect

Changing from word to character TF-IDF while holding Linear SVM fixed produces
a substantially larger robustness change.

### 3. Representation × perturbation × severity interaction

The character representation advantage:

- increases strongly with character-level corruption severity,
- reproduces across three datasets,
- but remains approximately zero for word deletion.

Therefore:

> Robustness is not simply an intrinsic scalar property of a classifier.
> It depends on the interaction between representation, corruption mechanism,
> and corruption severity.

---

# 11. Statistical interpretation

Matched perturbation seeds allow paired comparisons between representations
under the same corruption realizations.

The current t-based confidence intervals quantify variation over stochastic
perturbation realizations on a fixed test set.

They do not represent complete uncertainty arising from:

- sampling new test examples,
- training-set variation,
- training random seeds,
- model-selection uncertainty,
- alternative corruption generators.

Therefore, these intervals should be described as:

> confidence intervals across matched perturbation realizations

rather than general population confidence intervals.

For publication, additional example-level bootstrap analysis should be
considered.

---

# 12. Current figures

## Figure 1

TF-IDF Logistic Regression vs Linear SVM clean and corrupted performance.

Files:

- `results/figures/tfidf_lr_vs_svm_absolute_performance.png`
- `results/figures/tfidf_lr_vs_svm_retention.png`

---

## Figure 2

Clean performance under word and character TF-IDF representations.

File:

- `results/figures/representation_clean_accuracy.png`

---

## Figure 3

Paired representation effect at 20% corruption.

File:

- `results/figures/representation_retention_effect.png`

---

## Figure 4 — primary severity figure

Cross-dataset retention advantage as corruption severity increases.

File:

- `results/figures/representation_severity_retention_difference.png`

---

## Supporting severity figures

- `results/figures/representation_severity_typo.png`
- `results/figures/representation_severity_char_delete.png`
- `results/figures/representation_severity_word_delete.png`

---

# 13. What is already supported

The current experiments support the following claims:

1. Linear SVM improves absolute performance over Logistic Regression under a
   fixed word TF-IDF representation.

2. The normalized lexical-corruption retention profile changes much less than
   clean accuracy when only the classifier is changed.

3. Character TF-IDF improves clean accuracy modestly across all three datasets.

4. Character TF-IDF provides a substantial robustness advantage under typo and
   character-deletion corruption.

5. The advantage reproduces across BANKING77, CLINC150, and HWU64.

6. The advantage grows with corruption severity.

7. The same advantage does not appear under word deletion.

---

# 14. Claims that are NOT yet supported

Do not currently claim:

- character TF-IDF is universally more robust,
- representation completely determines robustness,
- Linear SVM is universally more reliable than Logistic Regression,
- transformer models are inherently less reliable,
- the findings generalize beyond intent classification,
- the confidence intervals quantify full population uncertainty,
- a single universal reliability score has been established.

---

# 15. Remaining high-value experiments

The next experiments should be selected for scientific necessity rather than
simply increasing the number of models.

Priority candidates:

## A. Example-level paired bootstrap

Quantify uncertainty over test examples rather than only corruption
realizations.

## B. Transformer comparison

Extend the controlled benchmark to a fundamentally different learned
representation.

DistilBERT results already exist for BANKING77 but currently use one training
seed.

A stronger transformer experiment should eventually include:

- multiple training seeds,
- the same perturbation protocol,
- ideally more than one dataset if compute permits.

## C. Calibration

Evaluate:

- negative log-likelihood,
- Brier score,
- expected calibration error,
- confidence under corruption.

## D. Computational efficiency

Record:

- training time,
- inference time,
- model size,
- possibly memory usage.

---

# 16. Provisional conclusion

The current ReliabilityLab experiments demonstrate that clean accuracy and
robustness are not interchangeable properties.

Classifier choice can substantially improve absolute predictive performance
without materially changing normalized degradation under corruption.

In contrast, representation choice can strongly alter robustness, but the
effect depends on how corruption interacts with the representation.

Character n-gram TF-IDF increasingly outperforms word TF-IDF as character-level
corruption becomes more severe, yet offers almost no normalized advantage when
complete words are removed.

These findings motivate reliability evaluation as a multidimensional process
in which clean accuracy, representation, perturbation mechanism, severity, and
uncertainty should be examined separately rather than collapsed into a single
performance number.