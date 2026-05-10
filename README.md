# Project 1: Neural Correlates of Emotion
## EEG-Based Classification of Affective Dimensions: Valence, Arousal, and Dominance

[![Paper Preprint](https://img.shields.io/badge/Zenodo-Paper%20Preprint-87CEEB?logo=zenodo)](https://doi.org/10.5281/zenodo.20108739)

---

**Author:** Shlok Khare | UC Davis | B.S. Cognitive Science & Computer Science, Minor Psychology

---

## What This Project Is

This project uses the DEAP dataset, a publicly available multimodal dataset containing 32-channel EEG recordings and peripheral physiological signals from 32 subjects watching music video stimuli, to build and evaluate a classification pipeline for emotional state across three affective dimensions: valence, arousal, and dominance. The central scientific question is whether these dimensions rely on neurally dissociable EEG oscillatory representations, with direct implications for the long-standing debate between Russell's (1980) two-axis circumplex model and Mehrabian and Russell's (1974) three-dimensional PAD model.

The pipeline extracts log-transformed Welch power spectral density features across four frequency bands (theta 4–8 Hz, alpha 8–13 Hz, beta 13–30 Hz, gamma 30–45 Hz) for all 32 EEG channels, plus three frontal alpha asymmetry features. SVM and Random Forest classifiers are trained independently per dimension under two evaluation regimes: leave-one-subject-out (LOSO) cross-subject generalization, and within-subject 5-fold cross-validation. A permutation importance analysis, anchored to held-out test data and validated against shuffled-label null distributions, identifies which frequency bands carry above-chance discriminative signal per dimension and whether those importance structures are shared or independent across classifier pairs.

---

## Scientific Motivation

Affective neuroscience predicts that valence, arousal, and dominance engage anatomically and functionally distinct neural circuits. Valence tracks approach-withdrawal motivation through frontal alpha asymmetry. Arousal reflects distributed thalamocortical and hippocampal-cortical dynamics across theta and parietal alpha. Dominance, as perceived situational control, implicates prefrontal regulatory networks anatomically separable from the limbic-frontal circuitry mediating valence. If these predictions hold in the oscillatory domain, classifiers trained independently on each dimension should recruit different frequency-band features, and pairwise importance correlation should be near zero.

Emotion classification in AI systems typically relies on behavioral proxies: text, facial expression, and voice. These reflect the surface of emotional experience rather than the neural activity generating it. A system informed by the frequency-domain dynamics of the brain generating an emotion is categorically different from one inferring affect from downstream behavior. This project is an early step toward that gap computationally, and more directly, it is the empirical foundation for grounding EMMCAI's emotion module in something more principled than sentiment scoring.

---

## Paper

**EEG-Based Classification of Affective Dimensions: Neural Dissociability of Valence, Arousal, and Dominance in Oscillatory EEG Features**
Shlok Khare, UC Davis, 2026.

Published: [https://doi.org/10.5281/zenodo.20108739](https://doi.org/10.5281/zenodo.20108739)

---

## Dataset

**DEAP Dataset** (Koelstra et al., 2012)
- Source: [eecs.qmul.ac.uk/mmv/datasets/deap](http://www.eecs.qmul.ac.uk/mmv/datasets/deap) (free via registration)
- Format: Preprocessed Python `.dat` files (`s01.dat` through `s32.dat`)
- 32 subjects, 40 trials per subject, 40 channels (32 EEG + 8 peripheral), 8064 samples per trial at 128 Hz
- Labels: valence, arousal, dominance, liking (self-reported, 1–9 SAM scale per trial)
- Preprocessing by dataset authors: downsampled to 128 Hz, bandpass filtered 4–45 Hz, EOG artifact removed, common-reference averaged, segmented into 63-second epochs (3 s baseline + 60 s signal)

```
data/
├── s01.dat
├── s02.dat
│   ...
└── s32.dat
```

---

## Project Structure

```
eegec-deap/
├── data/                        # Raw .dat files (32 subjects)
├── src/
│   ├── loader.py                # Load .dat files, baseline correction, shape validation
│   ├── features.py              # Constants (FS, BANDS, ELECTRODE_REGIONS, FAA_PAIRS),
│   │                            #   Welch PSD extraction, FAA features, feature naming
│   └── models.py                # LOSO loop, within-subject CV, permutation importance,
│                                #   band ablation, shuffled-label null distributions
├── notebooks/
│   ├── 01_data.ipynb            # Data exploration and validation; saves raw arrays
│   ├── 02_features.ipynb        # Feature extraction and sanity checks; saves feature matrix
│   ├── 03_model.ipynb           # Main LOSO + within-subject CV + null distributions
│   ├── 03.5_model.ipynb         # LOSO SVM importance + MI robustness checks
│   └── 04_results.ipynb         # All paper figures, RQ-specific tests, stimulus confound check
├── paper/
│   ├── figures/                 # Final paper figures (300 dpi)
│   ├── paper.tex
│   └── paper.pdf
├── results/
│   ├── figures/                 # All generated figures
│   ├── null_checkpoints/        # Batched null distribution checkpoints
│   ├── loso_results.pkl
│   ├── within_subject.pkl
│   ├── svm_importance.pkl
│   ├── importance_agg.pkl
│   ├── importance_corr.pkl
│   ├── null_results.pkl
│   ├── X_features.npy           # (1280, 131) EEG feature matrix
│   ├── X_peripheral_features.npy
│   ├── y.npy                    # (1280, 4) label array
│   └── subject_ids.npy
├── README.md
└── roadmap.md
```

---

## Research Questions

Five research questions organized the analysis. Each is listed with its conclusion.

**RQ1:** Do EEG frequency bands contribute asymmetrically across affective dimensions in cross-subject classification?
*Conclusive. Beta-only (0.537, p=0.010) and gamma-only (0.543, p=0.010) single-band models formally exceed matched per-band null distributions for cross-subject valence classification. No band exceeds its null for arousal or dominance at any threshold.*

**RQ2:** Does frontal alpha asymmetry function as a reliable valence discriminator under controlled conditions?
*Non-determinable on DEAP. FAA-only within-subject SVM yields chance-level valence accuracy (0.505, p=0.430). Direct Mann-Whitney U on raw FAA values shows negative mean effect sizes across all three electrode pairs, opposite to Davidson's prediction, attributable to DEAP's compressed valence distribution (median 3.08).*

**RQ3:** Does the generalization gap between within-subject and cross-subject accuracy differ systematically across affective dimensions?
*Non-determinable at N=32. Directional ordering is theoretically coherent (valence gap smallest, arousal larger) but paired Wilcoxon tests do not survive BH-FDR correction. Estimated 120–200 subjects required.*

**RQ4:** Do EEG and peripheral physiological channels contribute differentially per affective dimension?
*Directional only. EEG outperforms peripheral for valence (SVM: 0.525 vs. 0.495); peripheral outperforms EEG for arousal (SVM: 0.513 vs. 0.504). No comparison survived BH-FDR correction across 18 paired tests at N=32.*

**RQ5:** Are valence and dominance neurally dissociable in EEG oscillatory feature space, inconsistent with Russell's circumplex reduction?
*Exploratory, model-independent evidence for dissociability. Valence-dominance importance correlation is near zero across all four classifier conditions spanning two architectures and two evaluation regimes (range -0.040 to +0.040), with the pattern reproduced on a mutual-information-selected feature subset. No condition produces the positive overlap expected under shared feature basis. Arousal-dominance is sign-inconsistent across conditions and is treated as inconclusive.*

---

## Results

### Classification Accuracy (EEG-only)

| Regime | Model | Valence | Arousal | Dominance |
|--------|-------|---------|---------|-----------|
| Within-subject | SVM | 0.536 | 0.556 | 0.572 |
| Within-subject | RF  | 0.548 | 0.562 | 0.558 |
| LOSO           | SVM | 0.525 | 0.504 | 0.550 |
| LOSO           | RF  | 0.491 | 0.521 | 0.490 |

Binomial sign test (per-subject within-subject SVM): dominance 24/31 subjects above chance (p=0.0017), arousal 22/32 (p=0.0251), valence 17/32 (p=0.430, not significant). LOSO RF does not exceed the shuffled-label null for any target (null p95: valence 0.534, arousal 0.529, dominance 0.527).

### Primary Findings

**Finding 1 (RQ1, formally established):** Beta-only and gamma-only single-band models formally exceed matched per-band null distributions for cross-subject valence classification (both p=0.010). No equivalent result appears for arousal or dominance at any band. The full 131-feature model (0.491) underperforms both single-band valence models, confirming that theta and alpha introduce individually variable noise that degrades cross-subject valence transfer.

**Finding 2 (RQ5, exploratory):** Valence and dominance recruit distinct EEG oscillatory feature structures across four classifier conditions spanning two architectures and two evaluation regimes, reproduced under mutual-information-selected feature subsets. This constitutes model-independent exploratory evidence for neural dissociability, consistent with the PAD model and inconsistent with a feature-sharing prediction of Russell's circumplex reduction.

**Finding 3:** Dominance is simultaneously the most individually discriminable (highest within-subject accuracy, strongest binomial sign test) and the most individually idiosyncratic (LOSO RF falls below null mean at 0.490) of the three dimensions, reflecting neural encoding too specific to individual regulatory history to produce a transferable cross-subject template.

**Finding 4:** LOSO RF does not exceed the shuffled-label null for any affective target, replicating the cross-subject distribution shift problem in EEG-BCI and establishing the within-subject regime as the boundary of importance-based interpretability claims.

---

## Dependencies

```
numpy
scipy
scikit-learn
mne
matplotlib
seaborn
pickle (stdlib)
jupyter
```

---

## Part of a Larger Research Arc

This is Project 1 of a five-project research portfolio at the intersection of computational neuroscience, cognitive science, and AI. Project 1 establishes the empirical and computational foundation for what emotional state looks like at the neural signal level, and the subsequent projects build on both the pipeline and the findings.


