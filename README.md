# Project 1: Neural Correlates of Emotion
## EEG-Based Valence and Arousal Classification

**Author:** Shlok Khare | UC Davis | B.S. Computer Science & Cognitive Science
**Timeline:** April 15 – May 31, 2026 (~6 weeks)
**Status:** In Progress (Draft v1 — April 2026)

---

## What This Project Is

This project uses the DEAP dataset — a publicly available, widely cited multimodal dataset containing 32-channel EEG recordings and peripheral physiological signals from 32 subjects watching music video stimuli — to build a classification system for emotional state along the two-dimensional valence-arousal circumplex model of affect.

The pipeline involves frequency-band power feature extraction across theta (4–8 Hz), alpha (8–13 Hz), beta (13–30 Hz), and gamma (30+ Hz) bands, electrode region grouping consistent with established frontal asymmetry and parietal lateralization literature, and a machine learning classifier trained to predict self-reported valence and arousal ratings. A secondary interpretability analysis identifies which frequency signatures and electrode regions carry the most discriminative information per emotional dimension.

---

## Scientific Motivation

Affective neuroscience has established that emotional valence and arousal are neurobiologically dissociable: frontal alpha asymmetry correlates with approach-withdrawal motivation (valence), while parietal and occipital oscillations track arousal-driven attentional engagement. Translating these findings into a working computational pipeline bridges the gap between neuroimaging research and real-time AI applications, including affective computing systems, neuroprosthetics, and human-computer interaction.

Emotion classification from behavioral proxies — text, facial expression, voice — has received substantial research attention, but the underlying neural substrate of emotional experience remains underutilized in AI system design. There is a meaningful difference between a system that infers emotion from surface behavior and one that can be informed by the frequency-domain dynamics of the brain generating that emotion. This project is an early step toward bridging that gap computationally, and more personally, it is the empirical foundation for grounding EMMCAI's emotion module in something more principled than sentiment scoring.

---

## Dataset

**DEAP Dataset** (Koelstra et al., 2012)
- Source: [eecs.qmul.ac.uk/mmv/datasets/deap](http://www.eecs.qmul.ac.uk/mmv/datasets/deap) (free via registration)
- Format used: Preprocessed Python `.dat` files (`s01.dat` – `s32.dat`)
- 32 subjects, 40 trials per subject, 40 channels (32 EEG + 8 peripheral), 8064 samples per trial at 128 Hz
- Labels: valence, arousal, dominance, liking (self-reported, 1–9 scale per trial)
- Preprocessing already applied by dataset authors: downsampled to 128 Hz, bandpass filtered 4–45 Hz, EOG artifact removed, data segmented and baseline-normalized

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
project1-eeg-emotion/
├── data/                   # Raw .dat files (32 subjects)
├── src/
│   ├── loader.py           # Load and unify all .dat files
│   ├── features.py         # Frequency-band PSD extraction and electrode grouping
│   ├── classifier.py       # Model training, cross-validation, evaluation
│   ├── interpret.py        # Feature importance and electrode-level analysis
│   └── visualize.py        # All plotting and figure generation
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_analysis.ipynb
│   └── 03_results.ipynb
├── outputs/
│   ├── figures/
│   └── results/
├── main.py                 # End-to-end pipeline orchestration
├── README.md
└── roadmap.md
```

---

## Research Questions

1. Can band-limited EEG power features reliably discriminate high vs. low valence and arousal states across subjects?
2. Which frequency bands contribute most differentially to valence vs. arousal classification?
3. Does frontal alpha asymmetry emerge as a dominant feature for valence, consistent with the neurobiological approach-withdrawal hypothesis?
4. How does cross-subject (leave-one-subject-out) generalization compare to within-subject accuracy, and what does the gap reveal about individual differences in neural emotion representation?
5. Do EEG channels and peripheral physiological channels contribute differentially to each emotional dimension?

*Note: Research questions will be refined and finalized after the literature review phase. See roadmap.md.*

---

## Expected Outcomes

The classification system will produce accuracy and F1 metrics across valence and arousal dimensions, with arousal generally expected to be more separable from EEG signal based on prior literature. The interpretability analysis will produce electrode-level and frequency-band-level importance maps comparable against existing neuroimaging findings on emotional lateralization. Secondary outcomes include characterization of cross-subject generalization performance and an analysis of which physiological channels contribute most differentially to each emotional dimension.

---

## Dependencies

```
numpy
scipy
scikit-learn
mne
matplotlib
seaborn
shap
pickle (stdlib)
jupyter
```

---

## Part of a Larger Research Arc

This is Project 1 of a five-project research portfolio at the intersection of computational neuroscience, cognitive science, and AI. Project 1 establishes the empirical and computational foundation of what emotional state looks like at the neural signal level that the subsequent projects build on.

