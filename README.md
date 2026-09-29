# Conditional Diffusion for Rare Arrhythmia ECG Augmentation

**COSE362 Machine Learning · Korea University · Spring 2026 · Team I5**

Can a class-conditional diffusion model generate synthetic ECG beats that help a classifier detect *rare* arrhythmias? This project tests that question on the MIT-BIH Arrhythmia Database, and reports an honest negative result.

## The problem
In MIT-BIH, normal beats make up ~83% of the data. The most clinically dangerous arrhythmias are the rarest, so classifiers barely learn them.

## Approach
- **Data:** 48 MIT-BIH records → 186-sample beat windows around each R-peak (MLII lead), per-beat z-score normalization, stratified 80/20 split
- **Generator:** class-conditional 1D U-Net DDPM (T=200, linear β schedule, sinusoidal time embedding + class embedding)
- **Synthetic data:** 4,118 beats generated across 6 rare classes
- **Fair comparison:** all five settings train the *same* 1D CNN classifier, so differences come only from the augmentation method

| # | Method | Idea |
|---|---|---|
| 1 | Baseline | No augmentation |
| 2 | SMOTE | Linear interpolation in feature space |
| 3 | GAN (WGAN-GP) | Adversarial generation |
| 4 | Unconditional diffusion | Diffusion without class control |
| 5 | **Conditional diffusion (ours)** | Class-targeted generation |

## Results (rare-class metrics)

| Method | Rare-class macro F1 | Rare-class recall |
|---|---|---|
| Baseline | 0.842 | 0.768 |
| **SMOTE** | **0.868** | **0.853** |
| GAN | 0.847 | 0.805 |
| Unconditional diffusion | 0.796 | 0.736 |
| Conditional diffusion (ours) | 0.802 | 0.818 |

**The hypothesis was not supported.** Conditional diffusion improved rare-class *recall* over the baseline, but its F1 fell below both SMOTE and the baseline.

## Why, and what to try next
- **Weak conditioning:** the class embedding is simply added to the time embedding as one shared vector, with no dedicated pathway, so the label steers generation only weakly. Conditional was barely above unconditional (0.802 vs 0.796).
- **Short training:** 30 diffusion epochs, chosen for the Colab time budget.
- **Next steps:** FiLM-style or cross-attention class conditioning, confidence-based filtering of synthetic beats, more epochs and T=1000, 5-fold cross-validation, multi-lead input.

## Run it
Open `ecg_conditional_diffusion.ipynb` in Google Colab with a T4 GPU (Runtime → Change runtime type → GPU). The full run takes ~40–55 minutes and downloads MIT-BIH from PhysioNet automatically.

## Team and my role
Team I5: Ayaka (Runee Zubaida Zahid), Amani, Luai, Imran. Original team repository: [Bobbyy18/20261R0136COSE362](https://github.com/Bobbyy18/20261R0136COSE362)

**My role:** I implemented the code for the full pipeline: beat extraction and preprocessing, the conditional 1D U-Net DDPM, the WGAN-GP and unconditional-diffusion baselines, and the five-way evaluation.
- Assigned the evaluation part, but implemented the full pipeline end to end: preprocessing, conditional 1D U-Net DDPM, WGAN-GP and unconditional-diffusion baselines, and the five-way comparison.
