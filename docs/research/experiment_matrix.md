# Experiment Matrix

This document defines the evaluation runs required to benchmark the project's detection heuristics, behavioral analyzers, and attack generators.

## 1. LSB Steganography Evasion Rates
**Objective:** Measure the detectability of bit-level substitution attacks using baseline and advanced features.
* **Attacks:** FMLA (23-bit), HMLA (12-bit), HBLA (4-bit).
* **Detectors:** 
  * Baseline Mantissa Entropy
  * Model X-Ray GF Representation + Few-Shot Classifier
* **Metrics:** Attack Capacity (MB), Clean Accuracy Drop, True Positive Rate (TPR) at 1% False Positive Rate (FPR).

## 2. NPS vs. Mantissa Entropy & Structural Evidence
**Objective:** Verify that permutation-based steganography successfully evades simple bit-level heuristics but can be caught by structural analysis.
* **Attacks:** NPS (Neuron Permutation Steganography).
* **Detectors:**
  * Baseline Mantissa Entropy (Expected to evade/fail).
  * Gini/Spearman Rank Correlation (Structural Evidence).
* **Metrics:** Evasion Rate (against Mantissa Entropy), Detection Rate (Structural Evidence).

## 3. TransTroj Downstream Evaluation
**Objective:** Evaluate transferable trojans from Pre-Trained Models (PTMs) to downstream tasks.
* **Attacks:** TransTroj injected into victim PTMs.
* **Evaluation Pipeline:** Transfer poisoned PTM to downstream classification task -> Apply Trigger.
* **Metrics:** 
  * Clean Data Accuracy (CDA) on the downstream task.
  * Attack Success Rate (ASR) on triggered downstream data.
  * Detection rate using Implicit Behavioral Features (Paper 3) and Trojan Signatures (Paper 2).

## 4. Adaptive Trojan Signature Suppression
**Objective:** Evaluate the robustness of the Dixon Q-Test detector against an adaptive attacker.
* **Attacks:** Trojan insertion with adaptive loss $L_{reg} = L_{CE} + \gamma [E[W_t] - E[W]]$.
* **Variables:** Vary $\gamma$ (regularization strength).
* **Metrics:** Dixon Q-Statistic, Clean Accuracy, Attack Success Rate (ASR).

## 5. Disarming / CDR (Content Disarm and Reconstruction)
**Objective:** Measure the efficacy of zero-trust sanitization methods.
* **Attacks:** FMLA, HMLA, HBLA, TransTroj.
* **Mitigations:** Full LSB Prevention (FLP), K-LSB Random Bits Prevention (K-LRBP), Qint8 Quantization.
* **Metrics:** Post-mitigation Attack Success Rate (Sanitization Efficacy), Post-mitigation Clean Accuracy (Utility Loss).
