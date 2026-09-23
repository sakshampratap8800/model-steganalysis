# Research Mapping Audit

**Date:** 2026-09-23  
**Objective:** Audit the current codebase to identify where algorithms have been misattributed to cited academic papers, evaluate the algorithmic mismatches, and define the required corrections.

> **Principle:** "A mathematically related feature is not automatically an implementation of the cited paper."

---

## 1. Model X-Ray
* **Claimed Paper/Repo:** Model X-Ray (danigil/ModelXRay)
* **Actual Algorithm Implemented:** `ieee754_analyzer.cpp` calculates the Shannon entropy of the lower mantissa bits of floating-point weights.
* **Actual Source Support:** Mantissa entropy is a mathematically sound baseline feature for detecting LSB modification (e.g., FMLA), but it is *not* the detection algorithm proposed by Model X-Ray.
* **Mismatch:** Misattributed bit-level entropy as the primary Model X-Ray detector.
* **Correction Required:** 
  * Retain mantissa entropy explicitly labeled as a "Baseline Numerical Feature", not Model X-Ray.
  * Implement the *actual* Model X-Ray representation: FP32 parameter representation -> Grayscale-Fourpart (GF) image representation.
  * Implement the actual few-shot detection pathway on the GF representation.

---

## 2. Trojan Signatures
* **Claimed Paper/Repo:** Trojan Signatures in DNN Weights (Paper 2)
* **Actual Algorithm Implemented:** `trojan_q.cpp` implements the Dixon Q-Test.
* **Actual Source Support:** The paper does use Dixon Q-Test, but it applies it specifically to the *final classification-layer row means*.
* **Mismatch:** If applied with an arbitrary universal threshold or applied to arbitrary layers rather than the final classifier, it deviates from the paper. Furthermore, the adaptive Q-suppression regularization was conflated with detection rather than evaluation.
* **Correction Required:**
  * Keep `trojan_q.cpp` but enforce extraction of final classification-layer row means.
  * Use strictly tabulated/validated critical values based on sample size (classes) rather than an arbitrary threshold.
  * Move adaptive Q-suppression regularization strictly into an attack/evasion experiment script.

---

## 3. Implicit Features
* **Claimed Paper/Repo:** Steganalysis of Neural Networks Using Implicit Features (Paper 3)
* **Actual Algorithm Implemented:** `detector_model.py` reshapes raw weight arrays into an N x g matrix and passes it through the `model-06-long` CNN. Penultimate layer hooks were also used in related ML modules.
* **Actual Source Support:** Paper 3 does not analyze weights. It treats the model as a black box, feeding it a *fixed probe image sequence* to extract *output probabilities*, creating a 1000 x g behavioral feature matrix.
* **Mismatch:** Massive algorithmic deviation. Reshaping weights into a matrix is completely unrelated to the paper's behavioral image-probe methodology.
* **Correction Required:**
  * Remove any claim that penultimate-layer hooks or reshaped weights are the paper's implementation.
  * Implement: Fixed probe image sequence -> model outputs -> 1000 x g implicit feature matrix D_k.
  * Retain `model-06-long` as the CNN, but train/evaluate it strictly on the probability output matrix.

---

## 4. TransTroj
* **Claimed Paper/Repo:** TransTroj (haowang02/TransTroj)
* **Actual Algorithm Implemented:** Extractor hooks combined with trigger generation (`transtroj_pipeline.py`), previously conflated with the implicit features detector.
* **Actual Source Support:** TransTroj is an *attack methodology* (supply chain poisoning via embedding indistinguishability), not a steganalysis detector. 
* **Mismatch:** TransTroj was incorrectly mapped as a detection/feature component.
* **Correction Required:**
  * Separate completely from the Implicit Features pipeline.
  * Implement/use it exclusively as an attack/evaluation benchmark.
  * The pipeline must cleanly reflect: Trigger optimization -> victim Pre-Trained Model (PTM) optimization -> downstream task transfer.

---

## 5. Neuron Permutation Steganography (NPS)
* **Claimed Paper/Repo:** NPS (albblgb/NPS)
* **Actual Algorithm Implemented:** `generate_nps.py` for attack generation, and `structural_analyzer.cpp` (Gini/Spearman rank correlation) as the detector.
* **Actual Source Support:** The NPS repository/paper proposes the permutation *attack*. It does not propose Gini/Spearman as a detection mechanism. 
* **Mismatch:** Claiming Gini/Spearman is the NPS paper's detector.
* **Correction Required:**
  * Keep the NPS generator as the attack benchmark.
  * Relabel Gini/Spearman explicitly as a project-proposed "Structural Evidence" heuristic, *not* the NPS paper's detector.
  * Add explicit automated tests showing whether permutation attacks successfully evade ordinary weight-distribution statistics (like mantissa entropy).

---

## 6. Disarming Steganography
* **Claimed Paper/Repo:** Disarming Steganography Attacks Inside Neural Network Models (Paper 4)
* **Actual Algorithm Implemented:** FMLA, HMLA, and HBLA attack generation.
* **Actual Source Support:** The paper introduces both the LSB attacks *and* countermeasures (Full LSB Prevention, K-LSB Random Bits Prevention, Quantization).
* **Mismatch:** The countermeasures (Disarming/CDR) are missing or merged incorrectly.
* **Correction Required:**
  * Keep FMLA/HMLA/HBLA explicitly as controlled attack generators.
  * Add Full LSB Prevention (FLP), K-LSB Random Bits Prevention (K-LRBP), and 8-bit Quantization (Qint8) as separate mitigation/reconstruction experiments.
  * Ensure mitigation logic is strictly separated from detection logic.
