# Algorithms Inventory

## 1. Dixon Q-Test (Trojan Signature)
Identifies statistically significant outliers in the row means of the final classification layer. Extremely sensitive to standard poisoned triggers.

## 2. Grayscale-Fourpart (GF)
Interprets a 32-bit floating point number as 4 individual uint8 pixels. Fed into a CNN for Few-Shot Learning classification.

## 3. Implicit Features
Extracts a uniform tensor window `N x g` across a layer and feeds it into a 1D CNN to behaviorally classify the topological fingerprint of a steganographic embedding.

## 4. Neuron Permutation Steganography (NPS)
Hides a payload by sorting neurons according to a specific permutation matrix rather than altering values. Detected by our Structural Analyzer (adjacent variance metrics).
