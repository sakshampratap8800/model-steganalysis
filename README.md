# AI Model Steganalysis Platform

A research-grade platform for detecting steganographic payloads and backdoors embedded in neural network weights.

## Aim
The primary aim of this project is to provide a robust, fast, and highly extensible framework to detect malicious modifications—such as backdoors or hidden payloads—within deep learning models. By combining high-speed native C++ analysis with machine learning heuristics, it helps researchers and engineers audit third-party models before deployment, ensuring AI supply chain security.

## Overview
This platform implements multiple orthogonal detection branches to catch sophisticated attacks (including FMLA/HMLA, X-LSB-Fill, NPS, and TransTroj):
1. **IEEE-754 Bit Analysis:** Exposes LSB modifications via mantissa entropy.
2. **Structural / Topological Analysis:** Exposes Neuron Permutation Steganography (NPS).
3. **Behavioral Analysis:** Uses `model-06-long` to classify implicit features.
4. **Trojan Signatures:** Uses the Dixon Q-Test to find manipulated classification boundaries.
5. **Few-Shot Learning:** Prototypical networks operating on Grayscale-Fourpart (GF) representations.

## Architecture
- **C++ Core:** High-speed scanning and feature extraction without loading the model into GPU memory.
- **Python / FastAPI Backend:** Orchestrates the C++ core and provides machine-learning branches.
- **React Frontend:** Web UI for dragging and dropping ONNX files and reviewing risk reports.

## Usage
1. Start Backend: `uvicorn main:app --reload`
2. Start Frontend: `npm run dev`
3. Navigate to `http://localhost:3000`

## License / Attribution
This project includes independent reimplementations of algorithms published in peer-reviewed literature. No code was copied from restricted repositories (e.g., CC BY-NC-ND sources). See `docs/research/attribution.md` for full citations.
