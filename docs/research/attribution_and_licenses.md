# Attribution and Licenses

## Clean-Room Implementation Declaration

This project incorporates several algorithms and methodologies published in academic literature. We explicitly declare that all implementations within this codebase are **clean-room implementations** built entirely from scratch based on the mathematical descriptions, equations, and methodologies detailed in the published papers.

**We did NOT copy, modify, or derive any source code from the original authors' repositories.**

## CC BY-NC-ND and Restrictive Licenses

Several original academic repositories utilize restrictive licenses, such as Creative Commons Attribution-NonCommercial-NoDerivatives (CC BY-NC-ND). 

### Model X-Ray
* **Original Repo:** `danigil/ModelXRay`
* **Original License:** CC BY-NC-ND
* **Our Implementation:** We implemented the FP32 to Grayscale-Fourpart (GF) representation and the associated detection pathway entirely from the mathematical and conceptual descriptions provided in the academic paper. No source code from `danigil/ModelXRay` was viewed or used to produce our implementation.

### Other Referenced Papers
The following concepts were also independently implemented from their respective papers to ensure zero license contamination:
* **Paper 2 (Trojan Signatures):** Row means and Dixon Q-test formulations.
* **Paper 3 (Implicit Features):** Probe sequence generation and behavioral feature matrices.
* **Paper 4 (Disarming Steganography):** LSB attacks (FMLA/HMLA/HBLA) and bit-level mitigations (FLP, K-LRBP, Quantization).
* **TransTroj & NPS:** Attack methodologies reconstructed as independent evaluation benchmarks.

By adhering to a strict clean-room methodology, this codebase avoids license contamination from external repositories, ensuring all code remains fully distinct from any third-party CC BY-NC-ND materials.
