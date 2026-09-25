# AI Model Steganalysis Platform

A research-grade full-stack platform to detect hidden steganographic payloads and neural backdoors in deep learning models (ONNX/PyTorch), accurately reproducing 4 state-of-the-art academic papers.

**GitHub:** https://github.com/sakshampratap8800/model-steganalysis

---

## Architecture

```
React (TypeScript)  →  Express.js (Node.js)  →  C++ Core (scanner_cli + onnx_extract)
     Port 3000               Port 8000               Native binary
```

- **Frontend:** React + TypeScript + Tailwind CSS — cybersecurity-themed risk dashboard
- **Backend:** Express.js — receives model uploads, orchestrates C++ binaries via `child_process.spawn`
- **C++ Core:** High-performance tensor scanner with 6-stage Evidence Fusion Layer, IEEE-754 bit analysis, Dixon Q-test Trojan detection, and calibrated risk scoring
- **No Python dependency at runtime**

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React 18, TypeScript, Tailwind CSS, Recharts |
| Backend | Express.js, Node.js, Multer, better-sqlite3 |
| Analysis Engine | C++17 (CMake) |
| Model Format | ONNX (parsed natively in C++) |
| Database | SQLite |

---

## Running Locally

### Prerequisites
- Node.js ≥ 18
- CMake ≥ 3.20
- A C++17 compiler (MSVC / GCC / Clang)

### 1. Build the C++ core

```powershell
cmake -B build
cmake --build build --config Debug
```

This produces:
- `build/core/Debug/scanner_cli.exe` — full analysis engine
- `build/core/Debug/onnx_extract.exe` — ONNX → meta.json + weights.bin

### 2. Start the Express server

```powershell
cd server
npm install
npm start
# API running at http://localhost:8000
```

### 3. Start the React frontend

```powershell
cd frontend
npm install
npm run dev
# UI running at http://localhost:3000
```

Open **http://localhost:3000**, upload any `.onnx` model, and watch the platform scan it.

---

## Detection Methods

| Method | Source | What it detects |
|---|---|---|
| Baseline Bit-Level Entropy | IEEE-754 mantissa analysis | LSB steganography (FMLA, HMLA, HBLA) |
| Trojan Signature (Dixon Q-test) | Fields et al. (2021) | Neural backdoors in classification heads |
| Structural Evidence | Project-proposed | Neuron permutation anomalies (Gini / Spearman) |
| Evidence Fusion Layer | Project-proposed | Calibrated 6-stage risk probability |
| Grayscale-Fourpart (GF) | Model X-Ray (Gilkarov & Dubin 2024) | Few-shot detection via weight image representation |
| Implicit Features | Paper 3 | Softmax distribution steganography |

---

## Academic References

1. Gilkarov & Dubin — *Model X-Ray: Detection of Hidden Malware in AI Model Weights using Few Shot Learning* (arXiv:2409.19310)
2. Fields et al. — *Trojan Signatures in DNN Weights* (2021)
3. (Paper 3) — Implicit Feature Steganalysis of Neural Networks
4. Dubin — *Disarming Steganography Attacks Inside Neural Networks* (2022)
