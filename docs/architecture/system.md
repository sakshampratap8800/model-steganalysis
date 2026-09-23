# System Architecture

The AI Model Steganalysis platform is divided into three major boundaries:
1. **The C++ Scanner Core (`scanner_cli`)**
   - High speed binary execution.
   - Extracts Bit-Level, Statistical, and Topological metrics in milliseconds.
2. **The FastAPI Backend (`backend/`)**
   - Receives uploads, stages files, and orchestrates the CLI.
   - Holds the PyTorch Deep Learning models (`model-06-long` and `GFCNNBackbone`).
3. **The React UI (`frontend/`)**
   - Renders interactive dashboards based on the JSON reports produced by the C++ core.
