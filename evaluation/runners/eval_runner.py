"""
Evaluation Runner for AI Model Steganalysis.
Runs the C++ engine over a dataset of clean/attacked models and computes metrics.
"""

import json
import subprocess
from pathlib import Path
from typing import List, Dict
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def run_evaluation(
    dataset_csv: str, 
    cli_path: str, 
    extract_script: str, 
    python_exe: str,
    output_dir: str
):
    df = pd.read_csv(dataset_csv)
    results = []
    
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    for idx, row in df.iterrows():
        model_path = row["model_path"]
        true_label = row["label"]
        
        # 1. Extract tensors
        scan_dir = out_path / f"scan_{idx}"
        subprocess.run([
            python_exe, extract_script,
            model_path, "--output-dir", str(scan_dir)
        ], check=True)
        
        # 2. Run scanner
        report_path = scan_dir / "report.json"
        subprocess.run([
            cli_path, str(scan_dir / "meta.json"),
            "--output", str(report_path)
        ], check=True)
        
        # 3. Read verdict
        with open(report_path) as f:
            report = json.load(f)
            
        risk = report.get("global_risk", {}).get("risk_score", 0.0)
        verdict = report.get("global_risk", {}).get("verdict", "CLEAN")
        pred_label = 1 if verdict in ("SUSPICIOUS", "MALICIOUS") else 0
        
        results.append({
            "model_path": model_path,
            "true_label": true_label,
            "pred_label": pred_label,
            "risk_score": risk,
            "verdict": verdict
        })
        
    return pd.DataFrame(results)

def calculate_metrics(results_df: pd.DataFrame) -> Dict[str, float]:
    y_true = results_df["true_label"]
    y_pred = results_df["pred_label"]
    y_scores = results_df["risk_score"]
    
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_scores) if len(y_true.unique()) > 1 else 0.0
    }
