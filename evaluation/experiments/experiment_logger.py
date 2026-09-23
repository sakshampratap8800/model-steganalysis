"""
Experiment Logger.
Records experiment hyper-parameters and outputs to a JSONL file.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

class ExperimentLogger:
    def __init__(self, log_path: str):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        
    def log_run(self, experiment_name: str, config: dict, metrics: dict):
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "experiment": experiment_name,
            "config": config,
            "metrics": metrics
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
