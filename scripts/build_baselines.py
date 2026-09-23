"""
Baseline builder for Steganalysis Core.
Extracts statistics from known-clean models to produce a baseline dictionary.
"""

import json
import torch
import torch.nn as nn
import numpy as np

def compute_entropy(weights):
    hist, _ = np.histogram(weights, bins=256, density=False)
    total = np.sum(hist)
    if total == 0:
        return 0.0
    probs = hist / total
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))

def profile_model(model: nn.Module, arch_name: str) -> dict:
    profile = {}
    for name, param in model.named_parameters():
        if "weight" in name and param.dim() >= 2:
            w = param.detach().cpu().numpy()
            profile[name] = {
                "mean_expected": float(np.mean(w)),
                "std_expected": float(np.std(w)),
                "variance_expected": float(np.var(w)),
                "entropy_min": float(compute_entropy(w) * 0.9),
                "entropy_max": float(compute_entropy(w) * 1.1)
            }
    return {arch_name: profile}

def generate_real_baseline(output_path: str):
    # Create real, albeit small, reference models
    resnet_mock = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.Conv2d(16, 32, 3, padding=1)
    )
    # Initialize with Kaiming He (standard for ResNets)
    for m in resnet_mock.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            
    vgg_mock = nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, padding=1)
    )
    for m in vgg_mock.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)

    baselines = {}
    baselines.update(profile_model(resnet_mock, "resnet"))
    baselines.update(profile_model(vgg_mock, "vgg"))
    
    with open(output_path, "w") as f:
        json.dump(baselines, f, indent=2)
    print(f"Real baseline written to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    generate_real_baseline(args.output)
