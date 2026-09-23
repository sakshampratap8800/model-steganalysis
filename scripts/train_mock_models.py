"""
Train and Save ML Detectors (Phase 8 & Phase 7)
Generates the `.pt` weight files required by the inference engine.
Trains on separable synthetic distributions to ensure a real decision boundary is learned.
"""

import os
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'research/implicit_features'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'research/model_xray_adapter'))

from detector_model import Model06Long
from few_shot_detector import GFCNNBackbone

def train_and_save():
    models_dir = Path(__file__).parent.parent / "backend" / "models" / "weights"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    print("Training Implicit Features 1D CNN (model-06-long)...")
    model_implicit = Model06Long(in_channels=10)
    optimizer = torch.optim.Adam(model_implicit.parameters(), lr=0.005)
    criterion = nn.BCELoss()
    
    # Generate linearly separable synthetic training data
    # Clean: Mean=0.0, Std=0.1
    X_clean = torch.randn(64, 10, 1000) * 0.1
    y_clean = torch.zeros(64, 1)
    
    # Attacked (LSB shift): Mean=0.05, Std=0.15 + high frequency noise
    X_attacked = (torch.randn(64, 10, 1000) * 0.15) + 0.05
    y_attacked = torch.ones(64, 1)
    
    X_train = torch.cat([X_clean, X_attacked], dim=0)
    y_train = torch.cat([y_clean, y_attacked], dim=0)
    
    model_implicit.train()
    for epoch in range(15):
        optimizer.zero_grad()
        preds = model_implicit(X_train)
        loss = criterion(preds, y_train)
        loss.backward()
        optimizer.step()
        if epoch % 5 == 0:
            acc = ((preds > 0.5).float() == y_train).float().mean().item()
            print(f"Epoch {epoch}: Loss {loss.item():.4f}, Acc {acc:.2f}")
        
    implicit_path = models_dir / "detector_06_long.pt"
    torch.save(model_implicit.state_dict(), implicit_path)
    print(f"Saved -> {implicit_path}")
    
    print("Training ModelXRay Few-Shot Backbone (GF representation)...")
    model_gf = GFCNNBackbone(output_dim=128)
    optimizer_gf = torch.optim.Adam(model_gf.parameters(), lr=0.005)
    
    # Triplet margin loss for few-shot embedding learning
    criterion_gf = nn.TripletMarginLoss(margin=1.0)
    
    # Generate anchor, positive, negative triplets
    # Anchors (Clean)
    anchors = torch.randn(16, 1, 256, 256)
    # Positives (Clean, augmented slightly)
    positives = anchors + (torch.randn_like(anchors) * 0.05)
    # Negatives (Attacked, structural distortion)
    negatives = torch.randn(16, 1, 256, 256) + 1.0
    
    model_gf.train()
    for epoch in range(15):
        optimizer_gf.zero_grad()
        emb_a = model_gf(anchors)
        emb_p = model_gf(positives)
        emb_n = model_gf(negatives)
        
        loss = criterion_gf(emb_a, emb_p, emb_n)
        loss.backward()
        optimizer_gf.step()
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Triplet Loss {loss.item():.4f}")
        
    gf_path = models_dir / "few_shot_gf.pt"
    torch.save(model_gf.state_dict(), gf_path)
    print(f"Saved -> {gf_path}")
    
    print("All models successfully trained on separable data and saved!")

if __name__ == "__main__":
    train_and_save()
