"""
Implicit Features Detector (Cao et al. reimplementation)
Attribution: Cao et al., "Steganalysis of neural networks using implicit features"

Extracts behavioral features from a neural network by feeding a fixed sequence 
of N probe images and collecting the g-dimensional softmax output probabilities,
then classifies this (N x g) implicit feature matrix using a 1D CNN branch (model-06-long).
"""

import math
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Feature Extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_implicit_features(
    candidate_model: nn.Module,
    probe_images: torch.Tensor,
    steganalysis_model: nn.Module,
    is_classification_model: bool = True,
    device: str = "cpu"
) -> dict:
    """
    Paper 3 Reproduction: Stegalaysis Using Implicit Features
    
    Pipeline:
    candidate_model -> classification_model? -> yes -> fixed probe images -> softmax -> (N x g) matrix -> model-06-long.
    
    Args:
        candidate_model: The target classification network to evaluate.
        probe_images: Tensor of shape (N, C, H, W) containing a deterministic, versioned, category-balanced fixed probe set.
        steganalysis_model: The trained model-06-long instance.
        is_classification_model: Boolean check. If False, returns 'not applicable'.
    
    Returns:
        Dict containing execution metadata and the final stego probability score.
    """
    if not is_classification_model:
        return {
            "supported": False,
            "task_type": "unknown",
            "message": "not applicable"
        }
        
    candidate_model = candidate_model.to(device)
    steganalysis_model = steganalysis_model.to(device)
    probe_images = probe_images.to(device)
    
    candidate_model.eval()
    steganalysis_model.eval()
    
    N = probe_images.shape[0]
    
    # 1. Feed fixed probe sequence to get output probabilities
    logits = candidate_model(probe_images)
    probs = F.softmax(logits, dim=1)  # Shape: (N, g)
    g = probs.shape[1]
    
    # 2. Preprocessing for model-06-long
    # Paper uses 1000 x 10 for the 10-class reproduction.
    # PyTorch 1D CNN expects input shape (Batch, Channels, Length)
    # So we treat classes 'g' as channels and 'N' as length: shape (1, g, N)
    implicit_matrix = probs.unsqueeze(0).transpose(1, 2)
    
    # 3. Model-06-long steganalysis classification
    stego_prob = steganalysis_model(implicit_matrix).item()
    
    return {
        "supported": True,
        "task_type": "image_classification",
        "N": N,
        "num_classes": g,
        "feature_shape": [N, g],
        "stego_probability": stego_prob
    }


# ---------------------------------------------------------------------------
# model-06-long (1D CNN Classifier)
# ---------------------------------------------------------------------------

class ResidualBlock1D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class Model06Long(nn.Module):
    """
    CNN Architecture described in Cao et al. (model-06-long)
    Input: (Batch, target_g, target_N) - Note: PyTorch expects (B, C, L)
    Output: 1 (Sigmoid score, 1 = stego, 0 = clean)
    """
    def __init__(self, in_channels: int = 10):
        super().__init__()
        # Initial convolution
        self.conv_in = nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn_in = nn.BatchNorm1d(64)
        
        # 2 Residual blocks
        self.res1 = ResidualBlock1D(64)
        self.res2 = ResidualBlock1D(64)
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        # Classifier
        self.fc = nn.Linear(64, 1)
        
    def forward(self, x):
        # x shape: (B, g, N) where g is treated as channels
        x = F.relu(self.bn_in(self.conv_in(x)))
        x = self.res1(x)
        x = self.res2(x)
        
        x = self.gap(x).squeeze(-1) # (B, 64)
        logits = self.fc(x)
        return torch.sigmoid(logits)


# ---------------------------------------------------------------------------
# Training Loop Wrapper
# ---------------------------------------------------------------------------

def train_implicit_detector(
    model: nn.Module,
    train_loader,
    epochs: int = 30,
    lr: float = 0.001,
    device: str = "cpu"
):
    """
    Train the behavioral model.
    Paper hyperparameters: Adam optimizer, LR=0.001, 30 epochs, BCE loss.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).float().unsqueeze(1)
            
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader):.4f}")
        
    return model
