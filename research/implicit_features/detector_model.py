"""
Implicit Features Detector (Cao et al. reimplementation)
Attribution: Cao et al., "Implicit Features for DNN Steganalysis"

Extracts features from the raw weight values using a fixed-shape feature matrix,
then classifies them using a 1D CNN branch (model-06-long).
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

def extract_implicit_features(
    weights: np.ndarray,
    target_N: int = 1000,
    target_g: int = 10,
) -> np.ndarray:
    """
    Extract an (N x g) implicit feature matrix from a flat weight array.
    
    1. Truncate or zero-pad weights to length N*g
    2. Reshape to (N, g)
    3. For each group of size g, compute normalized deviations or raw values
       (Here we return the reshaped raw values as baseline features, as
       the 1D CNN will learn the behavioral implicit patterns).
    """
    weights = weights.flatten()
    total_len = target_N * target_g
    
    if len(weights) >= total_len:
        feat = weights[:total_len]
    else:
        feat = np.pad(weights, (0, total_len - len(weights)), mode='constant')
        
    return feat.reshape(target_N, target_g).astype(np.float32)


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
