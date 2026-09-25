"""
ModelXRay Few-Shot Detector (Independent Reimplementation)

Attribution: Gilkarov & Dubin, "Model X-Ray: Detection of Hidden Malware in
AI Model Weights using Few Shot Learning", arXiv:2409.19310.

Reimplemented from paper description. No code copied from danigil/ModelXRay.

Uses Prototypical Networks (Snell et al. 2017) to classify GF representations
of model weights as Clean (0) or Stego (1). Uses easy-few-shot-learning (MIT).
"""

import os
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Resize, ToTensor

# Fallback implementation of Prototypical Networks if easyfsl is unavailable
class PrototypicalNetwork(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, support_images, support_labels, query_images):
        """
        support_images: (n_support, C, H, W)
        support_labels: (n_support,)
        query_images: (n_query, C, H, W)
        """
        # Extract features
        support_features = self.backbone(support_images) # (n_support, feature_dim)
        query_features = self.backbone(query_images)     # (n_query, feature_dim)
        
        # Compute prototypes
        classes = torch.unique(support_labels)
        n_classes = len(classes)
        prototypes = torch.zeros(n_classes, support_features.shape[1], device=support_images.device)
        
        for i, c in enumerate(classes):
            mask = (support_labels == c)
            prototypes[i] = support_features[mask].mean(dim=0)
            
        # Compute distances (Euclidean)
        # ||q - p||^2 = ||q||^2 + ||p||^2 - 2<q, p>
        q_norm = (query_features ** 2).sum(dim=1, keepdim=True)
        p_norm = (prototypes ** 2).sum(dim=1, keepdim=True).t()
        dot = torch.mm(query_features, prototypes.t())
        
        distances = q_norm + p_norm - 2.0 * dot
        
        # Return negative distances as logits
        return -distances


class GFCNNBackbone(nn.Module):
    """
    Standard CNN backbone for ModelXRay.
    Accepts 1-channel GF images, outputs flat feature vectors.
    """
    def __init__(self, output_dim=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, output_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.output_dim = output_dim

    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1)


class GFDataset(Dataset):
    """Dataset for pre-computed GF images (.npy) or raw onnx files"""
    def __init__(self, items: List[Tuple[str, int]], transform=None):
        """
        items: List of (path, label)
        """
        self.items = items
        self.transform = transform
        
    def __len__(self):
        return len(self.items)
        
    def __getitem__(self, idx):
        path, label = self.items[idx]
        
        if path.endswith('.npy'):
            import numpy as np
            gf_array = np.load(path)
        else:
            # Assume onnx, generate GF on the fly
            # In a real training loop, this should be precomputed!
            import sys
            sys.path.append(str(Path(__file__).parent))
            from gf_representation import onnx_model_to_gf_image
            gf_array = onnx_model_to_gf_image(path)
            
        # Convert uint8 to PIL Image for torchvision transforms
        from PIL import Image
        img = Image.fromarray(gf_array, mode='L')
        
        if self.transform:
            img = self.transform(img)
        else:
            img = ToTensor()(img)
            
        return img, label

def build_few_shot_model(output_dim=128):
    backbone = GFCNNBackbone(output_dim=output_dim)
    return PrototypicalNetwork(backbone)

def get_default_transform(size=256):
    return Compose([
        Resize((size, size)),
        ToTensor(),
    ])
