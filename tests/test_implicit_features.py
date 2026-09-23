import pytest
import torch
import torch.nn as nn
from research.implicit_features.detector_model import evaluate_implicit_features, Model06Long

class DummyCandidate(nn.Module):
    def __init__(self, g):
        super().__init__()
        self.g = g

    def forward(self, x):
        # x is (N, C, H, W)
        N = x.shape[0]
        # output deterministic logits based on input spatial mean
        val = x.view(N, -1).mean(dim=1, keepdim=True)
        # Create different logits across classes to avoid uniform softmax
        offsets = torch.arange(self.g, device=x.device).float() * 0.1
        return val + offsets

def test_evaluate_implicit_features_not_classification():
    dummy_candidate = DummyCandidate(10)
    steg_model = Model06Long(10)
    probes = torch.randn(100, 3, 32, 32)
    
    result = evaluate_implicit_features(
        dummy_candidate,
        probes,
        steg_model,
        is_classification_model=False
    )
    assert result["supported"] is False
    assert result["message"] == "not applicable"

def test_evaluate_implicit_features_matrix_generation():
    g = 10
    N = 1000
    dummy_candidate = DummyCandidate(g)
    steg_model = Model06Long(g)
    
    # Generate a deterministic, non-random, category-balanced probe set.
    # For example, 10 categories, 100 images each.
    # We use linspace to avoid random noise and simulate structured "real" data.
    probes = torch.empty(N, 3, 32, 32)
    samples_per_class = N // g
    for c in range(g):
        start_idx = c * samples_per_class
        end_idx = start_idx + samples_per_class
        # Structured deterministic data representing class c
        probes[start_idx:end_idx] = torch.linspace(c, c+1, samples_per_class * 3 * 32 * 32).view(samples_per_class, 3, 32, 32)
        
    result = evaluate_implicit_features(
        dummy_candidate,
        probes,
        steg_model,
        is_classification_model=True
    )
    
    assert result["supported"] is True
    assert result["task_type"] == "image_classification"
    assert result["N"] == N
    assert result["num_classes"] == g
    assert result["feature_shape"] == [N, g]
    assert 0.0 <= result["stego_probability"] <= 1.0
