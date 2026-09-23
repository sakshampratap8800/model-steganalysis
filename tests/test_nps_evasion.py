import os
import pytest
import torch
import torch.nn as nn
import numpy as np
import onnx
import onnxruntime as ort
import tempfile
from collections import Counter
from attacks.nps.generate_nps import attack_onnx_with_nps

class DummyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(20, 5)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

def get_onnx_weights(onnx_path):
    model = onnx.load(onnx_path)
    from onnx import numpy_helper
    weights = {}
    for init in model.graph.initializer:
        weights[init.name] = numpy_helper.to_array(init)
    return weights

def compute_entropy(weights_flat, n_bits=4):
    # Same as gf_representation compute_lsb_entropy
    uint_view = weights_flat.astype('<f4').view('<u4')
    n = len(uint_view)
    if n == 0: return 0.0
    entropies = []
    for bit in range(n_bits):
        ones = int(np.sum((uint_view >> bit) & 1))
        p = ones / n
        if 0 < p < 1:
            entropies.append(-(p * np.log2(p) + (1 - p) * np.log2(1 - p)))
        else:
            entropies.append(0.0)
    return float(np.mean(entropies))

def test_nps_evasion_metrics():
    torch.manual_seed(42)
    model = DummyNet()
    
    # Generate dummy data
    X_val = torch.randn(100, 10)
    y_val = torch.randint(0, 5, (100,))
    
    with tempfile.TemporaryDirectory() as tmpdir:
        clean_onnx = os.path.join(tmpdir, "clean.onnx")
        nps_onnx = os.path.join(tmpdir, "nps.onnx")
        
        # Export clean model
        torch.onnx.export(
            model, X_val, clean_onnx,
            input_names=["input"], output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
        )
        
        # Attack with NPS
        # target_layer: 'fc1.weight', next_layer: 'fc2.weight'
        # In ONNX export from PyTorch, the names might be different. Let's find the names.
        clean_model_loaded = onnx.load(clean_onnx)
        init_names = [init.name for init in clean_model_loaded.graph.initializer]
        
        fc1_weight = next(n for n in init_names if "fc1.weight" in n or n.endswith(".weight") and "fc1" in n)
        fc2_weight = next(n for n in init_names if "fc2.weight" in n or n.endswith(".weight") and "fc2" in n)
        
        attack_onnx_with_nps(clean_onnx, nps_onnx, fc1_weight, fc2_weight)
        
        # Evaluate validation accuracy
        ort_sess_clean = ort.InferenceSession(clean_onnx)
        ort_sess_nps = ort.InferenceSession(nps_onnx)
        
        X_np = X_val.numpy()
        
        preds_clean = ort_sess_clean.run(None, {"input": X_np})[0]
        preds_nps = ort_sess_nps.run(None, {"input": X_np})[0]
        
        acc_clean = np.mean(np.argmax(preds_clean, axis=1) == y_val.numpy())
        acc_nps = np.mean(np.argmax(preds_nps, axis=1) == y_val.numpy())
        
        # Weight multiset equivalence & histogram distance
        w_clean = get_onnx_weights(clean_onnx)
        w_nps = get_onnx_weights(nps_onnx)
        
        # Flat arrays for the modified layers
        clean_w1 = w_clean[fc1_weight].flatten()
        nps_w1 = w_nps[fc1_weight].flatten()
        
        # Multiset equivalence (sets of weights are identical)
        # Using numpy sorting to check equivalence
        multiset_equiv = np.allclose(np.sort(clean_w1), np.sort(nps_w1), atol=1e-6)
        
        # Histogram distance
        hist_clean, bins = np.histogram(clean_w1, bins=50, density=True)
        hist_nps, _ = np.histogram(nps_w1, bins=bins, density=True)
        hist_distance = np.sum(np.abs(hist_clean - hist_nps))
        
        # Entropy
        entropy_clean = compute_entropy(clean_w1)
        entropy_nps = compute_entropy(nps_w1)
        
        # We just measure them, as requested.
        print(f"Clean Acc: {acc_clean}, NPS Acc: {acc_nps}")
        print(f"Multiset Equiv: {multiset_equiv}")
        print(f"Hist Distance: {hist_distance}")
        print(f"Entropy Clean: {entropy_clean}, NPS: {entropy_nps}")
        
        # We can assert multiset equivalence and acc match because mathematically they should,
        # but the prompt said "without asserting evasion success". Evasion success means whether
        # the detector catches it or not. Accuracy and multiset match are functional properties of NPS.
        assert np.isclose(acc_clean, acc_nps, atol=1e-5), "NPS should preserve accuracy!"
        assert multiset_equiv, "NPS should preserve multiset of weights!"
