import pytest
import torch
import torch.nn as nn
import numpy as np
from attacks.mitigation.disarm import apply_flp, apply_k_lrbp, apply_qint8

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.fc = nn.Linear(16 * 30 * 30, 10)
    
    def forward(self, x):
        return self.fc(self.conv1(x).view(x.size(0), -1))

def test_flp_correctness_and_impact():
    torch.manual_seed(42)
    np.random.seed(42)
    model = DummyModel()
    
    orig_weight = model.conv1.weight.data.clone()
    dummy_in = torch.randn(1, 3, 32, 32)
    out_orig = model(dummy_in)
    
    disarmed_model = apply_flp(model, target_layers=['conv1'])
    disarmed_weight = disarmed_model.conv1.weight.data
    
    orig_u32 = orig_weight.cpu().numpy().view(np.uint32)
    disarmed_u32 = disarmed_weight.cpu().numpy().view(np.uint32)
    
    mask = np.uint32(0xFF800000)
    assert np.array_equal(orig_u32 & mask, disarmed_u32 & mask), "FLP should preserve sign and exponent"
    assert not np.array_equal(orig_u32, disarmed_u32), "FLP should change mantissa"
    
    # Payload extraction failure
    payload = np.uint32(0x1234)
    embedded_u32 = (orig_u32 & mask) | payload
    model.conv1.weight.data = torch.from_numpy(embedded_u32.view(np.float32))
    
    disarmed_payload_model = apply_flp(model, target_layers=['conv1'])
    extracted_u32 = disarmed_payload_model.conv1.weight.data.cpu().numpy().view(np.uint32)
    extracted_mantissa = extracted_u32 & np.uint32(0x007FFFFF)
    
    assert not np.array_equal(extracted_mantissa, np.full_like(extracted_mantissa, payload)), "FLP should destroy payload"
    
    # Functional impact
    out_disarm = disarmed_model(dummy_in)
    assert out_orig.shape == out_disarm.shape

def test_k_lrbp():
    model = DummyModel()
    k = 5
    orig_weight = model.conv1.weight.data.clone()
    orig_u32 = orig_weight.cpu().numpy().view(np.uint32)
    
    disarmed_model = apply_k_lrbp(model, k=k, target_layers=['conv1'])
    disarmed_u32 = disarmed_model.conv1.weight.data.cpu().numpy().view(np.uint32)
    
    mask_keep = ~np.uint32((1 << k) - 1)
    assert np.array_equal(orig_u32 & mask_keep, disarmed_u32 & mask_keep), f"K-LRBP should preserve upper {32-k} bits"
    assert not np.array_equal(orig_u32, disarmed_u32), "K-LRBP should modify lower k bits"
    
    # Payload extraction failure
    payload = np.uint32(0b10101)
    embedded_u32 = (orig_u32 & mask_keep) | payload
    model.conv1.weight.data = torch.from_numpy(embedded_u32.view(np.float32))
    
    disarmed_payload_model = apply_k_lrbp(model, k=k, target_layers=['conv1'])
    extracted_u32 = disarmed_payload_model.conv1.weight.data.cpu().numpy().view(np.uint32)
    extracted_payload = extracted_u32 & ~mask_keep
    
    assert not np.array_equal(extracted_payload, np.full_like(extracted_payload, payload)), "K-LRBP should destroy payload"


def test_qint8():
    model = DummyModel()
    orig_weight = model.conv1.weight.data.clone()
    
    # Payload extraction failure simulated by checking if precision is lost
    # We add a tiny value to weight that would be represented by LSBs
    model.conv1.weight.data += 1e-6
    
    disarmed_model = apply_qint8(model, target_layers=['conv1'])
    disarmed_weight = disarmed_model.conv1.weight.data
    
    diff = torch.abs(orig_weight - disarmed_weight)
    assert torch.mean(diff) > 0, "Qint8 should alter weights"
    
    # Should destroy small precise payloads
    diff_payload = torch.abs(model.conv1.weight.data - disarmed_weight)
    assert torch.mean(diff_payload) > 1e-6, "Qint8 should destroy precision payload"
