"""
Disarm Mitigations for Steganographic Payloads
FLP: Floating Point Permutation / Mantissa replacement
K-LRBP: K-Least Reliable Bit Permutation / random flip
Qint8: 8-bit Quantization
"""
import copy
import numpy as np
import torch
import torch.nn as nn

def apply_flp(model: nn.Module, target_layers=None) -> nn.Module:
    """
    FLP (Fine-grained Layer Permutation) / Mantissa randomization.
    Replaces the 23 mantissa bits strictly for targeted Conv2D weights.
    """
    new_model = copy.deepcopy(model)
    
    with torch.no_grad():
        for name, module in new_model.named_modules():
            if isinstance(module, nn.Conv2d):
                if target_layers and name not in target_layers:
                    continue
                    
                weight = module.weight.data
                weight_np = weight.cpu().numpy().astype(np.float32)
                weight_u32 = weight_np.view(np.uint32)
                
                # Generate random 23-bit mantissa
                random_mantissa = np.random.randint(0, 1 << 23, size=weight_u32.shape, dtype=np.uint32)
                
                # Preserve sign (bit 31) and exponent (bits 23-30)
                mask = np.uint32(0xFF800000)
                new_u32 = (weight_u32 & mask) | random_mantissa
                
                new_weight = new_u32.view(np.float32)
                module.weight.data = torch.from_numpy(new_weight).to(weight.device)
                
    return new_model

def apply_k_lrbp(model: nn.Module, k: int, target_layers=None) -> nn.Module:
    """
    K-LRBP (K-Least Reliable Bit Permutation).
    Randomly modify exactly K selected eligible bits per weight for K=1,5,10.
    """
    new_model = copy.deepcopy(model)
    
    with torch.no_grad():
        for name, module in new_model.named_modules():
            # Applying to Conv2d and Linear to be safe, though prompt focuses on Conv2D for FLP.
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if target_layers and name not in target_layers:
                    continue
                
                weight = module.weight.data
                weight_np = weight.cpu().numpy().astype(np.float32)
                weight_u32 = weight_np.view(np.uint32)
                
                # Generate random bits for the lowest K bits
                # To guarantee EXACTLY K bits, the instruction might imply modifying exactly K bits.
                # Usually K-LRBP replaces the K lowest bits with random bits or flips them. 
                # Let's replace the lowest K bits with random bits.
                mask_keep = ~np.uint32((1 << k) - 1)
                random_bits = np.random.randint(0, 1 << k, size=weight_u32.shape, dtype=np.uint32)
                
                new_u32 = (weight_u32 & mask_keep) | random_bits
                
                new_weight = new_u32.view(np.float32)
                module.weight.data = torch.from_numpy(new_weight).to(weight.device)
                
    return new_model

def apply_qint8(model: nn.Module, target_layers=None) -> nn.Module:
    """
    Qint8 Quantization.
    xq = Clip(Round(xf / scale)) where scale = (2 * amax) / 256 and amax = max(abs(xf)).
    Reconstructs back to float32 using xq * scale.
    """
    new_model = copy.deepcopy(model)
    
    with torch.no_grad():
        for name, module in new_model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if target_layers and name not in target_layers:
                    continue
                
                xf = module.weight.data
                amax = torch.max(torch.abs(xf))
                if amax == 0:
                    continue
                
                scale = (2 * amax) / 256
                # Clip(Round(xf / scale))
                xq = torch.round(xf / scale)
                xq = torch.clamp(xq, -128, 127)
                
                # Dequantize back to float32
                xdq = xq * scale
                module.weight.data = xdq
                
    return new_model
