"""
Neuron Permutation Steganography (NPS) - ONNX Attacker
"""

import sys
from pathlib import Path

# Fix import path
sys.path.append(str(Path(__file__).parent.parent.parent / "research" / "nps"))

import numpy as np
import onnx
from onnx import numpy_helper
from topology import generate_permutation_matrix, apply_permutation

def attack_onnx_with_nps(input_path: str, output_path: str, target_layer: str, next_layer: str, bn_layer: str = None, seed: int = 42):
    """
    Applies NPS to a sequential pair of layers in an ONNX model.
    If bn_layer is provided, it permutes the BN weight, bias, running_mean, and running_var.
    """
    model = onnx.load(input_path)
    
    w_target, b_target, w_next = None, None, None
    init_target_idx, init_b_idx, init_next_idx = -1, -1, -1
    
    bn_params = {} # name_suffix -> (array, init_idx)
    
    for i, init in enumerate(model.graph.initializer):
        if init.name == target_layer:
            w_target = numpy_helper.to_array(init)
            init_target_idx = i
        elif init.name == next_layer:
            w_next = numpy_helper.to_array(init)
            init_next_idx = i
            
        if bn_layer and init.name.startswith(bn_layer):
            bn_params[init.name] = (numpy_helper.to_array(init), i)
            
    target_base = target_layer.removesuffix('.weight')
    for i, init in enumerate(model.graph.initializer):
        if init.name == f"{target_base}.bias":
            b_target = numpy_helper.to_array(init)
            init_b_idx = i
            break
            
    if w_target is None or w_next is None:
        raise ValueError("Target layer or next layer not found")
        
    c_out = w_target.shape[0]
    c_in_next = w_next.shape[1] if w_next.ndim > 1 else w_next.shape[0]
    
    if c_out != c_in_next:
        raise ValueError(f"Dimension mismatch: target out_channels ({c_out}) != next layer in_channels ({c_in_next})")
        
    P = generate_permutation_matrix(c_out, seed=seed).astype(np.float32)
    
    w_p, b_p, w_next_p = apply_permutation(w_target, b_target, w_next, P)
    
    model.graph.initializer[init_target_idx].CopyFrom(
        numpy_helper.from_array(w_p, name=model.graph.initializer[init_target_idx].name)
    )
    if b_p is not None and init_b_idx != -1:
        model.graph.initializer[init_b_idx].CopyFrom(
            numpy_helper.from_array(b_p, name=model.graph.initializer[init_b_idx].name)
        )
    model.graph.initializer[init_next_idx].CopyFrom(
        numpy_helper.from_array(w_next_p, name=model.graph.initializer[init_next_idx].name)
    )
    
    # Permute BatchNorm params if any
    for name, (arr, idx) in bn_params.items():
        if arr.ndim == 1 and arr.shape[0] == c_out:
            arr_p = P @ arr
            model.graph.initializer[idx].CopyFrom(
                numpy_helper.from_array(arr_p, name=name)
            )
    
    onnx.save(model, output_path)
    return {
        "attack_type": "NPS",
        "affected_layers": [target_layer, next_layer],
        "permutation_size": c_out,
        "bn_permuted": len(bn_params) > 0
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--target", required=True)
    parser.add_argument("--next", required=True)
    args = parser.parse_args()
    
    res = attack_onnx_with_nps(args.input, args.output, args.target, args.next)
    print(f"NPS applied: {res}")
