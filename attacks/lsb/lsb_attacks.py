"""
LSB Attack Generators for AI Model Steganalysis.

Attribution:
  Dubin, R. "Disarming Steganography Attacks Inside Neural Network Models"
  (conference / workshop paper, exact venue varies by edition).
  FMLA, HMLA, HBLA attack generators independently reimplemented from
  paper description. No code copied from any restricted source.

  Gilkarov, D. & Dubin, R. "Model X-Ray: Detection of Hidden Malware in
  AI Model Weights using Few Shot Learning", arXiv:2409.19310 / JISA 2026.
  XLSB (x-bit LSB fill) reimplemented from paper description.
"""

from __future__ import annotations

import hashlib
import json
import struct
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import onnx
from onnx import numpy_helper


# ---------------------------------------------------------------------------
# IEEE-754 float32 bit-field masks
# ---------------------------------------------------------------------------
# float32 layout (little-endian bit numbering):
#   bit 31     : sign
#   bits 30-23 : exponent  (8 bits)
#   bits 22-0  : mantissa  (23 bits)

_MASK_FMLA = np.uint32(0x007FFFFF)   # all 23 mantissa bits
_MASK_HMLA = np.uint32(0x00000FFF)   # lowest 12 mantissa bits
_MASK_HBLA = np.uint32(0x0000000F)   # lowest 4 mantissa bits
_PRESERVE_SIGN_EXP = np.uint32(0xFF800000)  # mask that keeps sign + exponent


# ---------------------------------------------------------------------------
# Payload expansion helpers
# ---------------------------------------------------------------------------

def _expand_payload_to_bits(payload: bytes, n_bits: int) -> np.ndarray:
    """
    Return a uint8 array of length *n_bits* containing the bit stream from
    *payload*, cycling if the payload is shorter than needed.
    """
    if not payload:
        # No payload: fill with pseudo-random bits for capacity measurement
        rng = np.random.default_rng(42)
        return rng.integers(0, 2, size=n_bits, dtype=np.uint8)

    # Convert bytes -> bits (MSB first per byte)
    payload_bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
    if len(payload_bits) >= n_bits:
        return payload_bits[:n_bits]
    # Tile to fill
    repeats = (n_bits + len(payload_bits) - 1) // len(payload_bits)
    tiled = np.tile(payload_bits, repeats)
    return tiled[:n_bits]


def _bits_to_uint32_mask(bits: np.ndarray, n_lsb: int) -> np.ndarray:
    """
    Pack a flat bit array into uint32 values where each value holds *n_lsb*
    bits in its least-significant positions.

    len(bits) must be == n_elements * n_lsb.
    """
    n_elements = len(bits) // n_lsb
    bits = bits[: n_elements * n_lsb].reshape(n_elements, n_lsb)
    # Build uint32 from bit columns; bit[:,0] is MSB of the n_lsb field
    result = np.zeros(n_elements, dtype=np.uint32)
    for i in range(n_lsb):
        result |= bits[:, i].astype(np.uint32) << np.uint32(n_lsb - 1 - i)
    return result


# ---------------------------------------------------------------------------
# Core embedding primitives
# ---------------------------------------------------------------------------

def embed_payload_xlsb(
    weights: np.ndarray,
    payload: bytes,
    x_bits: int,
) -> np.ndarray:
    """
    General X-LSB-Fill: replace the X lowest mantissa bits of every float32
    weight with bits from *payload* (cycled to fill all weights).

    Attribution: Gilkarov & Dubin, arXiv:2409.19310

    Args:
        weights: 1-D float32 numpy array (will be flattened internally).
        payload: raw bytes to embed.
        x_bits:  number of least-significant mantissa bits to replace (1-23).

    Returns:
        float32 array of the same shape as *weights* with modified LSBs.
    """
    if not (1 <= x_bits <= 23):
        raise ValueError(f"x_bits must be in [1, 23], got {x_bits}")

    original_shape = weights.shape
    flat = weights.flatten().astype(np.float32)
    n = len(flat)

    # View as uint32 for bit manipulation
    weights_flat = weights.flatten().astype(np.float32)
    n_params = weights_flat.size
    
    total_capacity_bits = n_params * x_bits
    payload_bits_len = len(payload) * 8
    
    if payload_bits_len > total_capacity_bits:
        raise ValueError(
            f"Payload too large. Model capacity: {total_capacity_bits} bits, "
            f"Payload: {payload_bits_len} bits."
        )

    # Convert weights to uint32 to manipulate bits
    uint_view = weights_flat.view(np.uint32)

    # Convert payload to bit string and slice to exact size needed
    payload_bin = "".join(f"{b:08b}" for b in payload)
    
    # Pad payload bit string if it doesn't evenly divide by x_bits
    rem = len(payload_bin) % x_bits
    if rem != 0:
        payload_bin += "0" * (x_bits - rem)

    # We need n_params * x_bits bits total to map one-to-one
    if len(payload_bin) < total_capacity_bits:
        repeats = (total_capacity_bits // len(payload_bin)) + 1
        payload_bin = (payload_bin * repeats)[:total_capacity_bits]
    else:
        payload_bin = payload_bin[:total_capacity_bits]

    # Chunk the payload bit string into x_bits blocks
    # Using list comprehension since payload_bin is a string
    chunks = [payload_bin[i:i + x_bits] for i in range(0, total_capacity_bits, x_bits)]

    # Convert chunks to integers
    payload_ints = [int(chunk, 2) for chunk in chunks]
    payload_uint = np.array(payload_ints, dtype=np.uint32)

    # Build mask for the x_bits LSBs of mantissa
    lsb_mask = np.uint32((1 << x_bits) - 1)
    keep_mask = ~lsb_mask  # preserve everything else

    uint_view = (uint_view & keep_mask) | (payload_uint & lsb_mask)
    return uint_view.view(np.float32).reshape(original_shape)


def apply_fmla(weights: np.ndarray, payload: bytes, seed: int = 42) -> np.ndarray:
    """
    Full Mantissa LSB Attack (FMLA).

    Replaces all 23 mantissa bits of every float32 weight with payload bits.
    The sign and exponent bits are preserved, so values retain their magnitude
    order but the mantissa becomes a carrier.

    Attribution: Dubin, "Disarming Steganography Attacks Inside Neural Network Models"
    """
    return embed_payload_xlsb(weights, payload, x_bits=23)


def apply_hmla(weights: np.ndarray, payload: bytes, seed: int = 42) -> np.ndarray:
    """
    Half Mantissa LSB Attack (HMLA).

    Replaces the lowest 12 mantissa bits with payload bits.
    Capacity is ~52% of FMLA with smaller floating-point perturbation.

    Attribution: Dubin, "Disarming Steganography Attacks Inside Neural Network Models"
    """
    return embed_payload_xlsb(weights, payload, x_bits=12)


def apply_hbla(weights: np.ndarray, payload: bytes, seed: int = 42) -> np.ndarray:
    """
    Half Byte LSB Attack (HBLA).

    Replaces the lowest 4 mantissa bits with payload bits.
    Minimal perturbation; lowest capacity of the three variants.

    Attribution: Dubin, "Disarming Steganography Attacks Inside Neural Network Models"
    """
    return embed_payload_xlsb(weights, payload, x_bits=4)


# ---------------------------------------------------------------------------
# Attack dispatch map
# ---------------------------------------------------------------------------

_ATTACK_REGISTRY: Dict[str, tuple] = {
    # name -> (function, n_bits_per_weight)
    "FMLA": (apply_fmla, 23),
    "HMLA": (apply_hmla, 12),
    "HBLA": (apply_hbla, 4),
}


def _xlsb_bits(x_bits: int):
    """Return a closure that calls embed_payload_xlsb with x_bits."""
    def _fn(w, p, seed=42):
        return embed_payload_xlsb(w, p, x_bits=x_bits)
    return _fn


# ---------------------------------------------------------------------------
# ONNX-level attack
# ---------------------------------------------------------------------------

def attack_onnx_model(
    input_path: str,
    output_path: str,
    attack_type: str,           # 'FMLA', 'HMLA', 'HBLA', or 'XLSB'
    payload: Optional[bytes] = None,
    x_bits: int = 4,            # used only when attack_type == 'XLSB'
    seed: int = 42,
    target_layers: Optional[List[str]] = None,   # None -> all float32 weight tensors
) -> dict:
    """
    Apply a steganographic LSB attack to an ONNX model and save the result.

    Returns a metadata dict describing what was modified.

    Supported attack_type values:
      'FMLA' - Full Mantissa LSB Attack (23 bits)
      'HMLA' - Half Mantissa LSB Attack (12 bits)
      'HBLA' - Half Byte LSB Attack (4 bits)
      'XLSB' - X-bit LSB Fill (x_bits parameter)
    """
    attack_type = attack_type.upper()

    if attack_type == "XLSB":
        attack_fn = _xlsb_bits(x_bits)
        bits_per_weight = x_bits
    elif attack_type in _ATTACK_REGISTRY:
        attack_fn, bits_per_weight = _ATTACK_REGISTRY[attack_type]
    else:
        raise ValueError(
            f"Unknown attack type '{attack_type}'. "
            f"Valid: {sorted(_ATTACK_REGISTRY.keys()) + ['XLSB']}"
        )

    if payload is None:
        # Generate random payload to fill capacity
        rng = np.random.default_rng(seed)
        payload = bytes(rng.integers(0, 256, size=1024, dtype=np.uint8).tolist())

    model = onnx.load(input_path, load_external_data=True)
    graph = model.graph

    affected_layer_names: List[str] = []
    modified_param_count = 0
    total_param_count = 0

    new_initializers = []

    for init in graph.initializer:
        # Only process float32 tensors, skip FLOAT16 since LSB on 32-bit float breaks on cast-back
        if init.data_type != onnx.TensorProto.FLOAT:
            total_param_count += int(np.prod(list(init.dims))) if init.dims else 0
            new_initializers.append(init)
            continue

        # Filter by target_layers if specified
        if target_layers is not None and init.name not in target_layers:
            arr = numpy_helper.to_array(init)
            total_param_count += arr.size
            new_initializers.append(init)
            continue

        # Convert to float32 for manipulation
        arr = numpy_helper.to_array(init).astype(np.float32)
        total_param_count += arr.size

        # Apply attack
        modified_arr = attack_fn(arr.flatten(), payload, seed=seed).reshape(arr.shape)

        # Pack back into ONNX initializer
        new_init = numpy_helper.from_array(modified_arr, name=init.name)
        new_initializers.append(new_init)

        affected_layer_names.append(init.name)
        modified_param_count += arr.size

    # Replace all initializers
    del graph.initializer[:]
    graph.initializer.extend(new_initializers)

    onnx.save(model, output_path)

    # Compute capacity and embedding rate
    capacity_bits = modified_param_count * bits_per_weight
    payload_bits = len(payload) * 8
    embedding_rate = payload_bits / capacity_bits if capacity_bits > 0 else 0.0

    # Detect architecture from tensor names
    all_names = [init.name for init in model.graph.initializer]
    architecture = _detect_arch_simple(all_names)

    metadata = {
        "source_model_path": str(input_path),
        "output_model_path": str(output_path),
        "attack_type": attack_type,
        "bits_modified": bits_per_weight,
        "payload_size_bytes": len(payload),
        "payload_bits": payload_bits,
        "capacity_bits": capacity_bits,
        "embedding_rate": round(embedding_rate, 6),
        "affected_layer_count": len(affected_layer_names),
        "affected_layers": affected_layer_names,
        "modified_param_count": modified_param_count,
        "total_param_count": total_param_count,
        "architecture": architecture,
        "format": "onnx",
        "dtype": "float32",
        "seed": seed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "label": 1,   # 1 = stego
    }

    return metadata


def _detect_arch_simple(tensor_names: List[str]) -> str:
    names_lower = [n.lower() for n in tensor_names]
    if any("layer" in n and "conv" in n for n in names_lower):
        return "resnet"
    if any("features.0" in n or "classifier" in n for n in names_lower):
        return "vgg"
    return "unknown"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Apply LSB steganographic attack to ONNX model.")
    parser.add_argument("input", help="Input .onnx file")
    parser.add_argument("output", help="Output .onnx file")
    parser.add_argument(
        "--attack",
        choices=["FMLA", "HMLA", "HBLA", "XLSB"],
        default="HBLA",
        help="Attack type (default: HBLA)",
    )
    parser.add_argument("--payload", help="Path to payload file (random if omitted)")
    parser.add_argument("--x-bits", type=int, default=4, help="Bits for XLSB attack (default: 4)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    payload_bytes = None
    if args.payload:
        with open(args.payload, "rb") as f:
            payload_bytes = f.read()

    meta = attack_onnx_model(
        input_path=args.input,
        output_path=args.output,
        attack_type=args.attack,
        payload=payload_bytes,
        x_bits=args.x_bits,
        seed=args.seed,
    )
    print(json.dumps(meta, indent=2))
