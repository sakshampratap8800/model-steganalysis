"""
GF (Grayscale-Fourpart) Representation of Model Weights.

Attribution: Gilkarov & Dubin, "Model X-Ray: Detection of Hidden Malware in
AI Model Weights using Few Shot Learning", arXiv:2409.19310 / JISA 2026.

This is an INDEPENDENT REIMPLEMENTATION from paper description only.
No code was copied from github.com/danigil/ModelXRay (CC BY-NC-ND 4.0).

Algorithm:
  1. Flatten all float32 weight tensors into a 1D array.
  2. View the float32 array as uint8 (4 bytes per float32 weight).
  3. Each byte becomes one grayscale pixel (value 0-255).
  4. Reshape the byte stream into a square 2D image (padded with zeros).

The 4-byte structure of each IEEE-754 float32 (little-endian / x86):
  byte 0 (LSB): lowest-order mantissa byte (bits 7-0)
  byte 1:       middle mantissa byte (bits 15-8)
  byte 2:       upper mantissa + lower exponent (bits 23-16)
  byte 3 (MSB): sign + upper exponent (bits 31-24)

This encoding means:
  - Clean model weights: bytes 2-3 dominated by exponent patterns
  - LSB-attacked models: byte 0 (and byte 1 for HMLA) contain payload bits
  - The GF image visually reveals payload injection in the low-byte planes
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import numpy as np


# ---------------------------------------------------------------------------
# Core conversion functions
# ---------------------------------------------------------------------------

def weights_to_gf_image(weights_flat: np.ndarray) -> np.ndarray:
    """
    Convert a 1D float32 weight array to a GF (Grayscale-Fourpart) image.

    Args:
        weights_flat: 1D float32 numpy array of model weights.

    Returns:
        2D uint8 numpy array (square grayscale image), padded with zeros
        to reach the nearest perfect square byte count.
    """
    # Enforce little-endian float32 ('<f4') to ensure reproducible chunk ordering
    # 4-byte chunk: [LSB(Mantissa), Mid(Mantissa), High(Mantissa)+Low(Exp), Sign+High(Exp)]
    weights_le = weights_flat.astype('<f4')

    # View float32 data as uint8 — 4 bytes per float
    byte_array = weights_le.view(np.uint8)
    n_bytes = len(byte_array)

    # Compute side length of smallest square that holds all bytes
    side = int(np.ceil(np.sqrt(n_bytes)))

    # Pad to side*side with zeros
    padded = np.zeros(side * side, dtype=np.uint8)
    padded[:n_bytes] = byte_array

    return padded.reshape(side, side)


def weights_to_gf_image_normalized(weights_flat: np.ndarray) -> np.ndarray:
    """
    Same as weights_to_gf_image but returns float32 in [0.0, 1.0].
    Suitable as CNN input without further normalization.
    """
    gf = weights_to_gf_image(weights_flat)
    return gf.astype(np.float32) / 255.0


def tensors_to_gf_image(
    tensors: List[np.ndarray],
    max_params: Optional[int] = None,
) -> np.ndarray:
    """
    Concatenate multiple weight tensors and convert to a single GF image.

    Args:
        tensors: List of numpy arrays (any shape, any float dtype).
        max_params: If set, truncate/pad total parameter count to this size.

    Returns:
        2D uint8 GF image.
    """
    flat_parts = [t.astype(np.float32).flatten() for t in tensors]
    combined = np.concatenate(flat_parts) if flat_parts else np.array([], dtype=np.float32)

    if max_params is not None:
        if len(combined) > max_params:
            combined = combined[:max_params]
        elif len(combined) < max_params:
            combined = np.concatenate([combined, np.zeros(max_params - len(combined), dtype=np.float32)])

    return weights_to_gf_image(combined)


def gf_image_to_tensor(gf_image: np.ndarray) -> "torch.Tensor":
    """
    Convert a GF image to a normalized torch.Tensor suitable for CNN input.

    Returns shape (1, H, W) — single-channel grayscale.
    """
    import torch  # type: ignore
    arr = gf_image.astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


# ---------------------------------------------------------------------------
# ONNX model → GF image
# ---------------------------------------------------------------------------

def onnx_model_to_gf_image(
    model_path: Union[str, Path],
    layer_names: Optional[List[str]] = None,
    max_params: Optional[int] = None,
) -> np.ndarray:
    """
    Load an ONNX model and convert all (or specified) weight tensors to a GF image.

    Args:
        model_path: Path to the .onnx file.
        layer_names: If provided, only include tensors whose names are in this list.
        max_params: Maximum number of float32 parameters to include.

    Returns:
        2D uint8 GF image.
    """
    import onnx  # type: ignore
    from onnx import numpy_helper, TensorProto  # type: ignore

    model = onnx.load(str(model_path))
    tensors = []

    for init in model.graph.initializer:
        if init.data_type != TensorProto.FLOAT:
            continue
        if layer_names is not None and init.name not in layer_names:
            continue
        arr = numpy_helper.to_array(init)
        tensors.append(arr.astype(np.float32))

    if not tensors:
        return np.zeros((1, 1), dtype=np.uint8)

    return tensors_to_gf_image(tensors, max_params=max_params)


# ---------------------------------------------------------------------------
# Bit-plane decomposition (for visualization / analysis)
# ---------------------------------------------------------------------------

def gf_image_to_byte_planes(weights_flat: np.ndarray) -> List[np.ndarray]:
    """
    Return the four byte planes of the float32 weights as separate images.

    Plane 0 (byte 0): lowest mantissa bits — where HBLA/HMLA payload appears
    Plane 1 (byte 1): middle mantissa bits — where HMLA payload appears
    Plane 2 (byte 2): upper mantissa + lower exponent
    Plane 3 (byte 3): sign + upper exponent — rarely modified by LSB attacks

    Returns: List of 4 square uint8 arrays (one per byte plane).
    """
    weights_le = weights_flat.astype('<f4')

    byte_view = weights_le.view(np.uint8)
    n = len(weights_le)

    planes = []
    for plane_idx in range(4):
        # Extract every 4th byte starting at plane_idx
        plane_bytes = byte_view[plane_idx::4]  # shape: (n,)

        side = int(np.ceil(np.sqrt(n)))
        padded = np.zeros(side * side, dtype=np.uint8)
        padded[:len(plane_bytes)] = plane_bytes
        planes.append(padded.reshape(side, side))

    return planes


def compute_lsb_entropy(weights_flat: np.ndarray, n_bits: int = 4) -> float:
    """
    Compute the mean per-bit binary entropy of the lowest n_bits mantissa bits.

    For clean models this should be close to 0.0 (non-random LSBs).
    For LSB-attacked models this approaches 1.0 (payload bits are random).

    Args:
        weights_flat: 1D float32 weight array.
        n_bits: Number of lowest mantissa bits to analyze (1-23).

    Returns:
        Mean binary entropy in [0.0, 1.0].
    """
    # Convert to little-endian uint32 to ensure bit shifts operate on consistent representation
    uint_view = weights_flat.astype('<f4').view('<u4')
    n = len(uint_view)
    if n == 0:
        return 0.0

    entropies = []
    for bit in range(n_bits):
        ones = int(np.sum((uint_view >> bit) & 1))
        p = ones / n
        if 0 < p < 1:
            h = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
        else:
            h = 0.0
        entropies.append(h)

    return float(np.mean(entropies))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: gf_representation.py <model.onnx> [--save <output.png>]")
        sys.exit(1)

    model_path = Path(sys.argv[1])
    save_path = None
    if "--save" in sys.argv:
        idx = sys.argv.index("--save")
        if idx + 1 < len(sys.argv):
            save_path = Path(sys.argv[idx + 1])

    gf = onnx_model_to_gf_image(model_path)
    lsb_entropy = compute_lsb_entropy(
        np.frombuffer(gf.tobytes(), dtype=np.uint8).view(np.float32)
        if gf.size >= 4 else np.array([], dtype=np.float32)
    )

    info = {
        "model": str(model_path),
        "gf_image_shape": list(gf.shape),
        "gf_image_pixel_count": int(gf.size),
        "lsb4_entropy": lsb_entropy,
    }
    print(json.dumps(info, indent=2))

    if save_path is not None:
        try:
            from PIL import Image  # type: ignore
            img = Image.fromarray(gf, mode="L")
            img.save(str(save_path))
            print(f"GF image saved to: {save_path}")
        except ImportError:
            print("[WARN] Pillow not installed; cannot save image. pip install Pillow")
