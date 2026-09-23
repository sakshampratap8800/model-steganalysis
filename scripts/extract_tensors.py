"""
Convert an ONNX model to the C++ intermediate format (meta.json + weights.bin).

Requirements: onnx, numpy, hashlib, uuid
Usage: python extract_tensors.py <model.onnx> --output-dir <scan_dir> [--scan-id <uuid>]

Output:
  <scan_dir>/meta.json     - tensor inventory + model metadata
  <scan_dir>/weights.bin  - all float32 tensors concatenated
"""

import argparse
import hashlib
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np
import onnx
from onnx import numpy_helper, TensorProto


# ---------------------------------------------------------------------------
# Architecture detection heuristic
# ---------------------------------------------------------------------------

def detect_architecture(tensor_names: List[str]) -> str:
    """
    Heuristic detection of model architecture from initializer names.

    Checks for known naming patterns from popular torchvision exports.
    """
    names_lower = [n.lower() for n in tensor_names]

    # ResNet family: layers with 'layer' and 'conv'
    if any("layer" in n and "conv" in n for n in names_lower):
        return "resnet"

    # VGG family: 'features.0' or 'classifier'
    if any("features.0" in n or "classifier" in n for n in names_lower):
        return "vgg"

    # EfficientNet family
    if any("blocks" in n and "depthwise" in n for n in names_lower):
        return "efficientnet"

    # MobileNet family
    if any("mobilenet" in n or "inverted_residuals" in n for n in names_lower):
        return "mobilenet"

    # BERT/transformer family
    if any("attention" in n and ("query" in n or "key" in n or "value" in n) for n in names_lower):
        return "transformer"

    # LSTM/RNN family
    if any("lstm" in n or "rnn" in n or "gru" in n for n in names_lower):
        return "rnn"

    return "unknown"


# ---------------------------------------------------------------------------
# Data type conversion helpers
# ---------------------------------------------------------------------------

# ONNX TensorProto.DataType enum -> descriptive string
_DTYPE_NAMES = {
    TensorProto.FLOAT: "float32",
    TensorProto.DOUBLE: "float64",
    TensorProto.FLOAT16: "float16",
    TensorProto.INT8: "int8",
    TensorProto.UINT8: "uint8",
    TensorProto.INT16: "int16",
    TensorProto.UINT16: "uint16",
    TensorProto.INT32: "int32",
    TensorProto.UINT32: "uint32",
    TensorProto.INT64: "int64",
    TensorProto.UINT64: "uint64",
    TensorProto.BOOL: "bool",
    TensorProto.STRING: "string",
    TensorProto.COMPLEX64: "complex64",
    TensorProto.COMPLEX128: "complex128",
    TensorProto.BFLOAT16: "bfloat16",
}

# Data types we can safely convert to float32 for analysis
_CONVERTIBLE_TYPES = {
    TensorProto.FLOAT,
    TensorProto.DOUBLE,
    TensorProto.FLOAT16,
    TensorProto.INT8,
    TensorProto.UINT8,
    TensorProto.INT16,
    TensorProto.UINT16,
    TensorProto.INT32,
    TensorProto.UINT32,
    TensorProto.INT64,
    TensorProto.UINT64,
}

# Types that represent non-weight constant scalars (skip or flag)
_SCALAR_LIKE_TYPES = {TensorProto.INT32, TensorProto.INT64, TensorProto.UINT32, TensorProto.UINT64}


def initializer_to_float32(init: onnx.TensorProto) -> Optional[np.ndarray]:
    """
    Convert an ONNX initializer to a float32 numpy array.

    Returns None for types we cannot meaningfully convert (strings, complex, etc.)
    Handles external data by catching exceptions gracefully.
    """
    dtype = init.data_type

    if dtype == 0:
        # data_type == 0 means UNDEFINED — constant-folded or opaque node; skip
        return None

    if dtype not in _CONVERTIBLE_TYPES:
        return None

    # numpy_helper.to_array handles external data loading automatically
    # when the model is loaded with load_external_data=True
    try:
        arr = numpy_helper.to_array(init)
    except Exception as e:
        print(f"  [WARN] Could not convert tensor '{init.name}': {e}", file=sys.stderr)
        return None

    if arr.size == 0:
        return None

    return arr.astype(np.float32)


# ---------------------------------------------------------------------------
# SHA-256
# ---------------------------------------------------------------------------

def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Main extraction logic
# ---------------------------------------------------------------------------

def extract_tensors(
    model_path: Path,
    output_dir: Path,
    scan_id: Optional[str] = None,
) -> dict:
    """
    Extract all weight tensors from an ONNX model into the intermediate format.

    Writes:
      <output_dir>/meta.json
      <output_dir>/weights.bin

    Returns the meta dict.
    """
    if scan_id is None:
        scan_id = str(uuid.uuid4())

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading ONNX model: {model_path}")
    # load_external_data=True so external tensor data is pulled in automatically
    model = onnx.load(str(model_path), load_external_data=True)

    model_sha256 = sha256_of_file(model_path)
    print(f"  SHA-256 : {model_sha256}")

    graph = model.graph
    initializers = list(graph.initializer)

    tensor_names = [init.name for init in initializers]
    architecture = detect_architecture(tensor_names)
    print(f"  Detected architecture : {architecture}")
    print(f"  Total initializers    : {len(initializers)}")

    weights_bin_path = output_dir / "weights.bin"
    meta_path = output_dir / "meta.json"

    tensor_entries = []
    byte_offset = 0
    total_params = 0
    skipped = 0

    with open(weights_bin_path, "wb") as bin_file:
        for init in initializers:
            dtype_name = _DTYPE_NAMES.get(init.data_type, f"unknown_{init.data_type}")
            dims = list(init.dims)

            # Skip undefined / un-convertible tensors
            if init.data_type == 0 or init.data_type not in _CONVERTIBLE_TYPES:
                print(f"  [SKIP] '{init.name}' dtype={dtype_name}")
                skipped += 1
                continue

            arr_f32 = initializer_to_float32(init)
            if arr_f32 is None:
                print(f"  [SKIP] '{init.name}' — conversion returned None")
                skipped += 1
                continue

            raw_bytes = arr_f32.tobytes()
            bin_file.write(raw_bytes)

            is_scalar_like = init.data_type in _SCALAR_LIKE_TYPES
            n_params = int(arr_f32.size)
            total_params += n_params

            entry = {
                # Keys matching C++ loader (model_loader.cpp) expectations:
                "name": init.name,
                "shape": dims,            # C++ reads "shape"
                "dtype": "float32",       # C++ reads "dtype" (stored as float32)
                "parameter_count": n_params,  # C++ reads "parameter_count"
                "byte_offset": byte_offset,
                "byte_size": len(raw_bytes),
                # Extra metadata (informational, not used by C++ core):
                "original_dtype": dtype_name,
                "is_scalar_like": is_scalar_like,
            }
            tensor_entries.append(entry)
            byte_offset += len(raw_bytes)

    weights_size = weights_bin_path.stat().st_size

    # Build ONNX model-level info
    opset_version = None
    if model.opset_import:
        for op in model.opset_import:
            if op.domain == "" or op.domain == "ai.onnx":
                opset_version = op.version
                break

    meta = {
        "scan_id": scan_id,
        "filename": model_path.name,
        "sha256": model_sha256,
        "file_size_bytes": model_path.stat().st_size,
        "architecture": architecture,
        "format": "onnx",
        "dtype": "float32",
        "opset_version": opset_version,
        "ir_version": model.ir_version,
        "tensor_count": len(tensor_entries),
        "skipped_tensors": skipped,
        "total_parameter_count": total_params,   # C++ reads "total_parameter_count"
        "weights_bin": "weights.bin",
        "weights_bin_size_bytes": weights_size,
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "tensors": tensor_entries,
    }

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nExtraction complete:")
    print(f"  Tensors extracted : {len(tensor_entries)}")
    print(f"  Tensors skipped   : {skipped}")
    print(f"  Total parameters  : {total_params:,}")
    print(f"  weights.bin size  : {weights_size:,} bytes")
    print(f"  meta.json written : {meta_path}")

    return meta


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert ONNX model to C++ intermediate format (meta.json + weights.bin)."
    )
    parser.add_argument("model_path", type=Path, help="Path to the .onnx file")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for meta.json and weights.bin output",
    )
    parser.add_argument(
        "--scan-id",
        type=str,
        default=None,
        help="UUID for this scan (auto-generated if omitted)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.model_path.exists():
        print(f"Error: model path does not exist: {args.model_path}", file=sys.stderr)
        sys.exit(1)

    extract_tensors(
        model_path=args.model_path,
        output_dir=args.output_dir,
        scan_id=args.scan_id,
    )


if __name__ == "__main__":
    main()
