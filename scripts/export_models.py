"""
Export ResNet-18 and VGG-16 from torchvision as ONNX files.

Requirements: torch, torchvision, onnx
Usage: python export_models.py --output-dir models/references
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torchvision.models as tv_models


def sha256_of_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def export_model(
    model_name: str,
    model: torch.nn.Module,
    output_dir: Path,
    opset_version: int = 17,
) -> dict:
    """Export a single PyTorch model to ONNX and write metadata JSON."""
    import onnx

    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / f"{model_name}.onnx"
    meta_path = output_dir / f"{model_name}_metadata.json"

    model.eval()
    dummy_input = torch.zeros(1, 3, 224, 224)

    print(f"[{model_name}] Exporting to {onnx_path} ...")
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        opset_version=opset_version,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        export_params=True,
        do_constant_folding=True,
        verbose=False,
    )

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print(f"[{model_name}] ONNX check passed.")

    sha = sha256_of_file(onnx_path)
    file_size = onnx_path.stat().st_size

    metadata = {
        "name": model_name,
        "source": "torchvision",
        "pretrained": True,
        "architecture": model_name,
        "task": "imagenet_classification",
        "num_classes": 1000,
        "input_shape": [1, 3, 224, 224],
        "dtype": "float32",
        "format": "onnx",
        "opset_version": opset_version,
        "filename": f"{model_name}.onnx",
        "sha256": sha,
        "file_size_bytes": file_size,
        "exported_at": datetime.now(timezone.utc).isoformat(),
    }

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[{model_name}] Metadata saved to {meta_path}")
    print(f"  SHA-256 : {sha}")
    print(f"  Size    : {file_size:,} bytes")
    return metadata


def build_models() -> dict:
    """Return a mapping from model name to loaded torchvision model."""
    return {
        "resnet18": tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1),
        "vgg16": tv_models.vgg16(weights=tv_models.VGG16_Weights.IMAGENET1K_V1),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export ResNet-18 and VGG-16 to ONNX with metadata."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/references"),
        help="Directory to write .onnx and _metadata.json files (default: models/references)",
    )
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version (default: 17)")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["resnet18", "vgg16"],
        choices=["resnet18", "vgg16"],
        help="Which models to export (default: both)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("=" * 60)
    print("Model Export Script")
    print(f"  Output dir : {args.output_dir.resolve()}")
    print(f"  ONNX opset : {args.opset}")
    print(f"  Models     : {args.models}")
    print("=" * 60)

    all_models = build_models()
    results = {}

    for name in args.models:
        if name not in all_models:
            print(f"[WARN] Unknown model '{name}', skipping.", file=sys.stderr)
            continue
        meta = export_model(name, all_models[name], args.output_dir, opset_version=args.opset)
        results[name] = meta
        print()

    print("=" * 60)
    print("Export summary:")
    for name, meta in results.items():
        print(f"  {name}: {meta['filename']}  ({meta['file_size_bytes']:,} B)  sha256={meta['sha256'][:16]}...")
    print("Done.")


if __name__ == "__main__":
    main()
