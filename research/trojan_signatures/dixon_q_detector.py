"""
Dixon Q-test Trojan Signature Detector

Attribution: Fields et al., "Trojan Signatures in DNN Weights"
Independent reimplementation from paper description.
No code was copied from any restricted source.

Algorithm:
  1. Locate the final linear (classification) layer of the model
  2. Compute row mean for each class: w_i = (1/d) * sum_j W_i,j
  3. Sort row means: w_i1 <= ... <= w_ic
  4. Q statistic: Q = |w_ic - w_i(c-1)| / (w_ic - w_i1)
  5. Compare Q with tabulated Dixon critical value for n_classes at alpha
  6. If Q > critical_value: Trojan signature detected; candidate class = argmax(w_i)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Dixon Q critical value table
# Tabulated values from Dixon (1950), Rorabacher (1991), and ASTM E178.
# Index = number of observations (= number of classes for our use case).
# Only r10 (90% CI) and r22 (95% CI) are commonly used; we use r22 (alpha=0.05).
# ---------------------------------------------------------------------------

# alpha = 0.05 (95% confidence interval)
_DIXON_Q_CRITICAL_95: Dict[int, float] = {
    3:  0.970,
    4:  0.829,
    5:  0.710,
    6:  0.625,
    7:  0.568,
    8:  0.526,
    9:  0.493,
    10: 0.466,
    11: 0.444,
    12: 0.426,
    13: 0.410,
    14: 0.396,
    15: 0.384,
    16: 0.374,
    17: 0.365,
    18: 0.356,
    19: 0.349,
    20: 0.342,
    25: 0.317,
    30: 0.298,
    40: 0.272,
    50: 0.252,
    100: 0.197,
    1000: 0.090,
}

# alpha = 0.10 (90% confidence interval)
_DIXON_Q_CRITICAL_90: Dict[int, float] = {
    3:  0.941,
    4:  0.765,
    5:  0.642,
    6:  0.560,
    7:  0.507,
    8:  0.468,
    9:  0.437,
    10: 0.412,
    11: 0.392,
    12: 0.376,
    13: 0.361,
    14: 0.349,
    15: 0.338,
    16: 0.329,
    17: 0.320,
    18: 0.313,
    19: 0.306,
    20: 0.300,
    25: 0.277,
    30: 0.260,
    40: 0.237,
    50: 0.219,
    100: 0.171,
    1000: 0.077,
}

_ALPHA_TABLE = {
    0.05: _DIXON_Q_CRITICAL_95,
    0.10: _DIXON_Q_CRITICAL_90,
}


def get_dixon_critical_value(n: int, alpha: float = 0.05) -> float:
    """
    Return Dixon Q critical value for n observations at significance level alpha.

    Interpolates linearly between tabulated values for unlisted n.
    Returns 0.0 for n < 3 (Q-test undefined).
    """
    if n < 3:
        return 0.0

    table = _ALPHA_TABLE.get(alpha, _DIXON_Q_CRITICAL_95)

    if n in table:
        return table[n]

    # Linear interpolation between nearest tabulated values
    keys = sorted(table.keys())
    if n > keys[-1]:
        # Extrapolate using asymptotic approximation: Q ≈ c / sqrt(n)
        # Fit c from last two tabulated points
        n1, n2 = keys[-2], keys[-1]
        q1, q2 = table[n1], table[n2]
        # Solve for c: c = Q * sqrt(n)
        c = (q1 * math.sqrt(n1) + q2 * math.sqrt(n2)) / 2.0
        return c / math.sqrt(n)

    # Interpolate
    for i in range(len(keys) - 1):
        lo, hi = keys[i], keys[i + 1]
        if lo <= n <= hi:
            t = (n - lo) / (hi - lo)
            return table[lo] + t * (table[hi] - table[lo])

    return table[keys[0]]  # fallback


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class TrojanSignatureResult:
    """Result of running the Dixon Q-test Trojan signature detector."""
    # Layer inspected
    final_layer_name: str

    # Raw per-class row means (in original class order)
    class_row_means: List[float]

    # Sorted ascending
    sorted_means: List[float]
    sorted_class_indices: List[int]

    # Dixon Q test values
    n_classes: int
    Q_statistic: float
    critical_value: float
    alpha: float

    # Detection outcome
    significant: bool           # Q > critical_value → Trojan signature detected
    candidate_target_class: int  # Class with the highest mean (most extreme outlier)
    candidate_mean: float        # That class's mean value
    range_value: float          # w_ic - w_i1 (denominator of Q)

    # Evidence score: normalized Q relative to critical value
    # > 1.0 means detected; bounded to [0, 2.0] for reporting
    evidence_score: float

    # Human-readable explanation
    explanation: str


# ---------------------------------------------------------------------------
# Core detector function
# ---------------------------------------------------------------------------

def _find_final_linear_layer(model) -> Optional[Tuple[str, np.ndarray]]:
    """
    Find the weight matrix of the final Linear (classification) layer.

    Works with:
    - torch.nn.Module (walks children in reverse to find last Linear)
    - state_dict (dict of name -> tensor): searches for last 'weight' key
      that matches FC layer naming conventions

    Returns:
        (layer_name, weight_matrix) or None if not found.
    """
    try:
        import torch
        import torch.nn as nn

        if isinstance(model, nn.Module):
            last_linear = None
            last_name = ""
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    last_linear = module
                    last_name = name
            if last_linear is not None:
                W = last_linear.weight.detach().cpu().numpy()
                return last_name, W

        if isinstance(model, dict):
            # Treat as state_dict
            linear_candidates = [
                k for k in model.keys()
                if k.endswith(".weight") and len(model[k].shape) == 2
            ]
            if not linear_candidates:
                return None
            # Take the last one (deepest in the network)
            last_key = linear_candidates[-1]
            W = model[last_key]
            if hasattr(W, "numpy"):
                W = W.detach().cpu().numpy()
            elif not isinstance(W, np.ndarray):
                W = np.array(W, dtype=np.float32)
            return last_key.removesuffix(".weight"), W

    except ImportError:
        pass

    return None


def detect_trojan_signature(
    model,  # torch.nn.Module OR state_dict dict OR np.ndarray (weight matrix directly)
    n_classes: Optional[int] = None,
    alpha: float = 0.05,
    layer_name: str = "auto",
) -> TrojanSignatureResult:
    """
    Apply the Dixon Q-test to detect Trojan signatures in the final classification layer.

    Args:
        model: PyTorch module, state dict, or raw weight matrix (numpy array).
        n_classes: Number of output classes. If None, inferred from weight matrix shape.
        alpha: Significance level for Dixon Q critical value (0.05 or 0.10).
        layer_name: Name for the inspected layer (used in result; 'auto' = detected).

    Returns:
        TrojanSignatureResult with all diagnostic fields populated.

    Algorithm (Fields et al.):
        1. W ∈ R^{c × d}: weight matrix of final linear layer
        2. w_i = (1/d) * Σ_j W_i,j  (mean of row i = class i row mean)
        3. Sort: w_i1 ≤ w_i2 ≤ ... ≤ w_ic
        4. Q = |w_ic - w_i(c-1)| / (w_ic - w_i1)
        5. Reject H0 (no outlier) if Q > Q_crit(c, alpha)
    """
    # ── Resolve weight matrix ────────────────────────────────────────────────
    if isinstance(model, np.ndarray):
        W = model.astype(np.float64)
        detected_layer_name = layer_name if layer_name != "auto" else "provided_matrix"
    else:
        found = _find_final_linear_layer(model)
        if found is None:
            return TrojanSignatureResult(
                final_layer_name="not_found",
                class_row_means=[],
                sorted_means=[],
                sorted_class_indices=[],
                n_classes=0,
                Q_statistic=0.0,
                critical_value=0.0,
                alpha=alpha,
                significant=False,
                candidate_target_class=-1,
                candidate_mean=0.0,
                range_value=0.0,
                evidence_score=0.0,
                explanation="Could not locate a final linear classification layer.",
            )
        detected_layer_name, W_raw = found
        W = W_raw.astype(np.float64)
        if layer_name != "auto":
            detected_layer_name = layer_name

    # W should be shape (n_classes, d)
    if W.ndim != 2:
        W = W.reshape(1, -1)

    c, d = W.shape

    if n_classes is not None:
        c = n_classes
        W = W[:c, :]  # trim if more rows than expected classes

    # ── Compute row means ────────────────────────────────────────────────────
    row_means = W.mean(axis=1)  # shape: (c,)

    # ── Sort ascending ───────────────────────────────────────────────────────
    sorted_indices = np.argsort(row_means)  # class indices sorted by mean
    sorted_means_arr = row_means[sorted_indices]

    # ── Dixon Q statistic ────────────────────────────────────────────────────
    w_min = sorted_means_arr[0]    # w_i1 (smallest)
    w_max = sorted_means_arr[-1]   # w_ic (largest = candidate target)
    w_prev = sorted_means_arr[-2]  # w_i(c-1) (second largest)

    range_val = w_max - w_min

    if range_val < 1e-12:
        # Degenerate: all row means equal — Q is undefined, no signature
        Q = 0.0
    else:
        Q = abs(w_max - w_prev) / range_val

    critical_value = get_dixon_critical_value(c, alpha)
    significant = (Q > critical_value) and (range_val > 1e-12)

    candidate_class = int(sorted_indices[-1])  # class with largest mean
    candidate_mean  = float(w_max)

    # Evidence score: Q / Q_crit, capped at 2.0
    if critical_value > 0:
        evidence_score = min(float(Q) / critical_value, 2.0)
    else:
        evidence_score = 0.0

    # ── Build explanation ────────────────────────────────────────────────────
    if significant:
        explanation = (
            f"Trojan signature DETECTED in layer '{detected_layer_name}'. "
            f"Class {candidate_class} is a statistical outlier: "
            f"row mean = {candidate_mean:.6f}, Q = {Q:.4f} > "
            f"Q_crit({c}, α={alpha}) = {critical_value:.4f}. "
            f"This class may be a Trojan target class."
        )
    elif range_val < 1e-12:
        explanation = (
            f"Layer '{detected_layer_name}': all class row means are equal "
            f"(range ≈ 0). Dixon Q-test undefined — no signature detectable."
        )
    else:
        explanation = (
            f"No Trojan signature detected in layer '{detected_layer_name}'. "
            f"Q = {Q:.4f} ≤ Q_crit({c}, α={alpha}) = {critical_value:.4f}. "
            f"Row means are consistent with a clean model."
        )

    return TrojanSignatureResult(
        final_layer_name=detected_layer_name,
        class_row_means=row_means.tolist(),
        sorted_means=sorted_means_arr.tolist(),
        sorted_class_indices=sorted_indices.tolist(),
        n_classes=c,
        Q_statistic=float(Q),
        critical_value=float(critical_value),
        alpha=alpha,
        significant=significant,
        candidate_target_class=candidate_class,
        candidate_mean=candidate_mean,
        range_value=float(range_val),
        evidence_score=evidence_score,
        explanation=explanation,
    )


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------

def detect_from_onnx(onnx_path: str, n_classes: int = 1000,
                     alpha: float = 0.05) -> TrojanSignatureResult:
    """
    Run Trojan signature detection directly on an ONNX model file.

    Finds the last 2D weight matrix (FC layer) in the initializers.
    """
    import onnx
    from onnx import numpy_helper

    model = onnx.load(onnx_path)
    graph = model.graph

    # Find last 2D float initializer (likely the FC classification weight)
    last_name, last_W = None, None
    for init in graph.initializer:
        if init.data_type != 1:  # 1 = FLOAT
            continue
        arr = numpy_helper.to_array(init)
        if arr.ndim == 2 and arr.shape[0] <= 10000:  # avoid huge matrices
            last_name = init.name
            last_W = arr

    if last_W is None:
        return TrojanSignatureResult(
            final_layer_name="not_found",
            class_row_means=[], sorted_means=[], sorted_class_indices=[],
            n_classes=0, Q_statistic=0.0, critical_value=0.0, alpha=alpha,
            significant=False, candidate_target_class=-1, candidate_mean=0.0,
            range_value=0.0, evidence_score=0.0,
            explanation="No 2D float weight tensor found in ONNX graph.",
        )

    return detect_trojan_signature(
        last_W, n_classes=min(n_classes, last_W.shape[0]),
        alpha=alpha, layer_name=last_name or "unknown"
    )


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: dixon_q_detector.py <model.onnx> [n_classes] [alpha]")
        sys.exit(1)

    onnx_path = sys.argv[1]
    n_cls = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    alp   = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05

    result = detect_from_onnx(onnx_path, n_classes=n_cls, alpha=alp)

    out = {
        "final_layer_name":      result.final_layer_name,
        "n_classes":             result.n_classes,
        "Q_statistic":           result.Q_statistic,
        "critical_value":        result.critical_value,
        "alpha":                 result.alpha,
        "significant":           result.significant,
        "candidate_target_class": result.candidate_target_class,
        "candidate_mean":        result.candidate_mean,
        "evidence_score":        result.evidence_score,
        "explanation":           result.explanation,
        "class_row_means":       result.class_row_means,
    }
    print(json.dumps(out, indent=2))
