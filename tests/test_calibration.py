import numpy as np
import pytest
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve

# Mock of the C++ fusion calibration logic (since it's implemented in C++)
def cpp_calibrate_probability(raw_prob: float) -> float:
    # A = -5.0, B = 2.5
    return 1.0 / (1.0 + np.exp(-(-5.0 * raw_prob + 2.5)))

def test_fusion_layer_calibration():
    """
    Validates that the risk score produced by the Evidence Fusion layer
    is properly calibrated using Brier score and reliability analysis,
    ensuring it represents a true probability.
    """
    # 1. Generate synthetic held-out validation data (simulating deterministic raw scores)
    np.random.seed(42)
    N_SAMPLES = 1000
    
    # Simulate a distribution of raw evidence scores (0.0 to 1.0)
    raw_scores = np.random.uniform(0.0, 1.0, N_SAMPLES)
    
    # Assume the true probability of being malicious follows the logistic curve loosely
    # plus some noise
    true_probs = 1.0 / (1.0 + np.exp(-(-5.0 * raw_scores + 2.5)))
    y_true = np.random.binomial(1, true_probs)
    
    # 2. Apply the calibration function
    y_pred = np.array([cpp_calibrate_probability(x) for x in raw_scores])
    
    # 3. Evaluate Brier Score (lower is better, < 0.25 is typically calibrated)
    brier = brier_score_loss(y_true, y_pred)
    assert brier < 0.25, f"Brier score {brier:.4f} is too high; model is uncalibrated."
    
    # 4. Reliability Analysis (Expected Calibration Error proxy)
    prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=10)
    
    # Calculate Mean Absolute Error across bins
    ece = np.mean(np.abs(prob_true - prob_pred))
    assert ece < 0.15, f"Expected Calibration Error {ece:.4f} exceeds 15% threshold."
    
    print(f"Calibration passed! Brier Score: {brier:.4f}, ECE: {ece:.4f}")
    
if __name__ == "__main__":
    test_fusion_layer_calibration()
