#include "fusion/evidence_fusion.h"
#include <algorithm>
#include <cmath>
#include <iostream>

namespace stega {

NormalizedEvidence EvidenceFusion::compute_normalized_evidence(
    const std::string& architecture,
    const std::string& tensor_name,
    const TensorStats& stats,
    const BitFeatures& bits,
    const StructuralFeatures& struct_feats,
    const BaselineManager& baselines
) const {
    NormalizedEvidence ev;
    
    // 1. Matched clean reference
    bool has_baseline = baselines.is_loaded() && baselines.get_baseline(architecture, tensor_name);
    
    if (has_baseline) {
        auto base = baselines.get_baseline(architecture, tensor_name).value();
        
        // 2. Family-specific deviation -> 3. Normalized evidence vector (0 to 1)
        
        // Bit Entropy Deviation (Expected LOB4 around 3.7 for FP32 weights, max 4.0)
        double bit_dev = 0.0;
        if (bits.lob4_entropy > 3.7) {
            bit_dev = std::min(1.0, (bits.lob4_entropy - 3.7) / 0.3); // Scales 3.7->0, 4.0->1.0
        }
        ev.bit_entropy_deviation = bit_dev;
        
        // Structural Deviation (Gini & adjacent diffs vs baseline)
        double struct_dev = struct_feats.anomaly_score; // anomaly_score is already pseudo-normalized
        if (struct_feats.cross_layer_rank_correlation < 0.2) {
            struct_dev = std::max(struct_dev, 0.8);
        }
        ev.structural_deviation = std::min(1.0, struct_dev);
        
        // Statistical Deviation (Variance shift vs baseline)
        if (base.variance_expected > 0) {
            double var_shift = std::abs(stats.variance - base.variance_expected) / base.variance_expected;
            ev.statistical_deviation = std::min(1.0, var_shift / 0.5); // 50% shift = 1.0
        }
    } else {
        // Fallback: heuristic deviation if no matched clean reference exists
        double bit_dev = 0.0;
        if (bits.lob4_entropy > 3.75) {
            bit_dev = std::min(1.0, (bits.lob4_entropy - 3.75) / 0.25);
        }
        ev.bit_entropy_deviation = bit_dev;
        ev.structural_deviation = std::min(1.0, struct_feats.anomaly_score);
        ev.statistical_deviation = 0.0; // Cannot compute without reference
    }
    
    return ev;
}

double EvidenceFusion::calibrate_probability(double raw_prob) const {
    // Stage 6: Calibration
    // Applies a logistic calibration curve (e.g. Platt scaling) learned from held-out validation data.
    // Assuming coefficients A = -5.0, B = 2.5 (Hypothetical trained values mapping raw output to calibrated Brier-optimized prob)
    double A = -5.0;
    double B = 2.5;
    double calibrated = 1.0 / (1.0 + std::exp(-(A * raw_prob + B)));
    return calibrated;
}

FusionResult EvidenceFusion::evaluate_tensor(
    const std::string& architecture,
    const std::string& tensor_name,
    const TensorStats& stats,
    const BitFeatures& bits,
    const StructuralFeatures& struct_feats,
    const BaselineManager& baselines
) const {
    FusionResult res;
    
    // Stages 1, 2, 3
    res.evidence_vector = compute_normalized_evidence(architecture, tensor_name, stats, bits, struct_feats, baselines);
    
    // Stage 4: Deterministic fusion + Learned fusion baseline
    // Deterministic rule: if bit entropy is extremely high, heavily weight it.
    double raw_prob = 0.0;
    
    double w_bit = 0.5;
    double w_struct = 0.3;
    double w_stat = 0.2;
    
    raw_prob = (w_bit * res.evidence_vector.bit_entropy_deviation) +
               (w_struct * res.evidence_vector.structural_deviation) +
               (w_stat * res.evidence_vector.statistical_deviation);
               
    // Non-linear trigger overrides (Deterministic)
    if (res.evidence_vector.bit_entropy_deviation > 0.8) {
        raw_prob = std::max(raw_prob, 0.9);
        res.triggers.push_back("Extreme bit entropy deviation detected in " + tensor_name + " (Possible LSB Attack).");
    }
    if (res.evidence_vector.structural_deviation > 0.8) {
        raw_prob = std::max(raw_prob, 0.85);
        res.triggers.push_back("Severe structural permutation detected in " + tensor_name + " (Possible NPS Attack).");
    }
    if (res.evidence_vector.statistical_deviation > 0.8) {
        res.triggers.push_back("Massive variance shift detected in " + tensor_name + ".");
    }
    
    // Stage 5: Calibrate probability
    res.calibrated_probability = calibrate_probability(raw_prob);
    
    // Stage 6: Risk score
    res.risk_score = res.calibrated_probability;
    
    if (res.risk_score >= 0.70) {
        res.verdict = Verdict::MALICIOUS;
    } else if (res.risk_score >= 0.35) {
        res.verdict = Verdict::SUSPICIOUS;
    } else {
        res.verdict = Verdict::CLEAN;
    }
    
    return res;
}

FusionResult EvidenceFusion::compute_global_risk(
    const std::vector<FusionResult>& tensor_results,
    const std::optional<TrojanQResult>& trojan_res
) const {
    FusionResult global_res;
    double max_tensor_raw = 0.0;
    
    for (const auto& tr : tensor_results) {
        // We aggregate the highest tensor-level evidence
        if (tr.risk_score > max_tensor_raw) {
            max_tensor_raw = tr.risk_score;
        }
        for (const auto& trig : tr.triggers) {
            global_res.triggers.push_back(trig);
        }
    }
    
    double raw_prob = max_tensor_raw;
    
    // Integrate Trojan Signature Deviation
    if (trojan_res && trojan_res->significant) {
        double trojan_dev = std::min(1.0, trojan_res->evidence_score / 2.0);
        global_res.evidence_vector.signature_deviation = trojan_dev;
        
        raw_prob = std::max(raw_prob, trojan_dev); // Deterministic fusion rule
        global_res.triggers.push_back(trojan_res->explanation + " Target class: " + std::to_string(trojan_res->candidate_class));
    }
    
    global_res.calibrated_probability = calibrate_probability(raw_prob);
    global_res.risk_score = global_res.calibrated_probability;
    
    if (global_res.risk_score >= 0.70) {
        global_res.verdict = Verdict::MALICIOUS;
    } else if (global_res.risk_score >= 0.35) {
        global_res.verdict = Verdict::SUSPICIOUS;
    } else {
        global_res.verdict = Verdict::CLEAN;
    }
    
    // Deduplicate triggers
    std::sort(global_res.triggers.begin(), global_res.triggers.end());
    global_res.triggers.erase(std::unique(global_res.triggers.begin(), global_res.triggers.end()), global_res.triggers.end());
    
    return global_res;
}

} // namespace stega
