#include "fusion/evidence_fusion.h"
#include <algorithm>
#include <cmath>
#include <iostream>

namespace stega {

FusionResult EvidenceFusion::evaluate_tensor(
    const std::string& architecture,
    const std::string& tensor_name,
    const TensorStats& stats,
    const BitFeatures& bits,
    const StructuralFeatures& struct_feats,
    const BaselineManager& baselines) const 
{
    FusionResult res;
    res.risk_score = 0.0;
    
    // 1. Bit-level analysis (Highly predictive of LSB replacement)
    // A clean floating point distribution rarely has LOB4 entropy above ~3.7.
    // Near 4.0 means payload injection.
    if (bits.lob4_entropy > 3.85) {
        res.risk_score += 0.8;
        res.triggers.push_back("Extremely high lower-order bit entropy (LOB4 > 3.85) in " + tensor_name + ". Highly indicative of LSB attack (HBLA/HMLA).");
    } else if (bits.lob4_entropy > 3.75) {
        res.risk_score += 0.4;
        res.triggers.push_back("Elevated lower-order bit entropy (LOB4 > 3.75) in " + tensor_name + ". Suspicious.");
    }
    
    if (bits.mantissa_overall_imbalance > 0.05) {
        res.risk_score += 0.2;
        res.triggers.push_back("High mantissa bit imbalance in " + tensor_name + ". Steganography often forces uniform distribution.");
    }
    
// 2. Structural anomalies (NPS)
    // If adjacent channel differences are too high, it might indicate rough permutation.
    // anomaly_score > 0 indicates divergence from baseline structure.
    if (struct_feats.anomaly_score > 0.5) {
        res.risk_score += 0.4;
        res.triggers.push_back("Severe topological divergence in " + tensor_name + " (Gini/Variance shift). Potential NPS permutation.");
    } else if (struct_feats.anomaly_score > 0.2) {
        res.risk_score += 0.15;
        res.triggers.push_back("Mild topological divergence in " + tensor_name + ".");
    }
    
    // 3. Baseline comparison (if available)
    if (baselines.is_loaded()) {
        auto baseline_opt = baselines.get_baseline(architecture, tensor_name);
        if (baseline_opt) {
            auto& base = *baseline_opt;
            
            // Variance shift check
            if (base.variance_expected > 0) {
                double var_shift = std::abs(stats.variance - base.variance_expected) / base.variance_expected;
                if (var_shift > 0.1) { // 10% shift
                    res.risk_score += 0.3;
                    res.triggers.push_back("Variance shift > 10% from known baseline in " + tensor_name);
                }
            }
            
            // Entropy check
            if (stats.entropy < base.entropy_min || stats.entropy > base.entropy_max) {
                res.risk_score += 0.3;
                res.triggers.push_back("Shannon entropy outside baseline bounds in " + tensor_name);
            }
        }
    }
    
    // Cap risk score at 1.0
    res.risk_score = std::min(res.risk_score, 1.0);
    
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
    const std::optional<TrojanQResult>& trojan_res) const 
{
    FusionResult global_res;
    double max_tensor_risk = 0.0;
    
    for (const auto& tr : tensor_results) {
        if (tr.risk_score > max_tensor_risk) {
            max_tensor_risk = tr.risk_score;
        }
        for (const auto& trig : tr.triggers) {
            global_res.triggers.push_back(trig);
        }
    }
    
    global_res.risk_score = max_tensor_risk;
    
    // Global Trojan Q-test check
    if (trojan_res && trojan_res->significant) {
        global_res.risk_score = std::max(global_res.risk_score, 0.9);
        global_res.triggers.push_back(trojan_res->explanation + " Target class: " + std::to_string(trojan_res->candidate_class));
    }
    
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
