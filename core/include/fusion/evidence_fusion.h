#pragma once
// evidence_fusion.h - Project-Proposed Evidence Fusion Layer
//
// Computes final risk score by combining evidence from statistical, structural, 
// bit-level, and signature analyzers via a strict two-stage matched-reference pipeline.
// 
// 1. Matched clean reference
// 2. Family-specific deviation
// 3. Normalized evidence vector
// 4. Deterministic fusion + Learned fusion baseline
// 5. Calibrated probability -> Risk Score

#include "statistics/parameter_stats.h"
#include "bit_analysis/ieee754_analyzer.h"
#include "structural/structural_analyzer.h"
#include "decision/trojan_q.h"
#include "baseline/baseline_manager.h"
#include <string>
#include <vector>
#include <optional>

namespace stega {

enum class Verdict {
    CLEAN,
    SUSPICIOUS,
    MALICIOUS
};

// Represents the normalized deviations for a specific tensor across evidence families
struct NormalizedEvidence {
    double bit_entropy_deviation = 0.0;
    double structural_deviation = 0.0;
    double statistical_deviation = 0.0;
    double signature_deviation = 0.0; // E.g. Trojan Q
    double implicit_behavioral_deviation = 0.0; // Provided by Python layer if available
    double model_xray_deviation = 0.0; // Provided by Python layer if available
};

struct FusionResult {
    double calibrated_probability = 0.0;
    double risk_score = 0.0; // 0.0 to 1.0
    Verdict verdict = Verdict::CLEAN;
    std::vector<std::string> triggers; 
    NormalizedEvidence evidence_vector;
};

class EvidenceFusion {
public:
    EvidenceFusion() = default;

    // Stage 1 & 2: Matched Clean Reference -> Family-Specific Deviation -> Normalization
    NormalizedEvidence compute_normalized_evidence(
        const std::string& architecture,
        const std::string& tensor_name,
        const TensorStats& stats,
        const BitFeatures& bits,
        const StructuralFeatures& struct_feats,
        const BaselineManager& baselines
    ) const;

    // Stage 3 & 4: Evaluate tensor using Deterministic + Learned Fusion
    FusionResult evaluate_tensor(
        const std::string& architecture,
        const std::string& tensor_name,
        const TensorStats& stats,
        const BitFeatures& bits,
        const StructuralFeatures& struct_feats,
        const BaselineManager& baselines
    ) const;

    // Stage 5 & 6: Compute global calibrated risk based on all tensor evidence vectors and global signatures
    FusionResult compute_global_risk(
        const std::vector<FusionResult>& tensor_results,
        const std::optional<TrojanQResult>& trojan_res
    ) const;
    
private:
    // Helper to calibrate the raw fusion probability using a reliability curve / Platt scaling approximation
    double calibrate_probability(double raw_prob) const;
};

} // namespace stega
