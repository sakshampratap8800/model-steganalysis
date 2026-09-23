#pragma once
// evidence_fusion.h — Computes final risk score by combining evidence
// from statistical, structural, bit-level, and signature analyzers.

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

struct FusionResult {
    double risk_score = 0.0; // 0.0 to 1.0
    Verdict verdict = Verdict::CLEAN;
    std::vector<std::string> triggers; // Text explanations of what was detected
};

class EvidenceFusion {
public:
    EvidenceFusion() = default;

    // Evaluates a single tensor and returns its risk contribution
    FusionResult evaluate_tensor(
        const std::string& architecture,
        const std::string& tensor_name,
        const TensorStats& stats,
        const BitFeatures& bits,
        const StructuralFeatures& struct_feats,
        const BaselineManager& baselines
    ) const;

    // Aggregates tensor-level risks and global features (like TrojanQ) into a final score
    FusionResult compute_global_risk(
        const std::vector<FusionResult>& tensor_results,
        const std::optional<TrojanQResult>& trojan_res
    ) const;
};

} // namespace stega
