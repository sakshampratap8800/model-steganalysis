#pragma once
// trojan_q.h — Dixon Q-test implementation for Trojan Signature detection
// Based on Fields et al., "Trojan Signatures in DNN Weights"

#include "common/tensor.h"
#include <vector>
#include <string>

namespace stega {

struct TrojanQResult {
    bool   significant = false;
    double q_statistic = 0.0;
    double critical_value = 0.0;
    int    candidate_class = -1;
    double candidate_mean = 0.0;
    double range_value = 0.0;
    double evidence_score = 0.0; // Q / Q_crit (capped at 2.0)
    std::string explanation;
};

class TrojanSignatureDetector {
public:
    // Alpha significance level (0.05 or 0.10)
    explicit TrojanSignatureDetector(double alpha = 0.05);

    // Detect on the final linear layer weight matrix
    // Expects a 2D float32 tensor (out_classes, in_features)
    TrojanQResult detect(const TensorView& tv) const;

private:
    double m_alpha;
    static double get_critical_value(int n_classes, double alpha);
};

} // namespace stega
