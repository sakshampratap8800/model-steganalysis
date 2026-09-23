#pragma once
// structural_analyzer.h — Topology and structural weight analysis
//
// Identifies anomalies such as Neuron Permutation Steganography (NPS),
// which preserves the parameter statistics perfectly (multiset invariant)
// but changes the order/topology of the weights.

#include "common/tensor.h"
#include <vector>
#include <string>

namespace stega {

struct StructuralFeatures {
    std::string tensor_name;
    
    double channel_norm_variance = 0.0;
    double max_adjacent_channel_diff = 0.0;
    double channel_norm_gini = 0.0;
    
    // Cross-layer rank correlation to detect permutation (NPS)
    double cross_layer_rank_correlation = 1.0; 
    
    double anomaly_score = 0.0;
};

class StructuralAnalyzer {
public:
    StructuralFeatures analyze(const TensorView& tv, 
                               const StructuralFeatures* reference = nullptr);

    void reset() { prev_layer_norms_.clear(); }

private:
    double compute_gini(std::vector<double>& values) const;
    double compute_spearman(const std::vector<double>& x, const std::vector<double>& y) const;
    
    // State to track between layers for cross-layer correlation
    std::vector<double> prev_layer_norms_;
};

} // namespace stega
