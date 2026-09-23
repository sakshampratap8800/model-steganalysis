#pragma once
#include "common/tensor.h"
#include <cstdint>

namespace stega {

// Full statistical description of a single tensor's weight distribution.
// Reference: Chandrasekaran et al., "Detecting Backdoor Attacks on Deep Neural
//            Networks by Activation Clustering" — statistical baseline section.
struct TensorStats {
    double mean      = 0.0;
    double std_dev   = 0.0;
    double variance  = 0.0;
    double min_val   = 0.0;
    double max_val   = 0.0;
    double median    = 0.0;
    double skewness  = 0.0;  // E[(x-mu)^3] / sigma^3
    double kurtosis  = 0.0;  // excess kurtosis: E[(x-mu)^4]/sigma^4 - 3
    double entropy   = 0.0;  // Shannon entropy (bits), 256-bin histogram
    double sparsity  = 0.0;  // fraction of |x| < 1e-6
    int64_t count    = 0;    // number of valid (non-NaN, non-Inf) elements
    int64_t nan_count = 0;   // elements that were NaN
    int64_t inf_count = 0;   // elements that were +/-Inf
};

// Stateless extractor; thread-safe to call from multiple threads.
class ParameterStatsExtractor {
public:
    // Extract full statistics from a float32 TensorView.
    // Non-float32 tensors return a zeroed TensorStats with count=0.
    TensorStats extract(const TensorView& tv) const;
};

} // namespace stega
