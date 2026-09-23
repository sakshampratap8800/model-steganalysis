#pragma once
#include "common/tensor.h"
#include <string>
#include <vector>

namespace stega {

// Per-tensor profile data computed during the profiling phase.
struct TensorProfile {
    std::string name;
    std::vector<int64_t> shape;
    std::string dtype_str;
    int64_t parameter_count = 0;

    // Basic statistics
    double min_val    = 0.0;
    double max_val    = 0.0;
    double mean       = 0.0;
    double std_dev    = 0.0;
    double variance   = 0.0;
    double median     = 0.0;
    double sparsity   = 0.0;  // fraction of |x| < 1e-8
};

// Aggregate profile for an entire model artifact.
struct ModelProfile {
    std::string scan_id;
    std::string filename;
    std::string sha256;
    std::string format;
    std::string architecture;
    std::string dtype_str;
    int64_t tensor_count         = 0;
    int64_t total_parameter_count = 0;
    std::vector<TensorProfile> tensors;
    std::string architecture_hint;  // heuristic detection result
};

// Stateless profiler; call profile() for each artifact.
class ModelProfiler {
public:
    ModelProfile profile(const ModelArtifact& artifact) const;

private:
    TensorProfile profile_tensor(const TensorView& tv) const;

    // Heuristic: inspect tensor names to guess architecture family.
    std::string detect_architecture(const ModelArtifact& artifact) const;
};

} // namespace stega
