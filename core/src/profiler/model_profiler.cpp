#include "profiler/model_profiler.h"
#include "statistics/parameter_stats.h"
#include <algorithm>
#include <cmath>
#include <vector>
#include <limits>

namespace stega {

// ── Architecture detection heuristics ─────────────────────────────────────
// We inspect the set of tensor names and look for well-known patterns.
// This is intentionally conservative — returns "unknown" if unsure.
std::string ModelProfiler::detect_architecture(const ModelArtifact& artifact) const {
    // Prefer the declared architecture from metadata if available
    if (!artifact.architecture.empty() && artifact.architecture != "unknown") {
        return artifact.architecture;
    }

    // Collect all tensor names into one searchable string
    std::string all_names;
    for (const auto& td : artifact.tensor_descriptors) {
        all_names += td.name + "|";
    }

    auto contains = [&](const std::string& pat) -> bool {
        return all_names.find(pat) != std::string::npos;
    };

    // ResNet family: layer1, layer2, conv1.weight, bn1
    if (contains("layer1") && contains("layer2") && contains("conv1.weight")) {
        return "resnet";
    }
    // VGG family: features.0, classifier.0 or features.0.weight
    if (contains("features.0") || contains("features.0.weight")) {
        return "vgg";
    }
    // EfficientNet / MobileNet
    if (contains("blocks.") && contains("depthwise")) {
        return "mobilenet";
    }
    if (contains("_blocks.") && contains("_expand_conv")) {
        return "efficientnet";
    }
    // BERT / transformer
    if (contains("encoder.layer") && contains("attention.self")) {
        return "bert";
    }
    // ViT
    if (contains("blocks.") && contains("attn.qkv")) {
        return "vit";
    }
    // AlexNet
    if (contains("features.0") && contains("classifier.1")) {
        return "alexnet";
    }

    return "unknown";
}

// ── Single-tensor profiling ────────────────────────────────────────────────
TensorProfile ModelProfiler::profile_tensor(const TensorView& tv) const {
    TensorProfile tp;
    tp.name            = tv.desc->name;
    tp.shape           = tv.desc->shape;
    tp.dtype_str       = dtype_to_string(tv.desc->dtype);
    tp.parameter_count = tv.desc->parameter_count;

    if (!tv.is_float32() || tv.num_elements() == 0) {
        return tp;  // leave stats zeroed for non-float or empty tensors
    }

    const float* data  = tv.as_float32();
    const int64_t n    = tv.num_elements();

    // Welford online algorithm for mean/variance
    double wf_mean = 0.0;
    double wf_M2   = 0.0;
    double min_val = std::numeric_limits<double>::max();
    double max_val = std::numeric_limits<double>::lowest();
    int64_t valid  = 0;
    int64_t sparse = 0;

    std::vector<double> sorted_vals;
    sorted_vals.reserve(static_cast<size_t>(n));

    for (int64_t i = 0; i < n; ++i) {
        double v = static_cast<double>(data[i]);
        if (std::isnan(v) || std::isinf(v)) continue;
        ++valid;
        sorted_vals.push_back(v);

        double delta  = v - wf_mean;
        wf_mean      += delta / static_cast<double>(valid);
        double delta2 = v - wf_mean;
        wf_M2        += delta * delta2;

        if (v < min_val) min_val = v;
        if (v > max_val) max_val = v;
        if (std::abs(v) < 1e-8) ++sparse;
    }

    if (valid == 0) return tp;

    tp.mean    = wf_mean;
    tp.min_val = min_val;
    tp.max_val = max_val;
    tp.sparsity = static_cast<double>(sparse) / static_cast<double>(valid);

    if (valid >= 2) {
        tp.variance = wf_M2 / static_cast<double>(valid - 1);
        tp.std_dev  = std::sqrt(std::max(0.0, tp.variance));
    }

    // Median
    std::sort(sorted_vals.begin(), sorted_vals.end());
    size_t sz = sorted_vals.size();
    if (sz % 2 == 0) {
        tp.median = (sorted_vals[sz / 2 - 1] + sorted_vals[sz / 2]) * 0.5;
    } else {
        tp.median = sorted_vals[sz / 2];
    }

    return tp;
}

// ── Full model profiling ───────────────────────────────────────────────────
ModelProfile ModelProfiler::profile(const ModelArtifact& artifact) const {
    ModelProfile mp;
    mp.scan_id               = artifact.scan_id;
    mp.filename              = artifact.filename;
    mp.sha256                = artifact.sha256;
    mp.format                = artifact.format;
    mp.architecture          = artifact.architecture;
    mp.dtype_str             = artifact.dtype_str;
    mp.tensor_count          = static_cast<int64_t>(artifact.tensor_count());
    mp.total_parameter_count = artifact.total_parameter_count;
    mp.architecture_hint     = detect_architecture(artifact);

    mp.tensors.reserve(artifact.tensor_count());
    for (size_t i = 0; i < artifact.tensor_count(); ++i) {
        TensorView tv = artifact.tensor_view(i);
        mp.tensors.push_back(profile_tensor(tv));
    }

    return mp;
}

} // namespace stega
