#include "structural/structural_analyzer.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace stega {

double StructuralAnalyzer::compute_gini(std::vector<double>& values) const {
    if (values.empty()) return 0.0;
    
    std::vector<double> sorted_vals = values;
    std::sort(sorted_vals.begin(), sorted_vals.end());
    
    double sum = 0.0;
    for (double v : sorted_vals) sum += v;
    
    if (sum == 0.0) return 0.0;
    
    int64_t n = sorted_vals.size();
    double gini_num = 0.0;
    
    for (int64_t i = 0; i < n; ++i) {
        gini_num += (i + 1) * sorted_vals[i];
    }
    
    return (2.0 * gini_num) / (n * sum) - (static_cast<double>(n + 1) / n);
}

double StructuralAnalyzer::compute_spearman(const std::vector<double>& x, const std::vector<double>& y) const {
    if (x.size() != y.size() || x.empty()) return 1.0;
    
    size_t sz = x.size();
    double nd = static_cast<double>(sz);

    auto get_ranks = [&](const std::vector<double>& v) {
        std::vector<std::pair<double, size_t>> indexed;
        indexed.reserve(sz);
        for (size_t i = 0; i < sz; ++i) indexed.push_back({v[i], i});
        std::sort(indexed.begin(), indexed.end());
        
        std::vector<double> ranks(sz);
        size_t i = 0;
        while (i < sz) {
            // Find the end of a tie group
            size_t j = i;
            while (j < sz && indexed[j].first == indexed[i].first) ++j;
            // Average rank for the group [i, j)
            double avg_rank = (static_cast<double>(i + 1) + static_cast<double>(j)) / 2.0;
            for (size_t k = i; k < j; ++k) {
                ranks[indexed[k].second] = avg_rank;
            }
            i = j;
        }
        return ranks;
    };
    
    std::vector<double> rank_x = get_ranks(x);
    std::vector<double> rank_y = get_ranks(y);
    
    double sum_d2 = 0.0;
    for (size_t i = 0; i < sz; ++i) {
        double d = rank_x[i] - rank_y[i];
        sum_d2 += d * d;
    }
    
    // Cast to double before multiplication to prevent int overflow for wide layers
    return 1.0 - (6.0 * sum_d2) / (nd * (nd * nd - 1.0));
}

StructuralFeatures StructuralAnalyzer::analyze(const TensorView& tv,
                                             const StructuralFeatures* reference) {
    StructuralFeatures sf;
    sf.tensor_name = tv.desc->name;
    
    if (!tv.is_float32() || tv.desc->shape.empty()) {
        return sf;
    }
    
    int64_t out_channels = tv.desc->shape[0];
    if (out_channels <= 1) {
        return sf;
    }
    
    int64_t in_channels = (tv.desc->shape.size() > 1) ? tv.desc->shape[1] : 1;
    int64_t out_in = out_channels * in_channels;
    if (out_in == 0) return sf;
    int64_t spatial = tv.num_elements() / out_in;
    
    const float* data = tv.as_float32();
    
    // Output channel norms (for Gini and next layer correlation)
    std::vector<double> out_norms(out_channels, 0.0);
    // Input channel norms (to correlate with previous layer)
    std::vector<double> in_norms(in_channels, 0.0);
    
    double sum_norms = 0.0;
    for (int64_t oc = 0; oc < out_channels; ++oc) {
        double sq_sum = 0.0;
        for (int64_t ic = 0; ic < in_channels; ++ic) {
            for (int64_t sp = 0; sp < spatial; ++sp) {
                double v = data[(oc * in_channels + ic) * spatial + sp];
                sq_sum += v * v;
                in_norms[ic] += v * v;
            }
        }
        out_norms[oc] = std::sqrt(sq_sum);
        sum_norms += out_norms[oc];
    }
    
    for (int64_t ic = 0; ic < in_channels; ++ic) {
        in_norms[ic] = std::sqrt(in_norms[ic]);
    }
    
    double mean_norm = sum_norms / out_channels;
    double var_sum = 0.0;
    for (double n : out_norms) {
        var_sum += (n - mean_norm) * (n - mean_norm);
    }
    sf.channel_norm_variance = var_sum / out_channels;
    
    double max_diff = 0.0;
    for (size_t i = 1; i < out_norms.size(); ++i) {
        double diff = std::abs(out_norms[i] - out_norms[i - 1]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    sf.max_adjacent_channel_diff = max_diff;
    sf.channel_norm_gini = compute_gini(out_norms);
    
    double base_score = 0.0;
    if (sf.channel_norm_gini > 0.7) {
        base_score += (sf.channel_norm_gini - 0.7) * 2.0; 
    }
    if (mean_norm > 1e-6 && (sf.channel_norm_variance / (mean_norm * mean_norm)) > 2.0) {
        base_score += 0.2;
    }

    // Compute cross layer rank correlation if dimensions match
    if (!prev_layer_norms_.empty() && prev_layer_norms_.size() == in_norms.size()) {
        sf.cross_layer_rank_correlation = compute_spearman(prev_layer_norms_, in_norms);
        
        // If correlation is suspiciously low (< 0.1) for adjacent layers, it's highly anomalous
        if (sf.cross_layer_rank_correlation < 0.2) {
            base_score += 0.6; // Triggers "Severe topological divergence"
        }
    }
    
    sf.anomaly_score = std::min(1.0, base_score);

    // Only update cross-layer state for genuinely multi-dimensional weight tensors
    // (ndim >= 2). Bias vectors and BN scalars (1D) must NOT overwrite the stored norms
    // or they will corrupt the Spearman correlation for the next real weight tensor.
    if (tv.desc->shape.size() >= 2) {
        prev_layer_norms_ = out_norms;
    }
    
    return sf;
}

} // namespace stega
