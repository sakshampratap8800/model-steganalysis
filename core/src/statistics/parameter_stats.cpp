#include "statistics/parameter_stats.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>
#include <numeric>

namespace stega {

// Welford online algorithm for numerically stable mean + M2
// Reference: Welford, B.P. (1962) "Note on a method for calculating corrected
//            sums of squares and products." Technometrics, 4(3), 419–420.
TensorStats ParameterStatsExtractor::extract(const TensorView& tv) const {
    TensorStats stats;

    if (!tv.is_float32() || tv.num_elements() == 0) {
        return stats;
    }

    const float* data    = tv.as_float32();
    const int64_t total  = tv.num_elements();

    // ── Pass 1: Welford mean/variance + min/max + NaN/Inf accounting ──────
    double wf_mean = 0.0;   // Welford running mean
    double wf_M2   = 0.0;   // Welford running M2
    double min_val = std::numeric_limits<double>::max();
    double max_val = std::numeric_limits<double>::lowest();
    int64_t valid_count = 0;
    int64_t nan_count   = 0;
    int64_t inf_count   = 0;
    int64_t sparse_count = 0; // |x| < 1e-6

    std::vector<double> valid_vals;
    valid_vals.reserve(static_cast<size_t>(total));

    for (int64_t i = 0; i < total; ++i) {
        double v = static_cast<double>(data[i]);

        if (std::isnan(v)) { ++nan_count; continue; }
        if (std::isinf(v)) { ++inf_count; continue; }

        ++valid_count;
        valid_vals.push_back(v);

        // Welford update
        double delta  = v - wf_mean;
        wf_mean      += delta / static_cast<double>(valid_count);
        double delta2 = v - wf_mean;
        wf_M2        += delta * delta2;

        if (v < min_val) min_val = v;
        if (v > max_val) max_val = v;

        if (std::abs(v) < 1e-6) ++sparse_count;
    }

    stats.count     = valid_count;
    stats.nan_count = nan_count;
    stats.inf_count = inf_count;

    if (valid_count == 0) {
        return stats;  // nothing valid to compute
    }

    stats.mean    = wf_mean;
    stats.min_val = min_val;
    stats.max_val = max_val;
    stats.sparsity = static_cast<double>(sparse_count) / static_cast<double>(valid_count);

    // Variance / std_dev — guard against degenerate case (single element)
    if (valid_count >= 2) {
        stats.variance = wf_M2 / static_cast<double>(valid_count - 1); // sample variance
        stats.std_dev  = std::sqrt(std::max(0.0, stats.variance));
    } else {
        stats.variance = 0.0;
        stats.std_dev  = 0.0;
    }

    // ── Median (sort a copy) ────────────────────────────────────────────────
    std::sort(valid_vals.begin(), valid_vals.end());
    size_t n = valid_vals.size();
    if (n % 2 == 0) {
        stats.median = (valid_vals[n / 2 - 1] + valid_vals[n / 2]) * 0.5;
    } else {
        stats.median = valid_vals[n / 2];
    }

    // ── Pass 2: Skewness & Kurtosis (standardised central moments) ─────────
    // Only meaningful when std_dev > 0
    if (stats.std_dev > 0.0) {
        double inv_sigma  = 1.0 / stats.std_dev;
        double sum_cube   = 0.0;
        double sum_fourth = 0.0;
        double mu         = stats.mean;

        for (double v : valid_vals) {
            double z    = (v - mu) * inv_sigma;
            double z2   = z * z;
            sum_cube   += z2 * z;
            sum_fourth += z2 * z2;
        }

        double nc        = static_cast<double>(valid_count);
        stats.skewness   = sum_cube   / nc;
        // Excess kurtosis (subtract 3 for normal distribution baseline)
        stats.kurtosis   = sum_fourth / nc - 3.0;
    }

    // ── Shannon Entropy (256-bin histogram over [min, max]) ─────────────────
    // Reference: Shannon, C.E. (1948) "A Mathematical Theory of Communication."
    if (min_val < max_val) {
        constexpr int NUM_BINS = 256;
        std::vector<int64_t> hist(NUM_BINS, 0);
        double range = max_val - min_val;
        double inv_range = static_cast<double>(NUM_BINS - 1) / range;

        for (double v : valid_vals) {
            int bin = static_cast<int>((v - min_val) * inv_range);
            if (bin < 0)        bin = 0;
            if (bin >= NUM_BINS) bin = NUM_BINS - 1;
            ++hist[bin];
        }

        double entropy = 0.0;
        double inv_n   = 1.0 / static_cast<double>(valid_count);
        for (int b = 0; b < NUM_BINS; ++b) {
            if (hist[b] == 0) continue;
            double p   = static_cast<double>(hist[b]) * inv_n;
            entropy   -= p * std::log2(p);
        }
        stats.entropy = entropy;
    } else {
        // All values identical → zero entropy
        stats.entropy = 0.0;
    }

    return stats;
}

} // namespace stega
