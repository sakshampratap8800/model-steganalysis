#include "decision/trojan_q.h"
#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>

namespace stega {

TrojanSignatureDetector::TrojanSignatureDetector(double alpha) : m_alpha(alpha) {}

double TrojanSignatureDetector::get_critical_value(int n, double alpha) {
    if (n < 3) return 0.0;

    // Tabulated values for alpha = 0.05
    static const std::map<int, double> table_95 = {
        {3, 0.970}, {4, 0.829}, {5, 0.710}, {6, 0.625}, {7, 0.568},
        {8, 0.526}, {9, 0.493}, {10, 0.466}, {11, 0.444}, {12, 0.426},
        {13, 0.410}, {14, 0.396}, {15, 0.384}, {16, 0.374}, {17, 0.365},
        {18, 0.356}, {19, 0.349}, {20, 0.342}, {25, 0.317}, {30, 0.298},
        {40, 0.272}, {50, 0.252}, {100, 0.197}, {1000, 0.090}
    };

    // Tabulated values for alpha = 0.10
    static const std::map<int, double> table_90 = {
        {3, 0.941}, {4, 0.765}, {5, 0.642}, {6, 0.560}, {7, 0.507},
        {8, 0.468}, {9, 0.437}, {10, 0.412}, {11, 0.392}, {12, 0.376},
        {13, 0.361}, {14, 0.349}, {15, 0.338}, {16, 0.329}, {17, 0.320},
        {18, 0.313}, {19, 0.306}, {20, 0.300}, {25, 0.277}, {30, 0.260},
        {40, 0.237}, {50, 0.219}, {100, 0.171}, {1000, 0.077}
    };

    const auto& table = (std::abs(alpha - 0.10) < 0.01) ? table_90 : table_95;

    auto it = table.find(n);
    if (it != table.end()) return it->second;

    auto upper = table.upper_bound(n);
    if (upper == table.end()) {
        auto last = std::prev(upper);
        auto prev = std::prev(last);
        double c = (last->second * std::sqrt(last->first) + prev->second * std::sqrt(prev->first)) / 2.0;
        return c / std::sqrt(n);
    }
    
    if (upper == table.begin()) return upper->second;
    
    auto lower = std::prev(upper);
    double t = static_cast<double>(n - lower->first) / (upper->first - lower->first);
    return lower->second + t * (upper->second - lower->second);
}

TrojanQResult TrojanSignatureDetector::detect(const TensorView& tv) const {
    TrojanQResult res;
    if (!tv.is_float32() || tv.desc->shape.size() != 2) {
        res.explanation = "Invalid tensor shape for final layer (must be 2D).";
        return res;
    }

    int64_t n_classes = tv.desc->shape[0];
    int64_t in_features = tv.desc->shape[1];

    if (n_classes < 3) {
        res.explanation = "Too few classes for Dixon Q-test (< 3).";
        return res;
    }
    
    if (in_features == 0) {
        res.explanation = "Zero features in tensor.";
        return res;
    }

    const float* data = tv.as_float32();
    std::vector<std::pair<double, int>> row_means;
    row_means.reserve(n_classes);

    for (int64_t i = 0; i < n_classes; ++i) {
        double sum = 0.0;
        int valid_count = 0;
        for (int64_t j = 0; j < in_features; ++j) {
            float val = data[i * in_features + j];
            if (std::isfinite(val)) {
                sum += val;
                valid_count++;
            }
        }
        if (valid_count > 0) {
            row_means.emplace_back(sum / valid_count, static_cast<int>(i));
        }
    }
    
    if (row_means.size() < 3) {
        res.explanation = "Too few finite rows after NaN filtering.";
        return res;
    }

    std::sort(row_means.begin(), row_means.end());

    int n = row_means.size();
    double w_min = row_means.front().first;
    double w_max = row_means.back().first;
    res.range_value = w_max - w_min;

    if (res.range_value < 1e-12) {
        res.q_statistic = 0.0;
        res.explanation = "Range is approximately zero; cannot compute test statistic.";
        return res;
    }

    // Grubbs' Test for N > 30 (Dixon is undefined)
    if (n > 30) {
        double mean = 0.0;
        for (const auto& p : row_means) mean += p.first;
        mean /= n;
        
        double var = 0.0;
        for (const auto& p : row_means) var += (p.first - mean) * (p.first - mean);
        double stddev = std::sqrt(var / (n - 1));
        
        if (stddev < 1e-12) {
            res.explanation = "Zero variance.";
            return res;
        }
        
        double g_max = std::abs(w_max - mean) / stddev;
        double g_min = std::abs(mean - w_min) / stddev;
        
        double g_stat = std::max(g_max, g_min);
        // Approximate inverse CDF of normal for p = 1 - alpha / (2N) (Bonferroni)
        double alpha = m_alpha;
        double tail = alpha / (2.0 * n);
        double w = std::sqrt(-2.0 * std::log(tail));
        // Abramowitz & Stegun 26.2.23
        double t_crit = w - (2.515517 + 0.802853 * w + 0.010328 * w * w) /
                            (1.0 + 1.432788 * w + 0.189269 * w * w + 0.001308 * w * w * w);
        double g_crit = (n - 1) * t_crit / std::sqrt(n * (n - 2 + t_crit * t_crit));
        
        res.q_statistic = g_stat;
        res.critical_value = g_crit;
        res.candidate_class = (g_max > g_min) ? row_means.back().second : row_means.front().second;
        res.candidate_mean = (g_max > g_min) ? w_max : w_min;
        res.significant = (g_stat > g_crit);
        res.evidence_score = std::min(g_stat / g_crit, 2.0);
        res.explanation = res.significant ? "Trojan signature DETECTED via Grubbs Test." : "No Trojan signature detected.";
        return res;
    }

    // Dixon's Q-test for N <= 30
    double q_max = 0.0, q_min = 0.0;
    
    if (n >= 3 && n <= 7) {
        q_max = (w_max - row_means[n-2].first) / (w_max - w_min + 1e-12);
        q_min = (row_means[1].first - w_min) / (w_max - w_min + 1e-12);
    } else if (n >= 8 && n <= 10) {
        q_max = (w_max - row_means[n-2].first) / (w_max - row_means[1].first + 1e-12);
        q_min = (row_means[1].first - w_min) / (row_means[n-2].first - w_min + 1e-12);
    } else if (n >= 11 && n <= 13) {
        q_max = (w_max - row_means[n-3].first) / (w_max - row_means[1].first + 1e-12);
        q_min = (row_means[2].first - w_min) / (row_means[n-2].first - w_min + 1e-12);
    } else if (n >= 14 && n <= 30) {
        q_max = (w_max - row_means[n-3].first) / (w_max - row_means[2].first + 1e-12);
        q_min = (row_means[2].first - w_min) / (row_means[n-3].first - w_min + 1e-12);
    }

    double q_stat = std::max(q_max, q_min);
    res.critical_value = get_critical_value(n, m_alpha);
    res.candidate_class = (q_max > q_min) ? row_means.back().second : row_means.front().second;
    res.candidate_mean = (q_max > q_min) ? w_max : w_min;
    res.q_statistic = q_stat;
    res.significant = (q_stat > res.critical_value);
    
    if (res.critical_value > 0) {
        res.evidence_score = std::min(q_stat / res.critical_value, 2.0);
    }

    if (res.significant) {
        res.explanation = "Trojan signature DETECTED: Q > Q_crit.";
    } else {
        res.explanation = "No Trojan signature detected: Q <= Q_crit.";
    }

    return res;
}

} // namespace stega
