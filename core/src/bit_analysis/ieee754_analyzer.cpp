// ieee754_analyzer.cpp — IEEE-754 bit-level analysis of float32 weights.
//
// Detects steganographic payload in the mantissa low-order bits of float32
// network parameters. Attacks targeted:
//   HBLA  — Dubin: lowest 4 mantissa bits
//   HMLA  — Dubin: lowest 12 mantissa bits
//   FMLA  — Dubin: all 23 mantissa bits
//   X-LSB-Fill — Gilkarov & Dubin (arXiv:2409.19310): lowest X bits, X∈{1..16}

#include "bit_analysis/ieee754_analyzer.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>
#include <limits>

namespace stega {

// ── Bit manipulation helpers ─────────────────────────────────────────────────

// IEEE-754 float32 bit layout (little-endian):
//   bits 22-0  = mantissa (23 bits)
//   bits 30-23 = exponent (8 bits)
//   bit  31    = sign

static constexpr uint32_t MANTISSA_MASK = 0x007FFFFFU;
static constexpr uint32_t EXPONENT_MASK = 0x7F800000U;
static constexpr uint32_t SIGN_MASK     = 0x80000000U;

static inline uint32_t float_as_uint(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    return u;
}

// ── Binary entropy ────────────────────────────────────────────────────────────

double IEEE754Analyzer::binary_entropy(double p) {
    if (p <= 0.0 || p >= 1.0) return 0.0;
    return -(p * std::log2(p) + (1.0 - p) * std::log2(1.0 - p));
}

// ── Low-order bit entropy (multi-bit window) ──────────────────────────────────

double IEEE754Analyzer::lob_entropy(const uint32_t* uint_data, int64_t n,
                                     int bit_lo, int bit_hi) {
    // Compute joint entropy of the bit window [bit_lo, bit_hi)
    // For small windows: exact via pattern histogram.
    // For large windows: approximate joint entropy as sum of per-bit marginal
    //   entropies (independence assumption). This gives range [0, width] bits.
    if (n == 0 || bit_lo >= bit_hi) return 0.0;

    const int width = bit_hi - bit_lo;
    if (width > 16) {
        // Sum of marginal entropies approximates joint entropy (independence assumption)
        double sum = 0.0;
        for (int b = bit_lo; b < bit_hi; ++b) {
            int64_t ones = 0;
            for (int64_t i = 0; i < n; ++i) {
                ones += (uint_data[i] >> b) & 1u;
            }
            sum += binary_entropy(static_cast<double>(ones) / static_cast<double>(n));
        }
        // Return sum (in bits), not mean — range is [0, width]
        return sum;
    }

    // Exact joint entropy via pattern histogram
    int32_t num_bins = 1 << width;
    std::vector<int64_t> hist(static_cast<size_t>(num_bins), 0);
    uint32_t mask = (width < 32) ? ((1u << width) - 1u) : 0xFFFFFFFFu;

    for (int64_t i = 0; i < n; ++i) {
        uint32_t pattern = (uint_data[i] >> bit_lo) & mask;
        ++hist[static_cast<size_t>(pattern)];
    }

    double entropy = 0.0;
    double inv_n   = 1.0 / static_cast<double>(n);
    for (int32_t b = 0; b < num_bins; ++b) {
        if (hist[static_cast<size_t>(b)] == 0) continue;
        double p = static_cast<double>(hist[static_cast<size_t>(b)]) * inv_n;
        entropy -= p * std::log2(p);
    }
    return entropy;
}

// ── Main analysis ─────────────────────────────────────────────────────────────

BitFeatures IEEE754Analyzer::analyze(const TensorView& tv,
                                      const BitFeatures* reference) const {
    BitFeatures bf;
    bf.tensor_name = tv.desc->name;

    if (!tv.is_float32() || tv.num_elements() == 0) {
        return bf;
    }

    const float*   fdata = tv.as_float32();
    const int64_t  n     = tv.num_elements();

    // View data as uint32 for bit manipulation
    std::vector<uint32_t> udata(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        udata[static_cast<size_t>(i)] = float_as_uint(fdata[i]);
    }
    const uint32_t* ud = udata.data();

    // ── Sign bit ────────────────────────────────────────────────────────────
    int64_t neg_count = 0;
    for (int64_t i = 0; i < n; ++i) {
        neg_count += (ud[i] >> 31) & 1u;
    }
    bf.sign_freq_negative = static_cast<double>(neg_count) / static_cast<double>(n);

    // ── Exponent field (bits 23-30) ─────────────────────────────────────────
    std::vector<int64_t> exp_hist(256, 0);
    double exp_sum  = 0.0;
    double exp_sum2 = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        uint32_t exp_val = (ud[i] & EXPONENT_MASK) >> 23;
        ++exp_hist[exp_val];
        exp_sum  += static_cast<double>(exp_val);
        exp_sum2 += static_cast<double>(exp_val) * static_cast<double>(exp_val);
    }
    double inv_n = 1.0 / static_cast<double>(n);
    bf.exponent_mean = exp_sum * inv_n;
    double exp_var   = exp_sum2 * inv_n - bf.exponent_mean * bf.exponent_mean;
    bf.exponent_std  = std::sqrt(std::max(0.0, exp_var));

    double exp_entropy = 0.0;
    for (int e = 0; e < 256; ++e) {
        if (exp_hist[e] == 0) continue;
        double p = static_cast<double>(exp_hist[e]) * inv_n;
        exp_entropy -= p * std::log2(p);
    }
    bf.exponent_entropy = exp_entropy;

    // ── Mantissa bit-plane statistics ────────────────────────────────────────
    bf.mantissa_bit_planes.resize(23);
    double imbalance_sum = 0.0;

    for (int b = 0; b < 23; ++b) {
        int64_t ones = 0;
        for (int64_t i = 0; i < n; ++i) {
            ones += (ud[i] >> b) & 1u;
        }
        double freq1 = static_cast<double>(ones) / static_cast<double>(n);

        BitPlaneStats bps;
        bps.bit_pos    = b;
        bps.freq_one   = freq1;
        bps.entropy    = binary_entropy(freq1);
        bps.imbalance  = std::abs(freq1 - 0.5);
        bf.mantissa_bit_planes[static_cast<size_t>(b)] = bps;
        imbalance_sum += bps.imbalance;
    }
    bf.mantissa_overall_imbalance = imbalance_sum / 23.0;

    // ── Low-order bit entropies ──────────────────────────────────────────────
    bf.lob4_entropy  = lob_entropy(ud, n, 0, 4);   // HBLA range
    bf.lob12_entropy = lob_entropy(ud, n, 0, 12);  // HMLA range (mean of marginals)
    bf.lob23_entropy = lob_entropy(ud, n, 0, 23);  // FMLA range (mean of marginals)

    // ── Run-length statistics on bit 0 ───────────────────────────────────────
    {
        int64_t total_runs_0 = 0, total_runs_1 = 0;
        int64_t num_runs_0 = 0, num_runs_1 = 0;
        int64_t cur_run = 1;
        uint32_t cur_bit = ud[0] & 1u;

        for (int64_t i = 1; i < n; ++i) {
            uint32_t bit = ud[i] & 1u;
            if (bit == cur_bit) {
                ++cur_run;
            } else {
                if (cur_bit == 0) { total_runs_0 += cur_run; ++num_runs_0; }
                else              { total_runs_1 += cur_run; ++num_runs_1; }
                cur_run = 1;
                cur_bit = bit;
            }
        }
        // Flush last run
        if (cur_bit == 0) { total_runs_0 += cur_run; ++num_runs_0; }
        else              { total_runs_1 += cur_run; ++num_runs_1; }

        bf.mean_run_length_zero = num_runs_0 > 0
            ? static_cast<double>(total_runs_0) / static_cast<double>(num_runs_0) : 0.0;
        bf.mean_run_length_one  = num_runs_1 > 0
            ? static_cast<double>(total_runs_1) / static_cast<double>(num_runs_1) : 0.0;
    }

    // ── Anomaly score vs. reference ──────────────────────────────────────────
    if (reference != nullptr && reference->tensor_name == bf.tensor_name) {
        // Simple L1 distance on key features
        double d = 0.0;
        d += std::abs(bf.lob4_entropy  - reference->lob4_entropy);
        d += std::abs(bf.lob12_entropy - reference->lob12_entropy);
        d += std::abs(bf.mantissa_overall_imbalance - reference->mantissa_overall_imbalance);
        bf.anomaly_score = d / 3.0;  // normalise to [0,1] range approximately
    }

    return bf;
}

} // namespace stega
