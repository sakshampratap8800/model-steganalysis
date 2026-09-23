#pragma once
// ieee754_analyzer.h - Baseline Bit-Level Entropy analysis of float32 weight tensors.
//
// Computes per-tensor bit-plane statistics, low-order bit entropy,
// NOTE: This serves purely as a Baseline Bit-Level Entropy feature.
// While it detects LSB replacement attacks, it is strictly distinct from
// the GF Few-Shot mechanism proposed in Model X-Ray.

#pragma once
#include "common/tensor.h"
#include <cstdint>
#include <vector>
#include <string>

namespace stega {

// Per-bit-plane statistics for a single float32 bit position (0-31)
struct BitPlaneStats {
    int     bit_pos;        // 0 = LSB of mantissa, 31 = sign bit
    double  freq_one;       // fraction of 1-bits in this position
    double  entropy;        // binary entropy H = -p*log2(p) - (1-p)*log2(1-p)
    double  imbalance;      // |freq_one - 0.5|, 0 = balanced, 0.5 = all 0 or all 1
};

// Full IEEE-754 bit analysis result for one tensor
struct BitFeatures {
    std::string tensor_name;

    // --- Sign field (bit 31) ---
    double sign_freq_negative;      // fraction of negative weights

    // --- Exponent field (bits 23-30) ---
    double exponent_mean;           // mean of 8-bit exponent values
    double exponent_std;
    double exponent_entropy;        // Shannon entropy of exponent histogram (256 bins)

    // --- Mantissa field (bits 0-22) ---
    // Per-bit statistics for each of the 23 mantissa bits
    std::vector<BitPlaneStats> mantissa_bit_planes;  // [0] = bit 0 (LSB), [22] = bit 22

    // --- Low-order bit summary (configurable window) ---
    // Default: bits 0-3 (HBLA range), 0-11 (HMLA range), 0-22 (FMLA range)
    double lob4_entropy;   // entropy of lowest 4 mantissa bits
    double lob12_entropy;  // entropy of lowest 12 mantissa bits
    double lob23_entropy;  // entropy of all 23 mantissa bits

    // --- Run statistics (on bit 0 sequence) ---
    double mean_run_length_zero;    // mean consecutive-zero run length in bit 0
    double mean_run_length_one;     // mean consecutive-one run length in bit 0

    // --- Overall 0/1 imbalance across all mantissa bits ---
    double mantissa_overall_imbalance;  // mean |freq_one - 0.5| across all 23 bits

    // Anomaly score vs. clean reference (set by baseline comparison, default 0)
    double anomaly_score = 0.0;
};

// Stateless analyzer — call analyze() for each tensor
class IEEE754Analyzer {
public:
    // Analyze a single float32 tensor.
    // If reference_features is provided, computes anomaly_score relative to it.
    BitFeatures analyze(const TensorView& tv,
                        const BitFeatures* reference = nullptr) const;

private:
    // Compute binary entropy H(p) = -p*log2(p) - (1-p)*log2(1-p)
    static double binary_entropy(double p);

    // Compute per-bit statistics for a range of mantissa bits [bit_lo, bit_hi)
    static double lob_entropy(const uint32_t* uint_data, int64_t n,
                               int bit_lo, int bit_hi);
};

} // namespace stega