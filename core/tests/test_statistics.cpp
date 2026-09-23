// test_statistics.cpp -- Unit tests for ParameterStatsExtractor.
//
// Tests use known synthetic tensors whose statistics are analytically
// computable to verify numerical correctness.

#include <gtest/gtest.h>
#include "statistics/parameter_stats.h"
#include "common/tensor.h"

#include <cmath>
#include <limits>
#include <vector>
#include <cstring>

namespace {

// ── Helpers ────────────────────────────────────────────────────────────────

// Build a TensorDescriptor + raw bytes for a float32 tensor.
struct SyntheticTensor {
    stega::TensorDescriptor desc;
    std::vector<uint8_t> buffer;

    stega::TensorView view() const {
        return stega::TensorView{&desc, buffer.data()};
    }
};

SyntheticTensor make_tensor(const std::string& name,
                            const std::vector<float>& values) {
    SyntheticTensor st;
    st.desc.name            = name;
    st.desc.dtype           = stega::DType::FLOAT32;
    st.desc.parameter_count = static_cast<int64_t>(values.size());
    st.desc.byte_offset     = 0;
    st.desc.byte_size       = values.size() * sizeof(float);
    st.desc.shape           = {static_cast<int64_t>(values.size())};

    st.buffer.resize(st.desc.byte_size);
    std::memcpy(st.buffer.data(), values.data(), st.desc.byte_size);
    return st;
}

// ── Tests ──────────────────────────────────────────────────────────────────

class ParameterStatsTest : public ::testing::Test {
protected:
    stega::ParameterStatsExtractor extractor;
};

// {1, 2, 3, 4, 5}: mean=3, sample variance=2.5 (Bessel's correction), std=sqrt(2.5), median=3
TEST_F(ParameterStatsTest, BasicSequentialTensor) {
    auto st  = make_tensor("t1", {1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    auto s   = extractor.extract(st.view());

    EXPECT_EQ(s.count, 5);
    EXPECT_NEAR(s.mean,     3.0,             1e-9);
    // Sample variance = Σ(x-μ)² / (n-1) = 10/4 = 2.5
    EXPECT_NEAR(s.variance, 2.5,             1e-9);
    EXPECT_NEAR(s.std_dev,  std::sqrt(2.5),  1e-9);
    EXPECT_NEAR(s.median,   3.0,             1e-9);
    EXPECT_NEAR(s.min_val,  1.0,             1e-9);
    EXPECT_NEAR(s.max_val,  5.0,             1e-9);
    // Symmetric distribution → skewness ≈ 0
    EXPECT_NEAR(s.skewness, 0.0, 1e-6);
    EXPECT_EQ(s.nan_count, 0);
    EXPECT_EQ(s.inf_count, 0);
}

// All-zeros: sparsity=1, mean=0, std=0, entropy=0
TEST_F(ParameterStatsTest, AllZerosTensor) {
    auto st = make_tensor("zeros", std::vector<float>(100, 0.0f));
    auto s  = extractor.extract(st.view());

    EXPECT_EQ(s.count, 100);
    EXPECT_NEAR(s.mean,     0.0, 1e-9);
    EXPECT_NEAR(s.std_dev,  0.0, 1e-9);
    EXPECT_NEAR(s.variance, 0.0, 1e-9);
    EXPECT_NEAR(s.sparsity, 1.0, 1e-9);
    EXPECT_NEAR(s.entropy,  0.0, 1e-9);
}

// Tensor with one NaN: should be counted in nan_count, stats on remaining
TEST_F(ParameterStatsTest, TensorWithNaN) {
    std::vector<float> vals = {1.0f, 2.0f, std::numeric_limits<float>::quiet_NaN(), 4.0f, 5.0f};
    auto st = make_tensor("nan_tensor", vals);
    auto s  = extractor.extract(st.view());

    EXPECT_EQ(s.nan_count, 1);
    EXPECT_EQ(s.count, 4);  // 5 total - 1 NaN
    // Mean of {1,2,4,5} = 3.0
    EXPECT_NEAR(s.mean, 3.0, 1e-6);
    EXPECT_EQ(s.inf_count, 0);
}

// Tensor with one Inf: should be counted in inf_count
TEST_F(ParameterStatsTest, TensorWithInf) {
    std::vector<float> vals = {1.0f, 2.0f, std::numeric_limits<float>::infinity(), 4.0f};
    auto st = make_tensor("inf_tensor", vals);
    auto s  = extractor.extract(st.view());

    EXPECT_EQ(s.inf_count, 1);
    EXPECT_EQ(s.count, 3);  // 4 - 1 Inf
    EXPECT_NEAR(s.mean, (1.0 + 2.0 + 4.0) / 3.0, 1e-6);
}

// Single element — variance should be 0 without crash
TEST_F(ParameterStatsTest, SingleElement) {
    auto st = make_tensor("single", {42.0f});
    auto s  = extractor.extract(st.view());

    EXPECT_EQ(s.count, 1);
    EXPECT_NEAR(s.mean,     42.0, 1e-9);
    EXPECT_NEAR(s.std_dev,   0.0, 1e-9);
    EXPECT_NEAR(s.variance,  0.0, 1e-9);
    EXPECT_NEAR(s.median,   42.0, 1e-9);
}

// Empty tensor — should return zeroed stats without crash
TEST_F(ParameterStatsTest, EmptyTensor) {
    auto st = make_tensor("empty", {});
    auto s  = extractor.extract(st.view());

    EXPECT_EQ(s.count, 0);
    EXPECT_NEAR(s.mean,    0.0, 1e-9);
    EXPECT_NEAR(s.std_dev, 0.0, 1e-9);
    EXPECT_NEAR(s.entropy, 0.0, 1e-9);
}

// Entropy: all same value → 0 bits; spread values → positive entropy
TEST_F(ParameterStatsTest, EntropyUniform) {
    // 256 distinct uniformly-spaced values → near-maximum entropy
    std::vector<float> unif;
    unif.reserve(2560);
    for (int i = 0; i < 2560; ++i) {
        unif.push_back(static_cast<float>(i % 256) / 255.0f);
    }
    auto st_unif = make_tensor("unif", unif);
    auto s_unif  = extractor.extract(st_unif.view());

    // All-same: entropy = 0
    auto st_same = make_tensor("same", std::vector<float>(100, 0.5f));
    auto s_same  = extractor.extract(st_same.view());

    EXPECT_NEAR(s_same.entropy, 0.0, 1e-9);
    // Uniform over 256 bins ≈ 8 bits; allow some tolerance
    EXPECT_GT(s_unif.entropy, 7.0);
}

// Sparsity: half values near-zero
TEST_F(ParameterStatsTest, Sparsity) {
    std::vector<float> vals(100, 1.0f);
    for (int i = 0; i < 50; ++i) vals[static_cast<size_t>(i)] = 1e-10f;  // below 1e-6 threshold

    auto st = make_tensor("sparse", vals);
    auto s  = extractor.extract(st.view());

    EXPECT_NEAR(s.sparsity, 0.5, 0.01);
}

// Median of even-size array
TEST_F(ParameterStatsTest, MedianEvenLength) {
    auto st = make_tensor("even", {1.0f, 3.0f, 5.0f, 7.0f});
    auto s  = extractor.extract(st.view());
    EXPECT_NEAR(s.median, 4.0, 1e-9);  // (3+5)/2
}

} // anonymous namespace

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
