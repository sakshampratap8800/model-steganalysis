#include "decision/trojan_q.h"
#include <gtest/gtest.h>

using namespace stega;

TEST(TrojanQTest, CleanModel) {
    std::vector<float> data = {
        0.1f, 0.2f, // class 0 mean: 0.15
        0.2f, 0.3f, // class 1 mean: 0.25
        0.3f, 0.4f, // class 2 mean: 0.35
        0.4f, 0.5f  // class 3 mean: 0.45
    };
    
    TensorDescriptor desc{"fc.weight", {4, 2}, DType::FLOAT32, 8, 0, 32};
    TensorView tv{&desc, reinterpret_cast<const uint8_t*>(data.data())};

    TrojanSignatureDetector det(0.05);
    auto res = det.detect(tv);
    
    // Q for {0.15, 0.25, 0.35, 0.45}:
    // range = 0.30
    // w_max = 0.45, w_prev = 0.35 -> diff = 0.10
    // Q = 0.10 / 0.30 = 0.333...
    // Critical value for n=4 at alpha 0.05 is 0.829
    
    EXPECT_NEAR(res.q_statistic, 0.333333, 1e-5);
    EXPECT_NEAR(res.critical_value, 0.829, 1e-5);
    EXPECT_FALSE(res.significant);
    EXPECT_EQ(res.candidate_class, 3);
}

TEST(TrojanQTest, TrojanModel) {
    std::vector<float> data = {
        0.1f, 0.2f, // class 0 mean: 0.15
        0.2f, 0.3f, // class 1 mean: 0.25
        0.3f, 0.4f, // class 2 mean: 0.35
        5.0f, 6.0f  // class 3 mean: 5.50 (outlier)
    };
    
    TensorDescriptor desc{"fc.weight", {4, 2}, DType::FLOAT32, 8, 0, 32};
    TensorView tv{&desc, reinterpret_cast<const uint8_t*>(data.data())};

    TrojanSignatureDetector det(0.05);
    auto res = det.detect(tv);
    
    // Q for {0.15, 0.25, 0.35, 5.50}:
    // range = 5.50 - 0.15 = 5.35
    // diff = 5.50 - 0.35 = 5.15
    // Q = 5.15 / 5.35 = 0.9626
    
    EXPECT_NEAR(res.q_statistic, 0.962617, 1e-5);
    EXPECT_NEAR(res.critical_value, 0.829, 1e-5);
    EXPECT_TRUE(res.significant);
    EXPECT_EQ(res.candidate_class, 3);
}
