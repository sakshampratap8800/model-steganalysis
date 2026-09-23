#include "structural/structural_analyzer.h"
#include <gtest/gtest.h>

using namespace stega;

TEST(StructuralTest, BasicAnalysis) {
    std::vector<float> data = {
        1.0f, 0.0f, // channel 0 norm: 1.0
        0.0f, 2.0f, // channel 1 norm: 2.0
        3.0f, 4.0f  // channel 2 norm: 5.0
    };
    
    TensorDescriptor desc{"conv.weight", {3, 2}, DType::FLOAT32, 6, 0, 24};
    TensorView tv{&desc, reinterpret_cast<const uint8_t*>(data.data())};

    StructuralAnalyzer analyzer;
    auto sf = analyzer.analyze(tv);
    
    // norms: [1.0, 2.0, 5.0], mean: 8/3 = 2.666
    // variance: ((1-2.66)^2 + (2-2.66)^2 + (5-2.66)^2)/3 = (2.777 + 0.444 + 5.444)/3 = 2.888
    EXPECT_NEAR(sf.channel_norm_variance, 2.8888, 1e-3);
    
    // max diff: 5.0 - 2.0 = 3.0
    EXPECT_NEAR(sf.max_adjacent_channel_diff, 3.0, 1e-5);
    
    EXPECT_GT(sf.channel_norm_gini, 0.0);
}

TEST(StructuralTest, GiniCalculation) {
    // If all norms are equal, Gini is 0
    std::vector<float> data = {
        2.0f, 0.0f, 
        0.0f, 2.0f, 
        0.0f, -2.0f  // all have L2 norm = 2.0
    };
    
    TensorDescriptor desc{"conv.weight", {3, 2}, DType::FLOAT32, 6, 0, 24};
    TensorView tv{&desc, reinterpret_cast<const uint8_t*>(data.data())};

    StructuralAnalyzer analyzer;
    auto sf = analyzer.analyze(tv);
    
    EXPECT_NEAR(sf.channel_norm_gini, 0.0, 1e-6);
    EXPECT_NEAR(sf.channel_norm_variance, 0.0, 1e-6);
    EXPECT_NEAR(sf.max_adjacent_channel_diff, 0.0, 1e-6);
}
