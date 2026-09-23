#include "fusion/evidence_fusion.h"
#include <gtest/gtest.h>

using namespace stega;

TEST(FusionTest, EvaluateCleanTensor) {
    TensorStats stats;
    stats.variance = 1.0;
    stats.entropy = 5.0;
    
    BitFeatures bits;
    bits.lob4_entropy = 3.5; // Normal clean
    bits.mantissa_overall_imbalance = 0.01;
    
    StructuralFeatures struct_feats;
    BaselineManager baselines; // Empty, no baseline available
    
    EvidenceFusion fusion;
    auto res = fusion.evaluate_tensor("resnet18", "conv1", stats, bits, struct_feats, baselines);
    
    EXPECT_NEAR(res.risk_score, 0.0, 1e-6);
    EXPECT_EQ(res.verdict, Verdict::CLEAN);
    EXPECT_TRUE(res.triggers.empty());
}

TEST(FusionTest, EvaluateAttackedTensor) {
    TensorStats stats;
    stats.variance = 1.0;
    stats.entropy = 5.0;
    
    BitFeatures bits;
    bits.lob4_entropy = 3.95; // highly elevated due to LSB fill
    bits.mantissa_overall_imbalance = 0.06;
    
    StructuralFeatures struct_feats;
    BaselineManager baselines;
    
    EvidenceFusion fusion;
    auto res = fusion.evaluate_tensor("resnet18", "conv1", stats, bits, struct_feats, baselines);
    
    EXPECT_GT(res.risk_score, 0.9); // Should cap at 1.0
    EXPECT_EQ(res.verdict, Verdict::MALICIOUS);
    EXPECT_FALSE(res.triggers.empty());
}

TEST(FusionTest, GlobalRiskAggregation) {
    FusionResult t1;
    t1.risk_score = 0.1;
    
    FusionResult t2;
    t2.risk_score = 0.95; // One highly malicious tensor
    t2.triggers.push_back("Extremely high LOB4");
    
    std::vector<FusionResult> t_results = {t1, t2};
    TrojanQResult trojan;
    trojan.significant = false; // No trojan
    
    EvidenceFusion fusion;
    auto global = fusion.compute_global_risk(t_results, trojan);
    
    EXPECT_NEAR(global.risk_score, 0.95, 1e-6);
    EXPECT_EQ(global.verdict, Verdict::MALICIOUS);
    
    // Test Trojan override
    t2.risk_score = 0.1;
    t_results = {t1, t2};
    trojan.significant = true;
    
    auto global2 = fusion.compute_global_risk(t_results, trojan);
    EXPECT_GE(global2.risk_score, 0.9);
    EXPECT_EQ(global2.verdict, Verdict::MALICIOUS);
}
