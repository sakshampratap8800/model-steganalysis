// test_profiler.cpp — Unit tests for ModelProfiler
// Tests architecture detection and per-tensor profiling with synthetic data.

#include "profiler/model_profiler.h"
#include "common/tensor.h"
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <cstring>

using namespace stega;

// ── Helper: build a synthetic ModelArtifact ────────────────────────────────

static ModelArtifact make_artifact(
    const std::vector<std::pair<std::string, std::vector<float>>>& tensors,
    const std::string& architecture = "unknown")
{
    ModelArtifact art;
    art.scan_id      = "test-scan-001";
    art.filename     = "test_model.onnx";
    art.sha256       = std::string(64, '0');
    art.format       = "onnx";
    art.architecture = architecture;
    art.dtype_str    = "float32";

    size_t offset = 0;
    for (const auto& [name, data] : tensors) {
        TensorDescriptor td;
        td.name            = name;
        td.dtype           = DType::FLOAT32;
        td.parameter_count = static_cast<int64_t>(data.size());
        td.shape           = {static_cast<int64_t>(data.size())};
        td.byte_offset     = offset;
        td.byte_size       = data.size() * sizeof(float);
        art.tensor_descriptors.push_back(td);
        offset += td.byte_size;
    }

    // Concatenate all float data into weight_buffer
    art.weight_buffer.resize(offset);
    size_t buf_off = 0;
    for (const auto& [name, data] : tensors) {
        std::memcpy(art.weight_buffer.data() + buf_off,
                    data.data(),
                    data.size() * sizeof(float));
        buf_off += data.size() * sizeof(float);
    }

    art.total_parameter_count = static_cast<int64_t>(offset / sizeof(float));
    return art;
}

// ── Basic profiling ────────────────────────────────────────────────────────

TEST(ModelProfilerTest, TensorCountAndNames) {
    auto art = make_artifact({
        {"layer1.weight", {1.0f, 2.0f, 3.0f}},
        {"layer1.bias",   {0.1f, 0.2f}},
    });
    ModelProfiler profiler;
    ModelProfile  mp = profiler.profile(art);

    EXPECT_EQ(mp.tensor_count, 2);
    ASSERT_EQ(mp.tensors.size(), 2u);
    EXPECT_EQ(mp.tensors[0].name, "layer1.weight");
    EXPECT_EQ(mp.tensors[1].name, "layer1.bias");
    EXPECT_EQ(mp.total_parameter_count, 5);
}

TEST(ModelProfilerTest, TensorMeanAndStd) {
    // {1, 2, 3, 4, 5} → mean=3, variance=2.5 (population) or 2.5 sample
    std::vector<float> vals = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    auto art = make_artifact({{"fc.weight", vals}});
    ModelProfiler profiler;
    ModelProfile  mp = profiler.profile(art);

    ASSERT_EQ(mp.tensors.size(), 1u);
    const TensorProfile& tp = mp.tensors[0];

    EXPECT_NEAR(tp.mean,    3.0, 1e-6);
    EXPECT_NEAR(tp.min_val, 1.0, 1e-6);
    EXPECT_NEAR(tp.max_val, 5.0, 1e-6);
    EXPECT_GT(tp.std_dev, 0.0);
    EXPECT_NEAR(tp.median,  3.0, 1e-6);
}

TEST(ModelProfilerTest, SparsityAllZeros) {
    std::vector<float> vals(100, 0.0f);
    auto art = make_artifact({{"sparse.weight", vals}});
    ModelProfiler profiler;
    ModelProfile  mp = profiler.profile(art);

    ASSERT_EQ(mp.tensors.size(), 1u);
    EXPECT_NEAR(mp.tensors[0].sparsity, 1.0, 1e-6);
    EXPECT_NEAR(mp.tensors[0].mean,     0.0, 1e-6);
    EXPECT_NEAR(mp.tensors[0].std_dev,  0.0, 1e-6);
}

// ── Architecture detection ─────────────────────────────────────────────────

TEST(ModelProfilerTest, ArchitectureDetect_ResNet) {
    auto art = make_artifact({
        {"layer1.0.conv1.weight", {1.0f, 2.0f}},
        {"layer2.1.conv2.weight", {3.0f, 4.0f}},
        {"fc.weight",             {0.5f}},
    });
    art.architecture = ""; // force detection
    ModelProfiler profiler;
    ModelProfile  mp = profiler.profile(art);
    // Should detect 'resnet' hint
    EXPECT_NE(mp.architecture_hint.find("resnet"), std::string::npos);
}

TEST(ModelProfilerTest, ArchitectureDetect_VGG) {
    auto art = make_artifact({
        {"features.0.weight",     {1.0f}},
        {"features.2.weight",     {2.0f}},
        {"classifier.6.weight",   {3.0f}},
    });
    art.architecture = "";
    ModelProfiler profiler;
    ModelProfile  mp = profiler.profile(art);
    EXPECT_NE(mp.architecture_hint.find("vgg"), std::string::npos);
}

// ── Edge cases ─────────────────────────────────────────────────────────────

TEST(ModelProfilerTest, SingleElementTensor) {
    auto art = make_artifact({{"bias", {42.0f}}});
    ModelProfiler profiler;
    ModelProfile  mp = profiler.profile(art);

    ASSERT_EQ(mp.tensors.size(), 1u);
    EXPECT_NEAR(mp.tensors[0].mean,    42.0, 1e-6);
    EXPECT_NEAR(mp.tensors[0].std_dev,  0.0, 1e-6);
}

TEST(ModelProfilerTest, ScanIdAndMetadataPreserved) {
    auto art = make_artifact({{"w", {1.0f}}});
    art.scan_id  = "scan-xyz-123";
    art.sha256   = "deadbeef";
    art.filename = "my_model.onnx";

    ModelProfiler profiler;
    ModelProfile  mp = profiler.profile(art);

    EXPECT_EQ(mp.scan_id,  "scan-xyz-123");
    EXPECT_EQ(mp.sha256,   "deadbeef");
    EXPECT_EQ(mp.filename, "my_model.onnx");
}
