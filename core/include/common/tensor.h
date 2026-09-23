#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include <cstddef>

namespace stega {

enum class DType { FLOAT32, FLOAT16, INT8, INT32, INT64, UNKNOWN };

DType dtype_from_string(const std::string& s);
std::string dtype_to_string(DType d);
size_t dtype_byte_size(DType d);

struct TensorDescriptor {
    std::string name;
    std::vector<int64_t> shape;
    DType dtype;
    int64_t parameter_count;
    size_t byte_offset;  // byte offset within weights.bin
    size_t byte_size;    // number of bytes for this tensor
};

// Non-owning view into a loaded tensor's data.
struct TensorView {
    const TensorDescriptor* desc;
    const uint8_t* data;  // raw bytes, not owned

    const float* as_float32() const {
        return reinterpret_cast<const float*>(data);
    }

    int64_t num_elements() const {
        return desc->parameter_count;
    }

    bool is_float32() const {
        return desc->dtype == DType::FLOAT32;
    }
};

// Owns all data for a loaded model artifact.
struct ModelArtifact {
    std::string scan_id;
    std::string filename;
    std::string sha256;
    uint64_t file_size_bytes = 0;
    std::string format;
    std::string architecture;
    std::string dtype_str;
    int64_t total_parameter_count = 0;
    std::vector<TensorDescriptor> tensor_descriptors;
    std::vector<uint8_t> weight_buffer;  // owns all tensor bytes

    TensorView tensor_view(size_t idx) const {
        const TensorDescriptor& d = tensor_descriptors[idx];
        return TensorView{&d, weight_buffer.data() + d.byte_offset};
    }

    size_t tensor_count() const { return tensor_descriptors.size(); }
};

} // namespace stega
