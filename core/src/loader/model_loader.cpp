#include "loader/model_loader.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <filesystem>

namespace stega {

namespace fs = std::filesystem;
using json   = nlohmann::json;

ModelArtifact load_from_directory(const std::string& dir_path) {
    fs::path dir(dir_path);

    // ── Read meta.json ────────────────────────────────────────────────────
    fs::path meta_path = dir / "meta.json";
    std::ifstream meta_file(meta_path);
    if (!meta_file.is_open()) {
        throw std::runtime_error("Cannot open meta.json at: " + meta_path.string());
    }

    json meta;
    try {
        meta_file >> meta;
    } catch (const json::parse_error& e) {
        throw std::runtime_error(std::string("Failed to parse meta.json: ") + e.what());
    }

    ModelArtifact artifact;
    artifact.scan_id               = meta.value("scan_id", "");
    artifact.filename              = meta.value("filename", "");
    artifact.sha256                = meta.value("sha256", "");
    artifact.file_size_bytes       = meta.value("file_size_bytes", uint64_t{0});
    artifact.format                = meta.value("format", "unknown");
    artifact.architecture          = meta.value("architecture", "unknown");
    artifact.dtype_str             = meta.value("dtype", "float32");
    artifact.total_parameter_count = meta.value("total_parameter_count", int64_t{0});

    // Parse tensor descriptors
    if (meta.contains("tensors") && meta["tensors"].is_array()) {
        for (const auto& t : meta["tensors"]) {
            TensorDescriptor td;
            td.name            = t.value("name", "");
            td.dtype           = dtype_from_string(t.value("dtype", "float32"));
            td.parameter_count = t.value("parameter_count", int64_t{0});
            td.byte_offset     = t.value("byte_offset", size_t{0});
            td.byte_size       = t.value("byte_size", size_t{0});

            if (t.contains("shape") && t["shape"].is_array()) {
                for (const auto& dim : t["shape"]) {
                    td.shape.push_back(dim.get<int64_t>());
                }
            }
            artifact.tensor_descriptors.push_back(std::move(td));
        }
    }

    // ── Read weights.bin ──────────────────────────────────────────────────
    fs::path weights_path = dir / "weights.bin";
    std::ifstream weights_file(weights_path, std::ios::binary | std::ios::ate);
    if (!weights_file.is_open()) {
        throw std::runtime_error("Cannot open weights.bin at: " + weights_path.string());
    }

    std::streamsize weights_size = weights_file.tellg();
    if (weights_size < 0) {
        throw std::runtime_error("Failed to determine size of weights.bin (tellg returned -1).");
    }
    weights_file.seekg(0, std::ios::beg);

    artifact.weight_buffer.resize(static_cast<size_t>(weights_size));
    if (!weights_file.read(
            reinterpret_cast<char*>(artifact.weight_buffer.data()),
            weights_size)) {
        throw std::runtime_error("Failed to read weights.bin: " + weights_path.string());
    }

    // Validate offsets don't exceed buffer
    for (const auto& td : artifact.tensor_descriptors) {
        // Split check to avoid size_t wraparound (integer overflow)
        if (td.byte_offset > artifact.weight_buffer.size() ||
            td.byte_size > artifact.weight_buffer.size() - td.byte_offset) {
            throw std::runtime_error(
                "Tensor '" + td.name + "' byte range [" +
                std::to_string(td.byte_offset) + ", " +
                std::to_string(td.byte_offset + td.byte_size) +
                ") exceeds weights.bin size " +
                std::to_string(artifact.weight_buffer.size()));
        }
        
        size_t expected_bytes = td.parameter_count * dtype_byte_size(td.dtype);
        if (expected_bytes > td.byte_size) {
            throw std::runtime_error("Tensor '" + td.name + "' parameter count exceeds byte size");
        }
        
        // Handle scalar tensors (empty shape = rank-0 tensor with 1 element, or 0 params)
        if (td.shape.empty()) {
            // scalar: 0 or 1 param is both valid depending on dtype
            continue;
        }
        
        int64_t shape_product = 1;
        for (int64_t dim : td.shape) {
            shape_product *= dim;
        }
        if (shape_product != td.parameter_count) {
             throw std::runtime_error("Tensor '" + td.name + "' shape product " +
                std::to_string(shape_product) + " != parameter_count " +
                std::to_string(td.parameter_count));
        }
    }

    return artifact;
}

} // namespace stega
