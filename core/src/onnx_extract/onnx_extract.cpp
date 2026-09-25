// onnx_extract.cpp
// Standalone C++ program: ONNX model file → meta.json + weights.bin
// Called by the Node.js scanner bridge via child_process.spawn.
//
// Usage:
//   onnx_extract <model.onnx> --output-dir <scan_dir> --scan-id <uuid>
//
// Output:
//   <scan_dir>/meta.json    — tensor inventory (matches scanner_cli expected format)
//   <scan_dir>/weights.bin  — all float32 tensors concatenated as little-endian bytes

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>

// ── Third-party: nlohmann/json (header-only, fetched by CMake) ───────────────
#include <nlohmann/json.hpp>
using json = nlohmann::json;

// ── ONNX Protobuf-lite parser ─────────────────────────────────────────────────
// We parse ONNX manually from its raw protobuf binary without linking protobuf.
// ONNX tensors are stored as protobuf messages; we only need the initializers.
//
// Protobuf wire types:
//   0 = Varint, 1 = 64-bit, 2 = Length-delimited, 5 = 32-bit
//
// ONNX ModelProto field numbers (from onnx.proto):
//   7 = graph (GraphProto)
//     GraphProto field 5 = initializer (TensorProto)
//       TensorProto fields:
//         1  = dims (int64, repeated)
//         2  = data_type (int32): 1=float, 10=float16, 3=int32, 7=int64
//         8  = name (string)
//         4  = float_data (float, repeated, packed)
//         9  = raw_data (bytes)

namespace proto {

// Read a varint from the buffer. Returns number of bytes consumed.
static uint64_t read_varint(const uint8_t* p, const uint8_t* end, int& bytes_read) {
    uint64_t result = 0;
    int shift = 0;
    bytes_read = 0;
    while (p < end) {
        uint8_t b = *p++;
        ++bytes_read;
        result |= (uint64_t)(b & 0x7F) << shift;
        if (!(b & 0x80)) break;
        shift += 7;
        if (shift >= 64) throw std::runtime_error("Varint overflow");
    }
    return result;
}

struct Field {
    int     field_number;
    int     wire_type;
    uint64_t varint_value;
    std::vector<uint8_t> bytes_value; // wire type 2
};

// Parse one level of protobuf fields from [data, data+size).
static std::vector<Field> parse_message(const uint8_t* data, size_t size) {
    std::vector<Field> fields;
    const uint8_t* p   = data;
    const uint8_t* end = data + size;

    while (p < end) {
        int tag_bytes = 0;
        uint64_t tag = read_varint(p, end, tag_bytes);
        p += tag_bytes;

        int field_number = (int)(tag >> 3);
        int wire_type    = (int)(tag & 0x7);

        Field f;
        f.field_number = field_number;
        f.wire_type    = wire_type;

        switch (wire_type) {
            case 0: { // Varint
                int vb = 0;
                f.varint_value = read_varint(p, end, vb);
                p += vb;
                break;
            }
            case 1: { // 64-bit
                if (p + 8 > end) throw std::runtime_error("Truncated 64-bit field");
                uint64_t v = 0;
                std::memcpy(&v, p, 8);
                f.varint_value = v;
                p += 8;
                break;
            }
            case 2: { // Length-delimited
                int lb = 0;
                uint64_t len = read_varint(p, end, lb);
                p += lb;
                if (p + len > end) throw std::runtime_error("Truncated length-delimited field");
                f.bytes_value.assign(p, p + len);
                p += len;
                break;
            }
            case 5: { // 32-bit
                if (p + 4 > end) throw std::runtime_error("Truncated 32-bit field");
                uint32_t v = 0;
                std::memcpy(&v, p, 4);
                f.varint_value = v;
                p += 4;
                break;
            }
            default:
                // Unknown wire type: skip rest of buffer (safe bail)
                p = end;
                break;
        }
        fields.push_back(std::move(f));
    }
    return fields;
}

} // namespace proto

// ── Tensor record ─────────────────────────────────────────────────────────────

struct TensorRecord {
    std::string           name;
    std::vector<int64_t>  shape;
    int                   data_type = 1; // 1 = float32
    std::vector<float>    float_data;    // populated from float_data or raw_data
};

static std::string dtype_string(int dt) {
    switch (dt) {
        case 1:  return "float32";
        case 10: return "float16";
        case 3:  return "int32";
        case 7:  return "int64";
        case 6:  return "int8";
        default: return "unknown";
    }
}

// Parse a TensorProto message into a TensorRecord.
static TensorRecord parse_tensor(const std::vector<uint8_t>& buf) {
    TensorRecord t;
    auto fields = proto::parse_message(buf.data(), buf.size());

    for (auto& f : fields) {
        switch (f.field_number) {
            case 1: // dims (int64, varint)
                if (f.wire_type == 0)
                    t.shape.push_back((int64_t)f.varint_value);
                break;
            case 2: // data_type
                t.data_type = (int)f.varint_value;
                break;
            case 8: // name
                if (f.wire_type == 2)
                    t.name.assign(f.bytes_value.begin(), f.bytes_value.end());
                break;
            case 4: { // float_data (packed floats)
                if (f.wire_type == 2) {
                    size_t n = f.bytes_value.size() / 4;
                    t.float_data.resize(n);
                    std::memcpy(t.float_data.data(), f.bytes_value.data(), n * 4);
                }
                break;
            }
            case 9: { // raw_data (bytes — most common in modern ONNX)
                if (f.wire_type == 2 && !f.bytes_value.empty()) {
                    // Convert to float32. If original dtype != float32, we still
                    // store as float32 for analysis (converting bit-by-bit is lossy
                    // for non-float types, but sufficient for bit-level analysis).
                    if (t.data_type == 1) {
                        // float32 raw bytes: just reinterpret
                        size_t n = f.bytes_value.size() / 4;
                        t.float_data.resize(n);
                        std::memcpy(t.float_data.data(), f.bytes_value.data(), n * 4);
                    } else if (t.data_type == 10) {
                        // float16 → float32 conversion
                        size_t n = f.bytes_value.size() / 2;
                        t.float_data.resize(n);
                        for (size_t i = 0; i < n; ++i) {
                            uint16_t h = 0;
                            std::memcpy(&h, f.bytes_value.data() + i * 2, 2);
                            // Simple fp16 → fp32 conversion
                            int sign     = (h >> 15) & 1;
                            int exponent = (h >> 10) & 0x1F;
                            int mantissa = h & 0x3FF;
                            float val = 0.0f;
                            if (exponent == 0) {
                                val = std::ldexp((float)mantissa, -24);
                            } else if (exponent == 31) {
                                val = mantissa ? std::numeric_limits<float>::quiet_NaN()
                                               : std::numeric_limits<float>::infinity();
                            } else {
                                val = std::ldexp((float)(mantissa + 1024), exponent - 25);
                            }
                            t.float_data[i] = sign ? -val : val;
                        }
                    } else {
                        // int32, int64, int8 — store as float for stats
                        if (t.data_type == 3 && f.bytes_value.size() % 4 == 0) {
                            size_t n = f.bytes_value.size() / 4;
                            t.float_data.resize(n);
                            for (size_t i = 0; i < n; ++i) {
                                int32_t iv = 0;
                                std::memcpy(&iv, f.bytes_value.data() + i * 4, 4);
                                t.float_data[i] = (float)iv;
                            }
                        } else if (t.data_type == 7 && f.bytes_value.size() % 8 == 0) {
                            size_t n = f.bytes_value.size() / 8;
                            t.float_data.resize(n);
                            for (size_t i = 0; i < n; ++i) {
                                int64_t iv = 0;
                                std::memcpy(&iv, f.bytes_value.data() + i * 8, 8);
                                t.float_data[i] = (float)iv;
                            }
                        }
                    }
                }
                break;
            }
            default: break;
        }
    }
    return t;
}

// ── Architecture heuristic ────────────────────────────────────────────────────

static std::string detect_architecture(const std::vector<TensorRecord>& tensors) {
    for (auto& t : tensors) {
        if (t.name.find("layer") != std::string::npos &&
            t.name.find("conv") != std::string::npos) return "resnet";
        if (t.name.find("features.0") != std::string::npos ||
            t.name.find("classifier") != std::string::npos) return "vgg";
        if (t.name.find("bert") != std::string::npos) return "bert";
        if (t.name.find("transformer") != std::string::npos) return "transformer";
    }
    return "unknown";
}

// ── Main ──────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    std::string model_path;
    std::string output_dir = ".";
    std::string scan_id    = "unknown";

    // Parse args
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--output-dir" && i + 1 < argc) output_dir = argv[++i];
        else if (arg == "--scan-id" && i + 1 < argc) scan_id = argv[++i];
        else if (arg[0] != '-') model_path = arg;
    }

    if (model_path.empty()) {
        std::cerr << "Usage: onnx_extract <model.onnx> [--output-dir <dir>] [--scan-id <uuid>]\n";
        return 1;
    }

    // ── Read ONNX file ────────────────────────────────────────────────────────
    std::ifstream fin(model_path, std::ios::binary);
    if (!fin.is_open()) {
        std::cerr << "Error: cannot open " << model_path << "\n";
        return 1;
    }

    std::vector<uint8_t> file_data((std::istreambuf_iterator<char>(fin)),
                                    std::istreambuf_iterator<char>());
    fin.close();

    if (file_data.empty()) {
        std::cerr << "Error: empty file\n";
        return 1;
    }

    // Compute SHA-256 (simple rolling FNV-64 as placeholder; real SHA-256 would need OpenSSL)
    // For a full SHA-256 we'd need a library; we use a deterministic hash instead and note it.
    // In production, pass the hash from Node.js (which computed it already during upload).
    // For now: emit a placeholder that the server fills in from its own crypto.createHash result.
    std::string sha256_placeholder = "computed_by_server";

    // ── Parse ModelProto ──────────────────────────────────────────────────────
    std::vector<TensorRecord> tensors;

    try {
        auto model_fields = proto::parse_message(file_data.data(), file_data.size());

        for (auto& mf : model_fields) {
            if (mf.field_number == 7 && mf.wire_type == 2) { // graph field
                auto graph_fields = proto::parse_message(mf.bytes_value.data(), mf.bytes_value.size());
                for (auto& gf : graph_fields) {
                    if (gf.field_number == 5 && gf.wire_type == 2) { // initializer
                        auto tensor = parse_tensor(gf.bytes_value);
                        if (!tensor.float_data.empty()) {
                            tensors.push_back(std::move(tensor));
                        }
                    }
                }
                break; // only need first graph
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error parsing ONNX protobuf: " << e.what() << "\n";
        return 1;
    }

    if (tensors.empty()) {
        std::cerr << "Error: no float tensors found in " << model_path << "\n";
        return 1;
    }

    // ── Write weights.bin ─────────────────────────────────────────────────────
    std::string weights_path = output_dir + "/weights.bin";
    std::ofstream wout(weights_path, std::ios::binary);
    if (!wout.is_open()) {
        std::cerr << "Error: cannot write " << weights_path << "\n";
        return 1;
    }

    // Build tensor descriptors + write float32 LE bytes
    json tensor_descs = json::array();
    size_t byte_offset = 0;
    int64_t total_params = 0;

    for (auto& t : tensors) {
        // Compute parameter_count from shape
        int64_t param_count = 1;
        for (auto d : t.shape) param_count *= d;
        if (param_count == 0) param_count = (int64_t)t.float_data.size();

        size_t byte_size = t.float_data.size() * 4;

        // Write float32 LE
        wout.write(reinterpret_cast<const char*>(t.float_data.data()), byte_size);

        json td;
        td["name"]            = t.name;
        td["shape"]           = t.shape;
        td["dtype"]           = dtype_string(t.data_type);
        td["parameter_count"] = (int64_t)t.float_data.size();
        td["byte_offset"]     = byte_offset;
        td["byte_size"]       = byte_size;
        tensor_descs.push_back(td);

        byte_offset  += byte_size;
        total_params += (int64_t)t.float_data.size();
    }
    wout.close();

    // ── Write meta.json ───────────────────────────────────────────────────────
    // Extract just the filename from the path
    std::string filename = model_path;
    size_t slash = filename.find_last_of("/\\");
    if (slash != std::string::npos) filename = filename.substr(slash + 1);

    json meta;
    meta["scan_id"]               = scan_id;
    meta["filename"]              = filename;
    meta["sha256"]                = sha256_placeholder;
    meta["file_size_bytes"]       = (int64_t)file_data.size();
    meta["format"]                = "onnx";
    meta["architecture"]          = detect_architecture(tensors);
    meta["dtype"]                 = "float32";
    meta["tensor_count"]          = (int)tensors.size();
    meta["total_parameter_count"] = total_params;
    meta["tensors"]               = tensor_descs;

    std::string meta_path = output_dir + "/meta.json";
    std::ofstream mout(meta_path);
    if (!mout.is_open()) {
        std::cerr << "Error: cannot write " << meta_path << "\n";
        return 1;
    }
    mout << meta.dump(2) << "\n";
    mout.close();

    std::cout << "Extracted " << tensors.size() << " tensors ("
              << total_params << " parameters) to " << output_dir << "\n";
    return 0;
}
