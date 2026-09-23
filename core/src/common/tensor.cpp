#include "common/tensor.h"
#include <stdexcept>

namespace stega {

DType dtype_from_string(const std::string& s) {
    if (s == "float32" || s == "FLOAT" || s == "fp32") return DType::FLOAT32;
    if (s == "float16" || s == "FLOAT16" || s == "fp16") return DType::FLOAT16;
    if (s == "int8"   || s == "INT8")   return DType::INT8;
    if (s == "int32"  || s == "INT32")  return DType::INT32;
    if (s == "int64"  || s == "INT64")  return DType::INT64;
    return DType::UNKNOWN;
}

std::string dtype_to_string(DType d) {
    switch (d) {
        case DType::FLOAT32: return "float32";
        case DType::FLOAT16: return "float16";
        case DType::INT8:    return "int8";
        case DType::INT32:   return "int32";
        case DType::INT64:   return "int64";
        default:             return "unknown";
    }
}

size_t dtype_byte_size(DType d) {
    switch (d) {
        case DType::FLOAT32: return 4;
        case DType::FLOAT16: return 2;
        case DType::INT8:    return 1;
        case DType::INT32:   return 4;
        case DType::INT64:   return 8;
        default: return 0;
    }
}

} // namespace stega
