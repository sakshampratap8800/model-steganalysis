#pragma once
#include "common/tensor.h"
#include <string>
#include <stdexcept>

namespace stega {

// Loads a ModelArtifact from a directory that contains:
//   meta.json   -- tensor inventory + model metadata
//   weights.bin -- raw float32 tensor bytes in meta.json order
//
// Throws std::runtime_error on any I/O or parse failure.
ModelArtifact load_from_directory(const std::string& dir_path);

} // namespace stega
