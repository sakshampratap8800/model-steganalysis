# CMakeLists.txt for WebAssembly build (Emscripten)
cmake_minimum_required(VERSION 3.20)
project(steganalysis_wasm)

# We use the same core source files
set(CORE_SOURCES
    ../core/src/common/tensor.cpp
    ../core/src/loader/model_loader.cpp
    ../core/src/profiler/model_profiler.cpp
    ../core/src/statistics/parameter_stats.cpp
    ../core/src/bit_analysis/ieee754_analyzer.cpp
    ../core/src/decision/trojan_q.cpp
    ../core/src/structural/structural_analyzer.cpp
    ../core/src/baseline/baseline_manager.cpp
    ../core/src/fusion/evidence_fusion.cpp
)

add_library(stega_wasm STATIC ${CORE_SOURCES})
target_include_directories(stega_wasm PUBLIC 
    ../core/include
    ../build/_deps/nlohmann_json-src/include
)

# WASM Bindings
add_executable(scanner_wasm steganalysis_wasm.cpp)
target_link_libraries(scanner_wasm PRIVATE stega_wasm)
target_include_directories(scanner_wasm PRIVATE 
    ../core/include
    ../build/_deps/nlohmann_json-src/include
)

# Emscripten specific flags
set_target_properties(scanner_wasm PROPERTIES 
    LINK_FLAGS "-s MODULARIZE=1 -s EXPORT_NAME='createStegaScanner' -s EXPORTED_RUNTIME_METHODS=['ccall','cwrap','FS'] -s ALLOW_MEMORY_GROWTH=1"
)
