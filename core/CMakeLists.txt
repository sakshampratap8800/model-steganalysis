cmake_minimum_required(VERSION 3.20)

# ── Library: steganalysis_core ─────────────────────────────────────────────
add_library(steganalysis_core STATIC
    src/common/tensor.cpp
    src/loader/model_loader.cpp
    src/profiler/model_profiler.cpp
    src/statistics/parameter_stats.cpp
    src/bit_analysis/ieee754_analyzer.cpp
    src/decision/trojan_q.cpp
    src/structural/structural_analyzer.cpp
    src/baseline/baseline_manager.cpp
    src/fusion/evidence_fusion.cpp
)

target_include_directories(steganalysis_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${nlohmann_json_SOURCE_DIR}/include
)

target_compile_features(steganalysis_core PUBLIC cxx_std_17)

# ── Executable: scanner_cli ────────────────────────────────────────────────
add_executable(scanner_cli src/main.cpp)
target_link_libraries(scanner_cli PRIVATE steganalysis_core)
target_include_directories(scanner_cli PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${nlohmann_json_SOURCE_DIR}/include
)

# ── Tests ──────────────────────────────────────────────────────────────────
add_executable(core_tests
    tests/test_profiler.cpp
    tests/test_statistics.cpp
    tests/test_trojan_q.cpp
    tests/test_structural.cpp
    tests/test_fusion.cpp
)

target_link_libraries(core_tests PRIVATE
    steganalysis_core
    GTest::gtest_main
)

target_include_directories(core_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${nlohmann_json_SOURCE_DIR}/include
)

include(GoogleTest)
gtest_discover_tests(core_tests)
