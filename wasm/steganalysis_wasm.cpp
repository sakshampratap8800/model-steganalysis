#include <emscripten/emscripten.h>
#include <emscripten/bind.h>
#include "loader/model_loader.h"
#include "fusion/evidence_fusion.h"
#include "statistics/parameter_stats.h"
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// This function can be called from JS to process a file stored in the Emscripten FS
extern "C" {

EMSCRIPTEN_KEEPALIVE
const char* scan_model(const char* meta_json_path) {
    stega::ModelLoader loader;
    stega::ModelArtifact artifact;
    
    if (!loader.load_from_directory(meta_json_path, artifact)) {
        return "{\"error\": \"failed to load artifact\"}";
    }
    
    stega::ParameterStatsExtractor stats_extractor;
    json report;
    report["scan_id"] = artifact.scan_id;
    report["status"] = "complete";
    report["tensor_count"] = artifact.tensor_count();
    
    // Convert JSON to string and return
    // Note: in a real app we need to manage this memory
    static std::string result;
    result = report.dump();
    return result.c_str();
}

} // extern "C"
