#pragma once
// baseline_manager.h — Clean Reference Baseline Manager
// Loads pre-computed statistical bounds for clean models.
// Used for anomaly detection against known clean weight distributions.

#include "statistics/parameter_stats.h"
#include <unordered_map>
#include <string>
#include <optional>

namespace stega {

struct BaselineDistribution {
    double mean_expected;
    double std_expected;
    double variance_expected;
    double entropy_min;
    double entropy_max;
};

class BaselineManager {
public:
    BaselineManager() = default;

    // Load baselines from a JSON file path
    bool load_from_file(const std::string& filepath);
    
    // Parse directly from JSON string (useful for testing or embedded WASM)
    bool load_from_json(const std::string& json_content);

    // Get expected distribution for a specific tensor in a specific architecture
    std::optional<BaselineDistribution> get_baseline(const std::string& architecture, 
                                                     const std::string& tensor_name) const;

    bool is_loaded() const { return !m_baselines.empty(); }

private:
    // map[architecture][tensor_name] -> BaselineDistribution
    std::unordered_map<std::string, std::unordered_map<std::string, BaselineDistribution>> m_baselines;
};

} // namespace stega
