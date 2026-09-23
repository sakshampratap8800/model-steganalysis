#include "baseline/baseline_manager.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>

using json = nlohmann::json;

namespace stega {

bool BaselineManager::load_from_file(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Failed to open baseline file: " << filepath << std::endl;
        return false;
    }
    
    try {
        json j;
        file >> j;
        return load_from_json(j.dump());
    } catch (const std::exception& e) {
        std::cerr << "JSON parse error in baseline file: " << e.what() << std::endl;
        return false;
    }
}

bool BaselineManager::load_from_json(const std::string& json_content) {
    try {
        json j = json::parse(json_content);
        
        for (auto& [arch, tensors_obj] : j.items()) {
            std::unordered_map<std::string, BaselineDistribution> arch_baselines;
            
            for (auto& [tensor_name, stats] : tensors_obj.items()) {
                BaselineDistribution dist;
                dist.mean_expected     = stats.value("mean_expected", 0.0);
                dist.std_expected      = stats.value("std_expected", 0.0);
                dist.variance_expected = stats.value("variance_expected", 0.0);
                dist.entropy_min       = stats.value("entropy_min", 0.0);
                dist.entropy_max       = stats.value("entropy_max", 0.0);
                arch_baselines[tensor_name] = dist;
            }
            
            m_baselines[arch] = std::move(arch_baselines);
        }
        return true;
    } catch (const std::exception& e) {
        std::cerr << "JSON parse error: " << e.what() << std::endl;
        return false;
    }
}

std::optional<BaselineDistribution> BaselineManager::get_baseline(
    const std::string& architecture, const std::string& tensor_name) const 
{
    auto arch_it = m_baselines.find(architecture);
    if (arch_it != m_baselines.end()) {
        auto tensor_it = arch_it->second.find(tensor_name);
        if (tensor_it != arch_it->second.end()) {
            return tensor_it->second;
        }
    }
    return std::nullopt;
}

} // namespace stega
