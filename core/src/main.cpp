// scanner_cli -- main entry point for the Phase 2 steganalysis scanner.
//
// Usage:
//   scanner_cli <meta.json> [--output <report.json>]
//
// Reads meta.json + sibling weights.bin, runs ModelProfiler and
// ParameterStatsExtractor, then emits a structured JSON report.

#include "loader/model_loader.h"
#include "profiler/model_profiler.h"
#include "statistics/parameter_stats.h"
#include "bit_analysis/ieee754_analyzer.h"
#include "decision/trojan_q.h"
#include "structural/structural_analyzer.h"
#include "baseline/baseline_manager.h"
#include "fusion/evidence_fusion.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace fs      = std::filesystem;
using json        = nlohmann::json;
using steady_clk  = std::chrono::steady_clock;

// ── Serialise helpers ──────────────────────────────────────────────────────

static json serialise_profile(const stega::ModelProfile& mp) {
    json j;
    j["scan_id"]               = mp.scan_id;
    j["filename"]              = mp.filename;
    j["sha256"]                = mp.sha256;
    j["format"]                = mp.format;
    j["architecture"]          = mp.architecture;
    j["architecture_hint"]     = mp.architecture_hint;
    j["dtype_str"]             = mp.dtype_str;
    j["tensor_count"]          = mp.tensor_count;
    j["total_parameter_count"] = mp.total_parameter_count;

    json tensors = json::array();
    for (const auto& tp : mp.tensors) {
        json t;
        t["name"]            = tp.name;
        t["shape"]           = tp.shape;
        t["dtype_str"]       = tp.dtype_str;
        t["parameter_count"] = tp.parameter_count;
        t["min_val"]         = tp.min_val;
        t["max_val"]         = tp.max_val;
        t["mean"]            = tp.mean;
        t["std_dev"]         = tp.std_dev;
        t["variance"]        = tp.variance;
        t["median"]          = tp.median;
        t["sparsity"]        = tp.sparsity;
        tensors.push_back(t);
    }
    j["tensors"] = tensors;
    return j;
}

static json serialise_stats(const std::string& name, 
                            const stega::TensorStats& s,
                            const stega::BitFeatures& bf,
                            const stega::StructuralFeatures& sf) {
    json j;
    j["name"] = name;
    
    json stats;
    stats["mean"]      = s.mean;
    stats["std_dev"]   = s.std_dev;
    stats["variance"]  = s.variance;
    stats["min_val"]   = s.min_val;
    stats["max_val"]   = s.max_val;
    stats["median"]    = s.median;
    stats["skewness"]  = s.skewness;
    stats["kurtosis"]  = s.kurtosis;
    stats["entropy"]   = s.entropy;
    stats["sparsity"]  = s.sparsity;
    stats["count"]     = s.count;
    j["stats"] = stats;
    
    json bit_stats;
    bit_stats["sign_freq_negative"] = bf.sign_freq_negative;
    bit_stats["exponent_mean"] = bf.exponent_mean;
    bit_stats["exponent_std"] = bf.exponent_std;
    bit_stats["exponent_entropy"] = bf.exponent_entropy;
    bit_stats["lob4_entropy"] = bf.lob4_entropy;
    bit_stats["lob12_entropy"] = bf.lob12_entropy;
    bit_stats["lob23_entropy"] = bf.lob23_entropy;
    bit_stats["mantissa_overall_imbalance"] = bf.mantissa_overall_imbalance;
    bit_stats["mean_run_length_zero"] = bf.mean_run_length_zero;
    bit_stats["mean_run_length_one"] = bf.mean_run_length_one;
    j["bit_features"] = bit_stats;
    
    json struct_stats;
    struct_stats["channel_norm_variance"] = sf.channel_norm_variance;
    struct_stats["max_adjacent_channel_diff"] = sf.max_adjacent_channel_diff;
    struct_stats["channel_norm_gini"] = sf.channel_norm_gini;
    struct_stats["cross_layer_rank_correlation"] = sf.cross_layer_rank_correlation;
    struct_stats["anomaly_score"] = sf.anomaly_score;
    j["structural_features"] = struct_stats;
    
    return j;
}

static json serialise_trojan_q(const stega::TrojanQResult& tq) {
    json j;
    j["significant"] = tq.significant;
    j["q_statistic"] = tq.q_statistic;
    j["critical_value"] = tq.critical_value;
    j["candidate_class"] = tq.candidate_class;
    j["candidate_mean"] = tq.candidate_mean;
    j["range_value"] = tq.range_value;
    j["evidence_score"] = tq.evidence_score;
    j["explanation"] = tq.explanation;
    return j;
}

// ── main ───────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: scanner_cli <meta.json> [--output <report.json>] [--baseline <baselines.json>]\n";
        return 1;
    }

    fs::path meta_path(argv[1]);
    std::string output_path;
    std::string baseline_path;

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--output" || arg == "-o") && i + 1 < argc) {
            output_path = argv[++i];
        } else if ((arg == "--baseline" || arg == "-b") && i + 1 < argc) {
            baseline_path = argv[++i];
        }
    }

    if (!fs::exists(meta_path)) {
        std::cerr << "Error: meta.json not found: " << meta_path << "\n";
        return 1;
    }

    auto t_start = steady_clk::now();

    // ── Load artifact ──────────────────────────────────────────────────────
    stega::ModelArtifact artifact;
    try {
        artifact = stega::load_from_directory(meta_path.parent_path().string());
    } catch (const std::exception& e) {
        std::cerr << "Error loading artifact: " << e.what() << "\n";
        return 1;
    }

    // ── Profile ────────────────────────────────────────────────────────────
    stega::ModelProfiler profiler;
    stega::ModelProfile  profile = profiler.profile(artifact);

    // ── Per-tensor statistics ──────────────────────────────────────────────
    stega::ParameterStatsExtractor stats_extractor;
    stega::IEEE754Analyzer         bit_analyzer;
    stega::StructuralAnalyzer      struct_analyzer;
    stega::TrojanSignatureDetector trojan_detector(0.05);
    stega::BaselineManager         baselines;
    if (!baseline_path.empty()) {
        try {
            baselines.load_from_file(baseline_path);
        } catch (const std::exception& e) {
            std::cerr << "Warning: Failed to load baselines from " << baseline_path << ": " << e.what() << "\n";
        }
    }
    stega::EvidenceFusion          fusion;
    
    json tensor_stats_arr = json::array();
    stega::TrojanQResult final_trojan_res;
    bool found_fc = false;
    
    std::vector<stega::FusionResult> fusion_results;
    
    for (size_t i = 0; i < artifact.tensor_count(); ++i) {
        stega::TensorView tv    = artifact.tensor_view(i);
        stega::TensorStats ts   = stats_extractor.extract(tv);
        stega::BitFeatures bf   = bit_analyzer.analyze(tv);
        stega::StructuralFeatures sf = struct_analyzer.analyze(tv);
        
        // Evaluate tensor risk
        stega::FusionResult fr = fusion.evaluate_tensor(profile.architecture_hint, tv.desc->name, ts, bf, sf, baselines);
        fusion_results.push_back(fr);
        
        json j_tensor = serialise_stats(tv.desc->name, ts, bf, sf);
        j_tensor["risk_score"] = fr.risk_score;
        j_tensor["verdict"] = (fr.verdict == stega::Verdict::CLEAN) ? "CLEAN" : 
                              (fr.verdict == stega::Verdict::SUSPICIOUS) ? "SUSPICIOUS" : "MALICIOUS";
        j_tensor["triggers"] = fr.triggers;
        
        tensor_stats_arr.push_back(j_tensor);
        
        // We defer Trojan Q-test to the end to ensure we get the LAST 2D layer.
    }

    // Find the final 2D layer for Trojan detection
    for (int i = artifact.tensor_count() - 1; i >= 0; --i) {
        stega::TensorView tv = artifact.tensor_view(i);
        if (tv.desc->shape.size() == 2) {
            std::string n = tv.desc->name;
            // It's the last 2D tensor, highly likely to be the classification head.
            final_trojan_res = trojan_detector.detect(tv);
            found_fc = true;
            break;
        }
    }

    auto t_end = steady_clk::now();
    auto ms     = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();

    // ── Compute Global Risk ────────────────────────────────────────────────
    stega::FusionResult global_risk;
    if (found_fc) {
        global_risk = fusion.compute_global_risk(fusion_results, final_trojan_res);
    } else {
        global_risk = fusion.compute_global_risk(fusion_results, std::nullopt);
    }

    // ── Assemble report ────────────────────────────────────────────────────
    json report;
    report["scan_id"]         = artifact.scan_id;
    report["status"]          = "complete";
    report["profile"]         = serialise_profile(profile);
    
    json final_verdict;
    final_verdict["calibrated_probability"] = global_risk.calibrated_probability;
    final_verdict["risk_score"] = global_risk.risk_score;
    final_verdict["verdict"] = (global_risk.verdict == stega::Verdict::CLEAN) ? "CLEAN" : 
                               (global_risk.verdict == stega::Verdict::SUSPICIOUS) ? "SUSPICIOUS" : "MALICIOUS";
    final_verdict["triggers"] = global_risk.triggers;
    
    json ev_vector;
    ev_vector["bit_entropy_deviation"] = global_risk.evidence_vector.bit_entropy_deviation;
    ev_vector["structural_deviation"] = global_risk.evidence_vector.structural_deviation;
    ev_vector["statistical_deviation"] = global_risk.evidence_vector.statistical_deviation;
    ev_vector["signature_deviation"] = global_risk.evidence_vector.signature_deviation;
    final_verdict["normalized_evidence_vector"] = ev_vector;
    
    report["global_risk"]     = final_verdict;
    
    report["tensor_stats"]    = tensor_stats_arr;
    if (found_fc) {
        report["trojan_signature"] = serialise_trojan_q(final_trojan_res);
    }
    report["scan_duration_ms"] = ms;

    std::string report_str = report.dump(2);

    if (output_path.empty()) {
        std::cout << report_str << "\n";
    } else {
        std::ofstream out(output_path);
        if (!out.is_open()) {
            std::cerr << "Error: cannot open output file: " << output_path << "\n";
            return 1;
        }
        out << report_str << "\n";
        std::cout << "Report written to: " << output_path << "\n";
    }

    return 0;
}
