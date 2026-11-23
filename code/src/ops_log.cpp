#include "ops_log.h"
#include "nlohmann/json.hpp"
#include <ctime>
#include <cstdio>
#include <filesystem>

namespace souris {
namespace {
std::string timestamp() {
    const std::time_t now = std::time(nullptr);
    std::tm utc{}; gmtime_r(&now, &utc);
    char text[32]; std::strftime(text, sizeof(text), "%Y-%m-%dT%H:%M:%SZ", &utc);
    return text;
}
}
OpsLog::OpsLog(const Config & config) : path_(config.log_file), errors_only_(config.log_level == "error"),
    model_(std::filesystem::path(config.model.srs1_path).stem().string()), mmq_group_(config.mmq_group) { open_file(); }
void OpsLog::open_file() {
    if (path_.empty()) { return; }
    const std::string day = timestamp().substr(0, 10);
    if (day_ != day) {
        file_.close();
        const auto path = std::filesystem::u8path(path_);
        if (path.has_parent_path()) {
            std::error_code error;
            std::filesystem::create_directories(path.parent_path(), error);
            if (error) { GGML_ABORT("cannot create operations log directory: %s", error.message().c_str()); }
        }
        file_.open(path_ + "." + day + ".jsonl", std::ios::app);
        if (!file_) { GGML_ABORT("cannot open operations log"); }
        day_ = day;
    }
}
void OpsLog::write(const std::string & line) {
    std::fwrite(line.data(), 1, line.size(), stdout); std::fputc('\n', stdout); std::fflush(stdout);
    open_file();
    if (path_.empty()) { return; }
    file_ << line << '\n'; file_.flush();
}
void OpsLog::request_end(const SessionManager & manager, const CompletionResult & result,
                         const std::string & id, const std::string & error_text) {
    if (errors_only_ && error_text.empty()) { return; }
    auto row = nlohmann::json::parse(manager.metrics());
    row.update({{"ts", timestamp()}, {"request_id", id}, {"session_id", manager.active_id()},
        {"model", manager.model_id()}, {"prompt_tokens", result.prompt_tokens},
        {"completion_tokens", result.completion_tokens}, {"ttft_ms", result.ttft_ms},
        {"error", error_text.empty() ? nlohmann::json(nullptr) : nlohmann::json(error_text)}});
    write(row.dump());
}
void OpsLog::error(const std::string & message, const std::string & request_id, const std::string & session_id) {
    write(nlohmann::json{{"ts", timestamp()}, {"request_id", request_id}, {"session_id", session_id}, {"model", model_},
        {"prompt_tokens", 0}, {"completion_tokens", 0}, {"prefill_ms", 0}, {"prefill_moe_ms", 0}, {"mmq_group", mmq_group_},
        {"ttft_ms", 0}, {"decode_ms", 0}, {"tok_s", 0}, {"mtp_accept", 0}, {"kv_hit_ratio", 0},
        {"expert_hit_ratio", 0}, {"error", message}}.dump());
}
}
