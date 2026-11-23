#include "config.h"
#include "toml.hpp"
#include "ggml.h"
#include <climits>
#include <unordered_set>

namespace souris {
Config read_config(const std::string & path) {
    toml::table table;
    try { table = toml::parse_file(path); }
    catch (const toml::parse_error & error) { GGML_ABORT("invalid config: %s", std::string(error.description()).c_str()); }
    const std::unordered_set<std::string> allowed = {"model.path", "context", "max_prompt_tokens",
        "gpu_expert_mib", "host_cache_mib", "staging_mib", "read_threads", "mmq_group", "mtp.path", "mtp.max_drafts",
        "server.port", "log.file", "log.level", "vision.mmproj", "vision.max_images", "vision.image_size", "vision.cache_entries"};
    auto visit = [&](const auto & self, const toml::table & entries, const std::string & prefix) -> void {
        for (const auto & item : entries) {
            const auto key = prefix + std::string(item.first.str());
            if (item.second.is_table()) {
                if (key != "model" && key != "mtp" && key != "server" && key != "log" && key != "vision") { GGML_ABORT("unknown config table: %s", key.c_str()); }
                self(self, *item.second.as_table(), key + ".");
            }
            else if (!allowed.count(key)) { GGML_ABORT("unknown config key: %s", key.c_str()); }
        }
    };
    visit(visit, table, "");
    auto number = [&](const char * key, int64_t fallback) {
        const auto node = table.at_path(key);
        if (!node) { return fallback; }
        if (!node.is_integer()) { GGML_ABORT("config key must be an integer: %s", key); }
        const auto value = node.value<int64_t>();
        if (!value) { GGML_ABORT("config key must be an integer: %s", key); }
        return *value;
    };
    auto text = [&](const char * key, const std::string & fallback) {
        const auto node = table.at_path(key);
        if (!node) { return fallback; }
        const auto value = node.value<std::string>();
        if (!value) { GGML_ABORT("config key must be a string: %s", key); }
        return *value;
    };
    auto bounded = [&](const char * key, int fallback, int low, int high) {
        const auto value = number(key, fallback);
        if (value < low || value > high) { GGML_ABORT("invalid config key: %s", key); }
        return int(value);
    };
    auto bytes = [&](const char * key, size_t fallback, bool zero) {
        const auto value = number(key, -1);
        if (value == -1 && !table.at_path(key)) { return fallback; }
        if (value < (zero ? 0 : 1) || uint64_t(value) > SIZE_MAX / fixed::mib) { GGML_ABORT("invalid byte budget: %s", key); }
        return size_t(value) * fixed::mib;
    };
    Config config;
    config.model.srs1_path = text("model.path", "");
    if (config.model.srs1_path.empty()) { GGML_ABORT("model.path is required"); }
    config.context = bounded("context", 32768, fixed::warmup_tokens + 1, INT_MAX - 4);
    config.max_prompt_tokens = bounded("max_prompt_tokens", config.context, 1, config.context);
    config.gpu_expert_bytes = bytes("gpu_expert_mib", SIZE_MAX, false);
    config.expert_cache_bytes = bytes("host_cache_mib", SIZE_MAX, true);
    config.staging_bytes = bytes("staging_mib", fixed::staging_bytes, false);
    config.read_threads = bounded("read_threads", fixed::default_read_threads, 1, INT_MAX);
    config.mmq_group = bounded("mmq_group", 16, 1, 64);
    config.mtp = text("mtp.path", "");
    config.mtp_max_drafts = bounded("mtp.max_drafts", fixed::mtp_max_drafts, 1, 4);
    if (config.mtp.empty() && table.at_path("mtp.max_drafts")) { GGML_ABORT("mtp.max_drafts requires mtp.path"); }
    config.model.vision_mmproj = text("vision.mmproj", "");
    config.model.max_images = bounded("vision.max_images", 4, 1, 4);
    bounded("vision.image_size", 768, 768, 768);
    config.model.image_cache_entries = bounded("vision.cache_entries", 8, 0, INT_MAX);
    config.port = bounded("server.port", 8000, 1, 65535);
    config.log_file = text("log.file", ""); config.log_level = text("log.level", "info");
    if (config.log_level != "info" && config.log_level != "error") { GGML_ABORT("log.level must be info or error"); }
    return config;
}
}
