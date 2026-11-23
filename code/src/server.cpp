#include "server.h"
#include "souris/runtime.h"
#include "ops_log.h"
#include "httplib/httplib.h"
#include "nlohmann/json.hpp"
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <memory>
#include <filesystem>
#include <random>
#include <thread>
#include <unordered_set>

namespace souris {
namespace {
using Json = nlohmann::json;
struct Delivery { bool started = false, delivered = false; RequestGate::Clock::time_point staged_at; };
volatile std::sig_atomic_t stopping = 0;
void stop_signal(int) { stopping = 1; }
void error(httplib::Response & response, const std::string & message) {
    response.status = 400;
    response.set_content(Json{{"error", {{"message", message}, {"type", "invalid_request_error"},
        {"param", nullptr}, {"code", nullptr}}}}.dump(), "application/json");
}
void admission_error(httplib::Response & response, const std::string & message,
                     const char * code, const std::string & request_id) {
    if (code) {
        const bool busy = std::string(code) == "model_busy";
        response.status = busy ? 409 : 429;
        if (!busy) { response.set_header("Retry-After", "1"); }
        response.set_content(Json{{"error", {{"message", message},
            {"type", busy ? "model_busy" : "rate_limit_error"}, {"code", code},
            {"phase", "admission"}, {"retryable", !busy}, {"request_id", request_id},
            {"param", nullptr}}}}.dump(), "application/json");
    } else { error(response, message); response.status = 503; }
}
bool parse(SessionManager & sessions, const Json & body, CompletionOptions & options, std::vector<Image> & images, std::string & why) {
    if (!body.is_object()) { why = "expected a JSON object"; return false; }
    const std::unordered_set<std::string> allowed = {"model", "prompt", "max_tokens", "temperature", "top_p", "top_k", "min_p", "stop", "stream", "seed", "images", "constraint"};
    for (const auto & item : body.items()) {
        if (!allowed.count(item.key())) { why = "unsupported raw completion field: " + item.key(); return false; }
    }
    if (!body.contains("prompt") || !body["prompt"].is_string()) { why = "prompt must be a string"; return false; }
    if (body.contains("model") && (!body["model"].is_string() || body["model"] != sessions.model_id())) { why = "unknown model"; return false; }
    if (body.contains("max_tokens")) {
        if (!body["max_tokens"].is_number_integer() || body["max_tokens"] < 1 || body["max_tokens"] > sessions.context_limit()) { why = "invalid max_tokens"; return false; }
        options.max_tokens = body["max_tokens"].get<int>();
    }
    for (const char * key : {"temperature", "top_p"}) {
        if (!body.contains(key)) { continue; }
        if (!body[key].is_number()) { why = std::string("invalid ") + key; return false; }
        const auto value = body[key].get<double>();
        if (!std::isfinite(value) || (std::string(key) == "temperature" ? value < 0 : value <= 0 || value > 1)) { why = std::string("invalid ") + key; return false; }
        if (std::string(key) == "temperature") { options.temperature = float(value); }
        else { options.top_p = float(value); }
    }
    if (body.contains("top_k")) {
        if (!body["top_k"].is_number_integer() || body["top_k"] < 0) { why = "invalid top_k"; return false; }
        options.top_k = int(std::min(body["top_k"].get<double>(), double(INT32_MAX)));
    }
    if (body.contains("min_p")) {
        if (!body["min_p"].is_number()) { why = "invalid min_p"; return false; }
        const double value = body["min_p"];
        if (!std::isfinite(value) || value < 0 || value > 1) { why = "invalid min_p"; return false; }
        options.min_p = float(value);
    }
    if (body.contains("stream") && !body["stream"].is_boolean()) { why = "stream must be boolean"; return false; }
    if (body.contains("seed")) {
        if (!body["seed"].is_number_integer() || body["seed"] < 0) { why = "seed must be a nonnegative integer"; return false; }
        options.seed = body["seed"].get<uint64_t>();
    } else { options.seed = std::random_device{}(); }
    if (body.contains("stop") && !body["stop"].is_null()) {
        const auto stops = body["stop"].is_string() ? Json::array({body["stop"]}) : body["stop"];
        if (!stops.is_array()) { why = "stop must be a string or array"; return false; }
        for (const auto & stop : stops) {
            if (!stop.is_string() || stop.get_ref<const std::string &>().empty()) { why = "stop strings must be nonempty"; return false; }
            options.stop.push_back(stop.get<std::string>());
        }
    }
    if (body.contains("images")) {
        const auto & list = body["images"];
        if (!list.is_array()) { why = "invalid image data"; return false; }
        if (list.size() > image::max_count) { why = "too many images"; return false; }
        for (const auto & item : list) {
            if (!item.is_object() || item.size() != 4 || !item.contains("format") || item["format"] != "rgb8" ||
                !item.contains("width") || !item["width"].is_number_integer() || !item.contains("height") || !item["height"].is_number_integer() ||
                !item.contains("data") || !item["data"].is_string()) { why = "invalid image data"; return false; }
            if (item["width"] != image::size || item["height"] != image::size) { why = "invalid image size"; return false; }
            images.emplace_back();
            if (!decode_rgb8(item["data"].get_ref<const std::string &>(), images.back().rgb)) { why = "invalid image data"; return false; }
        }
    }
    if (body.contains("constraint")) {
        options.constraint = body["constraint"].dump();
        if (!sessions.validate_constraint(options.constraint, why)) { return false; }
    }
    if (!sessions.validate_images(body["prompt"].get<std::string>(), images, why)) { return false; }
    const size_t count = sessions.token_count(body["prompt"].get<std::string>());
    if (!count || count > sessions.prompt_limit()) { why = "prompt exceeds max_prompt_tokens"; return false; }
    if (count + size_t(options.max_tokens) > size_t(sessions.context_limit())) { why = "prompt_tokens + max_tokens exceeds context"; return false; }
    return true;
}
}
int serve(Runtime & runtime, OpsLog & log) {
    const auto & config = runtime.config();
    httplib::Server http;
    // At most four workers wait on admission. Finalization RPCs must remain runnable.
    http.new_task_queue = [] { return new httplib::ThreadPool(RequestGate::max_waiters + 4); };
    http.set_keep_alive_max_count(1);
    http.Get("/health", [](const httplib::Request &, httplib::Response & response) {
        response.set_content("{\"status\":\"ok\"}", "application/json");
    });
    const auto model = std::filesystem::path(config.model.srs1_path).stem().string();
    http.Get("/v1/models", [model](const httplib::Request &, httplib::Response & response) {
        response.set_content(Json{{"object", "list"}, {"data", Json::array({{
            {"id", model}, {"object", "model"}, {"created", 0}, {"owned_by", "souris"}}})}}.dump(), "application/json");
    });
    std::atomic<uint64_t> sequence{0};
    auto transaction = [&](const httplib::Request & input, httplib::Response & response, bool commit) {
        const auto body = Json::parse(input.body, nullptr, false);
        if (!body.is_object() || body.size() != 3 || !body.contains("session_id") || !body["session_id"].is_string() ||
            !body.contains("request_id") || !body["request_id"].is_string() ||
            !body.contains("commit_token") || !body["commit_token"].is_string()) {
            error(response, "expected session_id, request_id and commit_token"); return;
        }
        const auto session = body["session_id"].get<std::string>(), request = body["request_id"].get<std::string>();
        const auto token = body["commit_token"].get<std::string>();
        size_t position = 0;
        std::string why;
        const bool ok = runtime.finalize(token, RequestGate::Clock::now() + RequestGate::admission_wait,
            [&] { return stopping || input.is_connection_closed(); }, [&](SessionManager & sessions) {
                if (!sessions.staged_matches(session, request, token) ||
                    !(commit ? sessions.commit_staged(token) : sessions.abort_staged(token))) { return false; }
                position = sessions.position(); return true;
            }, why, !commit && input.get_header_value("X-Retry-Continue") == "1");
        if (!ok) {
            error(response, why.empty() ? "commit token is unknown, expired or already finalized in the opposite state" : why);
            response.status = why.empty() ? 409 : 503; return;
        }
        response.set_content(Json{{"status", commit ? "committed" : "aborted"},
            {"committed_tokens", position}}.dump(), "application/json");
    };
    http.Post("/v1/session/commit", [&](const httplib::Request & input, httplib::Response & response) { transaction(input, response, true); });
    http.Post("/v1/session/abort", [&](const httplib::Request & input, httplib::Response & response) { transaction(input, response, false); });
    http.Post("/v1/completions", [&](const httplib::Request & input, httplib::Response & response) {
        const auto request_id = input.get_header_value("X-Request-ID", ("cmpl-souris-" + std::to_string(++sequence)).c_str());
        const auto session_id = input.get_header_value("X-Session-ID");
        response.set_header("X-Request-ID", request_id);
        std::string admission_message;
        const char * admission_code = nullptr;
        const auto retry_token = input.get_header_value("X-Retry-Token");
        auto lease = retry_token.empty() ?
            runtime.requests().acquire(RequestGate::Clock::now() + RequestGate::admission_wait,
                [&] { return stopping || input.is_connection_closed(); }, admission_message, &admission_code) :
            runtime.resume(session_id, input.get_header_value("X-Retry-Request-ID"), retry_token, admission_message);
        if (!lease) {
            if (!retry_token.empty()) { error(response, admission_message); response.status = 409; }
            else { admission_error(response, admission_message, admission_code, request_id); }
            return;
        }
        auto & sessions = runtime.session(*lease);
        const auto mode = input.get_header_value("X-Commit-Mode", "immediate");
        if ((mode != "immediate" && mode != "deferred") || (mode == "deferred" && session_id.empty()) ||
            (!retry_token.empty() && mode != "deferred")) {
            error(response, "X-Commit-Mode must be immediate or deferred; deferred requires X-Session-ID"); return;
        }
        const auto body = Json::parse(input.body, nullptr, false);
        CompletionOptions options;
        auto images = std::make_shared<std::vector<Image>>();
        std::string why;
        if (body.is_discarded() || !parse(sessions, body, options, *images, why)) {
            const auto message = why.empty() ? "invalid JSON" : why;
            error(response, message); log.error(message, request_id, session_id); return;
        }
        options.retry_prefix = !retry_token.empty();
        const bool stream = body.value("stream", false);
        const std::string prompt = body["prompt"];
        const int64_t created = std::time(nullptr);
        auto result = std::make_shared<CompletionResult>();
        auto delivery = std::make_shared<Delivery>();
        auto disconnected = input.is_connection_closed;
        sessions.open(session_id);
        const std::string commit_token = mode == "deferred" ? sessions.stage_commit(request_id) : "";
        if (mode == "deferred") {
            if (commit_token.empty()) { error(response, "cannot reserve deferred transaction"); response.status = 409; return; }
            response.set_header("X-Commit-Token", commit_token);
        }
        response.set_header("Cache-Control", "no-cache");
        response.set_chunked_content_provider(stream ? "text/event-stream" : "application/json",
            [&, result, delivery, prompt, options, images, request_id, session_id, stream, model, created, disconnected, lease]
            (size_t, httplib::DataSink & sink) {
                delivery->started = true;
                auto event = [&](const Json & object) {
                    const auto text = "data: " + object.dump(-1, ' ', false, Json::error_handler_t::replace) + "\n\n";
                    return sink.write(text.data(), text.size());
                };
                auto completion = [&](const std::string & text, const Json & reason) {
                    return Json{{"id", request_id}, {"object", "text_completion"},
                        {"created", created}, {"model", model}, {"choices", Json::array({{
                            {"index", 0}, {"text", text}, {"logprobs", nullptr}, {"finish_reason", reason}}})}};
                };
                auto cancelled = [&] { return stopping || disconnected(); };
                *result = sessions.generate(prompt, options,
                    [&](const std::string & text) { return !stream || event(completion(text, nullptr)); }, cancelled, std::move(*images));
                delivery->staged_at = RequestGate::Clock::now();
                if (!result->completed) {
                    if (!result->error.empty() && result->error != "cancelled") {
                        const Json failure{{"error", {{"message", result->error}, {"type", "server_error"},
                            {"code", result->error_code.empty() ? Json(nullptr) : Json(result->error_code)}}}};
                        if (stream) { event(failure); } else { const auto text = failure.dump(); sink.write(text.data(), text.size()); }
                        sink.done(); delivery->delivered = true; return true;
                    }
                    return false;
                }
                auto final = completion(stream ? "" : result->text, result->finish_reason);
                final["usage"] = {{"prompt_tokens", result->prompt_tokens}, {"completion_tokens", result->completion_tokens},
                    {"total_tokens", result->prompt_tokens + result->completion_tokens}};
                if (stream) {
                    if (!event(final)) { return false; }
                    constexpr char done[] = "data: [DONE]\n\n";
                    if (!sink.write(done, sizeof(done) - 1)) { return false; }
                } else {
                    const auto text = final.dump(-1, ' ', false, Json::error_handler_t::replace);
                    if (!sink.write(text.data(), text.size())) { return false; }
                }
                sink.done(); delivery->delivered = true; return true;
            },
            [&, result, delivery, request_id, session_id, commit_token, lease](bool success) {
                if (!delivery->started) {
                    lease->finish([&] { sessions.abandon(); });
                    return;
                }
                const bool completed = success && result->completed;
                auto record = [&] {
                    log.request_end(sessions, *result, request_id, completed ? "" :
                        result->error.empty() ? "client disconnected" : result->error);
                };
                if (success && delivery->delivered && !commit_token.empty()) {
                    record();
                    lease->defer(commit_token, delivery->staged_at);
                } else {
                    lease->finish([&] {
                        record();
                        if (!commit_token.empty()) { sessions.abort_staged(commit_token); }
                        else if (completed && !session_id.empty()) { sessions.commit(); }
                        else { sessions.abandon(); }
                    });
                }
            });
    });
    if (!http.bind_to_port("0.0.0.0", config.port)) { GGML_ABORT("server bind error"); }
    stopping = 0;
    const auto old_int = std::signal(SIGINT, stop_signal), old_term = std::signal(SIGTERM, stop_signal);
    std::atomic<bool> done{false};
    std::thread shutdown([&] {
        while (!done) {
            if (stopping) { http.stop(); return; }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    });
    const bool listened = http.listen_after_bind();
    done = true; shutdown.join();
    std::signal(SIGINT, old_int); std::signal(SIGTERM, old_term);
    if (!listened && !stopping) { GGML_ABORT("server listen error"); }
    return 0;
}
}
