#include "souris/session_manager.h"
#include "engine_internal.h"
#include "scheduler.h"
#include "draft_runner.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <random>
#include <sstream>
#include <unordered_map>

namespace souris {
struct SessionManager::Impl {
    Engine & engine;
    std::string active;
    Tokens committed;
    Tokens pending;
    /** Original Prefill boundary, retained so KV replay preserves tile shapes. */
    size_t committed_prompt_length = 0;
    std::vector<ImageIdentity> committed_images, pending_images;
    /** Bounded RGB sources for rebuilding a rewritten image prefix on abort. */
    std::vector<Image> committed_payload, pending_payload;
    int pending_prompt_slot = -1, pending_decode_slot = -1;
    size_t pending_prompt_length = 0, pending_decode_length = 0;
    struct Checkpoint { size_t length = 0; uint64_t stamp = 0; std::vector<ImageIdentity> images; };
    /** Published, rollback and pending metadata all refer to the same eight GPU slots. */
    std::array<Checkpoint, 8> checkpoints, saved_checkpoints, staged_checkpoints;
    enum class Outcome { Generating, Staged, Committed, Aborted };
    struct Receipt { std::string session, request; Outcome outcome; };
    /** Finalized receipts retain process-lifetime RPC idempotency and token binding. */
    std::unordered_map<std::string, Receipt> receipts;
    std::string staged_token;
    int rollback_slot = -1;
    unsigned overwritten = 0;
    bool rewritten = false;
    uint64_t stamp = 0;
    size_t last_prefix = 0;
    bool request_started = false;
    /** One unpublished prompt boundary, sharing the existing eight checkpoint slots. */
    Tokens scratch_prompt;
    std::string scratch_text;
    std::vector<ImageIdentity> scratch_images;
    int scratch_slot = -1;
    void clear_scratch() { scratch_prompt.clear(); scratch_text.clear(); scratch_images.clear(); scratch_slot = -1; }
    size_t length(int slot) const {
        return slot == scratch_slot ? scratch_prompt.size() : checkpoints[slot].length;
    }
    explicit Impl(Engine & owner) : engine(owner) {}
    int match(const Tokens & prompt, const std::vector<ImageIdentity> & images, bool retry = false) const {
        if (retry) {
            const size_t n = scratch_prompt.size();
            return scratch_slot >= 0 && n > 1 && n <= prompt.size() && scratch_images == images &&
                std::equal(scratch_prompt.begin(), scratch_prompt.end() - 1, prompt.begin()) ? scratch_slot : -1;
        }
        int best = -1;
        for (int slot = 0; slot < 8; ++slot) {
            const size_t length = checkpoints[slot].length;
            if (checkpoints[slot].images == images && length && length <= prompt.size() && length <= committed.size() &&
                (best < 0 || length > checkpoints[best].length) &&
                std::equal(prompt.begin(), prompt.begin() + length, committed.begin())) { best = slot; }
        }
        return best;
    }
    void invalidate_tail(const Tokens & next) {
        size_t common = 0;
        const Tokens & physical = scratch_slot >= 0 ? scratch_prompt : committed;
        while (common < next.size() && common < physical.size() && next[common] == physical[common]) { ++common; }
        // K/V and index are append-only and have no checkpoint copy. A state
        // checkpoint stays valid only while its consumed prefix stays intact.
        for (auto & cp : checkpoints) {
            if (cp.length && cp.length - 1 > common) { cp = {}; }
        }
    }
    int allocate(int pinned, int other = -1) {
        int oldest = -1;
        for (int slot = 0; slot < 8; ++slot) {
            if (slot == pinned || slot == other || slot == scratch_slot || (!staged_token.empty() && saved_checkpoints[slot].length)) { continue; }
            if (!checkpoints[slot].length) { return slot; }
            if (oldest < 0 || checkpoints[slot].stamp < checkpoints[oldest].stamp) { oldest = slot; }
        }
        GGML_ASSERT(oldest >= 0);
        checkpoints[oldest] = {};
        return oldest;
    }
    void clear_pending() {
        pending.clear(); pending_images.clear(); pending_payload.clear();
        pending_prompt_slot = pending_decode_slot = -1;
        pending_prompt_length = pending_decode_length = 0;
    }
};
SessionManager::SessionManager(Engine & engine) : impl_(std::make_unique<Impl>(engine)) {}
SessionManager::~SessionManager() = default;
void SessionManager::open(const std::string & id) {
    // An absent ID has no cross-request prefix identity.
    if (id.empty() || id != impl_->active) {
        abandon(); impl_->committed.clear(); impl_->committed_images.clear(); impl_->committed_payload.clear(); impl_->committed_prompt_length = 0; impl_->last_prefix = 0;
        for (auto & cp : impl_->checkpoints) { cp = {}; }
        impl_->clear_scratch();
        impl_->active = id;
        // generate resets GPU sequence state once before submitting any kernels.
    }
}
const std::string & SessionManager::active_id() const { return impl_->active; }
const Tokens & SessionManager::tokens() const { return impl_->committed; }
size_t SessionManager::position() const { return impl_->committed.size(); }
size_t SessionManager::match_prefix(const Tokens & prompt, const std::vector<ImageIdentity> & images) const {
    const int slot = impl_->match(prompt, images);
    return slot < 0 ? 0 : impl_->checkpoints[slot].length;
}
void SessionManager::reset() {
    auto & x = *impl_;
    abandon(); x.engine.reset(); x.committed.clear(); x.committed_images.clear(); x.committed_payload.clear(); x.committed_prompt_length = 0; x.last_prefix = 0;
    for (auto & cp : x.checkpoints) { cp = {}; }
    x.clear_scratch();
}
void SessionManager::close() { reset(); impl_->active.clear(); }
size_t SessionManager::token_count(const std::string & text) const { return impl_->engine.tokenize(text).size(); }
size_t SessionManager::prompt_limit() const { return impl_->engine.impl_->actual_max_prompt; }
int SessionManager::context_limit() const { return impl_->engine.impl_->config.context; }
std::string SessionManager::model_id() const {
    return std::filesystem::path(impl_->engine.impl_->config.model.srs1_path).stem().string();
}
bool SessionManager::validate_constraint(const std::string & json, std::string & error) const {
    auto & e = *impl_->engine.impl_;
    const bool ok = e.scheduler->grammar.configure({json}, e.model.tokenizer());
    if (!ok) { error = e.scheduler->grammar.error(); }
    return ok;
}
bool SessionManager::validate_images(const std::string & text, std::vector<Image> & images, std::string & error) const {
    const auto & model = impl_->engine.impl_->model;
    if (images.size() > size_t(model.max_images())) { error = "too many images"; return false; }
    if (!images.empty() && !model.has_vision()) { error = "vision not configured"; return false; }
    if (images.empty()) { return true; }
    const auto tokens = model.tokenizer().encode(text);
    size_t image_index = 0;
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (tokens[i] < 248053 || tokens[i] > 248057) { continue; }
        if (tokens[i] != 248053 || image_index >= images.size() || i + image::tokens + 1 >= tokens.size()) {
            error = "invalid image token count"; return false;
        }
        for (int row = 0; row < image::tokens; ++row) {
            if (tokens[i + row + 1] != 248056) { error = "invalid image token count"; return false; }
        }
        if (tokens[i + image::tokens + 1] != 248054) { error = "invalid image token count"; return false; }
        auto & data = images[image_index++];
        if (data.width != image::size || data.height != image::size) { error = "invalid image size"; return false; }
        if (data.rgb.size() != image::rgb_bytes) { error = "invalid image data"; return false; }
        data.hash = image_hash(data.rgb.data(), data.rgb.size());
        data.token_start = int(i + 1);
        i += image::tokens + 1;
    }
    if (image_index != images.size()) { error = "invalid image token count"; return false; }
    return true;
}
namespace {
size_t complete_utf8(const std::string & text, size_t limit) {
    size_t offset = 0;
    while (offset < limit) {
        const unsigned char c = text[offset];
        const size_t width = c < 0x80 ? 1 : c < 0xe0 ? 2 : c < 0xf0 ? 3 : 4;
        if (offset + width > limit) { break; }
        offset += width;
    }
    return offset;
}
}
CompletionResult SessionManager::generate(const std::string & text, const CompletionOptions & options,
    const std::function<bool(const std::string &)> & emit, const std::function<bool()> & cancelled, std::vector<Image> images) {
    auto & x = *impl_; auto & engine = x.engine; auto & e = *engine.impl_; auto & s = *e.scheduler;
    x.request_started = false;
    CompletionResult result;
    Tokens prompt = engine.tokenize(text);
    result.prompt_tokens = prompt.size();
    if (prompt.empty() || prompt.size() > e.actual_max_prompt) { result.error = "prompt exceeds max_prompt_tokens"; return result; }
    if (options.max_tokens < 1 || prompt.size() + size_t(options.max_tokens) > size_t(e.config.context)) {
        result.error = "prompt_tokens + max_tokens exceeds context"; return result;
    }
    if (!std::isfinite(options.temperature) || options.temperature < 0 || !std::isfinite(options.top_p) || options.top_p <= 0 || options.top_p > 1) {
        result.error = "invalid sampling parameters"; return result;
    }
    if (!validate_images(text, images, result.error)) { return result; }
    if (!validate_constraint(options.constraint, result.error)) { return result; }
    s.tool_request = !options.constraint.empty();
    s.tool_mask_tokens = s.tool_text_tokens = 0;
    s.tool_body_mtp_drafts = s.tool_mtp_entries = 0;
    if (s.tool_request) {
        const auto choice = nlohmann::json::parse(options.constraint).value("choice", nlohmann::json("auto"));
        if (choice != "auto" && !s.grammar.begin_tool_call()) { result.error = "cannot begin tool constraint"; return result; }
    }
    std::vector<ImageIdentity> identities;
    for (const auto & image : images) { identities.push_back({image.hash, image.token_start, image::tokens}); }
    if (!x.staged_token.empty() && x.rollback_slot >= 0) {
        const size_t boundary = x.saved_checkpoints[x.rollback_slot].length - 1;
        x.rewritten = prompt.size() < boundary ||
            !std::equal(x.committed.begin(), x.committed.begin() + boundary, prompt.begin()) ||
            identities != x.committed_images;
    }
    x.invalidate_tail(prompt);
    const int matched = x.match(prompt, identities, options.retry_prefix);
    if (options.retry_prefix && (matched < 0 || text.compare(0, x.scratch_text.size(), x.scratch_text) != 0)) {
        result.error = "retry prompt does not reuse its prompt checkpoint";
        result.error_code = "invalid_request"; return result;
    }
    if (!options.retry_prefix) { x.clear_scratch(); }
    if (matched < 0) { for (auto & cp : x.checkpoints) { cp = {}; } }
    GGML_ASSERT(x.pending_prompt_slot < 0);
    const int prompt_slot = x.allocate(matched);
    const int decode_slot = x.allocate(matched, prompt_slot);
    if (!x.staged_token.empty()) { x.overwritten = (1u << prompt_slot) | (1u << decode_slot); }
    const int start = matched < 0 ? 0 : int(x.length(matched) - 1);
    e.transfer->drain();
    e.transfer->begin_request(); e.stats.begin_request();
    x.request_started = true;
    if (matched < 0) { e.state->reset(); }
    else { e.state->restore_checkpoint(matched, start); }
    const auto began = StatsClock::now();
    s.reset_draft(); s.history.clear(); s.images.clear();
    for (size_t i = 0; i < images.size(); ++i) {
        if (cancelled && cancelled()) { result.error = "cancelled"; return result; }
        auto embedding = e.model.encode_image(images[i], int(i));
        s.images.push_back({images[i].token_start, image::tokens, 0, embedding.rows});
    }
    s.prompt_checkpoint = prompt_slot;
    s.cancelled = cancelled;
    engine.sampling(options.temperature, options.top_p, options.seed, options.top_k, options.min_p);
    x.last_prefix = result.prefix_tokens = start;
    bool ok = !cancelled || !cancelled();
    if (ok) { ok = e.run_prefill(prompt, start, false); }
    if (ok && !x.staged_token.empty()) {
        x.scratch_prompt = prompt; x.scratch_text = text; x.scratch_images = identities; x.scratch_slot = prompt_slot;
    } else if (!ok) { x.clear_scratch(); }
    std::string pending;
    size_t hold = 0;
    for (const auto & stop : options.stop) { if (!stop.empty()) { hold = std::max(hold, stop.size() - 1); } }
    auto output = [&](const std::string & part) {
        if (part.empty()) { return true; }
        result.text += part;
        if (!result.ttft_ms) { result.ttft_ms = milliseconds(began); }
        return !emit || emit(part);
    };
    Tokens completed = prompt;
    for (int i = 0; ok && i < options.max_tokens; ++i) {
        if (!e.stats.error.empty()) { ok = false; break; }
        if (cancelled && cancelled()) { ok = false; break; }
        const Token token = engine.next();
        if (engine.eos(token)) {
            completed.push_back(token); ++result.completion_tokens;
            result.finish_reason = "stop"; break;
        }
        completed.push_back(token); ++result.completion_tokens;
        pending += engine.piece(token);
        size_t stop_at = std::string::npos;
        for (const auto & stop : options.stop) { stop_at = std::min(stop_at, pending.find(stop)); }
        const bool stopped = stop_at != std::string::npos;
        const size_t available = stopped ? stop_at : pending.size() > hold ? pending.size() - hold : 0;
        const size_t count = complete_utf8(pending, available);
        ok = output(pending.substr(0, count)); pending.erase(0, count);
        if (!ok) { break; }
        if (stopped) { result.finish_reason = "stop"; pending.clear(); break; }
        // The final emitted token stays pending. A following request recomputes
        // it from a length-1 checkpoint with its own sampling parameters.
        if (i + 1 < options.max_tokens) { ok = e.run_decode(token, options.max_tokens - i); }
    }
    if (ok && s.tool_request && !s.grammar.accepting()) {
        ok = false; e.stats.error = "tool call was truncated; use a smaller write/edit";
        result.error_code = "tool_truncated";
    }
    if (ok) { ok = output(pending.substr(0, complete_utf8(pending, pending.size()))); }
    e.transfer->drain(); e.stats.flush_gpu();
    if (cancelled && cancelled()) { ok = false; }
    result.completed = ok;
    if (ok) {
        x.pending = std::move(completed);
        x.pending_images = std::move(identities);
        x.pending_payload = std::move(images);
        x.pending_prompt_slot = prompt_slot; x.pending_prompt_length = prompt.size();
        // A stop string can end inside an already verified MTP block. Keep the
        // prompt checkpoint in that case; its suffix overwrites the unused tail.
        if (size_t(e.state->position) + 1 == x.pending.size()) {
            e.state->capture_checkpoint(decode_slot);
            SOURIS_CUDA(cudaStreamSynchronize(e.device.compute));
            x.pending_decode_slot = decode_slot; x.pending_decode_length = x.pending.size();
        }
    } else { result.error = e.stats.error.empty() ? "cancelled" : e.stats.error; }
    e.stats.generated_tokens = result.completion_tokens;
    s.prompt_checkpoint = -1; s.cancelled = {};
    if (!x.staged_token.empty()) {
        x.staged_checkpoints = x.checkpoints;
        x.checkpoints = x.saved_checkpoints;
        if (ok) {
            x.receipts.at(x.staged_token).outcome = Impl::Outcome::Staged;
        }
    }
    return result;
}
void SessionManager::commit() {
    auto & x = *impl_;
    GGML_ASSERT(x.pending_prompt_slot >= 0);
    x.invalidate_tail(x.pending);
    for (auto & cp : x.checkpoints) {
        if (cp.length == x.pending_prompt_length || (x.pending_decode_slot >= 0 && cp.length == x.pending_decode_length)) { cp = {}; }
    }
    x.committed = std::move(x.pending);
    x.committed_images = std::move(x.pending_images);
    x.committed_payload = std::move(x.pending_payload);
    x.committed_prompt_length = x.pending_prompt_length;
    x.checkpoints[x.pending_prompt_slot] = {x.pending_prompt_length, ++x.stamp, x.committed_images};
    if (x.pending_decode_slot >= 0) { x.checkpoints[x.pending_decode_slot] = {x.pending_decode_length, ++x.stamp, x.committed_images}; }
    x.clear_pending(); x.clear_scratch();
}
void SessionManager::abandon() {
    if (!impl_->staged_token.empty()) { abort_staged(impl_->staged_token); }
    else { impl_->clear_pending(); }
}
std::string SessionManager::stage_commit(const std::string & request_id) {
    auto & x = *impl_;
    if (x.active.empty() || request_id.empty() || has_staged()) { return {}; }
    std::random_device random;
    std::ostringstream token;
    for (int i = 0; i < 4; ++i) { token << std::hex << std::setw(8) << std::setfill('0') << uint32_t(random()); }
    x.staged_token = token.str();
    x.receipts.emplace(x.staged_token, Impl::Receipt{x.active, request_id, Impl::Outcome::Generating});
    // Keep all published snapshots intact while two pending captures are in
    // flight. Reclaim only the oldest cache entries, never allocate extra GPU KV.
    while (std::count_if(x.checkpoints.begin(), x.checkpoints.end(), [&](const Impl::Checkpoint & cp) { return !cp.length && (x.scratch_slot < 0 || &cp != &x.checkpoints[x.scratch_slot]); }) < 2) {
        int oldest = -1;
        for (int i = 0; i < int(x.checkpoints.size()); ++i) {
            if (i != x.scratch_slot && x.checkpoints[i].length && (oldest < 0 || x.checkpoints[i].stamp < x.checkpoints[oldest].stamp)) { oldest = i; }
        }
        x.checkpoints[oldest] = {};
    }
    x.saved_checkpoints = x.checkpoints;
    x.rollback_slot = -1; x.overwritten = 0; x.rewritten = false;
    for (int i = 0; i < int(x.checkpoints.size()); ++i) {
        if (x.checkpoints[i].length && x.checkpoints[i].length <= x.committed.size() &&
            x.checkpoints[i].images == x.committed_images &&
            (x.rollback_slot < 0 || x.checkpoints[i].length > x.checkpoints[x.rollback_slot].length)) {
            x.rollback_slot = i;
        }
    }
    return x.staged_token;
}
bool SessionManager::has_staged() const { return !impl_->staged_token.empty(); }
bool SessionManager::staged_matches(const std::string & session, const std::string & request, const std::string & token) const {
    const auto found = impl_->receipts.find(token);
    return found != impl_->receipts.end() && found->second.session == session && found->second.request == request;
}
bool SessionManager::commit_staged(const std::string & token) {
    auto & x = *impl_;
    const auto found = x.receipts.find(token);
    if (found == x.receipts.end()) { return false; }
    if (found->second.outcome == Impl::Outcome::Committed) { return true; }
    if (found->second.outcome != Impl::Outcome::Staged || x.staged_token != token) { return false; }
    x.checkpoints = x.staged_checkpoints;
    commit();
    found->second.outcome = Impl::Outcome::Committed;
    x.staged_token.clear(); x.rollback_slot = -1;
    return true;
}
bool SessionManager::abort_staged(const std::string & token) {
    auto & x = *impl_;
    const auto found = x.receipts.find(token);
    if (found == x.receipts.end() || found->second.outcome == Impl::Outcome::Committed) { return false; }
    if (found->second.outcome == Impl::Outcome::Aborted) { return true; }
    if (x.staged_token != token) { return false; }
    auto & e = *x.engine.impl_; auto & s = *e.scheduler;
    e.transfer->drain();
    x.checkpoints = x.saved_checkpoints;
    // Pending captures used only the two reserved empty slots. Published
    // recurrent snapshots and append-only prefix rows are retained on abort.
    for (int i = 0; i < int(x.checkpoints.size()); ++i) {
        if (x.overwritten & (1u << i)) { x.checkpoints[i] = {}; }
    }
    if (x.scratch_slot >= 0) {
        // The scratch contains prompt inputs only. Keep its append-only KV rows,
        // while the published token history and its checkpoints remain unchanged.
        e.state->restore_checkpoint(x.scratch_slot, int(x.scratch_prompt.size() - 1));
        SOURIS_CUDA(cudaStreamSynchronize(e.device.compute));
    } else if (x.rollback_slot >= 0) {
        const int position = int(x.checkpoints[x.rollback_slot].length - 1);
        if (x.rewritten) {
            // A divergent request overwrote canonical KV rows. Rebuild only on
            // abort; normal extensions restore the recurrent checkpoint directly.
            auto draft = std::move(s.draft);
            s.grammar.reset(); s.tool_request = false; s.prompt_checkpoint = -1;
            e.stats.error.clear(); e.stats.ok = true;
            s.history.clear(); s.images.clear(); s.cancelled = {};
            e.state->reset();
            for (size_t i = 0; i < x.committed_payload.size(); ++i) {
                auto embedding = e.model.encode_image(x.committed_payload[i], int(i));
                s.images.push_back({x.committed_payload[i].token_start, image::tokens, 0, embedding.rows});
            }
            const size_t count = std::min(size_t(position), x.committed_prompt_length);
            bool ok = true;
            if (count) {
                // Preserve the original final-token tile boundary as well as
                // the token history. The capture uses a reserved, unpublished slot.
                for (int i = 0; i < int(x.saved_checkpoints.size()); ++i) {
                    if (!x.saved_checkpoints[i].length) { s.prompt_checkpoint = i; break; }
                }
                ok = e.run_prefill(Tokens(x.committed.begin(), x.committed.begin() + count), 0, false);
                s.prompt_checkpoint = -1;
            }
            for (size_t i = count; ok && i < size_t(position); ++i) { ok = e.run_decode(x.committed[i], 1); }
            s.draft = std::move(draft);
            if (!ok) { GGML_ABORT("cannot restore committed KV after deferred abort"); }
        }
        e.state->restore_checkpoint(x.rollback_slot, position);
        SOURIS_CUDA(cudaStreamSynchronize(e.device.compute));
    } else if (x.overwritten) { e.state->reset(); }
    s.reset_draft(); s.history.clear(); s.images.clear();
    x.clear_pending();
    found->second.outcome = Impl::Outcome::Aborted;
    x.staged_token.clear(); x.rollback_slot = -1;
    return true;
}
std::string SessionManager::metrics() const {
    auto & e = *impl_->engine.impl_; auto & s = e.stats;
    s.flush_gpu();
    if (!impl_->request_started) {
        return nlohmann::json{{"prefill_ms", 0}, {"prefill_moe_ms", 0}, {"mmq_group", e.config.mmq_group},
            {"decode_ms", 0}, {"tok_s", 0}, {"mtp_accept", 0}, {"kv_hit_ratio", 0}, {"expert_hit_ratio", 0}}.dump();
    }
    return nlohmann::json{
        {"prefill_ms", s.prefill_ms}, {"prefill_moe_ms", s.prefill_moe_ms}, {"mmq_group", e.config.mmq_group},
        {"decode_ms", s.decode_ms}, {"tok_s", s.decode_ms > 0 ? 1000.0 * s.generated_tokens / s.decode_ms : 0},
        {"mtp_accept", s.mtp_drafts ? double(s.mtp_accepted) / s.mtp_drafts : 0},
        {"mtp_drafts", s.mtp_drafts},
        {"prefill_sample_ms", s.prefill_sample_ms},
        {"tool_body_mtp_drafts", e.scheduler->tool_body_mtp_drafts},
        {"tool_mtp_entries", e.scheduler->tool_mtp_entries},
        {"tool_mask_tokens", e.scheduler->tool_mask_tokens}, {"tool_text_tokens", e.scheduler->tool_text_tokens},
        {"tool_calls", e.scheduler->tool_request ? e.scheduler->grammar.call_count() : 0},
        {"kv_hit_ratio", s.prompt_tokens ? double(impl_->last_prefix) / s.prompt_tokens : 0},
        {"expert_hit_ratio", s.decode_routes ? double(s.decode_hits) / s.decode_routes : 0}
    }.dump();
}
}
