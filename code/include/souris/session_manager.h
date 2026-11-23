#pragma once
#include "types.h"
#include "image.h"
#include <functional>
#include <memory>

namespace souris {
class Engine;
/** Sampling and output limits for one serial raw completion. */
struct CompletionOptions {
    int max_tokens = 128;
    float temperature = 0, top_p = 1, min_p = 0;
    int top_k = 0;
    uint64_t seed = 0;
    /** Require the authenticated prompt scratch; never Prefill a retry from zero. */
    bool retry_prefix = false;
    std::vector<std::string> stop;
    std::string constraint;
};
struct CompletionResult {
    std::string text, finish_reason = "length", error, error_code;
    size_t prompt_tokens = 0, completion_tokens = 0, prefix_tokens = 0;
    double ttft_ms = 0;
    bool completed = false;
};
/** Owns the last committed session and its bounded GPU recurrent checkpoints. */
class SessionManager {
public:
    explicit SessionManager(Engine &);
    ~SessionManager();
    void open(const std::string & session_id);
    const std::string & active_id() const;
    const Tokens & tokens() const;
    size_t position() const;
    size_t match_prefix(const Tokens &, const std::vector<ImageIdentity> & images = {}) const;
    /** Validate raw image/token pairing before starting a streaming response. */
    bool validate_images(const std::string &, std::vector<Image> &, std::string & error) const;
    /** Compile structured tool constraints before HTTP response headers. */
    bool validate_constraint(const std::string &, std::string & error) const;
    /** Return false from emit, or true from cancelled, to abandon without committing. */
    CompletionResult generate(const std::string & prompt, const CompletionOptions &,
        const std::function<bool(const std::string &)> & emit,
        const std::function<bool()> & cancelled = {}, std::vector<Image> images = {});
    /** Publish session history only after the caller completes output delivery. */
    void commit();
    /** Reserve a deferred transaction before streaming headers; generation stages its result. */
    std::string stage_commit(const std::string & request_id);
    /** Commit a ready token once; repeated commits succeed without changing state. */
    bool commit_staged(const std::string & commit_token);
    /** Discard pending assistant state but retain its unpublished prompt checkpoint. */
    bool abort_staged(const std::string & commit_token);
    /** Bind RPC credentials to this transaction, including finalized receipts. */
    bool staged_matches(const std::string & session_id, const std::string & request_id,
                        const std::string & commit_token) const;
    /** Whether the session has an unfinished deferred transaction. */
    bool has_staged() const;
    void abandon();
    void reset();
    void close();
    size_t token_count(const std::string &) const;
    size_t prompt_limit() const;
    int context_limit() const;
    std::string model_id() const;
    /** Restricted operational metrics, with no token or expert contents. */
    std::string metrics() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
