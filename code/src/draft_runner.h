#pragma once
#include "souris/types.h"
#include <memory>

struct ggml_tensor;

namespace souris {
class Model;
struct Device;
struct Statistics;

/** Resident single-layer drafter with private KV and fixed captured graphs. */
class DraftRunner {
public:
    static constexpr int max_drafts = fixed::mtp_max_drafts;
    struct Request {
        Token first_token;
        int first_pos, drafts;
        /** Proposed tokens in host memory, excluding first_token. */
        const Token * tokens;
        /** Complete row-major proposal probabilities in GPU memory. */
        const float * probabilities;
    };
    struct Emitted {
        /** Actually consumed inputs, including first_token. */
        Token tokens[max_drafts + 1]{};
        int count = 0;
        /** Correction or bonus for the next verify round, outside this commit. */
        Token next_token = -1;
        /** Target residuals for consumed inputs, valid through this commit. */
        ggml_tensor * residuals = nullptr;
    };
    using Verify = bool (*)(void *, const Request &, Emitted &);

    DraftRunner(const Model &, Device &, Statistics &, const float & temperature,
              const float & top_p, const int & top_k, const float & min_p, uint64_t (*random_bits)(void *), void * random_owner);
    ~DraftRunner();
    /** Allocate immutable assets, private KV and captured IO before target budgeting. */
    void init(const std::string & directory, int drafts, int context);
    /** Append 1..256 cells whose next tokens are known. */
    void append_prompt(ggml_tensor * residual, const Token * next_tokens, int rows, int first_pos);
    /** Draft, exactly verify and rewrite committed cells from target residuals. */
    bool step(Token first_token, ggml_tensor * last_residual, int first_pos,
              Verify verify, void * owner, Emitted & emitted);
    void reset();
private:
    friend class Scheduler;
    /** Startup-only capture after the target budget is locked. */
    void capture();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
