#pragma once
#include "types.h"
#include <memory>
#include <string>

struct ggml_tensor;

namespace souris {
class Device;
class Model;
class Scheduler;
class VerifyEngine;
class SessionManager;
/** Request execution over externally owned immutable model and device resources. */
class Engine {
public:
    Engine(Model &, Device &, Config &);
    ~Engine();
    struct Budget { int d_total, d_prefill, d_decode, workspace_slots, hidden_slots; };
    /** Plan the temporary hidden tail and lock all startup byte budgets once. */
    Budget plan_budget(size_t max_prompt_tokens);
    /** Validate 128-token Prefill and one Decode, then reset request state. */
    void warmup();
    void reset();
    /** Start direct complete-layer cache loading after warmup. */
    void preload();
    /** Locked startup budgets, printed once before serving requests. */
    std::string startup() const;
private:
    friend class Scheduler;
    friend class VerifyEngine;
    friend class SessionManager;
    friend class Runtime;
    /** Drain the retiring request, including its unconsumed output, before admission is released. */
    void finish_request();
    /** Borrow metadata over the single resident hidden allocation. */
    ggml_tensor * hidden_view(size_t offset, int count);
    /** Per-request GPU sampling; temperature=0 selects argmax, top_k=0/min_p=0 disable truncation. */
    void sampling(float temperature, float top_p, uint64_t seed, int top_k = 0, float min_p = 0);
    Tokens tokenize(const std::string & text) const;
    std::string piece(Token token) const;
    bool eos(Token token) const;
    Token next() const;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
