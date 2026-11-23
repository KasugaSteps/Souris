#pragma once
#include "souris/engine.h"
#include "expert.h"
#include "kv_cache.h"
#include "transfer.h"
#include "stats.h"
#include "tokenizer.h"

namespace souris {
/** Immutable startup accounting, owned and locked by Engine. */
struct MemoryBudget {
    size_t prefill_workspace = 0, startup_free = 0, cache_records = 0;
    size_t capture_free_before = 0, capture_free_after = 0, capture_scratch_allocations = 0;
    int slots = 0, prefill_slots = 0, workspace_slots = 0, hidden_slots = 0;
};
/** Single allocation. Its reusable leaf metadata is copied into each eager Frame. */
struct HiddenLease {
    ggml_context * context = nullptr;
    ggml_tensor * tensor = nullptr;
    ggml_tensor view{};
    size_t tokens = 0, bytes = 0;
    SlotPool::Tail tail;
    SlotPool * pool = nullptr;
    HiddenLease(int width, int hc, Device &, size_t total_tokens);
    ~HiddenLease();
    void borrow(SlotPool &, int slots);
    void release();
    ggml_tensor * slice(size_t offset, int count);
};
struct Engine::Impl {
    Engine & runner;
    Config & config;
    Statistics stats;
    Device & device;
    Model & model;
    const ExpertRepository & repository;
    std::unique_ptr<StagingRing> staging;
    std::unique_ptr<KVCacheManager> state;
    std::unique_ptr<HiddenLease> hidden;
    std::unique_ptr<ExpertCache> cache;
    std::unique_ptr<SlotPool> pool;
    MemoryBudget budget;
    std::unique_ptr<TransferEngine> transfer;
    std::unique_ptr<Scheduler> scheduler;
    size_t preload_layers = 0;
    double preload_ms = 0;
    bool preload_started = false;
    void preload();
    size_t actual_max_prompt = 0;
    Impl(Engine &, Model &, Device &, Config &);
    ~Impl();
    void plan_budget(size_t max_prompt_tokens);
    void reset_request();
    bool run_prefill(const Tokens &, int start = 0, bool reset = true);
    bool run_decode(Token, size_t remaining = SIZE_MAX);
    Token next() const;
    void warmup();
};
}
