#pragma once
#include "souris/engine.h"
#include "expert.h"
#include "transfer.h"
#include "graph_builder.h"
#include "native_layer.h"
#include "stats.h"
#include "draft_runner.h"
#include "grammar_verifier.h"
#include <memory>
#include <random>
#include <functional>

namespace souris {
struct MemoryBudget;
class VerifyEngine;
/** Eager layer-major Prefill and retained per-layer Decode captures. */
class Scheduler {
public:
    explicit Scheduler(Engine::Impl &);
    ~Scheduler();
    bool prefill(const Tokens &, int start = 0);
    bool decode(Token);
    /** Capture one real-hit count for a shared quantization signature. */
    void capture_hit(int dtype, int count);
private:
    friend class Engine;
    friend class SessionManager;
    friend class VerifyEngine;
    friend struct Engine::Impl;
    struct Fragment {
        Frame frame;
        explicit Fragment(size_t nodes = 1024) : frame(nodes) {}
        GraphKey key;
        cudaGraph_t source = nullptr;
        cudaGraphExec_t executable = nullptr;
        ~Fragment();
        void capture(Device &);
        void launch(Device &);
    };
    Engine::Impl & owner;
    const Config & config;
    Device & device;
    const Model & model;
    KVCacheManager & state;
    MemoryBudget & budget;
    TransferEngine & transfer;
    StagingRing & staging;
    SlotPool & pool;
    Statistics & stats;
    GraphBuilder graph;
    NativeLayerPlan native;
    MoEKernel moe;
    LayerIO decode_io, prefill_io;
    Arena decode_arena, sample_arena;
    Fragment greedy_sample, stochastic_sample, input_prologue;
    /** Persistent pinned scalar mailbox, independent of route/control ring leases. */
    struct SampleMailbox {
        struct Packet { float params[5]; Token token; } *host = nullptr;
        cudaStream_t stream = nullptr;
        cudaEvent_t sampled = nullptr, copied = nullptr, uploaded = nullptr;
        bool params_pending = false, result_pending = false, first = false, deferred = false;
        SampleMailbox();
        ~SampleMailbox();
    } sample_mailbox;
    bool device_input_ready = false;
    Token consume_sample(bool account = true);
    Token sampled_token = -1;
    float temperature = 0, top_p = 1, min_p = 0;
    int top_k = 0;
    std::mt19937_64 random{std::random_device{}()};
    Tokens history;
    /** Request state never enters KV or session checkpoints. */
    GrammarVerifier grammar;
    bool tool_request = false;
    size_t tool_mask_tokens = 0, tool_text_tokens = 0;
    size_t tool_body_mtp_drafts = 0, tool_mtp_entries = 0;
    bool draft_body = false;
    std::vector<ImageSlice> images;
    std::unique_ptr<DraftRunner> draft;
    std::unique_ptr<VerifyEngine> verify;
    Tokens emitted_queue;
    size_t emitted_cursor = 0;
    Token pending_token = -1;
    size_t output_remaining = SIZE_MAX;
    void build_draft();
    void capture_draft();
    void reset_draft();
    void append_draft_prompt(const Tokens &, int start = 0);
    bool decode_draft(Token);
    /** Independent Decode route mailbox; no expert queue or staging ownership. */
    struct RouteSlot {
        int32_t * host = nullptr;
        cudaEvent_t done = nullptr;
        cudaStream_t stream = nullptr;
        RouteSlot();
        ~RouteSlot();
        const int32_t * download(const void * source, cudaEvent_t ready, Statistics &);
    } route;
    cudaEvent_t computed;
    bool prefilling = false;
    int tile;
    int prompt_checkpoint = -1;
    std::function<bool()> cancelled;
    std::unique_ptr<Fragment> pre_[fixed::layers], post_[fixed::layers];
    std::unique_ptr<Fragment> expert_[fixed::layers];
    std::unique_ptr<Fragment> hit_[fixed::layers][fixed::topk];
    size_t workspace_bytes();
    void build_decode();
    void build_sampling();
    void capture_decode();
    StagingRing::Ring & control_ring();
    void inputs(LayerIO &, int offset, int count, int layer);
    void control(LayerIO &, int position, int layer);
    /** Upload all fixed layer controls through decode_ring in one transfer. */
    void upload_controls(int position);
    void upload(void * target, const void * source, size_t bytes);
    void download(void * target, const void * source, size_t bytes, double * copy_time = nullptr);
    void read_logits(LayerIO &);
    void observe_tool(Token);
    void eager(Frame &, Arena &);
    void launch(Fragment &);
    void launch(Fragment &, double &);
    void expert(int layer, TransferEngine::Lease);
};
}
