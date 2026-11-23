#pragma once
#include "device.h"
#include <array>
#include <chrono>
#include <string>
#include <vector>

namespace souris {
using StatsClock = std::chrono::steady_clock;
double milliseconds(StatsClock::time_point start);

/** Internal execution accounting; OpsLog selects the public request metrics. */
struct Statistics {
    bool ok = true;
    std::string error;
    double load_ms = 0, warmup_ms = 0, warmup_prefill_ms = 0, warmup_decode_ms = 0;
    double warmup_free_vram_mib = 0;
    double warmup_peak_vram_mib = 0, warmup_prefill_workspace_mib = 0;
    double prefill_ms = 0, decode_ms = 0, decode_gpu_ms = 0;
    /** First-token sampling, including grammar mask cache misses, after Prefill. */
    double prefill_sample_ms = 0;
    double decode_pre_ms = 0, decode_hit_ms = 0, decode_miss_ms = 0, decode_post_ms = 0;
    double decode_route_ms = 0, decode_route_d2h_ms = 0;
    /** Host submission duration, including the decode_ring control-slot wait. */
    double decode_control_h2d_ms = 0, decode_release_batch_ms = 0;
    /** Bytes per token's control block; request totals follow from generated_tokens. */
    size_t decode_control_bytes = 0;
    uint64_t decode_release_batches = 0, decode_release_experts = 0;
    double mtp_prompt_ms = 0, mtp_draft_ms = 0, mtp_verify_ms = 0, mtp_commit_ms = 0;
    double mtp_prompt_wall_ms = 0;
    uint64_t mtp_drafts = 0, mtp_accepted = 0, mtp_steps = 0, mtp_verify_tokens = 0, mtp_committed = 0;
    uint64_t mtp_verify_read_bytes = 0, mtp_verify_h2d_bytes = 0, mtp_verify_extents = 0;
    uint64_t mtp_verify_ram_hits = 0, mtp_verify_ssd_miss = 0;
    size_t mtp_prompt_tokens = 0;
    /** Unique provisional records remain in their final slot after acceptance. */
    size_t mtp_provisional_peak = 0;
    uint64_t mtp_provisional_experts = 0, mtp_committed_experts = 0, mtp_rolled_back_experts = 0;
    uint64_t mtp_accepted_hits = 0;
    double mtp_verify_read_ms = 0, mtp_verify_copy_submit_ms = 0, mtp_verify_wait_ms = 0;
    std::array<uint64_t, fixed::mtp_max_drafts> mtp_k{};
    double prefill_before_gpu_ms = 0, prefill_post_gpu_ms = 0;
    size_t prompt_tokens = 0, generated_tokens = 0;
    size_t prefill_workspace_peak = 0;
    double prefill_before_ms = 0, prefill_post_ms = 0;
    double prefill_demand_ms = 0, prefill_selected_upload_ms = 0;
    size_t vram_peak = 0, ram_peak = 0, swap_peak = 0;
    uint64_t prefill_hits = 0, prefill_routes = 0;
    uint64_t decode_hits = 0, decode_routes = 0;
    uint64_t prefill_eager_tiles = 0, prefill_moe_batches = 0;
    double prefill_group_ms = 0, prefill_moe_ms = 0, prefill_counts_d2h_ms = 0;
    uint64_t prefill_counts_d2h_bytes = 0;
    int gate_grid_min_full = 0, gate_grid_max = 0;
    uint64_t gate_full_batches = 0, gate_residual_batches = 0;
    uint64_t decode_plan_d2h_bytes = 0;
    uint64_t decode_activation_d2h_bytes = 0;
    uint64_t decode_activation_h2d_bytes = 0;
    double prefill_control_h2d_ms = 0, prefill_control_d2h_ms = 0;
    double decode_activation_d2h_ms = 0;
    double decode_activation_h2d_ms = 0;
    uint64_t decode_hit_graph_launches = 0, decode_miss_graph_launches = 0;
    uint64_t decode_hit_graph_experts = 0;
    std::array<uint64_t, fixed::topk> decode_hit_counts{};
    std::array<double, fixed::layers> prefill_layer_ms{};
    std::array<uint64_t, fixed::layers> prefill_union{}, prefill_admit{};
    void begin_request();
    void resources();
    void flush_gpu();
    struct GpuSample { cudaEvent_t begin, end; double * target; double * second; };
    std::vector<GpuSample> gpu_samples;
};

/** Only Prefill MoE duration is collected for the operational request record. */
class GpuTimer {
public:
    GpuTimer(Statistics &, cudaStream_t, double & target, double * second = nullptr);
    ~GpuTimer();
private:
    Statistics & stats_;
    cudaStream_t stream_;
    double & target_;
    double * second_;
    cudaEvent_t begin_ = nullptr, end_ = nullptr;
};
}
