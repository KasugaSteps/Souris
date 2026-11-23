#pragma once
#include "ggml.h"
#include "ggml-backend.h"
#include "souris/types.h"
#include <cuda_runtime.h>
#include <memory>

namespace souris {
struct Frame;
struct LayerIO;
class Model;

/** Packed grouping header followed by rows[n] and destinations[n]. */
constexpr int moe_task_prefix = 3 * fixed::experts + 1;
constexpr int moe_group_header = 4 * fixed::experts + 2;
constexpr int moe_tile_rows = 64;
constexpr int moe_quant_block = 32;

/** Per-layer projection geometry derived from the model's role types. */
struct MoeShape {
    int width, hidden;
    ggml_type type[3];
    size_t row_bytes[3];
    size_t record_bytes, expert_bytes;
    uint64_t quant_signature;
    bool gate_up_direct;
};

/** One selected expert interval and its fixed-address GPU leaves. */
struct MoeOp {
    MoeShape shape;
    /** Device addresses indexed by [layer][role][expert]. */
    const uint64_t * table;
    const int32_t * control;
    const int32_t * group;
    /** Real expert IDs in physical record order. */
    const int32_t * selected;
    const float * input;
    const float * weights;
    float * output;
    int routes;
    /** Half-open selected interval; tasks counts its nonempty 64-row tiles. */
    int begin, end, tasks;
};
class MoEKernel {
public:
    explicit MoEKernel(const Model &);
    ~MoEKernel();
    void group(LayerIO &, cudaStream_t) const;
    /** Compact real expert IDs on GPU in physical record order. */
    void select(LayerIO &, cudaStream_t) const;
    void prefill(LayerIO &, const void * table, int layer, int begin, int end, int tasks, cudaStream_t) const;
    void native_expert(Frame &, LayerIO &, ggml_tensor * table, ggml_backend_t, int layer, int count, bool miss) const;
private:
    friend class Scheduler;
    struct Workspace;
    Workspace * workspace_ = nullptr;
    void init_mmq();
    void begin_mmq();
    void end_mmq();
    size_t workspace_bytes_ = 0;
    void run_mmq(const MoeOp &, cudaStream_t) const;
    const Model & model_;
    MoeShape shapes_[fixed::layers]{};
    int sm_count_ = 0;
    MoeOp parameters(LayerIO &, const void *, int layer, int begin, int end, int tasks) const;
};
void cuda_moe_group(const int32_t *, int32_t *, int routes, cudaStream_t);
void cuda_moe_select(const int32_t *, int32_t *, cudaStream_t);
}
