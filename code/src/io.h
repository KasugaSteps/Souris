#pragma once
#include "device.h"
namespace souris {
struct NativeGdnReplay;
/** One intersection of an image span with this Prefill token tile. */
struct ImageSlice { int start, count, source; ggml_tensor * rows; };
/** Replay-dependent metadata, copied outside captured graphs. */
struct Control { int32_t position,tokens,layer,miss_index; };
/** Persistent input/output leaves; Decode shares one T=1 allocation across captures. */
struct LayerIO {
    ggml_context * ctx;
    ggml_backend_buffer_t buffer=nullptr;
    ggml_tensor * control,*tokens,*hidden,*residual,*inject,*expert_input;
    /** Decode controls are uploaded once, with fixed per-layer capture addresses. */
    ggml_tensor * control_block = nullptr, *control_views[fixed::layers]{};
    /** Shared scalar inputs for dtype/count hit and miss graphs. */
    ggml_tensor * layer_id = nullptr, *miss_index = nullptr;
    ggml_tensor * route_ids,*route_weights,*routed,*logits,*ple;
    /** Q8_1 source shared by router, shared expert and every routed expert launch. */
    ggml_tensor * native_expert_q8 = nullptr;
    ggml_tensor * expert_output, *moe_group, *moe_selected;
    /** State-free input prologue for target T=1; absent in verify/Prefill. */
    ggml_tensor * prepared_hidden = nullptr, * prepared_mixed = nullptr, * prepared_inject = nullptr;
    ggml_tensor * prepared_q8 = nullptr;
    ggml_tensor * decode_plan, *decode_miss, *sample_params, *sample_token;
    /** Optional verify inputs for accepted-prefix state replay; absent in normal Decode. */
    NativeGdnReplay * gdn_replay = nullptr;
    ggml_tensor * ple_input = nullptr;
    /** Independent view metadata over Engine's single hidden allocation. */
    ggml_tensor hidden_leaf{};
    std::vector<ImageSlice> images;
    int rows;
    size_t bytes=0;
    LayerIO(const Model &,int count,Device &,ggml_tensor * hidden,const LayerIO * storage=nullptr);
    ~LayerIO();
};
}
