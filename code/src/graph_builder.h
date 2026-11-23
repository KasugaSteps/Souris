#pragma once
#include "ops.h"
#include "kv_cache.h"

namespace souris {
/** Model math only. Callers own placement, copies, eager execution and capture. */
struct GraphBuilder {
    const Model & model;
    KVCacheManager & state;
    ggml_backend_t backend;
    GraphBuilder(const Model & model, KVCacheManager & state, ggml_backend_t backend) : model(model), state(state), backend(backend) {}
    enum class Part { Before, Post };
    /** Build one layer fragment against the caller's hidden view and IO leaves. */
    void build_layer(Frame &, LayerIO &, int layer, Part, bool output_head = true);
    /** Sum already weighted route contributions in their original top10 order. */
    static ggml_tensor * sum_experts(Frame & frame,ggml_tensor * values);
    void head(Frame & frame, LayerIO & io, ggml_tensor * hidden);
private:
    void before(Frame &, LayerIO &, int layer);
    void after(Frame &, LayerIO &, int layer, bool output_head);
    ggml_tensor * norm(Frame & frame, ggml_tensor * x, ggml_tensor * weight);
    ggml_tensor * mix(Frame & frame, ggml_tensor * x, int layer, const char * phase, ggml_tensor ** inject);
    ggml_tensor * ple(Frame & frame, LayerIO & io, ggml_tensor * hidden);
    ggml_tensor * gdn(Frame & frame, ggml_tensor * input, int layer);
    ggml_tensor * attention(Frame & frame, LayerIO & io, ggml_tensor * input, int layer);
    ggml_tensor * qsa(Frame & frame, ggml_tensor * input, int layer, const AuxInputs & aux);
};
}
