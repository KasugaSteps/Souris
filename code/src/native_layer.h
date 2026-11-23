#pragma once
#include "kv_cache.h"
#include "native_ops.h"
#include "ops.h"
#include <deque>

namespace souris {
/** Inputs actually used by verify, retained for exact accepted-prefix state replay. */
struct NativeGdnReplay {
    ggml_tensor *raw = nullptr, *conv = nullptr, *q = nullptr, *k = nullptr, *decay = nullptr, *beta = nullptr;
};
/** Shared small-row layer recipe. Captures own their Frames; metadata lives here. */
class NativeLayerPlan {
  public:
    NativeLayerPlan(const Model &model, KVCacheManager &state, ggml_backend_t backend)
        : model(model), state(state), backend(backend) {}
    enum class Part { Before, Post };
    void build_layer(Frame &, LayerIO &, int layer, Part, bool output_head = true);
    /** Compute the next token's state-free embedding/input mix before host detokenization. */
    void prepare_decode(Frame &, LayerIO &);
    void replay_gdn(Frame &, const NativeGdnReplay &, ggml_tensor *control, int layer, int accepted);
    void replay_ple(Frame &, ggml_tensor *normalized, ggml_tensor *control);
    static void sum_experts(Frame &, ggml_tensor *values, ggml_tensor *target);

  private:
    const Model &model;
    KVCacheManager &state;
    ggml_backend_t backend;
    struct Activation {
        ggml_tensor *value, *q8;
        NativeActivation *op;
        int q8_arg;
    };
    std::vector<Activation> activations_;
    std::deque<NativeActivation> metadata_;
    std::deque<float> router_scales_;
    std::deque<NativeHistory> histories_;
    ggml_tensor *quantized(Frame &, ggml_tensor *, ggml_tensor *target = nullptr);
    ggml_tensor *matmul(Frame &, ggml_tensor *, ggml_tensor *);
    ggml_tensor *producer(Frame &, ggml_tensor *, ggml_tensor *, int mode);
    ggml_tensor *norm(Frame &, ggml_tensor *, ggml_tensor *, ggml_tensor *gate = nullptr, float epsilon = -1,
                      float scale = 1);
    ggml_tensor *mix(Frame &, ggml_tensor *, int layer, const char *, ggml_tensor **);
    ggml_tensor *combine(Frame &, ggml_tensor *, ggml_tensor *, ggml_tensor *, ggml_tensor *);
    ggml_tensor *ple(Frame &, ggml_tensor *embedding, ggml_tensor *hidden, int rows, ggml_tensor *replay);
    ggml_tensor *gdn(Frame &, ggml_tensor *, int layer, NativeGdnReplay *replay);
    ggml_tensor *qsa(Frame &, ggml_tensor *, int layer, const AuxInputs &);
    ggml_tensor *attention(Frame &, LayerIO &, ggml_tensor *, int layer);
    void head(Frame &, LayerIO &, ggml_tensor *);
    void before(Frame &, LayerIO &, int layer);
    void after(Frame &, LayerIO &, int layer, bool output_head);
};
/** Compile-time row shapes share one implementation and stable capture inputs. */
template <int MAX_T> class NativeLayerKernels {
    static_assert(MAX_T >= 1 && MAX_T <= 5);

  public:
    explicit NativeLayerKernels(NativeLayerPlan &plan) : plan_(plan) {}
    void before(Frame &f, LayerIO &io, int layer) {
        GGML_ASSERT(io.rows == MAX_T);
        plan_.build_layer(f, io, layer, NativeLayerPlan::Part::Before, false);
    }
    void after(Frame &f, LayerIO &io, int layer, bool head = true) {
        GGML_ASSERT(io.rows == MAX_T);
        plan_.build_layer(f, io, layer, NativeLayerPlan::Part::Post, head);
    }

  private:
    NativeLayerPlan &plan_;
};
} // namespace souris
