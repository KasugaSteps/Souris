#pragma once
#include "device.h"

namespace souris {
/** Stable graph metadata for a fused activation producer and its optional Q8_1 output. */
struct NativeActivation {
    float epsilon = 0, scale = 1;
    int hc = 1, quant_columns = 0;
    bool weighted = true, gated = false;
};
struct NativeHistory {
    int width, history;
};
void cuda_native_history_commit(ggml_tensor *, void *, void *);
void cuda_native_norm(ggml_tensor *, void *, void *);
void cuda_native_hc_down(ggml_tensor *, void *, void *);
void cuda_native_hc_up(ggml_tensor *, void *, void *);
void cuda_native_swiglu(ggml_tensor *, void *, void *);
void cuda_native_gate(ggml_tensor *, void *, void *);
void cuda_native_gdn_gates(ggml_tensor *, void *, void *);
void cuda_native_router(ggml_tensor *, void *, void *);
void cuda_native_combine(ggml_tensor *, void *, void *);
void cuda_native_shared_combine(ggml_tensor *, void *, void *);
void cuda_native_sum(ggml_tensor *, void *, void *);
} // namespace souris
