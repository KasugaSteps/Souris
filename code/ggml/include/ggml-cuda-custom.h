#pragma once
#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif
/** Enqueue a CUDA-only custom operation on the backend's current stream. */
typedef void (*ggml_cuda_custom_op_t)(struct ggml_tensor * dst, void * stream, void * userdata);
GGML_BACKEND_API struct ggml_tensor * ggml_backend_cuda_custom_4d(
    struct ggml_context * ctx, enum ggml_type type, int64_t n0, int64_t n1, int64_t n2, int64_t n3,
    struct ggml_tensor ** args, int n_args, ggml_cuda_custom_op_t execute, void * userdata);
GGML_BACKEND_API struct ggml_tensor * ggml_backend_cuda_custom_inplace(
    struct ggml_context * ctx, struct ggml_tensor * target, struct ggml_tensor ** args, int n_args,
    ggml_cuda_custom_op_t execute, void * userdata);
/** Direct evaluation bypasses the backend graph cache; callers may capture the stream explicitly. */
GGML_BACKEND_API enum ggml_status ggml_backend_cuda_compute_eager(ggml_backend_t backend, struct ggml_cgraph * graph);
GGML_BACKEND_API void * ggml_backend_cuda_get_stream(ggml_backend_t backend);
/** Initialize handles and reserve reusable kernel scratch before any capture. */
GGML_BACKEND_API void ggml_backend_cuda_prepare(ggml_backend_t backend, size_t scratch_bytes);
/** Control physical scratch growth after preparation; return allocation count. */
GGML_BACKEND_API size_t ggml_backend_cuda_scratch_growth(ggml_backend_t backend, bool allow);
/** Packed shared gate/up MMQ with a shared-memory SwiGLU epilogue. */
GGML_BACKEND_API struct ggml_tensor * ggml_backend_cuda_prefill_shared(
    struct ggml_context *, ggml_backend_t, struct ggml_tensor * weights, struct ggml_tensor * input);
/** Native small-row projections consume one explicit, shared Q8_1 activation. */
GGML_BACKEND_API struct ggml_tensor * ggml_backend_cuda_native_quantize(
    struct ggml_context *, ggml_backend_t, struct ggml_tensor * input);
GGML_BACKEND_API struct ggml_tensor * ggml_backend_cuda_native_matmul(
    struct ggml_context *, ggml_backend_t, struct ggml_tensor * weights,
    struct ggml_tensor * input, struct ggml_tensor * quantized);
GGML_BACKEND_API struct ggml_tensor * ggml_backend_cuda_native_expert(
    struct ggml_context *, ggml_backend_t, struct ggml_tensor * weights,
    struct ggml_tensor * input, struct ggml_tensor * quantized, struct ggml_tensor * ids,
    struct ggml_tensor * pointers, struct ggml_tensor * layer, struct ggml_tensor * gate_pointers);
#ifdef __cplusplus
}
#endif
