#pragma once
#include "common.cuh"

/** Shared expert packed MMQ, followed by a shared-memory SwiGLU epilogue. */
void ggml_cuda_prefill_shared(ggml_backend_cuda_context &, const ggml_tensor *, const ggml_tensor *, ggml_tensor *);
