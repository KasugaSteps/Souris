#pragma once
#include "device.h"

namespace souris {
/** Keep the two HC matmuls separate; fuse only their elementwise epilogues. */
ggml_tensor * prefill_hc_down(Frame &, ggml_tensor *, int hc);
ggml_tensor * prefill_hc_up(Frame &, ggml_tensor * gate, ggml_tensor * normalized, int hc);
/** Combine directly into caller-owned GPU IO, preserving the FP32 operation order. */
ggml_tensor * prefill_combine(Frame &, ggml_tensor * residual, ggml_tensor * output,
                               ggml_tensor * inject, ggml_tensor * target);
ggml_tensor * prefill_shared_combine(Frame &, ggml_tensor * residual, ggml_tensor * routed,
                                      ggml_tensor * shared, ggml_tensor * gate, ggml_tensor * inject,
                                      ggml_tensor * target);
void cuda_prefill_hc_down(ggml_tensor *, void *, void *);
void cuda_prefill_hc_up(ggml_tensor *, void *, void *);
void cuda_prefill_combine(ggml_tensor *, void *, void *);
void cuda_prefill_shared_combine(ggml_tensor *, void *, void *);
}
