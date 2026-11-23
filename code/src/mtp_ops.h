#pragma once
#include "io.h"

namespace souris {
/** Replay metadata for the private MTP cache. Control holds position and valid rows. */
struct MtpInputs { int rows, capacity, head_dim, kv_heads, window, ring_tokens; };
void cuda_mtp_positions(ggml_tensor *, void *, void *);
void cuda_mtp_append(ggml_tensor *, void *, void *);
void cuda_mtp_mask(ggml_tensor *, void *, void *);
/** Fixed-T target routing plan and index-row transaction helpers. */
struct VerifyRows { int rows, width; };
void cuda_verify_plan(ggml_tensor *, void *, void *);
void cuda_verify_bind(ggml_tensor *, void *, void *);
void cuda_verify_index(ggml_tensor *, void *, void *);
void cuda_verify_weight(ggml_tensor *, void *, void *);
}
