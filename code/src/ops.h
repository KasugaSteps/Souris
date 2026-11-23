#pragma once
#include "io.h"

namespace souris {
struct AuxInputs { ggml_tensor * positions, * writes, * reads, * pool_mask, * tails, * pool_positions; };
/** Build replay-safe position and cache-selection inputs from the control buffer. */
AuxInputs auxiliary(Frame & frame, ggml_tensor * control, const Dimensions & dim, int tokens, int capacity);
ggml_tensor * node_view(Frame & frame, ggml_tensor * storage, ggml_type type, int64_t n0, int64_t n1, size_t offset);
void cuda_auxiliary(ggml_tensor * dst, void * stream, void * userdata);
/** Publish one completed record; stale entries are cleared only if they still alias it. */
void cuda_expert_bind(void * table, int32_t * resident, int slot, int key, int previous,
                      void * record, const Layout &, cudaStream_t, bool provisional = false);
/** Publish committed provisional records or invalidate rejected keys in one batch. */
void cuda_expert_finish(void * table, int32_t * resident, const int32_t * metadata, int count, cudaStream_t);
void cuda_expert_invalidate(void * table, int32_t * resident, int key, int slot, cudaStream_t);
/** GPU routing plan and original-rank weighted output for native Decode MMVQ. */
void cuda_decode_plan(ggml_tensor *, void *, void *);
void cuda_decode_select(ggml_tensor *, void *, void *);
void cuda_decode_scatter(ggml_tensor *, void *, void *);
/** Startup-reserved internal sampler workspace size. */
size_t sampling_scratch_bytes(int vocab);
/** Sample logits with [temperature, top_p, top_k, min_p, random].
 * F32 output writes a normalized distribution and src[2] receives its token.
 * The final source is persistent scratch. Non-null userdata selects residual
 * probability input, whose parameters disable all truncation.
 */
void cuda_sample(ggml_tensor *, void *, void *);
/** Apply a cached grammar bitset before sampling. */
void cuda_token_mask(float * logits, const uint32_t * bits, const Token * sparse, int allowed, int count, cudaStream_t);
}
