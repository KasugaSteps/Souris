#include "mtp_ops.h"
#include "cpy-utils.cuh"
#include <cuda_fp16.h>
#include <cassert>
#include <math_constants.h>

namespace souris {
static __global__ void positions_kernel(int32_t * positions, const Control * control, int rows) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 4 * rows) { positions[i] = control->position + i % rows; }
}
void cuda_mtp_positions(ggml_tensor * dst, void * stream, void *) {
    positions_kernel<<<(dst->ne[0] + 255) / 256, 256, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<int32_t *>(dst->data), static_cast<const Control *>(dst->src[0]->data), dst->ne[0] / 4);
}
static __global__ void append_kernel(block_q8_0 * cache, const float * input, const Control * control,
                                    int * positions, int rows, int head_dim, int heads, int ring_tokens) {
    const int blocks = head_dim / QK8_0;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * heads * blocks) { return; }
    const int token = i / (heads * blocks);
    if (token >= control->tokens) { return; }
    const int inner = i % (heads * blocks);
    const int position = control->position + token, slot = position % ring_tokens;
    if (!inner) { positions[slot] = position; }
    quantize_f32_q8_0_block(input + i * QK8_0,
                          cache + slot * heads * blocks + inner);
}
void cuda_mtp_append(ggml_tensor * dst, void * stream, void * metadata) {
    const auto & op = *static_cast<MtpInputs *>(metadata);
    const int blocks = op.rows * op.kv_heads * op.head_dim / QK8_0;
    append_kernel<<<(blocks + 255) / 256, 256, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<block_q8_0 *>(dst->data), static_cast<const float *>(dst->src[0]->data),
        static_cast<const Control *>(dst->src[1]->data), static_cast<int *>(dst->src[2]->data),
        op.rows, op.head_dim, op.kv_heads, op.ring_tokens);
}
static __global__ void window_mask_kernel(half * mask, const Control * control, const int * positions,
                                          int capacity, int rows, int window) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= capacity * rows) { return; }
    const int query = control->position + i / capacity, key = positions[i % capacity];
    mask[i] = __float2half(key >= 0 && key <= query && key > query - window ? 0.0f : -CUDART_INF_F);
}
void cuda_mtp_mask(ggml_tensor * dst, void * stream, void * metadata) {
    const auto & op = *static_cast<MtpInputs *>(metadata);
    window_mask_kernel<<<(op.capacity * op.rows + 255) / 256, 256, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<half *>(dst->data), static_cast<const Control *>(dst->src[0]->data),
        static_cast<const int *>(dst->src[1]->data), op.capacity, op.rows, op.window);
}
static __global__ void verify_plan_kernel(int * plan, const int * ids, const int * resident,
                                         const Control * control, int * unique, int * mapping, int count) {
    if (threadIdx.x) { return; }
    int hits = 0, misses = 0, used = 0;
    // Keep union keys and row membership so admission can select the committed
    // prefix later, without downloading per-token routes or admitting lookahead.
    for (int i = 0; i < count; ++i) {
        unique[i] = -1;
        plan[2 + 2 * count + i] = -1;
        plan[2 + 3 * count + i] = 0;
    }
    for (int i = 0; i < count; ++i) {
        int entry = 0;
        while (entry < used && unique[entry] != ids[i]) { ++entry; }
        mapping[i] = entry;
        plan[2 + 3 * count + entry] |= 1 << (i / fixed::topk);
        if (entry < used) { continue; }
        unique[used++] = ids[i];
        const int key = control->layer * fixed::experts + ids[i], slot = resident[key];
        plan[2 + 2 * count + entry] = key;
        if (slot >= 0) { plan[2 + hits++] = slot; }
        else { plan[2 + count + misses++] = key; }
    }
    plan[0] = hits; plan[1] = misses;
}
void cuda_verify_plan(ggml_tensor * dst, void * stream, void *) {
    verify_plan_kernel<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(static_cast<int *>(dst->data),
        static_cast<const int *>(dst->src[0]->data), static_cast<const int *>(dst->src[1]->data),
        static_cast<const Control *>(dst->src[2]->data), static_cast<int *>(dst->src[3]->data),
        static_cast<int *>(dst->src[4]->data), dst->src[0]->ne[0] * dst->src[0]->ne[1]);
}
static __global__ void verify_bind_kernel(uint64_t * pointers, const uint64_t * target,
                                         const int * target_resident, const int * unique,
                                         const Control * control, const char * zero, int count, uint64_t gate, uint64_t up, uint64_t down) {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= 3 * count) { return; }
    const int role = i / count, expert = unique[i % count];
    const uint64_t offsets[] = {gate, up, down};
    if (expert < 0) { pointers[i] = reinterpret_cast<uint64_t>(zero) + offsets[role]; return; }
    const int offset = (control->layer * 3 + role) * fixed::experts + expert;
    // -1 is absent. Provisional entries encode their slot as -(slot+2).
    assert(target_resident[control->layer * fixed::experts + expert] != -1);
    pointers[i] = target[offset];
    assert(pointers[i] != 0);
}
void cuda_verify_bind(ggml_tensor * dst, void * stream, void * metadata) {
    const auto & layout = *static_cast<const Layout *>(metadata);
    verify_bind_kernel<<<(dst->ne[0] + 255) / 256, 256, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<uint64_t *>(dst->data), static_cast<const uint64_t *>(dst->src[0]->data),
        static_cast<const int *>(dst->src[1]->data), static_cast<const int *>(dst->src[2]->data),
        static_cast<const Control *>(dst->src[3]->data), static_cast<const char *>(dst->src[4]->data),
        dst->ne[0] / 3, layout.role_offset[0], layout.role_offset[1], layout.role_offset[2]);
}
static __global__ void index_snapshot_kernel(char * snapshot, const char * index, const Control * control,
                                            int rows, int bytes) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows * bytes) { snapshot[i] = index[size_t(control->position) * bytes + i]; }
}
static __global__ void index_commit_kernel(char * index, const char * old, const char * updated,
                                          const Control * control, int rows, int bytes) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows * bytes) {
        index[size_t(control->position) * bytes + i] = i / bytes < control->tokens ? updated[i] : old[i];
    }
}
void cuda_verify_index(ggml_tensor * dst, void * stream, void * metadata) {
    const auto & op = *static_cast<VerifyRows *>(metadata);
    const int bytes = op.width * sizeof(ggml_fp16_t), blocks = (op.rows * bytes + 255) / 256;
    if (dst->src[3]) {
        index_commit_kernel<<<blocks, 256, 0, static_cast<cudaStream_t>(stream)>>>(static_cast<char *>(dst->data),
            static_cast<const char *>(dst->src[0]->data), static_cast<const char *>(dst->src[1]->data),
            static_cast<const Control *>(dst->src[2]->data), op.rows, bytes);
    } else {
        index_snapshot_kernel<<<blocks, 256, 0, static_cast<cudaStream_t>(stream)>>>(static_cast<char *>(dst->data),
            static_cast<const char *>(dst->src[0]->data), static_cast<const Control *>(dst->src[1]->data), op.rows, bytes);
    }
}
static __global__ void verify_weight_kernel(float * out, const float * input, const float * weights, int width, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < width * count) { out[i] = input[i] * weights[i / width]; }
}
void cuda_verify_weight(ggml_tensor * dst, void * stream, void *) {
    const int width = dst->ne[0], count = dst->ne[1];
    verify_weight_kernel<<<(width * count + 255) / 256, 256, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<float *>(dst->data), static_cast<const float *>(dst->src[0]->data),
        static_cast<const float *>(dst->src[1]->data), width, count);
}
}
