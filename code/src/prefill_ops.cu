#include "prefill_ops.h"
#include "unary.cuh"

namespace souris {
namespace {
constexpr int threads = 256;
__global__ void hc_down(const float * input, float * output, int n, float scale) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { output[i] = ggml_cuda_op_silu_single(__fmul_rn(input[i], scale)); }
}
__global__ void hc_up(const float * gate, const float * input, float * output, int width, int hc, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) { return; }
    const int row = i / width, col = i % width;
    float sum = 0;
    for (int h = 0; h < hc; ++h) {
        const int at = (row * hc + h) * width + col;
        const float value = __fmul_rn(input[at], 1.0f / (1.0f + expf(-gate[at])));
        sum = h ? __fadd_rn(sum, value) : value;
    }
    output[i] = __fmul_rn(sum, 1.0f / hc);
}
/** Row-local gates are evaluated once per CTA, then reused across width. */
__global__ void combine(const float * residual, const float * routed, const float * shared,
                         const float * shared_gate, const float * inject, void * output,
                         int width, int hc, bool half_output) {
    const int token = blockIdx.x, h = blockIdx.y;
    __shared__ float alpha, beta;
    if (!threadIdx.x) {
        const float value = __fmul_rn(inject[token * hc + h], 1.0f / hc);
        alpha = __fmul_rn(1.0f / (1.0f + expf(-value)), 2.0f);
        beta = shared_gate ? 1.0f / (1.0f + expf(-shared_gate[token])) : 0;
    }
    __syncthreads();
    for (int col = threadIdx.x; col < width; col += blockDim.x) {
        const size_t at = (size_t(token) * hc + h) * width + col;
        float value = routed[size_t(token) * width + col];
        if (shared) { value = __fadd_rn(value, __fmul_rn(shared[size_t(token) * width + col], beta)); }
        value = __fadd_rn(residual[at], __fmul_rn(value, alpha));
        if (half_output) { static_cast<half *>(output)[at] = __float2half(value); }
        else { static_cast<float *>(output)[at] = value; }
    }
}
}
void cuda_prefill_hc_down(ggml_tensor * dst, void * stream, void * userdata) {
    const int n = ggml_nelements(dst);
    hc_down<<<(n + threads - 1) / threads, threads, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const float *>(dst->src[0]->data), static_cast<float *>(dst->data), n, 1.0f / uintptr_t(userdata));
    SOURIS_CUDA(cudaGetLastError());
}
void cuda_prefill_hc_up(ggml_tensor * dst, void * stream, void * userdata) {
    const int n = ggml_nelements(dst);
    hc_up<<<(n + threads - 1) / threads, threads, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const float *>(dst->src[0]->data), static_cast<const float *>(dst->src[1]->data),
        static_cast<float *>(dst->data), dst->ne[0], uintptr_t(userdata), n);
    SOURIS_CUDA(cudaGetLastError());
}
void cuda_prefill_combine(ggml_tensor * dst, void * stream, void *) {
    combine<<<dim3(dst->ne[2], dst->ne[1]), threads, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const float *>(dst->src[0]->data), static_cast<const float *>(dst->src[1]->data),
        nullptr, nullptr, static_cast<const float *>(dst->src[2]->data), dst->data,
        dst->ne[0], dst->ne[1], dst->type == GGML_TYPE_F16);
    SOURIS_CUDA(cudaGetLastError());
}
void cuda_prefill_shared_combine(ggml_tensor * dst, void * stream, void *) {
    combine<<<dim3(dst->ne[2], dst->ne[1]), threads, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const float *>(dst->src[0]->data), static_cast<const float *>(dst->src[1]->data),
        static_cast<const float *>(dst->src[2]->data), static_cast<const float *>(dst->src[3]->data),
        static_cast<const float *>(dst->src[4]->data), dst->data,
        dst->ne[0], dst->ne[1], dst->type == GGML_TYPE_F16);
    SOURIS_CUDA(cudaGetLastError());
}
}
