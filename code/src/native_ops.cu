#include "common.cuh"
#include "io.h"
#include "native_ops.h"
#include "unary.cuh"
#include <cuda_fp16.h>
#include <math_constants.h>

namespace souris {
namespace {
constexpr int threads = 256;
__device__ void quantize_value(float value, block_q8_1 *output, int at) {
    const float maximum = warp_reduce_max<32>(fabsf(value));
    const float sum = warp_reduce_sum<32>(value);
    const float scale = maximum / 127.0f;
    output[at / 32].qs[at % 32] = maximum == 0 ? 0 : roundf(value / scale);
    if (!(at % 32)) {
        output[at / 32].ds = make_half2(scale, sum);
    }
}
__device__ float weight_value(const void *weights, bool half_weights, int at) {
    if (!weights) {
        return 1.0f;
    }
    return half_weights ? __half2float(static_cast<const half *>(weights)[at])
                        : static_cast<const float *>(weights)[at];
}
/** Use GGML's vectorized sum and block reduction order before the Q8 epilogue. */
template <int Block, int Vpt>
__global__ void norm_kernel(const float *input, const void *weight, const float *gate, float *output, block_q8_1 *q8,
                            int columns, int rows1, int rows2, int64_t s1, int64_t s2, int64_t s3, int weight_rows,
                            bool half_weights, NativeActivation op) {
    const int row = blockIdx.x, tid = threadIdx.x;
    const int i1 = row % rows1, i2 = (row / rows1) % rows2, i3 = row / (rows1 * rows2);
    const auto *x = reinterpret_cast<const float4 *>(input + i1 * s1 + i2 * s2 + i3 * s3);
    float4 values[Vpt];
    float total = 0;
    for (int k = 0; k < Vpt; ++k) {
        const int i = tid + k * Block;
        if (i < columns / 4) {
            values[k] = x[i];
            const auto v = values[k];
            total += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
    }
    __shared__ float reduction[32];
    total = block_reduce<block_reduce_method::SUM, Block>(total, reduction);
    const float scale = rsqrtf(total / columns + op.epsilon);
    for (int k = 0; k < Vpt; ++k) {
        const int i = tid + k * Block;
        if (i < columns / 4) {
            const auto v = values[k];
            const int at = row * columns + 4 * i, w = (row % weight_rows) * columns + 4 * i;
            output[at] = __fmul_rn(__fmul_rn(scale, v.x), weight_value(weight, half_weights, w)) * op.scale;
            output[at + 1] = __fmul_rn(__fmul_rn(scale, v.y), weight_value(weight, half_weights, w + 1)) * op.scale;
            output[at + 2] = __fmul_rn(__fmul_rn(scale, v.z), weight_value(weight, half_weights, w + 2)) * op.scale;
            output[at + 3] = __fmul_rn(__fmul_rn(scale, v.w), weight_value(weight, half_weights, w + 3)) * op.scale;
            if (op.gated) {
                for (int j = 0; j < 4; ++j)
                    output[at + j] = __fmul_rn(output[at + j], 1.0f / (1.0f + expf(-gate[at + j])));
            }
        }
    }
    if (!op.quant_columns)
        return;
    __syncthreads();
    const int padded = (op.quant_columns + MATRIX_ROW_PADDING - 1) / MATRIX_ROW_PADDING * MATRIX_ROW_PADDING;
    for (int col = tid; col < columns; col += Block) {
        const int flat = row * columns + col;
        const int at = flat / op.quant_columns * padded + flat % op.quant_columns;
        quantize_value(output[flat], q8, at);
    }
    if ((row + 1) * columns % op.quant_columns == 0) {
        const int qrow = (row + 1) * columns / op.quant_columns - 1;
        for (int b = tid; b < (padded - op.quant_columns) / 32; b += Block) {
            const int at = (qrow * padded + op.quant_columns) / 32 + b;
            q8[at].ds = make_half2(0.0f, 0.0f);
            for (int j = 0; j < 32; ++j)
                q8[at].qs[j] = 0;
        }
    }
}
template <int Mode>
__global__ void activation_kernel(const float *a, const float *b, float *output, block_q8_1 *q8, int columns,
                                  NativeActivation op) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x, row = blockIdx.y;
    const int padded = (columns + MATRIX_ROW_PADDING - 1) / MATRIX_ROW_PADDING * MATRIX_ROW_PADDING;
    if (col >= padded)
        return;
    float value = 0;
    if (col < columns) {
        if constexpr (Mode == 0) {
            value = ggml_cuda_op_silu_single(__fmul_rn(a[row * columns + col], op.scale));
        }
        if constexpr (Mode == 1) {
            for (int h = 0; h < op.hc; ++h) {
                const int at = (row * op.hc + h) * columns + col;
                const float part = __fmul_rn(b[at], 1.0f / (1.0f + expf(-a[at])));
                value = h ? __fadd_rn(value, part) : part;
            }
            value = __fmul_rn(value, 1.0f / op.hc);
        }
        if constexpr (Mode == 2) {
            value =
                __fmul_rn(ggml_cuda_op_silu_single(a[row * 2 * columns + col]), a[row * 2 * columns + columns + col]);
        }
        if constexpr (Mode == 3) {
            value = __fmul_rn(a[row * columns + col], 1.0f / (1.0f + expf(-b[row * columns + col])));
        }
        output[row * columns + col] = value;
    }
    if (op.quant_columns)
        quantize_value(value, q8, row * padded + col);
}
__global__ void gdn_gates_kernel(const float *alpha, const void *bias, const void *a, const float *beta, float *decay,
                                 float *sigmoid, int heads, int count, bool half_bias, bool half_a) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
        return;
    const float value = __fadd_rn(alpha[i], weight_value(bias, half_bias, i % heads));
    decay[i] = __fmul_rn((value > 20.0f ? value : log1pf(expf(value))), weight_value(a, half_a, i % heads));
    sigmoid[i] = 1.0f / (1.0f + expf(-beta[i]));
}
__global__ void router_kernel(const float *input, int *ids, float *weights, float output_scale) {
    const int tid = threadIdx.x, row = blockIdx.x;
    __shared__ float reduce[32], probabilities[512];
    __shared__ int selected[10], best_id;
    float value = input[row * 512 + tid];
    const float maximum = block_reduce<block_reduce_method::MAX, 512>(value, reduce);
    value = expf(value - maximum);
    __syncthreads();
    const float sum = block_reduce<block_reduce_method::SUM, 512>(value, reduce);
    probabilities[tid] = value * (1.0f / sum);
    float remaining = probabilities[tid];
    __syncthreads();
    for (int rank = 0; rank < 10; ++rank) {
        const float best = block_reduce<block_reduce_method::MAX, 512>(remaining, reduce);
        if (!tid)
            best_id = 512;
        __syncthreads();
        if (remaining == best)
            atomicMin(&best_id, tid);
        __syncthreads();
        if (!tid) {
            selected[rank] = best_id;
            ids[row * 10 + rank] = best_id;
        }
        if (tid == best_id)
            remaining = -CUDART_INF_F;
        __syncthreads();
    }
    if (tid < 32) {
        float p = tid < 10 ? probabilities[selected[tid]] : 0;
        const float total = warp_reduce_sum<32>(p);
        if (tid < 10)
            weights[row * 10 + tid] = __fmul_rn(p / total, output_scale);
    }
}
__global__ void combine_kernel(const float *residual, const float *routed, const float *shared, const float *gate,
                               const float *inject, float *output, int width, int hc) {
    const int row = blockIdx.x, h = blockIdx.y;
    __shared__ float alpha, beta;
    if (!threadIdx.x) {
        alpha = __fmul_rn(1.0f / (1.0f + expf(-__fmul_rn(inject[row * hc + h], 1.0f / hc))), 2.0f);
        beta = gate ? 1.0f / (1.0f + expf(-gate[row])) : 0;
    }
    __syncthreads();
    for (int col = threadIdx.x; col < width; col += blockDim.x) {
        const int at = (row * hc + h) * width + col;
        float value = routed[row * width + col];
        if (shared)
            value = __fadd_rn(value, __fmul_rn(shared[row * width + col], beta));
        output[at] = __fadd_rn(residual[at], __fmul_rn(value, alpha));
    }
}
__global__ void history_commit_kernel(float *state, const float *rows, const Control *control, NativeHistory op) {
    const int channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= op.width)
        return;
    const int accepted = control->tokens;
    for (int i = 0; i < op.history; ++i) {
        const int source = i + accepted;
        state[channel * op.history + i] = source < op.history ? state[channel * op.history + source]
                                                              : rows[(source - op.history) * op.width + channel];
    }
}
__global__ void sum_kernel(const float *input, float *output, int width, int rows) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width * rows)
        return;
    const int row = i / width, col = i % width;
    float sum = input[(row * fixed::topk) * width + col];
    for (int rank = 1; rank < fixed::topk; ++rank)
        sum = __fadd_rn(sum, input[(row * fixed::topk + rank) * width + col]);
    output[i] = sum;
}
} // namespace
void cuda_native_norm(ggml_tensor *dst, void *stream, void *userdata) {
    const auto &op = *static_cast<NativeActivation *>(userdata);
    const auto *x = dst->src[0], *w = dst->src[1];
    const int columns = x->ne[0], count = ggml_nelements(x) / columns, weights = w ? ggml_nelements(w) / columns : 1;
    const auto s = static_cast<cudaStream_t>(stream);
#define LAUNCH_NORM(B, V)                                                                                              \
    norm_kernel<B, V><<<count, B, 0, s>>>(                                                                             \
        static_cast<const float *>(x->data), op.weighted ? w->data : nullptr,                                          \
        op.gated ? static_cast<const float *>(dst->src[3]->data) : nullptr, static_cast<float *>(dst->data),           \
        static_cast<block_q8_1 *>(dst->src[2]->data), columns, x->ne[1], x->ne[2], x->nb[1] / 4, x->nb[2] / 4,         \
        x->nb[3] / 4, weights, w && w->type == GGML_TYPE_F16, op)
    if (columns / 4 <= 32) {
        LAUNCH_NORM(32, 1);
    } else if (columns / 4 <= 128) {
        LAUNCH_NORM(128, 1);
    } else if (columns / 4 <= 256) {
        LAUNCH_NORM(256, 1);
    } else if (columns / 4 <= 512) {
        LAUNCH_NORM(512, 1);
    } else if (columns / 4 <= 1024) {
        LAUNCH_NORM(1024, 1);
    } else {
        LAUNCH_NORM(1024, 4);
    }
#undef LAUNCH_NORM
    SOURIS_CUDA(cudaGetLastError());
}
void cuda_native_hc_down(ggml_tensor *dst, void *stream, void *userdata) {
    const auto &op = *static_cast<NativeActivation *>(userdata);
    const int padded = (dst->ne[0] + MATRIX_ROW_PADDING - 1) / MATRIX_ROW_PADDING * MATRIX_ROW_PADDING;
    activation_kernel<0>
        <<<dim3((padded + threads - 1) / threads, dst->ne[1]), threads, 0, static_cast<cudaStream_t>(stream)>>>(
            static_cast<const float *>(dst->src[0]->data), nullptr, static_cast<float *>(dst->data),
            static_cast<block_q8_1 *>(dst->src[1]->data), dst->ne[0], op);
    SOURIS_CUDA(cudaGetLastError());
}
void cuda_native_hc_up(ggml_tensor *dst, void *stream, void *userdata) {
    const auto &op = *static_cast<NativeActivation *>(userdata);
    const int padded = (dst->ne[0] + MATRIX_ROW_PADDING - 1) / MATRIX_ROW_PADDING * MATRIX_ROW_PADDING;
    activation_kernel<1>
        <<<dim3((padded + threads - 1) / threads, dst->ne[1]), threads, 0, static_cast<cudaStream_t>(stream)>>>(
            static_cast<const float *>(dst->src[0]->data), static_cast<const float *>(dst->src[1]->data),
            static_cast<float *>(dst->data), static_cast<block_q8_1 *>(dst->src[2]->data), dst->ne[0], op);
    SOURIS_CUDA(cudaGetLastError());
}
void cuda_native_swiglu(ggml_tensor *dst, void *stream, void *userdata) {
    const auto &op = *static_cast<NativeActivation *>(userdata);
    const int padded = (dst->ne[0] + MATRIX_ROW_PADDING - 1) / MATRIX_ROW_PADDING * MATRIX_ROW_PADDING;
    activation_kernel<2>
        <<<dim3((padded + threads - 1) / threads, dst->ne[1]), threads, 0, static_cast<cudaStream_t>(stream)>>>(
            static_cast<const float *>(dst->src[0]->data), nullptr, static_cast<float *>(dst->data),
            static_cast<block_q8_1 *>(dst->src[1]->data), dst->ne[0], op);
    SOURIS_CUDA(cudaGetLastError());
}
void cuda_native_gate(ggml_tensor *dst, void *stream, void *userdata) {
    const auto &op = *static_cast<NativeActivation *>(userdata);
    const int padded = (dst->ne[0] + MATRIX_ROW_PADDING - 1) / MATRIX_ROW_PADDING * MATRIX_ROW_PADDING;
    activation_kernel<3>
        <<<dim3((padded + threads - 1) / threads, dst->ne[1]), threads, 0, static_cast<cudaStream_t>(stream)>>>(
            static_cast<const float *>(dst->src[0]->data), static_cast<const float *>(dst->src[1]->data),
            static_cast<float *>(dst->data), static_cast<block_q8_1 *>(dst->src[2]->data), dst->ne[0], op);
    SOURIS_CUDA(cudaGetLastError());
}
void cuda_native_gdn_gates(ggml_tensor *dst, void *stream, void *) {
    const int n = ggml_nelements(dst);
    gdn_gates_kernel<<<(n + threads - 1) / threads, threads, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const float *>(dst->src[0]->data), dst->src[1]->data, dst->src[2]->data,
        static_cast<const float *>(dst->src[3]->data), static_cast<float *>(dst->data),
        static_cast<float *>(dst->src[4]->data), dst->ne[0], n, dst->src[1]->type == GGML_TYPE_F16,
        dst->src[2]->type == GGML_TYPE_F16);
    SOURIS_CUDA(cudaGetLastError());
}
void cuda_native_router(ggml_tensor *dst, void *stream, void *userdata) {
    router_kernel<<<dst->ne[1], 512, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const float *>(dst->src[0]->data), static_cast<int *>(dst->data),
        static_cast<float *>(dst->src[1]->data), *static_cast<const float *>(userdata));
    SOURIS_CUDA(cudaGetLastError());
}
void cuda_native_combine(ggml_tensor *dst, void *stream, void *) {
    combine_kernel<<<dim3(dst->ne[2], dst->ne[1]), threads, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const float *>(dst->src[0]->data), static_cast<const float *>(dst->src[1]->data), nullptr, nullptr,
        static_cast<const float *>(dst->src[2]->data), static_cast<float *>(dst->data), dst->ne[0], dst->ne[1]);
    SOURIS_CUDA(cudaGetLastError());
}
void cuda_native_shared_combine(ggml_tensor *dst, void *stream, void *) {
    combine_kernel<<<dim3(dst->ne[2], dst->ne[1]), threads, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const float *>(dst->src[0]->data), static_cast<const float *>(dst->src[1]->data),
        static_cast<const float *>(dst->src[2]->data), static_cast<const float *>(dst->src[3]->data),
        static_cast<const float *>(dst->src[4]->data), static_cast<float *>(dst->data), dst->ne[0], dst->ne[1]);
    SOURIS_CUDA(cudaGetLastError());
}
void cuda_native_history_commit(ggml_tensor *dst, void *stream, void *userdata) {
    const auto op = *static_cast<NativeHistory *>(userdata);
    history_commit_kernel<<<(op.width + threads - 1) / threads, threads, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<float *>(dst->data), static_cast<const float *>(dst->src[0]->data),
        static_cast<const Control *>(dst->src[1]->data), op);
    SOURIS_CUDA(cudaGetLastError());
}
void cuda_native_sum(ggml_tensor *dst, void *stream, void *) {
    const int n = ggml_nelements(dst);
    sum_kernel<<<(n + threads - 1) / threads, threads, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const float *>(dst->src[0]->data), static_cast<float *>(dst->data), dst->ne[0], dst->ne[1]);
    SOURIS_CUDA(cudaGetLastError());
}
} // namespace souris
