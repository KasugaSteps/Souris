#include "prefill-shared.cuh"
#include "mmq.cuh"
#include "quantize.cuh"
#include "unary.cuh"

namespace {
constexpr int shared_columns = 32;
/** Two row slices of the one packed GEMM share activation quantization and one CTA.
 * Native GGML MMQ accumulation writes only shared tiles; SwiGLU is the output epilogue.
 */
template <ggml_type type>
__global__ __launch_bounds__(ggml_cuda_mmq_get_nthreads(type, shared_columns, false, GGML_PREC_Q8))
void packed_swiglu(const char * weights, const int * input, float * output,
                   int width_blocks, int hidden, int tokens, int row_stride, int tile_words) {
    constexpr int J = shared_columns;
    constexpr int I = ggml_cuda_mmq_get_I(type, J, false, GGML_PREC_Q8);
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int tid = threadIdx.y * warp_size + threadIdx.x;
    extern __shared__ int data_mul_mat_q[];
    for (int j = tid; j < J; j += blockDim.y * warp_size) { data_mul_mat_q[j] = j; }
    __syncthreads();
    const int row = blockIdx.x * I, column = blockIdx.y * J;
    float * gate = reinterpret_cast<float *>(data_mul_mat_q + tile_words);
    float * up = reinterpret_cast<float *>(data_mul_mat_q + J);
    const int * y = input + column * (sizeof(block_q8_1_mmq) / sizeof(int));
    mul_mat_q_process_tile<type, J, false, false, GGML_PREC_Q8>(
        weights, row * row_stride, y, data_mul_mat_q, gate, nullptr, nullptr,
        row_stride, tokens, I, hidden - row - 1, tokens - column - 1, 0, width_blocks);
    __syncthreads();
    mul_mat_q_process_tile<type, J, false, false, GGML_PREC_Q8>(
        weights, (hidden + row) * row_stride, y, data_mul_mat_q, up, nullptr, nullptr,
        row_stride, tokens, I, hidden - row - 1, tokens - column - 1, 0, width_blocks);
    __syncthreads();
    for (int index = tid; index < I * J; index += blockDim.y * warp_size) {
        const int i = index % I, j = index / I;
        if (column + j < tokens) {
            output[size_t(column + j) * hidden + row + i] = ggml_cuda_op_silu_single(gate[index]) * up[index];
        }
    }
}
template <ggml_type type>
void launch_shared(ggml_backend_cuda_context & ctx, const ggml_tensor * weights,
                   const int * input, ggml_tensor * output) {
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    const auto config = ggml_cuda_mmq_get_config(type, shared_columns, false, cc, GGML_PREC_Q8);
    const int hidden = output->ne[0], tokens = output->ne[1];
    GGML_ASSERT(config.type != GGML_TYPE_COUNT && hidden % config.I == 0);
    const int base = std::max<int>(mmq_get_nbytes_shared(config, cc) / sizeof(int),
                                   shared_columns + config.I * shared_columns);
    const int bytes = (base + config.I * shared_columns) * sizeof(int);
    GGML_ASSERT(size_t(bytes) <= ggml_cuda_info().devices[ctx.device].smpbo);
    CUDA_CHECK(cudaFuncSetAttribute(packed_swiglu<type>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
    const dim3 block(ggml_cuda_info().devices[ctx.device].warp_size,
                     config.nthreads / ggml_cuda_info().devices[ctx.device].warp_size);
    const dim3 grid(hidden / config.I, (tokens + shared_columns - 1) / shared_columns);
    packed_swiglu<type><<<grid, block, bytes, ctx.stream()>>>(
        static_cast<const char *>(weights->data), input, static_cast<float *>(output->data),
        weights->ne[0] / ggml_blck_size(type), hidden, tokens,
        weights->nb[1] / ggml_type_size(type), base);
}
}
void ggml_cuda_prefill_shared(ggml_backend_cuda_context & ctx, const ggml_tensor * weights,
                              const ggml_tensor * input, ggml_tensor * output) {
    GGML_ASSERT(weights->ne[1] == 2 * output->ne[0] && input->type == GGML_TYPE_F32);
    const int width = input->ne[0], tokens = input->ne[1];
    const int padded = GGML_PAD(width, MATRIX_ROW_PADDING);
    const size_t padding = shared_columns * sizeof(block_q8_1_mmq) + 1024;
    ggml_cuda_pool_alloc<char> quantized(ctx.pool(), size_t(tokens) * padded * sizeof(block_q8_1) / QK8_1 + padding);
    quantize_mmq_q8_1_cuda(static_cast<const float *>(input->data), nullptr, quantized.get(), weights->type,
                           width, input->nb[1] / sizeof(float), input->nb[2] / sizeof(float),
                           input->nb[3] / sizeof(float), padded, tokens, 1, 1, ctx.stream());
    switch (weights->type) {
        case GGML_TYPE_IQ2_XXS: launch_shared<GGML_TYPE_IQ2_XXS>(ctx, weights, reinterpret_cast<const int *>(quantized.get()), output); break;
        case GGML_TYPE_Q4_0: launch_shared<GGML_TYPE_Q4_0>(ctx, weights, reinterpret_cast<const int *>(quantized.get()), output); break;
        default: GGML_ABORT("unsupported CUDA configuration: shared expert type");
    }
    CUDA_CHECK(cudaGetLastError());
}
