#include "moe.h"
#include "device.h"
#include "io.h"
#include "mmq.cuh"
#include "quantize.cuh"
#include "ggml-cuda.h"
#include "ggml-backend-impl.h"
#include <algorithm>

namespace souris {
namespace {
constexpr int group_threads = 256;
constexpr int max_group = 64;
constexpr int max_tokens = 1024;
// MMQ may issue unconditional loads through the final 128-column tile.
constexpr size_t quant_padding = GGML_PAD(128 * sizeof(block_q8_1_mmq), 1024);
int padded(int columns) { return GGML_PAD(columns, MATRIX_ROW_PADDING); }
size_t qbytes(int rows, int columns) { return size_t(rows) * padded(columns) * sizeof(block_q8_1_mmq) / QK8_1_MMQ + quant_padding; }
__global__ void group_count(const int32_t * ids, int32_t * counts, int routes) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < routes) { atomicAdd(counts + ids[i], 1); }
}
__global__ void group_scan(int32_t * group) {
    __shared__ int prefix[fixed::experts], tasks[fixed::experts];
    const int e = threadIdx.x, value = group[e];
    const int task_count = (value + moe_tile_rows - 1) / moe_tile_rows;
    prefix[e] = value;
    tasks[e] = task_count;
    __syncthreads();
    for (int offset = 1; offset < fixed::experts; offset *= 2) {
        const int add = e >= offset ? prefix[e - offset] : 0;
        const int task_add = e >= offset ? tasks[e - offset] : 0;
        __syncthreads();
        prefix[e] += add;
        tasks[e] += task_add;
        __syncthreads();
    }
    group[fixed::experts + e] = prefix[e] - value;
    group[2 * fixed::experts + 1 + e] = prefix[e] - value;
    group[moe_task_prefix + e] = tasks[e] - task_count;
    if (e == fixed::experts - 1) {
        group[2 * fixed::experts] = prefix[e];
        group[moe_task_prefix + fixed::experts] = tasks[e];
    }
}
__global__ void group_scatter(const int32_t * ids, int32_t * group, int routes) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= routes) { return; }
    const int position = atomicAdd(group + 2 * fixed::experts + 1 + ids[i], 1);
    group[moe_group_header + position] = i / fixed::topk;
    group[moe_group_header + routes + position] = i;
}

__global__ void select_experts(const int32_t * group, int32_t * selected) {
    __shared__ int prefix[fixed::experts];
    const int expert = threadIdx.x;
    prefix[expert] = group[expert] > 0;
    __syncthreads();
    for (int distance = 1; distance < fixed::experts; distance *= 2) {
        const int add = expert >= distance ? prefix[expert - distance] : 0;
        __syncthreads();
        prefix[expert] += add;
        __syncthreads();
    }
    if (group[expert]) { selected[prefix[expert] - 1] = expert; }
}

/** MMQ rows stay in expert order. Only the final scatter restores route order. */
__global__ void prepare_rows(const int * group, int * inverse, int * identity, int routes) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < routes) {
        inverse[group[moe_group_header + routes + row]] = row;
        identity[row] = row;
    }
}
__global__ void prepare_product(MoeOp op, uint64_t * gate, uint64_t * down,
                                int * bounds, char * gathered, size_t gu_bytes, bool direct) {
    const int index = blockIdx.x;
    const int expert = op.selected[op.begin + index];
    const int layer = op.control[2];
    const auto * table = op.table + layer * 3 * fixed::experts;
    const char * g = reinterpret_cast<const char *>(table[expert]);
    const char * u = reinterpret_cast<const char *>(table[fixed::experts + expert]);
    if (!threadIdx.x) {
        gate[index] = direct ? table[expert] : reinterpret_cast<uint64_t>(gathered + index * gu_bytes);
        down[index] = table[2 * fixed::experts + expert];
        bounds[index] = op.group[fixed::experts + expert];
        if (index == op.end - op.begin - 1) { bounds[index + 1] = op.group[fixed::experts + expert + 1]; }
    }
    if (!direct) {
        const size_t half = gu_bytes / 2;
        for (size_t byte = threadIdx.x; byte < gu_bytes; byte += blockDim.x) {
            gathered[index * gu_bytes + byte] = byte < half ? g[byte] : u[byte - half];
        }
    }
}
/** GGML MMQ's D4/DS4/D2S6 layout, with SwiGLU as the quantizer's producer. */
__global__ void swiglu_quant(const float * gu, block_q8_1_mmq * out, const int * bounds,
                            int experts, int routes, int hidden, int pad, mmq_q8_1_ds_layout layout) {
    const int row = bounds[0] + blockIdx.x, col = (blockIdx.y * blockDim.x + threadIdx.x) * 4;
    if (row < bounds[0] || row >= bounds[experts] || col >= pad) { return; }
    float values[4] = {};
    if (col < hidden) {
        for (int k = 0; k < 4; ++k) {
            const float gate = gu[size_t(row) * 2 * hidden + col + k];
            values[k] = gate / (1.0f + expf(-gate)) * gu[size_t(row) * 2 * hidden + hidden + col + k];
        }
    }
    float maximum = fmaxf(fmaxf(fabsf(values[0]), fabsf(values[1])), fmaxf(fabsf(values[2]), fabsf(values[3])));
    const int scale_values = layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 64 : 32;
    for (int offset = scale_values / 8; offset; offset >>= 1) {
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, offset));
    }
    float sum = values[0] + values[1] + values[2] + values[3];
    const int sum_values = layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 16 : 32;
    for (int offset = sum_values / 8; offset; offset >>= 1) { sum += __shfl_xor_sync(0xffffffff, sum, offset); }
    const float inverse = maximum ? 127.0f / maximum : 0;
    const float scale = maximum ? 1.0f / inverse : 0;
    char4 codes = {char(roundf(values[0] * inverse)), char(roundf(values[1] * inverse)),
                   char(roundf(values[2] * inverse)), char(roundf(values[3] * inverse))};
    auto & block = out[size_t(col / QK8_1_MMQ) * routes + row];
    reinterpret_cast<char4 *>(block.qs)[col % QK8_1_MMQ / 4] = codes;
    const int at = col % QK8_1_MMQ;
    if (layout == MMQ_Q8_1_DS_LAYOUT_D2S6) {
        if (at % 16 == 0 && at < 96) {
            block.d2s6[2 + at / 16] = sum;
            if (at % 64 == 0) { block.d2s6[at / 64] = scale; }
        }
    } else if (at % 32 == 0) {
        if (layout == MMQ_Q8_1_DS_LAYOUT_DS4) { block.ds4[at / 32] = make_half2(scale, sum); }
        else { block.d4[at / 32] = scale; }
    }
}
__global__ void weighted_scatter(MoeOp op, const float * values, const int * bounds, int experts) {
    const int first = bounds[0], row = first + blockIdx.x, last = bounds[experts];
    if (row < first || row >= last) { return; }
    const int destination = op.group[moe_group_header + op.routes + row];
    for (int col = threadIdx.x; col < op.shape.width; col += blockDim.x) {
        op.output[size_t(destination) * op.shape.width + col] = values[size_t(row) * op.shape.width + col] * op.weights[destination];
    }
}
struct Product {
    ggml_type type;
    int columns, rows, experts, activation_rows, max_rows;
    const void * q8;
    const uint64_t * table;
    const int * zero, *bounds, *identity;
    float * output;
};
void product(ggml_backend_cuda_context & ctx, const Product & p, cudaStream_t stream) {
    mmq_args args{};
    args.type_x = p.type; args.y = static_cast<const int *>(p.q8);
    args.ids_dst = p.identity; args.expert_bounds = p.bounds; args.dst = p.output;
    args.ncols_x = p.columns; args.nrows_x = p.rows; args.ncols_dst = p.activation_rows;
    args.stride_row_x = p.columns / ggml_blck_size(p.type);
    args.ncols_y = p.activation_rows; args.nrows_dst = p.rows;
    args.nchannels_x = args.nchannels_y = p.experts;
    args.nsamples_x = args.nsamples_y = 1; args.ncols_max = p.max_rows;
    args.pointers = {reinterpret_cast<const void * const *>(p.table), p.zero, 0};
    const auto & device = ggml_cuda_info().devices[ggml_cuda_get_device()];
    int tiles = INT_MAX;
    for (int width = 8; width <= 128 && tiles > 1; width += 8) {
        const auto config = ggml_cuda_mmq_get_config(p.type, width, ggml_cuda_mmq_needs_fallback(p.rows), device.cc, GGML_PREC_Q8);
        if (config.type == GGML_TYPE_COUNT || mmq_get_nbytes_shared(config, device.cc) > device.smpbo) { continue; }
        const int count = (p.max_rows + config.J - 1) / config.J;
        if (count < tiles) { args.J_best = width; tiles = count; }
    }
    if (!args.J_best) { GGML_ABORT("unsupported quant configuration: MMQ tile"); }
#define MMQ_CASE(T) case GGML_TYPE_##T: mul_mat_q_case<GGML_TYPE_##T, GGML_PREC_Q8>(ctx, args, stream); break
    switch (p.type) {
        MMQ_CASE(Q2_0); MMQ_CASE(Q4_0); MMQ_CASE(IQ2_XXS); MMQ_CASE(IQ2_XS); MMQ_CASE(IQ2_S);
        MMQ_CASE(IQ3_XXS); MMQ_CASE(IQ3_S); MMQ_CASE(IQ4_NL); MMQ_CASE(IQ4_XS); MMQ_CASE(Q8_0);
        MMQ_CASE(Q4_K); MMQ_CASE(Q5_K); MMQ_CASE(Q5_1); MMQ_CASE(Q6_K);
        default: GGML_ABORT("unsupported quant configuration: %s", ggml_type_name(p.type));
    }
#undef MMQ_CASE
}
}
struct MoEKernel::Workspace {
    ggml_backend_t backend = nullptr;
    void * activation = nullptr, *hidden = nullptr, *gathered = nullptr;
    float * gate_up = nullptr, *down = nullptr;
    int * inverse = nullptr, *identity = nullptr, *bounds = nullptr, *zero = nullptr;
    uint64_t *gate_table = nullptr, *down_table = nullptr;
    bool direct = false;
    ~Workspace() {
        for (void * pointer : {activation, hidden, gathered, static_cast<void *>(gate_up), static_cast<void *>(down),
            static_cast<void *>(inverse), static_cast<void *>(identity), static_cast<void *>(bounds), static_cast<void *>(zero),
            static_cast<void *>(gate_table), static_cast<void *>(down_table)}) { SOURIS_CUDA(cudaFree(pointer)); }
        ggml_backend_free(backend);
    }
};
MoEKernel::~MoEKernel() { delete workspace_; }
void MoEKernel::begin_mmq() {
    GGML_ASSERT(!workspace_->backend);
    auto & w = *workspace_;
    int device; SOURIS_CUDA(cudaGetDevice(&device)); w.backend = ggml_backend_cuda_init(device); GGML_ASSERT(w.backend);
    const auto & d = model_.dims(); const int routes = max_tokens * fixed::topk;
    size_t maximum_gate_bytes = 0;
    w.direct = std::all_of(std::begin(shapes_), std::end(shapes_),
        [](const MoeShape & shape) { return shape.gate_up_direct; });
    for (const auto & shape : shapes_) {
        GGML_ASSERT(shape.type[0] == shape.type[1] && shape.row_bytes[0] == shape.row_bytes[1]);
        maximum_gate_bytes = std::max(maximum_gate_bytes, 2 * shape.row_bytes[0] * d.expert_ff);
    }
    SOURIS_CUDA(cudaMalloc(&w.activation, qbytes(routes, d.width)));
    SOURIS_CUDA(cudaMalloc(&w.hidden, qbytes(routes, d.expert_ff)));
    SOURIS_CUDA(cudaMalloc(&w.gate_up, size_t(routes) * 2 * d.expert_ff * sizeof(float)));
    SOURIS_CUDA(cudaMalloc(&w.down, size_t(routes) * d.width * sizeof(float)));
    if (!w.direct) { SOURIS_CUDA(cudaMalloc(&w.gathered, max_group * maximum_gate_bytes + quant_padding)); }
    SOURIS_CUDA(cudaMalloc(&w.inverse, routes * sizeof(int)));
    SOURIS_CUDA(cudaMalloc(&w.identity, (routes + 128) * sizeof(int)));
    SOURIS_CUDA(cudaMalloc(&w.bounds, (max_group + 1) * sizeof(int)));
    SOURIS_CUDA(cudaMalloc(&w.zero, sizeof(int))); SOURIS_CUDA(cudaMemset(w.zero, 0, sizeof(int)));
    SOURIS_CUDA(cudaMalloc(&w.gate_table, max_group * sizeof(uint64_t)));
    SOURIS_CUDA(cudaMalloc(&w.down_table, max_group * sizeof(uint64_t)));
    SOURIS_CUDA(cudaDeviceGetAttribute(&sm_count_, cudaDevAttrMultiProcessorCount, device));
}
void MoEKernel::end_mmq() {
    delete workspace_;
    workspace_ = new Workspace;
}
void MoEKernel::init_mmq() {
    workspace_ = new Workspace;
    const auto & d = model_.dims();
    const int routes = max_tokens * fixed::topk;
    size_t gate_bytes = 0;
    for (const auto & shape : shapes_) {
        gate_bytes = std::max(gate_bytes, 2 * shape.row_bytes[0] * d.expert_ff);
    }
    const bool direct = std::all_of(std::begin(shapes_), std::end(shapes_),
        [](const MoeShape & shape) { return shape.gate_up_direct; });
    workspace_bytes_ = qbytes(routes, d.width) + qbytes(routes, d.expert_ff) +
        size_t(routes) * (2 * d.expert_ff + d.width) * sizeof(float) +
        (direct ? 0 : max_group * gate_bytes + quant_padding) +
        size_t(2 * routes + 128 + max_group + 2) * sizeof(int) + 2 * max_group * sizeof(uint64_t);
    SOURIS_CUDA(cudaDeviceGetAttribute(&sm_count_, cudaDevAttrMultiProcessorCount, 0));
    // GGML stream-K may retain one maximum-size fixup allocation in its pool.
    constexpr size_t max_tile = 128;
    constexpr size_t pool_granularity = 2 * 1024 * 1024;
    workspace_bytes_ += size_t(sm_count_) * max_tile * max_tile * sizeof(float) + pool_granularity;
    workspace_bytes_ = GGML_PAD(workspace_bytes_, pool_granularity);
}
void MoEKernel::run_mmq(const MoeOp & op, cudaStream_t stream) const {
    GGML_ASSERT(op.routes <= max_tokens * fixed::topk && op.end > op.begin && op.end - op.begin <= max_group);
    auto & w = *workspace_; const auto & shape = op.shape;
    const int tokens = op.routes / fixed::topk, experts = op.end - op.begin;
    // Every selected expert contributes at least one 64-row task. The
    // remaining tasks bound the largest expert without another host download.
    const int max_rows = std::min(tokens, (op.tasks - experts + 1) * moe_tile_rows);
    if (!op.begin) {
        prepare_rows<<<(op.routes + 255) / 256, 256, 0, stream>>>(op.group, w.inverse, w.identity, op.routes);
        quantize_scatter_mmq_q8_1_cuda(op.input, w.inverse, w.activation, shape.type[0], shape.width,
            shape.width, padded(shape.width), tokens, op.routes, fixed::topk, stream);
    }
    prepare_product<<<experts, 256, 0, stream>>>(op, w.gate_table, w.down_table, w.bounds, static_cast<char *>(w.gathered),
        2 * shape.hidden * shape.row_bytes[0], shape.gate_up_direct);
    auto & context = *static_cast<ggml_backend_cuda_context *>(w.backend->context);
    product(context, {shape.type[0], shape.width, 2 * shape.hidden, experts, op.routes, max_rows,
        w.activation, w.gate_table, w.zero, w.bounds, w.identity, w.gate_up}, stream);
    const int batch_rows = std::min(op.routes, experts * max_rows);
    swiglu_quant<<<dim3(batch_rows, padded(shape.hidden) / 512), 128, 0, stream>>>(w.gate_up,
        static_cast<block_q8_1_mmq *>(w.hidden), w.bounds, experts, op.routes, shape.hidden,
        padded(shape.hidden), mmq_get_q8_1_ds_layout(shape.type[2]));
    product(context, {shape.type[2], shape.hidden, shape.width, experts, op.routes, max_rows,
        w.hidden, w.down_table, w.zero, w.bounds, w.identity, w.down}, stream);
    weighted_scatter<<<batch_rows, 256, 0, stream>>>(op, w.down, w.bounds, experts);
    SOURIS_CUDA(cudaGetLastError());
}
void cuda_moe_group(const int32_t * ids, int32_t * group, int routes, cudaStream_t stream) {
    SOURIS_CUDA(cudaMemsetAsync(group, 0, fixed::experts * sizeof(int32_t), stream));
    group_count<<<(routes + group_threads - 1) / group_threads, group_threads, 0, stream>>>(ids, group, routes);
    group_scan<<<1, fixed::experts, 0, stream>>>(group);
    group_scatter<<<(routes + group_threads - 1) / group_threads, group_threads, 0, stream>>>(ids, group, routes);
    SOURIS_CUDA(cudaGetLastError());
}
void cuda_moe_select(const int32_t * group, int32_t * selected, cudaStream_t stream) {
    select_experts<<<1, fixed::experts, 0, stream>>>(group, selected);
    SOURIS_CUDA(cudaGetLastError());
}
}
