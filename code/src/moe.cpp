#include "moe.h"
#include "io.h"
#include "ops.h"

namespace souris {
MoEKernel::MoEKernel(const Model & model) : model_(model) {
    for (int layer = 0; layer < fixed::layers; ++layer) {
        auto & shape = shapes_[layer];
        shape.width = model.dims().width;
        shape.hidden = model.dims().expert_ff;
        for (int role = 0; role < 3; ++role) {
            auto * matrix = model.expert(layer, role);
            shape.type[role] = matrix->type;
            shape.row_bytes[role] = quant_traits(matrix->type).row_bytes(matrix->ne[0]);
            GGML_ASSERT(matrix->ne[0] % moe_quant_block == 0);
        }
        if (shape.type[0] != shape.type[1] || shape.row_bytes[0] != shape.row_bytes[1]) {
            GGML_ABORT("gate/up quantization and row layout must match");
        }
        shape.record_bytes = model.store().layout().record_bytes;
        shape.expert_bytes = shape.record_bytes;
        shape.quant_signature = model.store().layout().quant_signature(layer);
        const auto & layout = model.store().layout();
        shape.gate_up_direct = layout.role_offset[1] == layout.role_offset[0] +
            shape.row_bytes[0] * shape.hidden;
    }
    init_mmq();
}
MoeOp MoEKernel::parameters(LayerIO & io, const void * table, int layer,
                            int begin, int end, int tasks) const {
    return {shapes_[layer], static_cast<const uint64_t *>(table), static_cast<const int32_t *>(io.control->data),
        static_cast<const int32_t *>(io.moe_group->data),
        static_cast<const int32_t *>(io.moe_selected->data),
        static_cast<const float *>(io.expert_input->data), static_cast<const float *>(io.route_weights->data),
        static_cast<float *>(io.expert_output->data),
        io.rows * fixed::topk, begin, end, tasks};
}
void MoEKernel::native_expert(Frame & frame, LayerIO & io, ggml_tensor * table, ggml_backend_t backend,
                       int layer, int count, bool miss) const {
    frame.key = {-1, 1, count, model_.store().layout().quant_signature(layer)};
    auto * ids = frame.bind(miss ? io.decode_miss : io.moe_selected);
    if (miss) {
        ggml_tensor * args[] = {frame.bind(io.decode_plan), frame.bind(io.miss_index)};
        frame.expand(ggml_backend_cuda_custom_inplace(frame.ctx, ids, args, 2, cuda_decode_select, nullptr));
    }
    ids = ggml_view_2d(frame.ctx, ids, count, 1, count * sizeof(int32_t), 0);
    auto * layer_id = frame.bind(io.layer_id);
    ggml_tensor * pointers[3];
    ggml_tensor * weights[3];
    for (int role = 0; role < 3; ++role) {
        pointers[role] = ggml_view_2d(frame.ctx, frame.bind(table), fixed::experts, fixed::layers,
                                      3 * fixed::experts * sizeof(void *), role * fixed::experts * sizeof(void *));
        weights[role] = frame.bind(model_.expert(layer, role));
        // Native MMVQ follows ExpertTable; this descriptor supplies shape/type only.
        weights[role]->data = pointers[role]->data;
        weights[role]->buffer = table->buffer;
    }
    auto * input = ggml_reshape_3d(frame.ctx, frame.bind(io.expert_input), shapes_[layer].width, 1, 1);
    GGML_ASSERT(weights[0]->type == weights[1]->type && ggml_are_same_stride(weights[0], weights[1]));
    auto * hidden = ggml_backend_cuda_native_expert(frame.ctx, backend, weights[1], input, frame.bind(io.native_expert_q8), ids,
                                                  pointers[1], layer_id, pointers[0]);
    auto * quantized = ggml_backend_cuda_native_quantize(frame.ctx, backend, hidden);
    auto * output = ggml_backend_cuda_native_expert(frame.ctx, backend, weights[2], hidden, quantized, ids,
                                                  pointers[2], layer_id, nullptr);
    ggml_tensor * args[] = {output, ids, frame.bind(io.moe_group), frame.bind(io.route_weights)};
    frame.expand(ggml_backend_cuda_custom_inplace(frame.ctx, frame.bind(io.expert_output), args, 4,
                                                  cuda_decode_scatter, nullptr));
}
void MoEKernel::group(LayerIO & io, cudaStream_t stream) const {
    cuda_moe_group(static_cast<const int32_t *>(io.route_ids->data), static_cast<int32_t *>(io.moe_group->data),
                   io.rows * fixed::topk, stream);
}
void MoEKernel::select(LayerIO & io, cudaStream_t stream) const {
    cuda_moe_select(static_cast<const int32_t *>(io.moe_group->data), static_cast<int32_t *>(io.moe_selected->data), stream);
}
void MoEKernel::prefill(LayerIO & io, const void * table, int layer, int begin, int end, int tasks, cudaStream_t stream) const {
    run_mmq(parameters(io, table, layer, begin, end, tasks), stream);
}
}
