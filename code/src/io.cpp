#include "io.h"
#include "ggml-cuda.h"
#include <cstring>

namespace souris {
LayerIO::LayerIO(const Model & model,int count,Device & device,ggml_tensor * hidden_view,const LayerIO * storage) : rows(count) {
    const auto & d=model.dims();
    const size_t metadata = (hidden_view ? 34 : 34 + fixed::layers) + (!hidden_view && count == 1 ? 4 : 0);
    ctx=ggml_init({ggml_tensor_overhead()*metadata,nullptr,true}); GGML_ASSERT(ctx);
    if (hidden_view) {
        control=ggml_new_tensor_1d(ctx,GGML_TYPE_I8,sizeof(Control));
    } else {
        control_block = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, fixed::layers * sizeof(Control));
        for (int layer = 0; layer < fixed::layers; ++layer) {
            control_views[layer] = ggml_view_1d(ctx, control_block, sizeof(Control), size_t(layer) * sizeof(Control));
        }
        control = control_views[0];
        layer_id = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        miss_index = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    }
    tokens=ggml_new_tensor_1d(ctx,GGML_TYPE_I32,count);
    if (hidden_view) {
        hidden_leaf = *hidden_view;
        hidden_leaf.ne[2] = count; hidden_leaf.nb[3] = hidden_leaf.nb[2] * count;
        hidden = &hidden_leaf;
    } else {
        hidden = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d.width, d.hc, count);
    }
    if (!hidden_view && count == 1) {
        prepared_hidden = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d.width, d.hc, 1);
        prepared_mixed = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d.width, 1);
        prepared_inject = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d.hc, 1);
        prepared_q8 = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, size_t((d.width + 511) / 512 * 512) * 36 / 32);
    }
    residual=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,d.width,d.hc,count);
    inject=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,d.hc,count); expert_input=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,d.width,count);
    route_ids=ggml_new_tensor_2d(ctx,GGML_TYPE_I32,fixed::topk,count); route_weights=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,fixed::topk,count);
    native_expert_q8=ggml_new_tensor_1d(ctx,GGML_TYPE_I8,size_t((d.width+511)/512*512)*count*36/32);
    moe_group=ggml_new_tensor_1d(ctx,GGML_TYPE_I32,moe_group_header+2*count*fixed::topk);
    sample_params=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,5);
    sample_token=ggml_new_tensor_1d(ctx,GGML_TYPE_I32,1);
    decode_plan=ggml_new_tensor_1d(ctx,GGML_TYPE_I32,3+fixed::topk);
    decode_miss=ggml_new_tensor_1d(ctx,GGML_TYPE_I32,1);
    moe_selected=ggml_new_tensor_1d(ctx,GGML_TYPE_I32,fixed::experts);
    routed=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,d.width,count);
    expert_output=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,d.width,count*fixed::topk);
    logits=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,model.tensor("output.weight")->ne[1],hidden_view?1:count);
    ple=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,d.ple_dim*d.ple_heads,count);
    bytes=ggml_backend_alloc_ctx_tensors_from_buft_size(ctx,ggml_backend_cuda_buffer_type(0));
    if (storage) {
        // Chunk shapes borrow prefix rows from one maximum-size allocation.
        // Their metadata can expire after submission; device addresses stay fixed.
        GGML_ASSERT(count<=storage->rows);
        auto * source=ggml_get_first_tensor(storage->ctx);
        for (auto * tensor=ggml_get_first_tensor(ctx); tensor; tensor=ggml_get_next_tensor(ctx,tensor)) {
            GGML_ASSERT(source);
            tensor->data=source->data; tensor->buffer=source->buffer;
            source=ggml_get_next_tensor(storage->ctx,source);
        }
        GGML_ASSERT(!source);
    } else {
        buffer=ggml_backend_alloc_ctx_tensors(ctx,device.backend); GGML_ASSERT(buffer); bytes=ggml_backend_buffer_get_size(buffer); ggml_backend_buffer_clear(buffer,0);

    }
}
LayerIO::~LayerIO() { ggml_backend_buffer_free(buffer); ggml_free(ctx); }
}
