#include "ops.h"

namespace souris {
ggml_tensor * node_view(Frame & frame, ggml_tensor * storage, ggml_type type, int64_t n0, int64_t n1, size_t offset) {
    auto * tensor=ggml_new_tensor_2d(frame.ctx,type,n0,n1);
    tensor->op=GGML_OP_VIEW; tensor->src[0]=storage; tensor->view_src=storage; tensor->view_offs=offset; return tensor;
}
AuxInputs auxiliary(Frame & frame, ggml_tensor * control, const Dimensions & dim, int tokens, int capacity) {
    auto & op=frame.add_op(); op.tokens=tokens; op.capacity=capacity; op.pool=dim.pool;
    const int pools=(capacity+dim.pool-1)/dim.pool; size_t bytes=0;
    op.positions=bytes; bytes+=4*tokens*sizeof(int32_t);
    bytes=(bytes+sizeof(int64_t)-1)/sizeof(int64_t)*sizeof(int64_t); op.write_ids=bytes; bytes+=tokens*sizeof(int64_t);
    op.read_ids=bytes; bytes+=capacity*sizeof(int32_t); op.pool_mask=bytes; bytes+=pools*tokens*sizeof(ggml_fp16_t);
    bytes=(bytes+sizeof(int32_t)-1)/sizeof(int32_t)*sizeof(int32_t); op.tail_ids=bytes; bytes+=(dim.pool-1)*tokens*sizeof(int32_t);
    op.pool_positions=bytes; bytes+=4*pools*sizeof(int32_t);
    ggml_tensor * args[]={control};
    auto * node=ggml_backend_cuda_custom_4d(frame.ctx,GGML_TYPE_I8,bytes,1,1,1,args,1,cuda_auxiliary,&op);
    return {node_view(frame,node,GGML_TYPE_I32,4*tokens,1,op.positions),
        node_view(frame,node,GGML_TYPE_I64,tokens,1,op.write_ids), node_view(frame,node,GGML_TYPE_I32,capacity,1,op.read_ids),
        node_view(frame,node,GGML_TYPE_F16,pools,tokens,op.pool_mask), node_view(frame,node,GGML_TYPE_I32,dim.pool-1,tokens,op.tail_ids),
        node_view(frame,node,GGML_TYPE_I32,4*pools,1,op.pool_positions)};
}
}
