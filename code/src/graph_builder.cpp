#include "graph_builder.h"
#include "prefill_ops.h"
#include <cmath>
#include <cstdio>

namespace souris {
static ggml_tensor * mm(ggml_context * ctx, ggml_tensor * weight, ggml_tensor * input) { return ggml_mul_mat(ctx,weight,input); }
ggml_tensor * GraphBuilder::norm(Frame & frame, ggml_tensor * input, ggml_tensor * source) {
    auto * weight=frame.bind(source);
    if (ggml_nelements(weight)==model.dims().width*model.dims().hc) { weight=ggml_reshape_2d(frame.ctx,weight,model.dims().width,model.dims().hc); }
    return ggml_mul(frame.ctx,ggml_rms_norm(frame.ctx,input,model.dims().epsilon),weight);
}
ggml_tensor * GraphBuilder::mix(Frame & frame, ggml_tensor * input, int layer, const char * phase, ggml_tensor ** inject) {
    auto * ctx=frame.ctx; const auto & d=model.dims(); const int tokens=input->ne[2];
    auto weight=[&](const char * name) {
        const std::string prefix=layer<0 ? "output_hc_" : "blk."+std::to_string(layer)+".hc_"+phase+"_";
        return model.tensor(prefix+name+".weight");
    };
    auto * normalized=ggml_reshape_2d(ctx,norm(frame,input,weight("norm")),d.width*d.hc,tokens);
    auto * down = mm(ctx, frame.bind(weight("down")), normalized);
    auto * low=prefill_hc_down(frame,down,d.hc);
    auto * up=mm(ctx,frame.bind(weight("up")),low);
    auto * result=prefill_hc_up(frame,up,normalized,d.hc);
    if (inject) { *inject=mm(ctx,frame.bind(weight("inject")),normalized); }
    return result;
}
ggml_tensor * GraphBuilder::ple(Frame & frame, LayerIO & io, ggml_tensor * hidden) {
    auto * ctx=frame.ctx; const auto & d=model.dims(); const int layer=d.ple_layer, wide=d.width*d.hc, tokens=io.rows;
    auto * embedding=frame.bind(io.ple);
    auto * key=mm(ctx,frame.bind(model.layer(layer,"ple_key.weight")),embedding);
    auto * value=mm(ctx,frame.bind(model.layer(layer,"ple_value.weight")),embedding);
    key=norm(frame,ggml_reshape_3d(ctx,key,d.width,d.hc,tokens),model.layer(layer,"ple_norm_key.weight"));
    auto * query=norm(frame,hidden,model.layer(layer,"ple_norm_query.weight"));
    auto * score=ggml_scale(ctx,ggml_sum_rows(ctx,ggml_mul(ctx,key,query)),1.0f/std::sqrt(float(d.width)));
    auto * magnitude=ggml_sqrt(ctx,ggml_clamp(ctx,ggml_abs(ctx,score),1e-6f,INFINITY));
    auto * gate=ggml_sigmoid(ctx,ggml_mul(ctx,ggml_sgn(ctx,score),magnitude));
    auto * expanded=ggml_repeat_4d(ctx,ggml_reshape_3d(ctx,value,d.width,1,tokens),d.width,d.hc,tokens,1);
    auto * gated=ggml_mul(ctx,expanded,gate);
    auto * normalized=norm(frame,gated,model.layer(layer,"ple_norm_conv.weight"));
    normalized=ggml_reshape_2d(ctx,normalized,wide,tokens);
    const int history=(d.ple_kernel-1)*d.ple_ngrams;
    auto * cache=frame.bind(state.ple_history);
    auto * padded=ggml_concat(ctx,cache,ggml_transpose(ctx,normalized),0);
    auto * tail=ggml_view_2d(ctx,padded,history,wide,padded->nb[1],tokens*sizeof(float));
    frame.expand(ggml_cpy(ctx,ggml_cont(ctx,tail),cache));
    auto * kernel=frame.bind(model.layer(layer,"ple_conv1d.weight")); ggml_tensor * convolution=nullptr;
    for (int tap=0; tap<d.ple_kernel; ++tap) {
        const int start=history-(d.ple_kernel-1-tap)*d.ple_ngrams;
        auto * shifted=ggml_view_2d(ctx,padded,tokens,wide,padded->nb[1],start*sizeof(float));
        shifted=ggml_cont(ctx,ggml_transpose(ctx,shifted));
        auto * weight=ggml_cont(ctx,ggml_view_2d(ctx,kernel,1,wide,kernel->nb[1],tap*kernel->nb[0]));
        weight=ggml_reshape_1d(ctx,weight,wide); if (weight->type!=GGML_TYPE_F32) { weight=ggml_cast(ctx,weight,GGML_TYPE_F32); }
        auto * term=ggml_mul(ctx,shifted,weight); convolution=convolution ? ggml_add(ctx,convolution,term) : term;
    }
    convolution=ggml_reshape_3d(ctx,ggml_silu(ctx,convolution),d.width,d.hc,tokens);
    return ggml_add(ctx,hidden,ggml_add(ctx,gated,convolution));
}
ggml_tensor * GraphBuilder::gdn(Frame & frame, ggml_tensor * input, int layer) {
    auto * ctx=frame.ctx; const auto & d=model.dims(); const int tokens=input->ne[1], channels=d.state_dim*(2*d.key_heads+d.value_heads);
    auto weight=[&](const char * name) { return frame.bind(model.layer(layer,name)); };
    auto * qkv=mm(ctx,weight("attn_qkv.weight"),input); auto * z=mm(ctx,weight("attn_gate.weight"),input);
    auto * beta=ggml_sigmoid(ctx,mm(ctx,weight("ssm_beta.weight"),input)); beta=ggml_reshape_4d(ctx,beta,1,d.value_heads,tokens,1);
    auto * decay=ggml_add(ctx,mm(ctx,weight("ssm_alpha.weight"),input),weight("ssm_dt.bias"));
    decay=ggml_mul(ctx,ggml_softplus(ctx,decay),weight("ssm_a")); decay=ggml_reshape_4d(ctx,decay,1,d.value_heads,tokens,1);
    auto * conv_state=frame.bind(state.layers[layer].conv);
    auto * padded=ggml_concat(ctx,conv_state,ggml_transpose(ctx,qkv),0);
    auto * tail=ggml_view_2d(ctx,padded,d.conv_kernel-1,channels,padded->nb[1],tokens*sizeof(float));
    frame.expand(ggml_cpy(ctx,ggml_cont(ctx,tail),conv_state));
    auto * convolution=ggml_silu(ctx,ggml_ssm_conv(ctx,ggml_reshape_3d(ctx,padded,padded->ne[0],channels,1),weight("ssm_conv1d.weight")));
    const size_t row=channels*sizeof(float), head=d.state_dim*sizeof(float);
    auto * q=ggml_view_4d(ctx,convolution,d.state_dim,d.key_heads,tokens,1,head,row,row*tokens,0);
    auto * k=ggml_view_4d(ctx,convolution,d.state_dim,d.key_heads,tokens,1,head,row,row*tokens,head*d.key_heads);
    auto * v=ggml_view_4d(ctx,convolution,d.state_dim,d.value_heads,tokens,1,head,row,row*tokens,2*head*d.key_heads);
    q=ggml_scale(ctx,ggml_rms_norm(ctx,q,d.epsilon/d.state_dim),1.0f/std::sqrt(float(d.state_dim)));
    k=ggml_scale(ctx,ggml_rms_norm(ctx,k,d.epsilon/d.state_dim),1.0f/std::sqrt(float(d.state_dim)));
    auto * memory=frame.bind(state.layers[layer].recurrent);
    auto * result=ggml_gated_delta_net(ctx,q,k,v,decay,beta,memory,1);
    auto * output=ggml_view_4d(ctx,result,d.state_dim,d.value_heads,tokens,1,head,head*d.value_heads,head*d.value_heads*tokens,0);
    auto * next=ggml_view_4d(ctx,result,d.state_dim,d.state_dim,d.value_heads,1,head,head*d.state_dim,head*d.state_dim*d.value_heads,head*d.value_heads*tokens);
    frame.expand(ggml_cpy(ctx,next,memory));
    output=ggml_mul(ctx,ggml_rms_norm(ctx,output,d.epsilon),weight("ssm_norm.weight"));
    output=ggml_mul(ctx,output,ggml_sigmoid(ctx,ggml_reshape_4d(ctx,z,d.state_dim,d.value_heads,tokens,1)));
    return mm(ctx,weight("ssm_out.weight"),ggml_reshape_2d(ctx,output,d.state_dim*d.value_heads,tokens));
}
static ggml_tensor * rope(Frame & frame, const Dimensions & d, ggml_tensor * input, ggml_tensor * positions) {
    int sections[4]; for (int i=0; i<4; ++i) { sections[i]=d.rope_sections[i]; }
    return ggml_rope_multi(frame.ctx,input,positions,nullptr,d.rotary,sections,GGML_ROPE_TYPE_IMROPE,d.original_context,d.rope_base,1.0f,0.0f,1.0f,32.0f,1.0f);
}
ggml_tensor * GraphBuilder::qsa(Frame & frame, ggml_tensor * input, int layer, const AuxInputs & aux) {
    auto * ctx=frame.ctx; const auto & d=model.dims(); const int tokens=input->ne[1], pools=(state.capacity+d.pool-1)/d.pool;
    auto * raw=mm(ctx,frame.bind(model.layer(layer,"indexer.k_proj.weight")),input);
    auto * cache=frame.bind(state.layers[layer].index); frame.expand(ggml_set_rows(ctx,cache,raw,aux.writes));
    auto * rows=ggml_get_rows(ctx,cache,aux.reads);
    const int padding=pools*d.pool-state.capacity; if (padding) { rows=ggml_pad(ctx,rows,0,padding,0,0); }
    rows=ggml_reshape_3d(ctx,rows,d.index_dim,d.pool,pools); ggml_tensor * pooled=nullptr;
    for (int i=0; i<d.pool; ++i) {
        auto * part=ggml_view_2d(ctx,rows,d.index_dim,pools,rows->nb[2],i*rows->nb[1]);
        pooled=pooled ? ggml_add(ctx,pooled,part) : ggml_cont(ctx,part);
    }
    pooled=norm(frame,ggml_scale(ctx,pooled,1.0f/d.pool),model.layer(layer,"indexer.k_norm.weight"));
    pooled=rope(frame,d,ggml_reshape_3d(ctx,pooled,d.index_dim,1,pools),aux.pool_positions);
    auto * query=mm(ctx,frame.bind(model.layer(layer,"indexer.q_proj.weight")),input);
    query=ggml_reshape_3d(ctx,query,d.index_dim,d.index_heads,tokens);
    query=norm(frame,query,model.layer(layer,"indexer.q_norm.weight")); query=rope(frame,d,query,aux.positions);
    auto * weights=ggml_fill(ctx,ggml_new_tensor_2d(ctx,GGML_TYPE_F32,d.index_heads,tokens),1.0f/std::sqrt(float(d.index_dim)));
    auto * scores=ggml_lightning_indexer(ctx,query,ggml_cast(ctx,pooled,GGML_TYPE_F16),weights,aux.pool_mask);
    auto * selected=ggml_top_k(ctx,scores,std::min(pools,d.index_topk/d.pool));
    auto * block_ids=ggml_clamp(ctx,ggml_arange(ctx,0,float(pools*d.pool),1),0,float(state.capacity));
    block_ids=ggml_reshape_2d(ctx,ggml_cast(ctx,block_ids,GGML_TYPE_I32),d.pool,pools);
    auto * zeros=ggml_fill(ctx,ggml_new_tensor_1d(ctx,GGML_TYPE_F16,state.capacity),0.0f);
    return ggml_top_k_mask(ctx,zeros,selected,scores,block_ids,aux.tails);
}
ggml_tensor * GraphBuilder::attention(Frame & frame, LayerIO & io, ggml_tensor * input, int layer) {
    auto * ctx=frame.ctx; const auto & d=model.dims(); const int tokens=io.rows;
    const auto aux=auxiliary(frame,frame.bind(io.control),d,tokens,state.capacity);
    auto * mask=qsa(frame,input,layer,aux);
    ggml_tensor * full, * key, * value;
    const std::string prefix="blk."+std::to_string(layer)+".";
    if (auto * weight=model.optional(prefix+"attn_q.weight")) {
        full=mm(ctx,frame.bind(weight),input);
        key=mm(ctx,frame.bind(model.layer(layer,"attn_k.weight")),input);
        value=mm(ctx,frame.bind(model.layer(layer,"attn_v.weight")),input);
    } else {
        auto * qkv=mm(ctx,frame.bind(model.layer(layer,"attn_qkv.weight")),input);
        const int nq=2*d.head_dim*d.heads, nk=d.head_dim*d.kv_heads;
        full=ggml_view_2d(ctx,qkv,nq,tokens,qkv->nb[1],0);
        key=ggml_cont(ctx,ggml_view_2d(ctx,qkv,nk,tokens,qkv->nb[1],nq*sizeof(float)));
        value=ggml_cont(ctx,ggml_view_2d(ctx,qkv,nk,tokens,qkv->nb[1],(nq+nk)*sizeof(float)));
    }
    auto * query=ggml_view_3d(ctx,full,d.head_dim,d.heads,tokens,2*d.head_dim*sizeof(float),full->nb[1],0);
    auto * gate=ggml_view_3d(ctx,full,d.head_dim,d.heads,tokens,2*d.head_dim*sizeof(float),full->nb[1],d.head_dim*sizeof(float));
    gate=ggml_cont_2d(ctx,gate,d.head_dim*d.heads,tokens);
    query=norm(frame,query,model.layer(layer,"attn_q_norm.weight"));
    key=norm(frame,ggml_reshape_3d(ctx,key,d.head_dim,d.kv_heads,tokens),model.layer(layer,"attn_k_norm.weight"));
    query=rope(frame,d,query,aux.positions); key=rope(frame,d,key,aux.positions);
    // Quantize complete head blocks directly into the persistent Q8 cache.
    // Token rows are strided across heads; SET_ROWS preserves those strides.
    key = ggml_cont(ctx, ggml_permute(ctx, key, 0, 2, 1, 3));
    value = ggml_cont(ctx, ggml_permute(ctx,
        ggml_reshape_3d(ctx, value, d.head_dim, d.kv_heads, tokens), 0, 2, 1, 3));
    const auto cache = state.append(frame, layer, key, value, aux.writes);
    auto * result=ggml_flash_attn_ext(ctx,ggml_permute(ctx,query,0,2,1,3),cache.key,cache.value,mask,1.0f/std::sqrt(float(d.head_dim)),0,0);
    ggml_flash_attn_ext_set_n_kv_max(result,std::min(state.capacity,d.index_topk+d.pool-1));
    auto * device = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(state.layers[layer].key->buffer));
    if (!ggml_backend_dev_supports_op(device, result)) { GGML_ABORT("unsupported KV type: q8_0/q8_0 FlashAttention"); }
    result=ggml_reshape_2d(ctx,result,d.head_dim*d.heads,tokens);
    result=ggml_mul(ctx,result,ggml_sigmoid(ctx,gate)); return mm(ctx,frame.bind(model.layer(layer,"attn_output.weight")),result);
}
void GraphBuilder::before(Frame & frame, LayerIO & io, int layer) {
    auto * ctx=frame.ctx; const auto & d=model.dims();
    ggml_tensor * hidden;
    if (layer==0) {
        ggml_tensor * embedding = nullptr;
        if (io.images.empty()) {
            embedding = ggml_get_rows(ctx, frame.bind(model.tensor("token_embd.weight")), frame.bind(io.tokens));
        } else {
            auto append = [&](ggml_tensor * rows) { embedding = embedding ? ggml_concat(ctx, embedding, rows, 1) : rows; };
            auto text = [&](int start, int count) {
                if (!count) { return; }
                auto * ids = ggml_view_1d(ctx, frame.bind(io.tokens), count, size_t(start) * sizeof(Token));
                append(ggml_get_rows(ctx, frame.bind(model.tensor("token_embd.weight")), ids));
            };
            int cursor = 0;
            for (const auto & image : io.images) {
                text(cursor, image.start - cursor);
                auto * source = frame.bind(image.rows);
                append(ggml_view_2d(ctx, source, d.width, image.count, source->nb[1], size_t(image.source) * source->nb[1]));
                cursor = image.start + image.count;
            }
            text(cursor, io.rows - cursor);
        }
        hidden=ggml_repeat_4d(ctx,ggml_reshape_3d(ctx,embedding,d.width,1,io.rows),d.width,d.hc,io.rows,1);
    } else { hidden=frame.bind(io.hidden); if (hidden->type != GGML_TYPE_F32) { hidden=ggml_cast(ctx,hidden,GGML_TYPE_F32); } }
    if (layer==d.ple_layer) {
        hidden=ple(frame,io,hidden);
    }
    ggml_tensor * inject;
    auto * mixed=mix(frame,hidden,layer,"attn",&inject);
    auto * attention_out=d.recurrent[layer] ? gdn(frame,mixed,layer) : attention(frame,io,mixed,layer);
    auto * residual=prefill_combine(frame,hidden,attention_out,inject,io.residual);
    auto * expert_input=mix(frame,residual,layer,"ffn",&inject);
    frame.expand(residual);
    frame.expand(ggml_cpy(ctx,inject,frame.bind(io.inject)));
    frame.expand(ggml_cpy(ctx,expert_input,frame.bind(io.expert_input)));
    auto * scores=ggml_soft_max(ctx,mm(ctx,frame.bind(model.layer(layer,"ffn_gate_inp.weight")),expert_input));
    auto * ids=ggml_top_k(ctx,scores,fixed::topk);
    auto * weights=ggml_get_rows(ctx,ggml_reshape_3d(ctx,scores,1,fixed::experts,io.rows),ids);
    weights=ggml_reshape_2d(ctx,weights,fixed::topk,io.rows); weights=ggml_div(ctx,weights,ggml_sum_rows(ctx,weights));
    const float scale=model.number("qwen4exp.expert_weights_scale",1.0f); if (scale!=1.0f) { weights=ggml_scale(ctx,weights,scale); }
    frame.expand(ggml_cpy(ctx,ids,frame.bind(io.route_ids))); frame.expand(ggml_cpy(ctx,weights,frame.bind(io.route_weights)));
}
ggml_tensor * GraphBuilder::sum_experts(Frame & frame,ggml_tensor * values) {
    auto * ctx=frame.ctx; const int width=values->ne[0], used=values->ne[1], tokens=values->ne[2]; GGML_ASSERT(used<=fixed::topk);
    ggml_tensor * views[fixed::topk];
    // This canonical GGML view/add pattern also enables its CUDA weighted-MoE fusion.
    for (int i=0; i<used; ++i) {
        views[i]=ggml_view_2d(ctx,values,width,tokens,values->nb[2],i*values->nb[1]); ggml_build_forward_expand(frame.graph,views[i]);
    }
    auto * result=views[0];
    for (int i=1; i<used; ++i) { result=ggml_add(ctx,result,views[i]); ggml_build_forward_expand(frame.graph,result); }
    return result;
}

void GraphBuilder::head(Frame & frame, LayerIO & io, ggml_tensor * hidden) {
    auto * ctx=frame.ctx; const auto & d=model.dims();
    if (io.rows>1) { hidden=ggml_view_3d(ctx,hidden,d.width,d.hc,1,hidden->nb[1],hidden->nb[2],(io.rows-1)*hidden->nb[2]); }
    auto * output=mix(frame,hidden,-1,"",nullptr);
    frame.expand(ggml_cpy(ctx,mm(ctx,frame.bind(model.tensor("output.weight")),output),frame.bind(io.logits)));
}
void GraphBuilder::after(Frame & frame, LayerIO & io, int layer, bool output_head) {
    auto * ctx=frame.ctx; auto * input=frame.bind(io.expert_input);
    auto * packed=frame.bind(model.layer(layer,"ffn_gate_up_shexp.weight"));
    auto * shared=ggml_backend_cuda_prefill_shared(ctx,backend,packed,input);
    shared=mm(ctx,frame.bind(model.layer(layer,"ffn_down_shexp.weight")),shared);
    auto * shared_gate=mm(ctx,frame.bind(model.layer(layer,"ffn_gate_inp_shexp.weight")),input);
    auto * hidden=prefill_shared_combine(frame,frame.bind(io.residual),frame.bind(io.routed),shared,shared_gate,frame.bind(io.inject),io.hidden);
    frame.expand(hidden);
    if(output_head&&layer==fixed::layers-1)head(frame,io,hidden);
}
void GraphBuilder::build_layer(Frame & frame,LayerIO & io,int layer,Part part,bool output_head){
    if(part==Part::Before)before(frame,io,layer);else after(frame,io,layer,output_head);
}
}
