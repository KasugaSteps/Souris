#include "native_layer.h"
#include <cmath>
#include <cstdio>

namespace souris {
ggml_tensor *NativeLayerPlan::quantized(Frame &f, ggml_tensor *input, ggml_tensor *target) {
    ggml_tensor *root = input;
    while (root->view_src && !root->view_offs && ggml_is_contiguous(root))
        root = root->view_src;
    for (auto &entry : activations_) {
        ggml_tensor *origin = entry.value;
        while (origin->view_src && !origin->view_offs && ggml_is_contiguous(origin))
            origin = origin->view_src;
        if (origin != root)
            continue;
        if (entry.op) {
            GGML_ASSERT(!entry.op->quant_columns || entry.op->quant_columns == input->ne[0]);
            entry.op->quant_columns = input->ne[0];
            if (target) {
                entry.q8 = f.bind(target);
                entry.value->src[entry.q8_arg] = entry.q8;
            }
        }
        return entry.q8;
    }
    auto *q8 = ggml_backend_cuda_native_quantize(f.ctx, backend, input);
    GGML_ASSERT(!target);
    activations_.push_back({input, q8, nullptr, 0});
    return q8;
}
ggml_tensor *NativeLayerPlan::matmul(Frame &f, ggml_tensor *weight, ggml_tensor *input) {
    ggml_tensor *q8 = ggml_is_quantized(weight->type) ? quantized(f, input) : nullptr;
    return ggml_backend_cuda_native_matmul(f.ctx, backend, weight, input, q8);
}
ggml_tensor *NativeLayerPlan::producer(Frame &f, ggml_tensor *input, ggml_tensor *normalized, int mode) {
    const int width = mode == 1 ? input->ne[0] / model.dims().hc : mode == 2 ? input->ne[0] / 2 : input->ne[0];
    auto *q8 = ggml_new_tensor_1d(f.ctx, GGML_TYPE_I8, size_t((width + 511) / 512 * 512) * input->ne[1] * 36 / 32);
    metadata_.push_back({0, 1.0f / model.dims().hc, model.dims().hc, 0});
    auto *op = &metadata_.back();
    ggml_tensor *args[] = {input, normalized, q8};
    ggml_cuda_custom_op_t callback = mode == 0   ? cuda_native_hc_down
                                     : mode == 1 ? cuda_native_hc_up
                                     : mode == 2 ? cuda_native_swiglu
                                                 : cuda_native_gate;
    if (mode == 0 || mode == 2) {
        args[1] = q8;
    }
    auto *value = ggml_backend_cuda_custom_4d(f.ctx, GGML_TYPE_F32, width, input->ne[1], 1, 1, args,
                                              (mode == 1 || mode == 3) ? 3 : 2, callback, op);
    activations_.push_back({value, q8, op, (mode == 1 || mode == 3) ? 2 : 1});
    return value;
}
ggml_tensor *NativeLayerPlan::norm(Frame &f, ggml_tensor *input, ggml_tensor *source, ggml_tensor *gate, float epsilon,
                                   float scale) {
    auto *weight = source ? f.bind(source) : input;
    const int width = input->ne[0];
    auto *q8 = ggml_new_tensor_1d(f.ctx, GGML_TYPE_I8,
                                  size_t((width + 511) / 512 * 512) * ggml_nelements(input) / width * 36 / 32);
    metadata_.push_back({epsilon < 0 ? model.dims().epsilon : epsilon, scale, 1, 0, source != nullptr, gate != nullptr});
    auto *op = &metadata_.back();
    ggml_tensor *args[] = {input, weight, q8, gate ? gate : input};
    auto *value = ggml_backend_cuda_custom_4d(f.ctx, GGML_TYPE_F32, input->ne[0], input->ne[1], input->ne[2],
                                              input->ne[3], args, 4, cuda_native_norm, op);
    activations_.push_back({value, q8, op, 2});
    return value;
}

ggml_tensor *NativeLayerPlan::mix(Frame &frame, ggml_tensor *input, int layer, const char *phase,
                                  ggml_tensor **inject) {
    auto *ctx = frame.ctx;
    const auto &d = model.dims();
    const int tokens = input->ne[2];
    auto weight = [&](const char *name) {
        const std::string prefix = layer < 0 ? "output_hc_" : "blk." + std::to_string(layer) + ".hc_" + phase + "_";
        return model.tensor(prefix + name + ".weight");
    };
    auto *normalized = ggml_reshape_2d(ctx, norm(frame, input, weight("norm")), d.width * d.hc, tokens);
    auto *down = matmul(frame, frame.bind(weight("down")), normalized);
    auto *low = producer(frame, down, nullptr, 0);
    auto *up = matmul(frame, frame.bind(weight("up")), low);
    auto *result = producer(frame, up, normalized, 1);
    if (inject) {
        *inject = matmul(frame, frame.bind(weight("inject")), normalized);
    }
    return result;
}
ggml_tensor *NativeLayerPlan::combine(Frame &frame, ggml_tensor *residual, ggml_tensor *output, ggml_tensor *inject,
                                      ggml_tensor *target) {
    ggml_tensor *args[] = {residual, output, inject};
    return ggml_backend_cuda_custom_inplace(frame.ctx, frame.bind(target), args, 3, cuda_native_combine, nullptr);
}

ggml_tensor *NativeLayerPlan::ple(Frame &frame, ggml_tensor *source, ggml_tensor *hidden, int tokens,
                                  ggml_tensor *replay) {
    auto *ctx = frame.ctx;
    const auto &d = model.dims();
    const int layer = d.ple_layer, wide = d.width * d.hc;
    auto *embedding = frame.bind(source);
    auto *key = matmul(frame, frame.bind(model.layer(layer, "ple_key.weight")), embedding);
    auto *value = matmul(frame, frame.bind(model.layer(layer, "ple_value.weight")), embedding);
    key = norm(frame, ggml_reshape_3d(ctx, key, d.width, d.hc, tokens), model.layer(layer, "ple_norm_key.weight"));
    auto *query = norm(frame, hidden, model.layer(layer, "ple_norm_query.weight"));
    auto *score = ggml_scale(ctx, ggml_sum_rows(ctx, ggml_mul(ctx, key, query)), 1.0f / std::sqrt(float(d.width)));
    auto *magnitude = ggml_sqrt(ctx, ggml_clamp(ctx, ggml_abs(ctx, score), 1e-6f, INFINITY));
    auto *gate = ggml_sigmoid(ctx, ggml_mul(ctx, ggml_sgn(ctx, score), magnitude));
    auto *expanded = ggml_repeat_4d(ctx, ggml_reshape_3d(ctx, value, d.width, 1, tokens), d.width, d.hc, tokens, 1);
    auto *gated = ggml_mul(ctx, expanded, gate);
    auto *normalized = norm(frame, gated, model.layer(layer, "ple_norm_conv.weight"));
    if (replay) {
        normalized->data = replay->data;
        normalized->buffer = replay->buffer;
    }
    normalized = ggml_reshape_2d(ctx, normalized, wide, tokens);
    const int history = (d.ple_kernel - 1) * d.ple_ngrams;
    auto *cache = frame.bind(state.ple_history);
    auto *padded = ggml_concat(ctx, cache, ggml_transpose(ctx, normalized), 0);
    auto *tail = ggml_view_2d(ctx, padded, history, wide, padded->nb[1], tokens * sizeof(float));
    frame.expand(ggml_cpy(ctx, ggml_cont(ctx, tail), cache));
    auto *kernel = frame.bind(model.layer(layer, "ple_conv1d.weight"));
    ggml_tensor *convolution = nullptr;
    for (int tap = 0; tap < d.ple_kernel; ++tap) {
        const int start = history - (d.ple_kernel - 1 - tap) * d.ple_ngrams;
        auto *shifted = ggml_view_2d(ctx, padded, tokens, wide, padded->nb[1], start * sizeof(float));
        shifted = ggml_cont(ctx, ggml_transpose(ctx, shifted));
        auto *weight = ggml_cont(ctx, ggml_view_2d(ctx, kernel, 1, wide, kernel->nb[1], tap * kernel->nb[0]));
        weight = ggml_reshape_1d(ctx, weight, wide);
        if (weight->type != GGML_TYPE_F32) {
            weight = ggml_cast(ctx, weight, GGML_TYPE_F32);
        }
        auto *term = ggml_mul(ctx, shifted, weight);
        convolution = convolution ? ggml_add(ctx, convolution, term) : term;
    }
    convolution = ggml_reshape_3d(ctx, ggml_silu(ctx, convolution), d.width, d.hc, tokens);
    return ggml_add(ctx, hidden, ggml_add(ctx, gated, convolution));
}
ggml_tensor *NativeLayerPlan::gdn(Frame &frame, ggml_tensor *input, int layer, NativeGdnReplay *replay) {
    auto *ctx = frame.ctx;
    const auto &d = model.dims();
    const int tokens = input->ne[1], channels = d.state_dim * (2 * d.key_heads + d.value_heads);
    auto weight = [&](const char *name) { return frame.bind(model.layer(layer, name)); };
    auto *qkv = matmul(frame, weight("attn_qkv.weight"), input);
    auto *z = matmul(frame, weight("attn_gate.weight"), input);
    if (replay) {
        qkv->data = replay->raw->data;
        qkv->buffer = replay->raw->buffer;
    }
    auto *beta_input = matmul(frame, weight("ssm_beta.weight"), input);
    auto *alpha = matmul(frame, weight("ssm_alpha.weight"), input);
    auto *beta = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d.value_heads, tokens);
    ggml_tensor *gate_args[] = {alpha, weight("ssm_dt.bias"), weight("ssm_a"), beta_input, beta};
    auto *decay = ggml_backend_cuda_custom_4d(ctx, GGML_TYPE_F32, d.value_heads, tokens, 1, 1, gate_args, 5,
                                              cuda_native_gdn_gates, nullptr);
    if (replay) {
        decay->data = replay->decay->data;
        decay->buffer = replay->decay->buffer;
        beta->data = replay->beta->data;
        beta->buffer = replay->beta->buffer;
    }
    decay = ggml_reshape_4d(ctx, decay, 1, d.value_heads, tokens, 1);
    beta = ggml_reshape_4d(ctx, beta, 1, d.value_heads, tokens, 1);
    auto *conv_state = frame.bind(state.layers[layer].conv);
    auto *padded = ggml_concat(ctx, conv_state, ggml_transpose(ctx, qkv), 0);
    auto *tail = ggml_view_2d(ctx, padded, d.conv_kernel - 1, channels, padded->nb[1], tokens * sizeof(float));
    frame.expand(ggml_cpy(ctx, ggml_cont(ctx, tail), conv_state));
    auto *convolution = ggml_silu(
        ctx, ggml_ssm_conv(ctx, ggml_reshape_3d(ctx, padded, padded->ne[0], channels, 1), weight("ssm_conv1d.weight")));
    if (replay) {
        convolution->data = replay->conv->data;
        convolution->buffer = replay->conv->buffer;
    }
    const size_t row = channels * sizeof(float), head = d.state_dim * sizeof(float);
    auto *q = ggml_view_4d(ctx, convolution, d.state_dim, d.key_heads, tokens, 1, head, row, row * tokens, 0);
    auto *k = ggml_view_4d(ctx, convolution, d.state_dim, d.key_heads, tokens, 1, head, row, row * tokens,
                           head * d.key_heads);
    auto *v = ggml_view_4d(ctx, convolution, d.state_dim, d.value_heads, tokens, 1, head, row, row * tokens,
                           2 * head * d.key_heads);
    q = norm(frame, q, nullptr, nullptr, d.epsilon / d.state_dim, 1.0f / std::sqrt(float(d.state_dim)));
    k = norm(frame, k, nullptr, nullptr, d.epsilon / d.state_dim, 1.0f / std::sqrt(float(d.state_dim)));
    if (replay) {
        q->data = replay->q->data;
        q->buffer = replay->q->buffer;
        k->data = replay->k->data;
        k->buffer = replay->k->buffer;
    }
    auto *memory = frame.bind(state.layers[layer].recurrent);
    ggml_tensor *output = nullptr;
    // State transitions remain sequential. Projections and normalization are
    // multi-row, while the recurrence preserves the M=1 reduction order.
    for (int row_index = 0; row_index < tokens; ++row_index) {
        auto row_view = [&](ggml_tensor *value) {
            return ggml_view_4d(ctx, value, value->ne[0], value->ne[1], 1, 1, value->nb[1], value->nb[2], value->nb[3],
                                row_index * value->nb[2]);
        };
        auto *result = ggml_gated_delta_net(ctx, row_view(q), row_view(k), row_view(v), row_view(decay), row_view(beta),
                                            memory, 1);
        auto *current = ggml_view_4d(ctx, result, d.state_dim, d.value_heads, 1, 1, head, head * d.value_heads,
                                     head * d.value_heads, 0);
        auto *next = ggml_view_4d(ctx, result, d.state_dim, d.state_dim, d.value_heads, 1, head, head * d.state_dim,
                                  head * d.state_dim * d.value_heads, head * d.value_heads);
        memory = ggml_cpy(ctx, next, memory);
        frame.expand(memory);
        output = output ? ggml_concat(ctx, output, current, 2) : current;
    }
    output = norm(frame, output, model.layer(layer, "ssm_norm.weight"), z);
    return matmul(frame, weight("ssm_out.weight"), ggml_reshape_2d(ctx, output, d.state_dim * d.value_heads, tokens));
}
static ggml_tensor *rope(Frame &frame, const Dimensions &d, ggml_tensor *input, ggml_tensor *positions) {
    int sections[4];
    for (int i = 0; i < 4; ++i) {
        sections[i] = d.rope_sections[i];
    }
    return ggml_rope_multi(frame.ctx, input, positions, nullptr, d.rotary, sections, GGML_ROPE_TYPE_IMROPE,
                           d.original_context, d.rope_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
}
ggml_tensor *NativeLayerPlan::qsa(Frame &frame, ggml_tensor *input, int layer, const AuxInputs &aux) {
    auto *ctx = frame.ctx;
    const auto &d = model.dims();
    const int tokens = input->ne[1], pools = (state.capacity + d.pool - 1) / d.pool;
    auto *raw = matmul(frame, frame.bind(model.layer(layer, "indexer.k_proj.weight")), input);
    auto *cache = frame.bind(state.layers[layer].index);
    frame.expand(ggml_set_rows(ctx, cache, raw, aux.writes));
    auto *rows = ggml_get_rows(ctx, cache, aux.reads);
    const int padding = pools * d.pool - state.capacity;
    if (padding) {
        rows = ggml_pad(ctx, rows, 0, padding, 0, 0);
    }
    rows = ggml_reshape_3d(ctx, rows, d.index_dim, d.pool, pools);
    ggml_tensor *pooled = nullptr;
    for (int i = 0; i < d.pool; ++i) {
        auto *part = ggml_view_2d(ctx, rows, d.index_dim, pools, rows->nb[2], i * rows->nb[1]);
        pooled = pooled ? ggml_add(ctx, pooled, part) : ggml_cont(ctx, part);
    }
    pooled = norm(frame, ggml_scale(ctx, pooled, 1.0f / d.pool), model.layer(layer, "indexer.k_norm.weight"));
    pooled = rope(frame, d, ggml_reshape_3d(ctx, pooled, d.index_dim, 1, pools), aux.pool_positions);
    auto *query = matmul(frame, frame.bind(model.layer(layer, "indexer.q_proj.weight")), input);
    query = ggml_reshape_3d(ctx, query, d.index_dim, d.index_heads, tokens);
    query = norm(frame, query, model.layer(layer, "indexer.q_norm.weight"));
    query = rope(frame, d, query, aux.positions);
    auto *weights = ggml_fill(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d.index_heads, tokens),
                              1.0f / std::sqrt(float(d.index_dim)));
    auto *scores = ggml_lightning_indexer(ctx, query, ggml_cast(ctx, pooled, GGML_TYPE_F16), weights, aux.pool_mask);
    auto *selected = ggml_top_k(ctx, scores, std::min(pools, d.index_topk / d.pool));
    auto *block_ids = ggml_clamp(ctx, ggml_arange(ctx, 0, float(pools * d.pool), 1), 0, float(state.capacity));
    block_ids = ggml_reshape_2d(ctx, ggml_cast(ctx, block_ids, GGML_TYPE_I32), d.pool, pools);
    auto *zeros = ggml_fill(ctx, ggml_new_tensor_1d(ctx, GGML_TYPE_F16, state.capacity), 0.0f);
    return ggml_top_k_mask(ctx, zeros, selected, scores, block_ids, aux.tails);
}
ggml_tensor *NativeLayerPlan::attention(Frame &frame, LayerIO &io, ggml_tensor *input, int layer) {
    auto *ctx = frame.ctx;
    const auto &d = model.dims();
    const int tokens = io.rows;
    const auto aux = auxiliary(frame, frame.bind(io.control), d, tokens, state.capacity);
    auto *mask = qsa(frame, input, layer, aux);
    ggml_tensor *full, *key, *value;
    const std::string prefix = "blk." + std::to_string(layer) + ".";
    if (auto *weight = model.optional(prefix + "attn_q.weight")) {
        full = matmul(frame, frame.bind(weight), input);
        key = matmul(frame, frame.bind(model.layer(layer, "attn_k.weight")), input);
        value = matmul(frame, frame.bind(model.layer(layer, "attn_v.weight")), input);
    } else {
        auto *qkv = matmul(frame, frame.bind(model.layer(layer, "attn_qkv.weight")), input);
        const int nq = 2 * d.head_dim * d.heads, nk = d.head_dim * d.kv_heads;
        full = ggml_view_2d(ctx, qkv, nq, tokens, qkv->nb[1], 0);
        key = ggml_cont(ctx, ggml_view_2d(ctx, qkv, nk, tokens, qkv->nb[1], nq * sizeof(float)));
        value = ggml_cont(ctx, ggml_view_2d(ctx, qkv, nk, tokens, qkv->nb[1], (nq + nk) * sizeof(float)));
    }
    auto *query = ggml_view_3d(ctx, full, d.head_dim, d.heads, tokens, 2 * d.head_dim * sizeof(float), full->nb[1], 0);
    auto *gate = ggml_view_3d(ctx, full, d.head_dim, d.heads, tokens, 2 * d.head_dim * sizeof(float), full->nb[1],
                              d.head_dim * sizeof(float));
    gate = ggml_cont_2d(ctx, gate, d.head_dim * d.heads, tokens);
    query = norm(frame, query, model.layer(layer, "attn_q_norm.weight"));
    key = norm(frame, ggml_reshape_3d(ctx, key, d.head_dim, d.kv_heads, tokens),
               model.layer(layer, "attn_k_norm.weight"));
    query = rope(frame, d, query, aux.positions);
    key = rope(frame, d, key, aux.positions);
    // Quantize complete head blocks directly into the persistent Q8 cache.
    // Token rows are strided across heads; SET_ROWS preserves those strides.
    key = ggml_cont(ctx, ggml_permute(ctx, key, 0, 2, 1, 3));
    value = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, value, d.head_dim, d.kv_heads, tokens), 0, 2, 1, 3));
    const auto cache = state.append(frame, layer, key, value, aux.writes);
    auto *result = ggml_flash_attn_ext(ctx, ggml_permute(ctx, query, 0, 2, 1, 3), cache.key, cache.value, mask,
                                       1.0f / std::sqrt(float(d.head_dim)), 0, 0);
    auto *device = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(state.layers[layer].key->buffer));
    if (!ggml_backend_dev_supports_op(device, result)) {
        GGML_ABORT("unsupported KV type: q8_0/q8_0 FlashAttention");
    }
    result = ggml_reshape_2d(ctx, result, d.head_dim * d.heads, tokens);
    result = producer(frame, result, gate, 3);
    return matmul(frame, frame.bind(model.layer(layer, "attn_output.weight")), result);
}
void NativeLayerPlan::prepare_decode(Frame &frame, LayerIO &io) {
    activations_.clear();
    const auto &d = model.dims();
    frame.key = {0, 1, 0, model.store().layout().quant_signature(0)};
    auto *embedding = ggml_get_rows(frame.ctx, frame.bind(model.tensor("token_embd.weight")), frame.bind(io.tokens));
    auto *hidden = ggml_repeat_4d(frame.ctx, ggml_reshape_3d(frame.ctx, embedding, d.width, 1, 1), d.width, d.hc, 1, 1);
    ggml_tensor *inject;
    auto *mixed = mix(frame, hidden, 0, "attn", &inject);
    quantized(frame, mixed, io.prepared_q8);
    auto store = [&](ggml_tensor *value, ggml_tensor *target) {
        value->data = target->data; value->buffer = target->buffer; frame.expand(value);
    };
    store(hidden, io.prepared_hidden);
    store(mixed, io.prepared_mixed);
    store(inject, io.prepared_inject);
}
void NativeLayerPlan::before(Frame &frame, LayerIO &io, int layer) {
    auto *ctx = frame.ctx;
    const auto &d = model.dims();
    ggml_tensor *hidden;
    if (layer == 0 && io.prepared_hidden) {
        hidden = frame.bind(io.prepared_hidden);
    } else if (layer == 0) {
        auto *embedding = ggml_get_rows(ctx, frame.bind(model.tensor("token_embd.weight")), frame.bind(io.tokens));
        hidden = ggml_repeat_4d(ctx, ggml_reshape_3d(ctx, embedding, d.width, 1, io.rows), d.width, d.hc, io.rows, 1);
    } else {
        hidden = frame.bind(io.hidden);
        if (hidden->type != GGML_TYPE_F32) {
            hidden = ggml_cast(ctx, hidden, GGML_TYPE_F32);
        }
    }
    if (layer == d.ple_layer) {
        hidden = ple(frame, io.ple, hidden, io.rows, io.ple_input);
    }
    ggml_tensor *inject;
    ggml_tensor *mixed;
    if (layer == 0 && io.prepared_hidden) {
        mixed = frame.bind(io.prepared_mixed);
        activations_.push_back({mixed, frame.bind(io.prepared_q8), nullptr, 0});
        inject = frame.bind(io.prepared_inject);
    } else {
        mixed = mix(frame, hidden, layer, "attn", &inject);
    }
    auto *attention_out =
        d.recurrent[layer] ? gdn(frame, mixed, layer, io.gdn_replay) : attention(frame, io, mixed, layer);
    auto *residual = combine(frame, hidden, attention_out, inject, io.residual);
    auto *expert_input = mix(frame, residual, layer, "ffn", &inject);
    frame.expand(residual);
    inject->data = io.inject->data;
    inject->buffer = io.inject->buffer;
    frame.expand(inject);
    expert_input->data = io.expert_input->data;
    expert_input->buffer = io.expert_input->buffer;
    frame.expand(expert_input);
    quantized(frame, expert_input, io.native_expert_q8);
    auto *scores = matmul(frame, frame.bind(model.layer(layer, "ffn_gate_inp.weight")), expert_input);
    router_scales_.push_back(model.number("qwen4exp.expert_weights_scale", 1.0f));
    ggml_tensor *args[] = {scores, frame.bind(io.route_weights)};
    frame.expand(ggml_backend_cuda_custom_inplace(ctx, frame.bind(io.route_ids), args, 2, cuda_native_router,
                                                  &router_scales_.back()));
}
void NativeLayerPlan::replay_gdn(Frame &frame, const NativeGdnReplay &replay, ggml_tensor *control, int layer,
                                 int accepted) {
    auto *ctx = frame.ctx;
    const auto &d = model.dims();
    const size_t head = d.state_dim * sizeof(float);
    const int channels = d.state_dim * (2 * d.key_heads + d.value_heads);
    histories_.push_back({channels, d.conv_kernel - 1});
    ggml_tensor *args[] = {frame.bind(replay.raw), frame.bind(control)};
    frame.expand(ggml_backend_cuda_custom_inplace(ctx, frame.bind(state.layers[layer].conv), args, 2,
                                                  cuda_native_history_commit, &histories_.back()));
    auto *memory = frame.bind(state.layers[layer].recurrent);
    for (int row = 0; row < accepted; ++row) {
        auto view = [&](ggml_tensor *tensor, int width, int heads) {
            return ggml_view_4d(ctx, frame.bind(tensor), width, heads, 1, 1, width * sizeof(float),
                                width * heads * sizeof(float), width * heads * sizeof(float),
                                size_t(row) * width * heads * sizeof(float));
        };
        auto *v =
            ggml_view_4d(ctx, frame.bind(replay.conv), d.state_dim, d.value_heads, 1, 1, head, channels * sizeof(float),
                         channels * sizeof(float), size_t(row) * channels * sizeof(float) + 2 * head * d.key_heads);
        auto *result = ggml_gated_delta_net(
            ctx, view(replay.q, d.state_dim, d.key_heads), view(replay.k, d.state_dim, d.key_heads), v,
            view(replay.decay, 1, d.value_heads), view(replay.beta, 1, d.value_heads), memory, 1);
        auto *next = ggml_view_4d(ctx, result, d.state_dim, d.state_dim, d.value_heads, 1, head, head * d.state_dim,
                                  head * d.state_dim * d.value_heads, head * d.value_heads);
        memory = ggml_cpy(ctx, next, memory);
        frame.expand(memory);
    }
}
void NativeLayerPlan::replay_ple(Frame &frame, ggml_tensor *normalized, ggml_tensor *control) {
    histories_.push_back({model.dims().width * model.dims().hc, (model.dims().ple_kernel - 1) * model.dims().ple_ngrams});
    ggml_tensor *args[] = {frame.bind(normalized), frame.bind(control)};
    frame.expand(ggml_backend_cuda_custom_inplace(frame.ctx, frame.bind(state.ple_history), args, 2,
                                                  cuda_native_history_commit, &histories_.back()));
}
void NativeLayerPlan::sum_experts(Frame &f, ggml_tensor *values, ggml_tensor *target) {
    ggml_tensor *args[] = {values};
    f.expand(ggml_backend_cuda_custom_inplace(f.ctx, f.bind(target), args, 1, cuda_native_sum, nullptr));
}

void NativeLayerPlan::head(Frame &frame, LayerIO &io, ggml_tensor *hidden) {
    auto *ctx = frame.ctx;
    const auto &d = model.dims();
    auto *output = mix(frame, hidden, -1, "", nullptr);
    frame.expand(
        ggml_cpy(ctx, matmul(frame, frame.bind(model.tensor("output.weight")), output), frame.bind(io.logits)));
}
void NativeLayerPlan::after(Frame &frame, LayerIO &io, int layer, bool output_head) {
    auto *ctx = frame.ctx;
    auto *input = frame.bind(io.expert_input);
    activations_.push_back({input, frame.bind(io.native_expert_q8), nullptr, 0});
    auto *packed = matmul(frame, frame.bind(model.layer(layer, "ffn_gate_up_shexp.weight")), input);
    auto *shared = producer(frame, packed, nullptr, 2);
    shared = matmul(frame, frame.bind(model.layer(layer, "ffn_down_shexp.weight")), shared);
    auto *gate = matmul(frame, frame.bind(model.layer(layer, "ffn_gate_inp_shexp.weight")), input);
    ggml_tensor *args[] = {frame.bind(io.residual), frame.bind(io.routed), shared, gate, frame.bind(io.inject)};
    auto *hidden = ggml_backend_cuda_custom_inplace(frame.ctx, frame.bind(io.hidden), args, 5,
                                                    cuda_native_shared_combine, nullptr);
    frame.expand(hidden);
    if (output_head && layer == fixed::layers - 1)
        head(frame, io, hidden);
}
void NativeLayerPlan::build_layer(Frame &frame, LayerIO &io, int layer, Part part, bool output_head) {
    activations_.clear();
    frame.key = {layer, io.rows, 0, model.store().layout().quant_signature(layer)};
    if (part == Part::Before)
        before(frame, io, layer);
    else
        after(frame, io, layer, output_head);
}
} // namespace souris
