#include "kv_cache.h"
#include "ggml-cuda.h"
#include <algorithm>
#include <climits>
#include <cmath>

namespace souris {
KVCacheManager::KVCacheManager(const Dimensions & dim, int context, Device & device) :
    capacity((std::max(context, fixed::minimum_context) + fixed::kv_token_alignment - 1) /
             fixed::kv_token_alignment * fixed::kv_token_alignment), device_(device) {
    constexpr size_t tensors=512; context_=ggml_init({ggml_tensor_overhead()*tensors,nullptr,true}); GGML_ASSERT(context_);
    kv_context_ = ggml_init({ggml_tensor_overhead() * fixed::layers * 2, nullptr, true});
    GGML_ASSERT(kv_context_);
    allocate_q8_0(dim);
    for (int i = 0; i < fixed::layers; ++i) {
        auto & layer = layers[i];
        if (!dim.recurrent[i]) {
            layer.index = ggml_new_tensor_2d(context_, GGML_TYPE_F16, dim.index_dim, capacity);
        } else {
            const int channels = dim.state_dim * (dim.key_heads * 2 + dim.value_heads);
            layer.conv = ggml_new_tensor_2d(context_, GGML_TYPE_F32, dim.conv_kernel - 1, channels);
            layer.recurrent = ggml_new_tensor_4d(context_, GGML_TYPE_F32, dim.state_dim, dim.state_dim, dim.value_heads, 1);
        }
    }
    ple_history=ggml_new_tensor_2d(context_,GGML_TYPE_F32,(dim.ple_kernel-1)*dim.ple_ngrams,dim.width*dim.hc);
    fixed_buffer_=ggml_backend_alloc_ctx_tensors(context_,device.backend); GGML_ASSERT(fixed_buffer_);
    fixed_bytes=ggml_backend_buffer_get_size(fixed_buffer_); reset();
}
KVCacheManager::KVCacheManager(const Dimensions & dim, KVCacheManager & canonical, Device & device) :
    capacity(canonical.capacity), device_(device) {
    context_ = ggml_init({ggml_tensor_overhead() * (fixed::layers * 2 + 1), nullptr, true}); GGML_ASSERT(context_);
    for (int layer = 0; layer < fixed::layers; ++layer) {
        layers[layer].key = canonical.layers[layer].key;
        layers[layer].value = canonical.layers[layer].value;
        layers[layer].index = canonical.layers[layer].index;
        if (dim.recurrent[layer]) {
            layers[layer].conv = ggml_dup_tensor(context_, canonical.layers[layer].conv);
            layers[layer].recurrent = ggml_dup_tensor(context_, canonical.layers[layer].recurrent);
        }
    }
    ple_history = ggml_dup_tensor(context_, canonical.ple_history);
    fixed_buffer_ = ggml_backend_alloc_ctx_tensors(context_, device.backend); GGML_ASSERT(fixed_buffer_);
    fixed_bytes = ggml_backend_buffer_get_size(fixed_buffer_);
    ggml_backend_buffer_clear(fixed_buffer_, 0);
}
void KVCacheManager::allocate_q8_0(const Dimensions & dim) {
    GGML_ASSERT(dim.head_dim % ggml_blck_size(GGML_TYPE_Q8_0) == 0);
    for (int layer = 0; layer < fixed::layers; ++layer) {
        if (dim.recurrent[layer]) { continue; }
        layers[layer].key = ggml_new_tensor_3d(kv_context_, GGML_TYPE_Q8_0, dim.head_dim, dim.kv_heads, capacity);
        layers[layer].value = ggml_new_tensor_3d(kv_context_, GGML_TYPE_Q8_0, dim.head_dim, dim.kv_heads, capacity);
    }
    kv_buffer_ = ggml_backend_alloc_ctx_tensors(kv_context_, device_.backend);
    if (!kv_buffer_) { GGML_ABORT("unsupported KV type: q8_0 allocation failed"); }
    kv_bytes = ggml_backend_buffer_get_size(kv_buffer_);
}
KVCacheManager::~KVCacheManager() { ggml_backend_buffer_free(checkpoints_); ggml_backend_buffer_free(fixed_buffer_); ggml_backend_buffer_free(kv_buffer_); ggml_free(context_); ggml_free(kv_context_); }
void KVCacheManager::prepare_checkpoints() {
    GGML_ASSERT(!checkpoints_);
    for (const auto & layer : layers) {
        for (auto * tensor : {layer.conv, layer.recurrent}) {
            if (tensor) {
                checkpoint_tensors_.push_back({tensor, checkpoint_stride_});
                checkpoint_stride_ = GGML_PAD(checkpoint_stride_ + ggml_nbytes(tensor), 256);
            }
        }
    }
    checkpoint_tensors_.push_back({ple_history, checkpoint_stride_});
    checkpoint_stride_ = GGML_PAD(checkpoint_stride_ + ggml_nbytes(ple_history), 256);
    checkpoints_ = ggml_backend_buft_alloc_buffer(ggml_backend_get_default_buffer_type(device_.backend), 8 * checkpoint_stride_);
    GGML_ASSERT(checkpoints_);
}
void KVCacheManager::capture_checkpoint(int slot) {
    GGML_ASSERT(checkpoints_ && slot >= 0 && slot < 8);
    auto * target = static_cast<char *>(ggml_backend_buffer_get_base(checkpoints_)) + slot * checkpoint_stride_;
    for (const auto & entry : checkpoint_tensors_) {
        SOURIS_CUDA(cudaMemcpyAsync(target + entry.offset, entry.tensor->data, ggml_nbytes(entry.tensor), cudaMemcpyDeviceToDevice, device_.compute));
    }
}
void KVCacheManager::capture_checkpoint_layer(int slot, int layer, bool ple) {
    GGML_ASSERT(checkpoints_ && slot >= 0 && slot < 8 && layer >= 0 && layer < fixed::layers);
    auto * target = static_cast<char *>(ggml_backend_buffer_get_base(checkpoints_)) + slot * checkpoint_stride_;
    for (const auto & entry : checkpoint_tensors_) {
        if (entry.tensor == layers[layer].conv || entry.tensor == layers[layer].recurrent || (ple && entry.tensor == ple_history)) {
            SOURIS_CUDA(cudaMemcpyAsync(target + entry.offset, entry.tensor->data, ggml_nbytes(entry.tensor), cudaMemcpyDeviceToDevice, device_.compute));
        }
    }
}
void KVCacheManager::restore_checkpoint(int slot, int offset) {
    GGML_ASSERT(checkpoints_ && slot >= 0 && slot < 8 && offset >= 0 && offset < capacity);
    auto * source = static_cast<char *>(ggml_backend_buffer_get_base(checkpoints_)) + slot * checkpoint_stride_;
    for (const auto & entry : checkpoint_tensors_) {
        SOURIS_CUDA(cudaMemcpyAsync(entry.tensor->data, source + entry.offset, ggml_nbytes(entry.tensor), cudaMemcpyDeviceToDevice, device_.compute));
    }
    position = offset;
}
void KVCacheManager::reset() {
    SOURIS_CUDA(cudaStreamSynchronize(device_.compute)); ggml_backend_buffer_clear(kv_buffer_,0); ggml_backend_buffer_clear(fixed_buffer_,0); position=0;
}
KVCacheManager::Views KVCacheManager::append(Frame & frame, int layer, ggml_tensor * key,
                                             ggml_tensor * value, ggml_tensor * rows) {
    auto * ctx = frame.ctx;
    auto * keys = ggml_permute(ctx, frame.bind(layers[layer].key), 0, 2, 1, 3);
    auto * values = ggml_permute(ctx, frame.bind(layers[layer].value), 0, 2, 1, 3);
    GGML_ASSERT(keys->type == GGML_TYPE_Q8_0 && values->type == GGML_TYPE_Q8_0);
    return {ggml_set_rows(ctx, keys, key, rows), ggml_set_rows(ctx, values, value, rows)};
}
}
