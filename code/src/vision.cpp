#include "vision.h"
#include "device.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <unordered_map>

namespace souris {
struct VisionEncoder::Impl {
    ggml_backend_t backend;
    ggml_context * weights = nullptr, * storage = nullptr;
    ggml_backend_buffer_t weight_buffer = nullptr, storage_buffer = nullptr;
    std::unordered_map<std::string, ggml_tensor *> tensors;
    Frame frame{4096};
    Arena arena;
    ggml_tensor * pixels = nullptr, * positions = nullptr, * output = nullptr;
    ggml_tensor * request[image::max_count]{};
    struct Entry { std::string hash; uint64_t stamp = 0; ggml_tensor * rows = nullptr; };
    std::vector<Entry> cache;
    std::vector<float> normalized;
    uint64_t stamp = 0;
    float epsilon = 1e-6f;

    Impl(const std::string & path, ggml_backend_t gpu, int capacity) : backend(gpu), cache(capacity),
        normalized(image::rgb_bytes) {
        MappedShard source(path, 0, std::filesystem::file_size(path));
        auto integer = [&](const char * key, int expected) {
            const int id = gguf_find_key(source.metadata, key);
            if (id < 0 || gguf_get_kv_type(source.metadata, id) != GGUF_TYPE_UINT32 ||
                gguf_get_val_u32(source.metadata, id) != uint32_t(expected)) {
                GGML_ABORT("invalid vision model pairing: %s", key);
            }
        };
        auto text = [&](const char * key, const char * expected) {
            const int id = gguf_find_key(source.metadata, key);
            if (id < 0 || gguf_get_kv_type(source.metadata, id) != GGUF_TYPE_STRING ||
                std::string(gguf_get_val_str(source.metadata, id)) != expected) {
                GGML_ABORT("invalid vision model pairing: %s", key);
            }
        };
        text("general.name", "Qwen3.8 Flash Next");
        text("clip.projector_type", "qwen3vl_merger");
        integer("clip.vision.projection_dim", image::width);
        integer("clip.vision.image_size", image::size);
        integer("clip.vision.patch_size", 16);
        integer("clip.vision.embedding_length", 1152);
        integer("clip.vision.feed_forward_length", 4304);
        integer("clip.vision.block_count", 27);
        integer("clip.vision.attention.head_count", 16);
        integer("clip.vision.spatial_merge_size", 2);
        const int has = gguf_find_key(source.metadata, "clip.has_vision_encoder");
        if (has < 0 || gguf_get_kv_type(source.metadata, has) != GGUF_TYPE_BOOL || !gguf_get_val_bool(source.metadata, has)) {
            GGML_ABORT("invalid vision model pairing: no vision encoder");
        }
        for (const char * key : {"clip.vision.image_mean", "clip.vision.image_std"}) {
            const int id = gguf_find_key(source.metadata, key);
            if (id < 0 || gguf_get_arr_type(source.metadata, id) != GGUF_TYPE_FLOAT32 || gguf_get_arr_n(source.metadata, id) != 3) {
                GGML_ABORT("invalid vision normalization: %s", key);
            }
            const auto * values = static_cast<const float *>(gguf_get_arr_data(source.metadata, id));
            for (int i = 0; i < 3; ++i) { if (values[i] != 0.5f) { GGML_ABORT("invalid vision normalization: %s", key); } }
        }
        const int deep = gguf_find_key(source.metadata, "clip.vision.is_deepstack_layers");
        if (deep < 0 || gguf_get_arr_type(source.metadata, deep) != GGUF_TYPE_BOOL || gguf_get_arr_n(source.metadata, deep) != 27) {
            GGML_ABORT("invalid vision deepstack metadata");
        }
        const auto * layers = static_cast<const bool *>(gguf_get_arr_data(source.metadata, deep));
        for (int i = 0; i < 27; ++i) { if (layers[i]) { GGML_ABORT("unsupported vision deepstack"); } }
        const int eps = gguf_find_key(source.metadata, "clip.vision.attention.layer_norm_epsilon");
        if (eps < 0 || gguf_get_kv_type(source.metadata, eps) != GGUF_TYPE_FLOAT32) { GGML_ABORT("invalid vision epsilon"); }
        epsilon = gguf_get_val_f32(source.metadata, eps);
        if (!(epsilon > 0) || !std::isfinite(epsilon)) { GGML_ABORT("invalid vision epsilon"); }

        const int count = gguf_get_n_tensors(source.metadata);
        weights = ggml_init({ggml_tensor_overhead() * size_t(count), nullptr, true});
        GGML_ASSERT(weights);
        for (auto * tensor = ggml_get_first_tensor(source.tensors); tensor; tensor = ggml_get_next_tensor(source.tensors, tensor)) {
            auto * target = ggml_dup_tensor(weights, tensor);
            ggml_set_name(target, ggml_get_name(tensor));
            tensors.emplace(ggml_get_name(target), target);
        }
        weight_buffer = ggml_backend_alloc_ctx_tensors(weights, backend);
        if (!weight_buffer) { GGML_ABORT("insufficient vision memory"); }
        ggml_backend_buffer_set_usage(weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        for (auto * tensor = ggml_get_first_tensor(source.tensors); tensor; tensor = ggml_get_next_tensor(source.tensors, tensor)) {
            ggml_backend_tensor_set(tensors.at(ggml_get_name(tensor)), tensor->data, 0, ggml_nbytes(tensor));
        }
        storage = ggml_init({ggml_tensor_overhead() * size_t(6 + capacity), nullptr, true});
        GGML_ASSERT(storage);
        pixels = ggml_new_tensor_3d(storage, GGML_TYPE_F32, image::size, image::size, 3);
        positions = ggml_new_tensor_1d(storage, GGML_TYPE_I32, 2304 * 4);
        for (auto & rows : request) { rows = ggml_new_tensor_2d(storage, GGML_TYPE_F32, image::width, image::tokens); }
        for (auto & entry : cache) { entry.rows = ggml_new_tensor_2d(storage, GGML_TYPE_F32, image::width, image::tokens); }
        storage_buffer = ggml_backend_alloc_ctx_tensors(storage, backend);
        if (!storage_buffer) { GGML_ABORT("insufficient vision memory"); }
        std::vector<int32_t> pos(2304 * 4);
        int at = 0;
        for (int y = 0; y < 48; y += 2) { for (int x = 0; x < 48; x += 2) {
            for (int dy = 0; dy < 2; ++dy) { for (int dx = 0; dx < 2; ++dx, ++at) {
                pos[at] = pos[4608 + at] = y + dy;
                pos[2304 + at] = pos[6912 + at] = x + dx;
            } }
        } }
        ggml_backend_tensor_set(positions, pos.data(), 0, pos.size() * sizeof(int32_t));
        build();
        arena.reserve(frame); arena.allocate(frame); arena.freeze();
    }
    ~Impl() {
        ggml_backend_synchronize(backend);
        ggml_backend_buffer_free(storage_buffer); ggml_free(storage);
        ggml_backend_buffer_free(weight_buffer); ggml_free(weights);
    }
    ggml_tensor * weight(const std::string & name) {
        const auto found = tensors.find(name);
        if (found == tensors.end()) { GGML_ABORT("missing vision tensor: %s", name.c_str()); }
        return frame.bind(found->second);
    }
    ggml_tensor * norm(ggml_tensor * input, const std::string & name) {
        return ggml_add(frame.ctx, ggml_mul(frame.ctx, ggml_norm(frame.ctx, input, epsilon), weight(name + ".weight")), weight(name + ".bias"));
    }
    ggml_tensor * linear(ggml_tensor * input, const std::string & name) {
        return ggml_add(frame.ctx, ggml_mul_mat(frame.ctx, weight(name + ".weight"), input), weight(name + ".bias"));
    }
    ggml_tensor * spatial_order(ggml_tensor * input) {
        auto * ctx = frame.ctx;
        input = ggml_cont_4d(ctx, input, 1152 * 2, 24, 48, 1);
        input = ggml_reshape_4d(ctx, input, 1152 * 2, 24, 2, 24);
        return ggml_cont_3d(ctx, ggml_permute(ctx, input, 0, 2, 1, 3), 1152, 2304, 1);
    }
    void build() {
        auto * ctx = frame.ctx;
        // mtmd Qwen3VL: a static frame feeds both temporal Conv2D halves.
        auto * input = frame.bind(pixels);
        auto * patches = ggml_add(ctx,
            ggml_conv_2d(ctx, weight("v.patch_embd.weight"), input, 16, 16, 0, 0, 1, 1),
            ggml_conv_2d(ctx, weight("v.patch_embd.weight.1"), input, 16, 16, 0, 0, 1, 1));
        auto * hidden = spatial_order(ggml_permute(ctx, patches, 1, 2, 0, 3));
        hidden = ggml_add(ctx, hidden, weight("v.patch_embd.bias"));
        hidden = ggml_add(ctx, hidden, spatial_order(weight("v.position_embd.weight")));
        int sections[4] = {18, 18, 18, 18};
        for (int layer = 0; layer < 27; ++layer) {
            const auto name = "v.blk." + std::to_string(layer) + ".";
            auto * qkv = linear(norm(hidden, name + "ln1"), name + "attn_qkv");
            auto * q = ggml_view_3d(ctx, qkv, 72, 16, 2304, 72 * sizeof(float), qkv->nb[1], 0);
            auto * k = ggml_view_3d(ctx, qkv, 72, 16, 2304, 72 * sizeof(float), qkv->nb[1], 1152 * sizeof(float));
            auto * v = ggml_view_3d(ctx, qkv, 72, 16, 2304, 72 * sizeof(float), qkv->nb[1], 2304 * sizeof(float));
            q = ggml_rope_multi(ctx, q, frame.bind(positions), nullptr, 36, sections, GGML_ROPE_TYPE_VISION, 32768, 10000, 1, 0, 1, 32, 1);
            k = ggml_rope_multi(ctx, k, frame.bind(positions), nullptr, 36, sections, GGML_ROPE_TYPE_VISION, 32768, 10000, 1, 0, 1, 32, 1);
            // The existing CUDA FlashAttention supports 80-wide heads. Zero pad
            // the 72-wide head and retain its original attention scale.
            q = ggml_cont(ctx, ggml_permute(ctx, ggml_pad(ctx, q, 8, 0, 0, 0), 0, 2, 1, 3));
            k = ggml_cast(ctx, ggml_cont(ctx, ggml_permute(ctx, ggml_pad(ctx, k, 8, 0, 0, 0), 0, 2, 1, 3)), GGML_TYPE_F16);
            v = ggml_cast(ctx, ggml_cont(ctx, ggml_permute(ctx, ggml_pad(ctx, v, 8, 0, 0, 0), 0, 2, 1, 3)), GGML_TYPE_F16);
            auto * attention = ggml_flash_attn_ext(ctx, q, k, v, nullptr, 1.0f / std::sqrt(72.0f), 0, 0);
            ggml_prec_set_acc(attention, GGML_PREC_F32);
            attention = ggml_view_3d(ctx, attention, 72, 16, 2304, attention->nb[1], attention->nb[2], 0);
            attention = ggml_cont_2d(ctx, attention, 1152, 2304);
            hidden = ggml_add(ctx, hidden, linear(attention, name + "attn_out"));
            auto * ff = ggml_gelu(ctx, linear(norm(hidden, name + "ln2"), name + "ffn_up"));
            hidden = ggml_add(ctx, hidden, linear(ff, name + "ffn_down"));
        }
        hidden = norm(hidden, "v.post_ln");
        hidden = ggml_reshape_2d(ctx, hidden, 4608, image::tokens);
        hidden = ggml_gelu(ctx, linear(hidden, "mm.0"));
        output = linear(hidden, "mm.2");
        frame.expand(output);
    }
};
std::unique_ptr<VisionEncoder> VisionEncoder::load(const std::string & path, ggml_backend_t backend, int entries) {
    return std::unique_ptr<VisionEncoder>(new VisionEncoder(path, backend, entries));
}
VisionEncoder::VisionEncoder(const std::string & path, ggml_backend_t backend, int entries) :
    impl_(std::make_unique<Impl>(path, backend, entries)) {}
VisionEncoder::~VisionEncoder() = default;
size_t VisionEncoder::resident_bytes() const {
    return ggml_backend_buffer_get_size(impl_->weight_buffer) + ggml_backend_buffer_get_size(impl_->storage_buffer) + impl_->arena.bytes();
}
ImageEmbedding VisionEncoder::encode_rgb8(const uint8_t * rgb, int width, int height, int index) {
    GGML_ASSERT(width == image::size && height == image::size && index >= 0 && index < image::max_count);
    auto & x = *impl_;
    const auto hash = image_hash(rgb, image::rgb_bytes);
    auto * rows = x.request[index];
    for (auto & entry : x.cache) {
        if (entry.hash == hash) {
            entry.stamp = ++x.stamp;
            ggml_backend_tensor_copy_async(x.backend, x.backend, entry.rows, rows);
            ggml_backend_synchronize(x.backend);
            return {rows};
        }
    }
    for (int y = 0; y < image::size; ++y) { for (int col = 0; col < image::size; ++col) {
        for (int channel = 0; channel < 3; ++channel) {
            x.normalized[size_t(channel) * image::size * image::size + size_t(y) * image::size + col] =
                float(rgb[(size_t(y) * image::size + col) * 3 + channel]) / 127.5f - 1.0f;
        }
    } }
    ggml_backend_tensor_set(x.pixels, x.normalized.data(), 0, x.normalized.size() * sizeof(float));
    if (ggml_backend_cuda_compute_eager(x.backend, x.frame.graph) != GGML_STATUS_SUCCESS) { GGML_ABORT("vision encode failed"); }
    // Buffer-level copy uses cudaStreamPerThread and does not fence the
    // nonblocking encoder stream. Keep output/cache copies with their producer.
    ggml_backend_tensor_copy_async(x.backend, x.backend, x.output, rows);
    if (!x.cache.empty()) {
        auto & entry = *std::min_element(x.cache.begin(), x.cache.end(), [](const auto & a, const auto & b) { return a.stamp < b.stamp; });
        ggml_backend_tensor_copy_async(x.backend, x.backend, rows, entry.rows);
        entry.hash = hash; entry.stamp = ++x.stamp;
    }
    // The next image reuses pixels, graph temporaries and CPU normalization.
    ggml_backend_synchronize(x.backend);
    return {rows};
}
}
