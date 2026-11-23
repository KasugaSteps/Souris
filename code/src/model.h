#pragma once
#include "souris/types.h"
#include "gguf.h"
#include "ggml-backend.h"
#include "repository.h"
#include <memory>
#include <unordered_map>

namespace souris {
class Tokenizer;
class VisionEncoder;
struct Image;
struct ImageEmbedding;
/** SRS1 v1 little-endian model header. Dense shard offsets follow at byte 2048. */
struct Srs1Header {
    char magic[4];
    uint32_t version;
    uint64_t header_size;
    uint32_t shard_count, layers, experts, topk, role_count, reserved;
    uint64_t expert_count, expert_offset, expert_size, shard_table_offset;
    uint64_t role_offset[3], role_bytes[3];
    QuantDesc quant[fixed::layers][3];
};
static_assert(sizeof(Srs1Header) == 696, "SRS1 v1 header ABI");

/** Qwen4Exp dimensions validated against the SRS1 role directory. */
struct Dimensions {
    int width, heads, kv_heads, head_dim, rotary, hc, hc_rank;
    int state_dim, key_heads, value_heads, conv_kernel, expert_ff;
    int index_dim, index_heads, index_topk, pool, original_context;
    int ple_dim, ple_heads, ple_ngrams, ple_per_gram, ple_kernel, ple_eos, ple_layer;
    float epsilon, rope_base;
    int rope_sections[4];
    bool recurrent[fixed::layers];
    std::vector<uint64_t> ple_multipliers, ple_offsets, ple_vocab;
};

/** Embedded GGUF metadata, dense mappings and the direct expert repository. */
class Model {
public:
    /** Load SRS1, its embedded tokenizer and optional paired vision artifacts. */
    static std::unique_ptr<Model> load(const ModelConfig &, ggml_backend_t backend);
    const Srs1Header & header() const { return store_->header(); }
    QuantDesc quant(int layer, int role) const { return store_->layout().quant[layer][role]; }
    ~Model();
    const Dimensions & dims() const { return dim; }
    Tokenizer & tokenizer() const { return *tokenizer_; }
    bool has_vision() const { return bool(vision_); }
    VisionEncoder * vision() const { return vision_.get(); }
    ImageEmbedding encode_image(const Image &, int index = 0) const;
    int image_tokens() const { return 576; }
    int max_images() const { return config_.max_images; }
    size_t dense_bytes=0;
    ggml_tensor * tensor(const std::string & name) const;
    ggml_tensor * optional(const std::string & name) const;
    ggml_tensor * layer(int layer, const char * suffix) const;
    ggml_tensor * expert(int layer, int role) const;
    const ExpertRepository & store() const { return *store_; }
    gguf_context * metadata() const { return store_->metadata(); }
    int64_t integer(const char * key, int64_t fallback=-1) const;
    float number(const char * key, float fallback=0) const;
    std::vector<uint64_t> integers(const char * key) const;
    /** Gather actual n-gram PLE rows directly from mmap into the caller's staging memory. */
    void ple_rows(const Tokens & prefix, int offset, int count, float * output) const;
private:
    Model(const ModelConfig &, ggml_backend_t);
    ModelConfig config_;
    std::unique_ptr<ExpertRepository> store_;
    std::unique_ptr<Tokenizer> tokenizer_;
    std::unique_ptr<VisionEncoder> vision_;
    Dimensions dim{};
    std::unordered_map<std::string,ggml_tensor *> tensors_;
    ggml_context * dense_context_=nullptr, *expert_context_=nullptr;
    ggml_backend_buffer_t dense_buffer_=nullptr;
    void dimensions();
};

}
