#include "model.h"
#include "tokenizer.h"
#include "vision.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include "ggml-alloc.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
namespace souris {
size_t QuantTraits::row_bytes(int64_t columns) const {
    if (columns <= 0 || columns % block_values) { GGML_ABORT("invalid quant row geometry"); }
    return size_t(columns / block_values) * block_bytes;
}
QuantTraits quant_traits(QuantType type) {
    switch (type) {
        case GGML_TYPE_Q2_0: case GGML_TYPE_Q4_0: case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS: case GGML_TYPE_IQ2_S: case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S: case GGML_TYPE_IQ4_NL: case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_Q8_0: case GGML_TYPE_Q4_K: case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q5_1: case GGML_TYPE_Q6_K:
            break;
        default: GGML_ABORT("unsupported quant configuration: %d", int(type));
    }
    return {type, int(ggml_blck_size(type)), ggml_type_size(type)};
}
uint64_t quant_signature(const QuantType types[3]) {
    uint64_t signature = 0;
    for (int role = 0; role < 3; ++role) {
        quant_traits(types[role]);
        signature |= uint64_t(types[role]) << (role * 16);
    }
    return signature;
}


static bool is_expert(const std::string & name) { return name.find("_exps.weight")!=std::string::npos; }
std::unique_ptr<Model> Model::load(const ModelConfig & config, ggml_backend_t backend) {
    for (const auto * path : {&config.srs1_path, &config.vision_mmproj}) {
        if (path == &config.vision_mmproj && path->empty()) { continue; }
        std::error_code error;
        if (!std::filesystem::is_regular_file(std::filesystem::u8path(*path), error)) {
            GGML_ABORT("cannot load model artifact: %s", path->c_str());
        }
    }
    return std::unique_ptr<Model>(new Model(config, backend));
}
Model::Model(const ModelConfig & config, ggml_backend_t gpu) : config_(config),
    store_(std::make_unique<ExpertRepository>(config.srs1_path)) {
    for (const auto & shard : store_->mappings()) {
        for (auto * tensor=ggml_get_first_tensor(shard->tensors); tensor; tensor=ggml_get_next_tensor(shard->tensors,tensor)) {
            GGML_ASSERT(tensors_.emplace(ggml_get_name(tensor),tensor).second);
        }
    }
    ggml_cpu_init(); // Quantized dense PLE rows use GGML conversion tables.
    dimensions();
    const int name = gguf_find_key(metadata(), "general.name");
    if (name < 0 || gguf_get_kv_type(metadata(), name) != GGUF_TYPE_STRING ||
        std::string(gguf_get_val_str(metadata(), name)) != "Qwen3.8 Flash Next" || dim.width != 2560) {
        GGML_ABORT("invalid model pairing: SRS1 general.name/width");
    }
    tokenizer_ = std::make_unique<Tokenizer>(*this);
    tokenizer_->validate_vision();
    expert_context_ = ggml_init({ggml_tensor_overhead() * fixed::layers * 3, nullptr, true});
    GGML_ASSERT(expert_context_);
    const char * roles[]={"ffn_gate_exps.weight","ffn_up_exps.weight","ffn_down_exps.weight"};
    for (int layer=0; layer<fixed::layers; ++layer) {
        for (int role=0; role<3; ++role) {
            auto * matrix=this->layer(layer,roles[role]);
            const auto span = store_->role(layer, 0, role);
            auto * descriptor = ggml_new_tensor_3d(expert_context_, span.type, matrix->ne[0], matrix->ne[1], matrix->ne[2]);
            ggml_set_name(descriptor, ggml_get_name(matrix));
            tensors_[ggml_get_name(matrix)] = descriptor;
            matrix = descriptor;
            GGML_ASSERT(matrix->nb[2] == span.bytes);
            GGML_ASSERT(matrix->ne[2]==fixed::experts && matrix->nb[2]<=store_->layout().role_bytes[role]);
            GGML_ASSERT(matrix->ne[0]==(role==2 ? dim.expert_ff : dim.width));
            GGML_ASSERT(matrix->ne[1]==(role==2 ? dim.width : dim.expert_ff));

        }
    }
    dense_context_=ggml_init({ggml_tensor_overhead()*(tensors_.size()+fixed::layers*3),nullptr,true});
    GGML_ASSERT(dense_context_);
    std::vector<std::pair<ggml_tensor *,ggml_tensor *>> copies;
    for (int layer = 0; layer < fixed::layers; ++layer) {
        auto * gate = this->layer(layer, "ffn_gate_shexp.weight");
        auto * up = this->layer(layer, "ffn_up_shexp.weight");
        GGML_ASSERT(gate->type == up->type && ggml_are_same_shape(gate, up));
        const auto name = "blk." + std::to_string(layer) + ".ffn_gate_up_shexp.weight";
        auto * packed = ggml_new_tensor_2d(dense_context_, gate->type, gate->ne[0], 2 * gate->ne[1]);
        ggml_set_name(packed, name.c_str()); tensors_[name] = packed;
        auto * g = ggml_view_2d(dense_context_, packed, gate->ne[0], gate->ne[1], gate->nb[1], 0);
        auto * u = ggml_view_2d(dense_context_, packed, up->ne[0], up->ne[1], up->nb[1], ggml_nbytes(gate));
        ggml_set_name(g, ggml_get_name(gate)); ggml_set_name(u, ggml_get_name(up));
        tensors_[ggml_get_name(gate)] = g; tensors_[ggml_get_name(up)] = u;
        copies.push_back({g, gate}); copies.push_back({u, up});
    }
    for (auto & item : tensors_) {
        const auto & name=item.first;
        auto * source=item.second;
        if (source->view_src || name.find("ffn_gate_up_shexp.weight") != std::string::npos ||
            is_expert(name) || name=="per_layer_token_embd.weight") { continue; }
        if (name.rfind("blk.",0)==0 && std::stoi(name.substr(4))>=fixed::layers) { continue; }
        auto * target=ggml_dup_tensor(dense_context_,source);
        ggml_set_name(target,name.c_str());
        copies.push_back({target,source});
        item.second=target;
    }
    dense_buffer_=ggml_backend_alloc_ctx_tensors(dense_context_,gpu);
    GGML_ASSERT(dense_buffer_);
    ggml_backend_buffer_set_usage(dense_buffer_,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    dense_bytes=ggml_backend_buffer_get_size(dense_buffer_);
    for (const auto & copy : copies) { ggml_backend_tensor_set(copy.first,copy.second->data,0,ggml_nbytes(copy.second)); }
    if (!optional("output.weight")) { tensors_["output.weight"]=tensor("token_embd.weight"); }
    if (!config.vision_mmproj.empty()) {
        vision_ = VisionEncoder::load(config.vision_mmproj, gpu, config.image_cache_entries);
    }
}

ImageEmbedding Model::encode_image(const Image & image, int index) const {
    GGML_ASSERT(vision_);
    if (image.width != image::size || image.height != image::size) { GGML_ABORT("invalid image size"); }
    if (image.rgb.size() != image::rgb_bytes) { GGML_ABORT("invalid image data"); }
    return vision_->encode_rgb8(image.rgb.data(), image.width, image.height, index);
}
Model::~Model() { ggml_backend_buffer_free(dense_buffer_); ggml_free(dense_context_); ggml_free(expert_context_); }
ggml_tensor * Model::optional(const std::string & name) const { auto found=tensors_.find(name); return found==tensors_.end() ? nullptr : found->second; }
ggml_tensor * Model::tensor(const std::string & name) const { auto * result=optional(name); if (!result) { GGML_ABORT("Missing tensor: %s",name.c_str()); } return result; }
ggml_tensor * Model::layer(int index, const char * suffix) const { return tensor("blk."+std::to_string(index)+"."+suffix); }
ggml_tensor * Model::expert(int index,int role) const {
    const char * names[]={"ffn_gate_exps.weight","ffn_up_exps.weight","ffn_down_exps.weight"};
    return layer(index,names[role]);
}
int64_t Model::integer(const char * key, int64_t fallback) const {
    auto * meta=metadata(); const int64_t index=gguf_find_key(meta,key);
    if (index<0) {
        if (fallback<0) { GGML_ABORT("unsupported model metadata: %s",key); }
        return fallback;
    }
    switch (gguf_get_kv_type(meta,index)) {
        case GGUF_TYPE_UINT32: return gguf_get_val_u32(meta,index);
        case GGUF_TYPE_INT32: return gguf_get_val_i32(meta,index);
        case GGUF_TYPE_UINT64: return gguf_get_val_u64(meta,index);
        case GGUF_TYPE_INT64: return gguf_get_val_i64(meta,index);
        case GGUF_TYPE_BOOL: return gguf_get_val_bool(meta,index);
        default: GGML_ABORT("Invalid integer metadata: %s",key);
    }
}
float Model::number(const char * key, float fallback) const {
    auto * meta=metadata(); const auto index=gguf_find_key(meta,key); return index<0 ? fallback : gguf_get_val_f32(meta,index);
}
std::vector<uint64_t> Model::integers(const char * key) const {
    auto * meta=metadata(); const auto index=gguf_find_key(meta,key); if (index<0) { return {}; }
    const size_t count=gguf_get_arr_n(meta,index); const auto * data=gguf_get_arr_data(meta,index); std::vector<uint64_t> result(count);
    for (size_t i=0; i<count; ++i) {
        switch (gguf_get_arr_type(meta,index)) {
            case GGUF_TYPE_INT32: result[i]=static_cast<const int32_t *>(data)[i]; break;
            case GGUF_TYPE_UINT32: result[i]=static_cast<const uint32_t *>(data)[i]; break;
            case GGUF_TYPE_INT64: result[i]=static_cast<const int64_t *>(data)[i]; break;
            case GGUF_TYPE_UINT64: result[i]=static_cast<const uint64_t *>(data)[i]; break;
            case GGUF_TYPE_BOOL: result[i]=static_cast<const uint8_t *>(data)[i]; break;
            default: GGML_ABORT("Invalid integer array: %s",key);
        }
    }
    return result;
}
void Model::dimensions() {
    if (integer("qwen4exp.block_count")!=fixed::layers) { GGML_ABORT("unsupported model metadata: qwen4exp.block_count"); }
    if (integer("qwen4exp.expert_count")!=fixed::experts) { GGML_ABORT("unsupported model metadata: qwen4exp.expert_count"); }
    if (integer("qwen4exp.expert_used_count")!=fixed::topk) { GGML_ABORT("unsupported model metadata: qwen4exp.expert_used_count"); }
    auto & d=dim;
    d.width=integer("qwen4exp.embedding_length"); d.heads=integer("qwen4exp.attention.head_count");
    d.kv_heads=integer("qwen4exp.attention.head_count_kv"); d.head_dim=integer("qwen4exp.attention.key_length");
    d.rotary=integer("qwen4exp.rope.dimension_count"); d.rope_base=number("qwen4exp.rope.freq_base");
    d.original_context=integer("qwen4exp.context_length"); d.epsilon=number("qwen4exp.attention.layer_norm_rms_epsilon");
    d.hc=integer("qwen4exp.hyper_connection.count"); d.hc_rank=integer("qwen4exp.hyper_connection.low_rank");
    d.state_dim=integer("qwen4exp.ssm.state_size"); d.key_heads=integer("qwen4exp.ssm.group_count");
    d.value_heads=integer("qwen4exp.ssm.time_step_rank"); d.conv_kernel=integer("qwen4exp.ssm.conv_kernel");
    d.expert_ff=integer("qwen4exp.expert_feed_forward_length");
    if (d.expert_ff<=0) { GGML_ABORT("unsupported model metadata: qwen4exp.expert_feed_forward_length"); }
    d.index_dim=integer("qwen4exp.attention.indexer.key_length"); d.index_heads=integer("qwen4exp.attention.indexer.head_count");
    d.index_topk=integer("qwen4exp.attention.indexer.top_k");
    const auto sections=integers("qwen4exp.rope.dimension_sections"); GGML_ASSERT(sections.size()==4);
    for (int i=0; i<4; ++i) { d.rope_sections[i]=sections[i]; }
    const auto & ratios = store_->compress_ratios();
    d.pool = 0;
    for (int i = 0; i < fixed::layers; ++i) {
        if (ratios[i] && !d.pool) { d.pool = ratios[i]; }
        d.recurrent[i] = ratios[i] == 0;
    }
    d.ple_dim=integer("qwen4exp.embedding_length_per_layer_input"); d.ple_ngrams=integer("qwen4exp.ple.ngram_size");
    d.ple_per_gram=integer("qwen4exp.ple.heads_per_ngram"); d.ple_heads=(d.ple_ngrams-1)*d.ple_per_gram;
    d.ple_kernel=integer("qwen4exp.ple.conv_kernel"); d.ple_eos=integer("qwen4exp.ple.eos_token_id");
    const auto ple_layers=integers("qwen4exp.ple.layers"); GGML_ASSERT(ple_layers.size()==1); d.ple_layer=ple_layers.front();
    if (d.ple_layer != 1) { GGML_ABORT("invalid metadata: qwen4exp.ple.layers"); }
    d.ple_multipliers=integers("qwen4exp.ple.layer_multipliers"); d.ple_offsets=integers("qwen4exp.ple.head_offsets");
    d.ple_vocab=integers("qwen4exp.ple.head_vocab_sizes");
    GGML_ASSERT(d.ple_multipliers.size()>=size_t(d.ple_ngrams) && d.ple_offsets.size()==size_t(d.ple_heads) && d.ple_vocab.size()==size_t(d.ple_heads));
}
void Model::ple_rows(const Tokens & prefix, int offset, int count, float * output) const {
    auto * table=tensor("per_layer_token_embd.weight"); const auto * traits=ggml_get_type_traits(table->type); GGML_ASSERT(traits->to_float);
    std::vector<uint64_t> context(dim.ple_ngrams);
    for (int token=0; token<count; ++token) {
        context[0]=prefix[offset+token]; bool cut=false;
        for (int n=1; n<dim.ple_ngrams; ++n) {
            const int position=offset+token-n; const int previous=position>=0 ? prefix[position] : dim.ple_eos;
            cut|=position<0 || previous==dim.ple_eos; context[n]=cut ? dim.ple_eos : previous;
        }
        for (int n=2; n<=dim.ple_ngrams; ++n) {
            uint64_t mixed=context[0]*dim.ple_multipliers[0];
            for (int j=1; j<n; ++j) { mixed^=context[j]*dim.ple_multipliers[j]; }
            for (int h=0; h<dim.ple_per_gram; ++h) {
                const int head=(n-2)*dim.ple_per_gram+h; const uint64_t row=mixed%dim.ple_vocab[head]+dim.ple_offsets[head];
                const auto * source=static_cast<const char *>(table->data)+row*table->nb[1];
                MappedFile::prefetch(source,table->nb[1]); traits->to_float(source,output+(size_t(token)*dim.ple_heads+head)*dim.ple_dim,dim.ple_dim);
            }
        }
    }
}
}
