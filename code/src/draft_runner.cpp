#include "draft_runner.h"
#include "mtp_ops.h"
#include "ops.h"
#include "stats.h"
#include "ggml-cuda.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <unordered_map>

namespace souris {
namespace {
constexpr int prompt_rows = 256;
constexpr float h_greedy = 0.90f;
constexpr float h_sample = 0.90f;
constexpr size_t load_block = 16 * 1024 * 1024;

struct Captured {
    Frame frame{2048};
    Arena arena;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    ~Captured() { release(); }
    void release() {
        if (executable) { SOURIS_CUDA(cudaGraphExecDestroy(executable)); }
        if (graph) { SOURIS_CUDA(cudaGraphDestroy(graph)); }
        executable = nullptr; graph = nullptr;
    }
    void prepare() {
        arena.reserve(frame); arena.allocate(frame); arena.freeze();
    }
    void capture(Device & device) {
        SOURIS_CUDA(cudaStreamBeginCapture(device.compute, cudaStreamCaptureModeRelaxed));
        GGML_ASSERT(ggml_backend_cuda_compute_eager(device.backend, frame.graph) == GGML_STATUS_SUCCESS);
        SOURIS_CUDA(cudaStreamEndCapture(device.compute, &graph));
        SOURIS_CUDA(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    }
    void launch(Device & device) {
        GGML_ASSERT(executable);
        SOURIS_CUDA(cudaGraphLaunch(executable, device.compute));
    }
};
}

struct DraftRunner::Impl {
    const Model & model;
    Device & device;
    Statistics & stats;
    const float & temperature, & top_p, & min_p;
    const int & top_k;
    uint64_t (*random_bits)(void *);
    void * random_owner;
    ggml_context * weights_ctx = nullptr, * io_ctx = nullptr;
    ggml_backend_buffer_t weights_buffer = nullptr, io_buffer = nullptr;
    std::unordered_map<std::string, ggml_tensor *> weights;
    ggml_tensor * keys = nullptr, * values = nullptr, * proposals = nullptr, * positions = nullptr;
    ggml_tensor * probability = nullptr, * parameters = nullptr, * sample_token = nullptr, * entropy = nullptr;
    int context = 0, capacity = 0, window = 0, max_drafts = 0, cursor = -1;
    struct Scalars { float params[5]; Token token; float entropy; } *sample_host = nullptr;
    Token pending = -1;
    bool initialized = false, primed = false;
    struct IO {
        ggml_tensor * control = nullptr, * tokens = nullptr, * residual = nullptr;
        ggml_tensor * output = nullptr, * logits = nullptr;
        char * host = nullptr;
        cudaEvent_t uploaded = nullptr;
        MtpInputs inputs{};
    } prompt, draft;
    Captured prompt_graph, draft_graph, greedy_graph, sampling_graph;

    Impl(const Model & m, Device & d, Statistics & s, const float & t, const float & p, const int & k, const float & minimum,
         uint64_t (*random)(void *), void * owner) : model(m), device(d), stats(s),
        temperature(t), top_p(p), min_p(minimum), top_k(k), random_bits(random), random_owner(owner) {}
    ~Impl() {
        SOURIS_CUDA(cudaStreamSynchronize(device.compute));
        prompt_graph.release(); draft_graph.release(); greedy_graph.release(); sampling_graph.release();
        for (auto * io : {&prompt, &draft}) {
            if (io->uploaded) { SOURIS_CUDA(cudaEventDestroy(io->uploaded)); }
            if (io->host) { SOURIS_CUDA(cudaFreeHost(io->host)); }
        }
        if (sample_host) { SOURIS_CUDA(cudaFreeHost(sample_host)); }
        ggml_backend_buffer_free(io_buffer); ggml_backend_buffer_free(weights_buffer);
        ggml_free(io_ctx); ggml_free(weights_ctx);
    }
    ggml_tensor * weight(Frame & frame, const std::string & suffix) {
        const auto item = weights.find("mtp." + suffix);
        if (item == weights.end()) { GGML_ABORT("MTP tensor missing: %s", suffix.c_str()); }
        return frame.bind(item->second);
    }
    void load(const std::string & directory) {
        std::ifstream manifest(directory + "/manifest.json");
        if (!manifest) { GGML_ABORT("MTP manifest cannot be opened"); }
        const auto document = nlohmann::json::parse(manifest);
        if (document.at("format") != "souris-mtp-q4" || document.at("version") != 1 ||
            document.at("quant") != "Q4_0") { GGML_ABORT("Unsupported MTP manifest"); }
        std::ifstream data(directory + "/weights.bin", std::ios::binary | std::ios::ate);
        if (!data) { GGML_ABORT("MTP weights cannot be opened"); }
        const size_t file_bytes = size_t(data.tellg());
        weights_ctx = ggml_init({ggml_tensor_overhead() * 128, nullptr, true}); GGML_ASSERT(weights_ctx);
        size_t previous_end = 0;
        for (const auto & entry : document.at("tensors")) {
            const std::string name = entry.at("name"), type = entry.at("type");
            const auto shape = entry.at("ne").get<std::vector<int64_t>>();
            const auto source = entry.at("source_shape").get<std::vector<int64_t>>();
            if (shape.empty() || shape.size() > GGML_MAX_DIMS || source.empty()) { GGML_ABORT("Invalid MTP shape"); }
            const bool router = name == "mtp.layers.0.mlp.gate.weight";
            const bool norm = name.find("norm") != std::string::npos;
            const ggml_type dtype = type == "Q4_0" ? GGML_TYPE_Q4_0 : GGML_TYPE_F16;
            if ((norm || router) ? type != "F16" : type != "Q4_0") { GGML_ABORT("Invalid MTP dtype: %s", name.c_str()); }
            if (dtype == GGML_TYPE_F16) {
                if (shape.size() != source.size() || !std::equal(shape.begin(), shape.end(), source.rbegin())) {
                    GGML_ABORT("Invalid MTP F16 dimensions: %s", name.c_str());
                }
            }
            for (int64_t n : shape) { if (n <= 0) { GGML_ABORT("Invalid MTP extent"); } }
            auto * tensor = ggml_new_tensor(weights_ctx, dtype, shape.size(), shape.data());
            const size_t offset = entry.at("offset"), bytes = entry.at("bytes");
            if (offset % 256 || bytes != ggml_nbytes(tensor) || offset < previous_end ||
                offset > file_bytes || bytes > file_bytes - offset || !weights.emplace(name, tensor).second) {
                GGML_ABORT("Invalid MTP tensor range: %s", name.c_str());
            }
            previous_end = offset + bytes;
        }
        weights_buffer = ggml_backend_alloc_ctx_tensors(weights_ctx, device.backend); GGML_ASSERT(weights_buffer);
        ggml_backend_buffer_set_usage(weights_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        std::vector<char> block(load_block);
        for (const auto & entry : document.at("tensors")) {
            auto * tensor = weights.at(entry.at("name").get<std::string>());
            data.seekg(entry.at("offset").get<size_t>());
            for (size_t offset = 0; offset < ggml_nbytes(tensor);) {
                const size_t bytes = std::min(load_block, ggml_nbytes(tensor) - offset);
                if (!data.read(block.data(), bytes)) { GGML_ABORT("MTP tensor short read"); }
                ggml_backend_tensor_set(tensor, block.data(), offset, bytes);
                offset += bytes;
            }
        }
    }
    void make_io(IO & io, int rows) {
        const auto & d = model.dims();
        io.inputs = {rows, capacity, d.head_dim, d.kv_heads, window, window + DraftRunner::max_drafts};
        io.control = ggml_new_tensor_1d(io_ctx, GGML_TYPE_I8, sizeof(Control));
        io.tokens = ggml_new_tensor_1d(io_ctx, GGML_TYPE_I32, rows);
        io.residual = ggml_new_tensor_3d(io_ctx, GGML_TYPE_F32, d.width, d.hc, rows);
        io.output = ggml_new_tensor_3d(io_ctx, GGML_TYPE_F32, d.width, d.hc, rows);
        io.logits = ggml_new_tensor_1d(io_ctx, GGML_TYPE_F32, model.tensor("output.weight")->ne[1]);
        SOURIS_CUDA(cudaHostAlloc(&io.host, sizeof(Control) + rows * sizeof(Token), cudaHostAllocDefault));
        SOURIS_CUDA(cudaEventCreateWithFlags(&io.uploaded, cudaEventDisableTiming));
    }
    ggml_tensor * norm(Frame & f, ggml_tensor * x, ggml_tensor * w) {
        if (w->type != GGML_TYPE_F32) { w = ggml_cast(f.ctx, w, GGML_TYPE_F32); }
        return ggml_mul(f.ctx, ggml_rms_norm(f.ctx, x, model.dims().epsilon), w);
    }
    ggml_tensor * mix(Frame & f, ggml_tensor * x, const std::string & prefix, ggml_tensor ** inject) {
        const auto & d = model.dims(); auto * ctx = f.ctx;
        const int rows = x->ne[2];
        auto * flat = ggml_reshape_2d(ctx, x, d.width * d.hc, rows);
        auto * normalized = norm(f, flat, weight(f, prefix + ".hc_norm.weight"));
        auto * low = ggml_silu(ctx, ggml_scale(ctx, ggml_mul_mat(ctx,
            weight(f, prefix + ".input_mix_weight_down.weight"), normalized), 1.0f / d.hc));
        auto * gate = ggml_sigmoid(ctx, ggml_mul_mat(ctx, weight(f, prefix + ".input_mix_weight_up.weight"), low));
        auto * gated = ggml_reshape_3d(ctx, ggml_mul(ctx, normalized, gate), d.width, d.hc, rows);
        auto * out = ggml_cont(ctx, ggml_view_2d(ctx, gated, d.width, rows, gated->nb[2], 0));
        for (int i = 1; i < d.hc; ++i) {
            out = ggml_add(ctx, out, ggml_view_2d(ctx, gated, d.width, rows, gated->nb[2], i * gated->nb[1]));
        }
        if (inject) { *inject = ggml_mul_mat(ctx, weight(f, prefix + ".block_inject_weight.weight"), normalized); }
        return ggml_scale(ctx, out, 1.0f / d.hc);
    }
    ggml_tensor * combine(Frame & f, ggml_tensor * residual, ggml_tensor * output, ggml_tensor * inject) {
        const auto & d = model.dims(); auto * ctx = f.ctx; const int rows = residual->ne[2];
        auto * scale = ggml_reshape_3d(ctx, ggml_scale(ctx, ggml_sigmoid(ctx,
            ggml_scale(ctx, inject, 1.0f / d.hc)), 2), 1, d.hc, rows);
        return ggml_add(ctx, residual, ggml_mul(ctx, ggml_repeat_4d(ctx,
            ggml_reshape_3d(ctx, output, d.width, 1, rows), d.width, d.hc, rows, 1), scale));
    }
    ggml_tensor * rope(Frame & f, ggml_tensor * x, ggml_tensor * positions) {
        const auto & d = model.dims(); int sections[4]; std::copy(d.rope_sections, d.rope_sections + 4, sections);
        return ggml_rope_multi(f.ctx, x, positions, nullptr, d.rotary, sections, GGML_ROPE_TYPE_IMROPE,
            d.original_context, d.rope_base, 1, 0, 1, 32, 1);
    }
    /** The prompt graph records only the front needed to append private K/V. */
    void build(Captured & captured, IO & io, bool full) {
        auto & f = captured.frame; auto * ctx = f.ctx; const auto & d = model.dims(); const int rows = io.inputs.rows;
        auto * control = f.bind(io.control);
        ggml_tensor * args[] = {control};
        auto * positions = ggml_backend_cuda_custom_4d(ctx, GGML_TYPE_I32, 4 * rows, 1, 1, 1,
            args, 1, cuda_mtp_positions, nullptr);
        auto * embedding = ggml_get_rows(ctx, f.bind(model.tensor("token_embd.weight")), f.bind(io.tokens));
        embedding = ggml_mul_mat(ctx, weight(f, "fc_embedding.weight"),
            norm(f, embedding, weight(f, "pre_fc_norm_embedding.weight")));
        auto * hidden = norm(f, ggml_reshape_2d(ctx, f.bind(io.residual), d.width * d.hc, rows),
                            weight(f, "pre_fc_norm_hidden.weight"));
        hidden = ggml_reshape_2d(ctx, hidden, d.width, d.hc * rows);
        hidden = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, weight(f, "fc_hidden.weight"), hidden), d.width, d.hc, rows);
        hidden = ggml_add(ctx, hidden, ggml_repeat_4d(ctx, ggml_reshape_3d(ctx, embedding, d.width, 1, rows),
                                                   d.width, d.hc, rows, 1));
        ggml_tensor * inject;
        auto * input = mix(f, hidden, "layers.0.attn_hyper_connection", &inject);
        auto * k = ggml_mul_mat(ctx, weight(f, "layers.0.self_attn.k_proj.weight"), input);
        auto * v = ggml_mul_mat(ctx, weight(f, "layers.0.self_attn.v_proj.weight"), input);
        k = norm(f, ggml_reshape_3d(ctx, k, d.head_dim, d.kv_heads, rows), weight(f, "layers.0.self_attn.k_norm.weight"));
        k = rope(f, k, positions);
        k = ggml_cont(ctx, ggml_reshape_2d(ctx, k, d.head_dim * d.kv_heads, rows));
        ggml_tensor * key_args[] = {k, control, f.bind(this->positions)}, * value_args[] = {v, control, f.bind(this->positions)};
        auto * key = ggml_backend_cuda_custom_inplace(ctx, f.bind(keys), key_args, 3, cuda_mtp_append, &io.inputs);
        auto * value = ggml_backend_cuda_custom_inplace(ctx, f.bind(values), value_args, 3, cuda_mtp_append, &io.inputs);
        f.expand(key); f.expand(value);
        if (!full) { return; }
        auto * qfull = ggml_mul_mat(ctx, weight(f, "layers.0.self_attn.q_proj.weight"), input);
        auto * q = ggml_view_3d(ctx, qfull, d.head_dim, d.heads, rows, 2 * d.head_dim * sizeof(float), qfull->nb[1], 0);
        auto * gate = ggml_cont_2d(ctx, ggml_view_3d(ctx, qfull, d.head_dim, d.heads, rows,
            2 * d.head_dim * sizeof(float), qfull->nb[1], d.head_dim * sizeof(float)), d.head_dim * d.heads, rows);
        q = rope(f, norm(f, q, weight(f, "layers.0.self_attn.q_norm.weight")), positions);
        ggml_tensor * mask_args[] = {control, f.bind(this->positions), key, value};
        auto * mask = ggml_backend_cuda_custom_4d(ctx, GGML_TYPE_F16, capacity, rows, 1, 1,
            mask_args, 4, cuda_mtp_mask, &io.inputs);
        auto * attention = ggml_flash_attn_ext(ctx, ggml_permute(ctx, q, 0, 2, 1, 3),
            ggml_permute(ctx, key, 0, 2, 1, 3), ggml_permute(ctx, value, 0, 2, 1, 3),
            mask, 1.0f / std::sqrt(float(d.head_dim)), 0, 0);
        attention = ggml_mul(ctx, ggml_reshape_2d(ctx, attention, d.head_dim * d.heads, rows), ggml_sigmoid(ctx, gate));
        attention = ggml_mul_mat(ctx, weight(f, "layers.0.self_attn.o_proj.weight"), attention);
        auto * residual = combine(f, hidden, attention, inject);
        input = mix(f, residual, "layers.0.mlp_hyper_connection", &inject);
        auto * scores = ggml_soft_max(ctx, ggml_mul_mat(ctx, weight(f, "layers.0.mlp.gate.weight"), input));
        auto * ids = ggml_top_k(ctx, scores, fixed::topk);
        auto * route_weights = ggml_get_rows(ctx, ggml_reshape_3d(ctx, scores, 1, fixed::experts, rows), ids);
        route_weights = ggml_reshape_2d(ctx, route_weights, fixed::topk, rows);
        route_weights = ggml_div(ctx, route_weights, ggml_sum_rows(ctx, route_weights));
        const float route_scale = model.number("qwen4exp.expert_weights_scale", 1);
        route_weights = ggml_scale(ctx, route_weights, route_scale);
        auto * x = ggml_reshape_3d(ctx, input, d.width, 1, rows);
        auto * expert_gate = ggml_mul_mat_id(ctx, weight(f, "layers.0.mlp.switch_mlp.gate_proj.weight"), x, ids);
        auto * expert_up = ggml_mul_mat_id(ctx, weight(f, "layers.0.mlp.switch_mlp.up_proj.weight"), x, ids);
        auto * expert = ggml_mul_mat_id(ctx, weight(f, "layers.0.mlp.switch_mlp.down_proj.weight"),
                                      ggml_swiglu_split(ctx, expert_gate, expert_up), ids);
        expert = ggml_mul(ctx, expert, ggml_reshape_3d(ctx, route_weights, 1, fixed::topk, rows));
        auto * routed = ggml_cont(ctx, ggml_view_2d(ctx, expert, d.width, rows, expert->nb[2], 0));
        for (int rank = 1; rank < fixed::topk; ++rank) {
            routed = ggml_add(ctx, routed, ggml_view_2d(ctx, expert, d.width, rows, expert->nb[2], rank * expert->nb[1]));
        }
        auto * shared_gate = ggml_mul_mat(ctx, weight(f, "layers.0.mlp.shared_expert.gate_proj.weight"), input);
        auto * shared_up = ggml_mul_mat(ctx, weight(f, "layers.0.mlp.shared_expert.up_proj.weight"), input);
        auto * shared = ggml_mul_mat(ctx, weight(f, "layers.0.mlp.shared_expert.down_proj.weight"), ggml_swiglu_split(ctx, shared_gate, shared_up));
        shared = ggml_mul(ctx, shared, ggml_sigmoid(ctx, ggml_mul_mat(ctx, weight(f, "layers.0.mlp.shared_expert_gate.weight"), input)));
        hidden = combine(f, residual, ggml_add(ctx, routed, shared), inject);
        f.expand(ggml_cpy(ctx, hidden, f.bind(io.output)));
        auto * output = mix(f, hidden, "hyper_connection_mixer", nullptr);
        f.expand(ggml_cpy(ctx, ggml_mul_mat(ctx, f.bind(model.tensor("output.weight")), output), f.bind(io.logits)));
    }
    void build_distribution(Captured & captured, bool sampling) {
        auto & f = captured.frame; auto * ctx = f.ctx;
        auto * scores = f.bind(draft.logits); auto * params = f.bind(parameters);
        auto * q = sampling ? scores : ggml_soft_max(ctx, scores);
        if (sampling) {
            auto * scratch = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, sampling_scratch_bytes(scores->ne[0]));
            ggml_tensor * args[] = {scores, params, f.bind(sample_token), scratch};
            q = ggml_backend_cuda_custom_4d(ctx, GGML_TYPE_F32, q->ne[0], 1, 1, 1, args, 4, cuda_sample, nullptr);
        } else { f.expand(ggml_cpy(ctx, ggml_argmax(ctx, scores), f.bind(sample_token))); }
        f.expand(ggml_cpy(ctx, q, f.bind(probability)));
        auto * logarithm = ggml_log(ctx, ggml_clamp(ctx, q, 1e-30f, 1));
        f.expand(ggml_cpy(ctx, ggml_sqrt(ctx, ggml_scale(ctx, ggml_sum(ctx, ggml_mul(ctx, q, logarithm)), -1)), f.bind(entropy)));
    }
    void append(IO & io, Captured & captured, ggml_tensor * residual, const Token * tokens, int rows, int position) {
        GGML_ASSERT(residual && residual->type == GGML_TYPE_F32 &&
                    residual->nb[2] == io.residual->nb[2]);
        if (rows < 1 || rows > io.inputs.rows || position < 0 || position + rows > context + max_drafts) {
            GGML_ABORT("MTP append outside private KV capacity");
        }
        const Control control{position, rows, 0, 0};
        SOURIS_CUDA(cudaEventSynchronize(io.uploaded));
        std::memcpy(io.host, &control, sizeof(control));
        std::memcpy(io.host + sizeof(control), tokens, rows * sizeof(Token));
        SOURIS_CUDA(cudaMemcpyAsync(io.control->data, io.host, sizeof(control), cudaMemcpyHostToDevice, device.compute));
        if (rows < io.inputs.rows) {
            SOURIS_CUDA(cudaMemsetAsync(io.tokens->data, 0, ggml_nbytes(io.tokens), device.compute));
            SOURIS_CUDA(cudaMemsetAsync(io.residual->data, 0, ggml_nbytes(io.residual), device.compute));
        }
        SOURIS_CUDA(cudaMemcpyAsync(io.tokens->data, io.host + sizeof(control), rows * sizeof(Token), cudaMemcpyHostToDevice, device.compute));
        SOURIS_CUDA(cudaEventRecord(io.uploaded, device.compute));
        SOURIS_CUDA(cudaMemcpyAsync(io.residual->data, residual->data, rows * io.residual->nb[2], cudaMemcpyDeviceToDevice, device.compute));
        captured.launch(device);
    }
    void tail(ggml_tensor * residual, Token token, int position) { append(draft, draft_graph, residual, &token, 1, position); }
};

DraftRunner::DraftRunner(const Model & m, Device & d, Statistics & s, const float & t, const float & p, const int & k, const float & minimum,
                     uint64_t (*random)(void *), void * owner) : impl_(std::make_unique<Impl>(m, d, s, t, p, k, minimum, random, owner)) {}
DraftRunner::~DraftRunner() = default;
void DraftRunner::init(const std::string & directory, int drafts, int context) {
    auto & x = *impl_; GGML_ASSERT(!x.initialized && drafts >= 1 && drafts <= max_drafts && context > 0);
    x.context = context; x.window = std::min(context, fixed::mtp_window);
    x.capacity = (x.window + max_drafts + fixed::kv_token_alignment - 1) /
        fixed::kv_token_alignment * fixed::kv_token_alignment;
    x.max_drafts = drafts;
    x.load(directory);
    const auto & d = x.model.dims();
    for (const char * role : {"gate_proj", "up_proj", "down_proj"}) {
        const auto * tensor = x.weights.at(std::string("mtp.layers.0.mlp.switch_mlp.") + role + ".weight");
        const bool down = std::strcmp(role, "down_proj") == 0;
        GGML_ASSERT(tensor->ne[0] == (down ? d.expert_ff : d.width) &&
                    tensor->ne[1] == (down ? d.width : d.expert_ff) && tensor->ne[2] == fixed::experts);
    }
    x.io_ctx = ggml_init({ggml_tensor_overhead() * 64, nullptr, true}); GGML_ASSERT(x.io_ctx);
    x.positions = ggml_new_tensor_1d(x.io_ctx, GGML_TYPE_I32, x.capacity);
    x.keys = ggml_new_tensor_3d(x.io_ctx, GGML_TYPE_Q8_0, d.head_dim, d.kv_heads, x.capacity);
    x.values = ggml_new_tensor_3d(x.io_ctx, GGML_TYPE_Q8_0, d.head_dim, d.kv_heads, x.capacity);
    const int vocab = x.model.tensor("output.weight")->ne[1];
    x.proposals = ggml_new_tensor_2d(x.io_ctx, GGML_TYPE_F32, vocab, max_drafts);
    x.probability = ggml_new_tensor_1d(x.io_ctx, GGML_TYPE_F32, vocab);
    x.parameters = ggml_new_tensor_1d(x.io_ctx, GGML_TYPE_F32, 5);
    x.sample_token = ggml_new_tensor_1d(x.io_ctx, GGML_TYPE_I32, 1);
    x.entropy = ggml_new_tensor_1d(x.io_ctx, GGML_TYPE_F32, 1);
    SOURIS_CUDA(cudaHostAlloc(&x.sample_host, sizeof(Impl::Scalars), cudaHostAllocDefault));
    x.make_io(x.prompt, prompt_rows); x.make_io(x.draft, 1);
    x.io_buffer = ggml_backend_alloc_ctx_tensors(x.io_ctx, x.device.backend); GGML_ASSERT(x.io_buffer);
    ggml_backend_buffer_clear(x.io_buffer, 0);
    x.build(x.prompt_graph, x.prompt, false); x.build(x.draft_graph, x.draft, true);
    x.build_distribution(x.greedy_graph, false); x.build_distribution(x.sampling_graph, true);
    x.prompt_graph.prepare(); x.draft_graph.prepare();
    x.greedy_graph.prepare(); x.sampling_graph.prepare();
    x.initialized = true;
}
void DraftRunner::capture() {
    auto & x = *impl_; GGML_ASSERT(x.initialized && !x.prompt_graph.executable);
    x.prompt_graph.capture(x.device); x.draft_graph.capture(x.device);
    x.greedy_graph.capture(x.device); x.sampling_graph.capture(x.device);
}
void DraftRunner::append_prompt(ggml_tensor * residual, const Token * next_tokens, int rows, int first_pos) {
    auto & x = *impl_; GGML_ASSERT(x.initialized && !x.primed);
    if (x.cursor < 0) { x.cursor = first_pos; }
    GGML_ASSERT(first_pos == x.cursor);
    GpuTimer timer(x.stats, x.device.compute, x.stats.mtp_prompt_ms);
    x.append(x.prompt, x.prompt_graph, residual, next_tokens, rows, first_pos);
    ggml_tensor last = *residual; last.data = static_cast<char *>(residual->data) + (rows - 1) * residual->nb[2];
    x.tail(&last, next_tokens[rows - 1], first_pos + rows - 1);
    x.cursor += rows;
    x.pending = next_tokens[rows - 1];
}
bool DraftRunner::step(Token first_token, ggml_tensor * last_residual, int first_pos,
                     Verify verify, void * owner, Emitted & emitted) {
    auto & x = *impl_; GGML_ASSERT(x.initialized && verify && first_pos > 0);
    const auto draft_start = StatsClock::now();
    // Prefill or the preceding commit already paired this residual and pending token.
    GGML_ASSERT(last_residual && last_residual->type == GGML_TYPE_F32 && x.cursor == first_pos && first_token == x.pending);
    x.primed = true;
    Token tokens[max_drafts]; int count = 0;
    do {
        const float uniform = float((x.random_bits(x.random_owner) >> 40) * 0x1.0p-24);
        const float parameters[] = {x.temperature > 0 ? x.temperature : 1, x.top_p, float(x.top_k), x.min_p, uniform};
        std::memcpy(x.sample_host->params, parameters, sizeof(parameters));
        SOURIS_CUDA(cudaMemcpyAsync(x.parameters->data, x.sample_host->params, sizeof(parameters),
                                   cudaMemcpyHostToDevice, x.device.compute));
        (x.temperature > 0 ? x.sampling_graph : x.greedy_graph).launch(x.device);
        const size_t bytes = ggml_nbytes(x.probability);
        SOURIS_CUDA(cudaMemcpyAsync(static_cast<char *>(x.proposals->data) + count * bytes,
            x.probability->data, bytes, cudaMemcpyDeviceToDevice, x.device.compute));
        SOURIS_CUDA(cudaMemcpyAsync(&x.sample_host->token, x.sample_token->data, sizeof(Token),
                                   cudaMemcpyDeviceToHost, x.device.compute));
        SOURIS_CUDA(cudaMemcpyAsync(&x.sample_host->entropy, x.entropy->data, sizeof(float),
                                   cudaMemcpyDeviceToHost, x.device.compute));
        SOURIS_CUDA(cudaStreamSynchronize(x.device.compute));
        tokens[count] = x.sample_host->token;
        const float root_entropy = x.sample_host->entropy;
        if (!std::isfinite(root_entropy) || root_entropy < 0) { GGML_ABORT("MTP entropy invalid"); }
        ++count;
        if (count == x.max_drafts || root_entropy > (x.temperature > 0 ? h_sample : h_greedy)) { break; }
        x.tail(x.draft.output, tokens[count - 1], first_pos + count - 1);
    } while (true);
    x.stats.mtp_draft_ms += milliseconds(draft_start);
    x.stats.mtp_drafts += count; ++x.stats.mtp_steps; ++x.stats.mtp_k[count - 1];
    const Request request{first_token, first_pos, count, tokens, static_cast<const float *>(x.proposals->data)};
    const auto verify_start = StatsClock::now();
    if (!verify(owner, request, emitted)) { GGML_ABORT("MTP target verify failed"); }
    x.stats.mtp_verify_ms += milliseconds(verify_start);
    if (emitted.count < 1 || emitted.count > count + 1 || emitted.tokens[0] != first_token ||
        emitted.next_token < 0 || !emitted.residuals) { GGML_ABORT("MTP emitted prefix invalid"); }
    Token next[max_drafts + 1];
    GpuTimer timer(x.stats, x.device.compute, x.stats.mtp_commit_ms);
    for (int i = 0; i < emitted.count; ++i) { next[i] = i + 1 < emitted.count ? emitted.tokens[i + 1] : emitted.next_token; }
    x.append(x.prompt, x.prompt_graph, emitted.residuals, next, emitted.count, first_pos);
    ggml_tensor last = *emitted.residuals;
    last.data = static_cast<char *>(last.data) + (emitted.count - 1) * last.nb[2];
    x.tail(&last, emitted.next_token, first_pos + emitted.count - 1);
    x.cursor = first_pos + emitted.count;
    x.pending = emitted.next_token;
    return true;
}
void DraftRunner::reset() {
    auto & x = *impl_; GGML_ASSERT(x.initialized);
    SOURIS_CUDA(cudaStreamSynchronize(x.device.compute));
    SOURIS_CUDA(cudaMemsetAsync(x.keys->data, 0, ggml_nbytes(x.keys), x.device.compute));
    SOURIS_CUDA(cudaMemsetAsync(x.values->data, 0, ggml_nbytes(x.values), x.device.compute));
    SOURIS_CUDA(cudaMemsetAsync(x.positions->data, 0xff, ggml_nbytes(x.positions), x.device.compute));
    x.cursor = -1; x.primed = false; x.pending = -1;
}
}

