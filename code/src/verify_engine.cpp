#include "verify_engine.h"
#include "engine_internal.h"
#include "native_layer.h"
#include "mtp_ops.h"
#include "stats.h"
#include "scheduler.h"
#include "ggml-cuda.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>

namespace souris {
struct VerifyEngine::Impl {
    static constexpr size_t mailbox_bytes = 1024;
    Engine::Impl & owner;
    const Model & model;
    Device & device;
    KVCacheManager & state;
    SlotPool & pool;
    StagingRing & staging;
    Statistics & stats;
    const Config & config;
    Tokens & history;
    const float & temperature, & top_p, & min_p;
    const int & top_k;
    std::mt19937_64 & random;
    const size_t & output_remaining;
    /** Captured target fragments share one startup-reserved arena. */
    struct Fragment {
        Frame frame;
        GraphKey key;
        cudaGraph_t source = nullptr;
        cudaGraphExec_t executable = nullptr;
        explicit Fragment(size_t nodes = 1024) : frame(nodes) {}
        ~Fragment() {
            if (executable) { SOURIS_CUDA(cudaGraphExecDestroy(executable)); }
            if (source) { SOURIS_CUDA(cudaGraphDestroy(source)); }
        }
        void capture(Device & device) {
            key = frame.key;
            SOURIS_CUDA(cudaStreamBeginCapture(device.compute, cudaStreamCaptureModeRelaxed));
            GGML_ASSERT(ggml_backend_cuda_compute_eager(device.backend, frame.graph) == GGML_STATUS_SUCCESS);
            SOURIS_CUDA(cudaStreamEndCapture(device.compute, &source));
            SOURIS_CUDA(cudaGraphInstantiate(&executable, source, nullptr, nullptr, 0));
        }
        void launch(Device & device) { GGML_ASSERT(key == frame.key); SOURIS_CUDA(cudaGraphLaunch(executable, device.compute)); }
    };
    KVCacheManager shadow;
    Impl(Engine::Impl & r, Tokens & h, const float & t, const float & p, const int & k, const float & m,
         std::mt19937_64 & rng, const size_t & remaining) : owner(r), model(r.model), device(r.device),
        state(*r.state), pool(*r.pool), staging(*r.staging), stats(r.stats),
        config(r.config), history(h), temperature(t), top_p(p), min_p(m), top_k(k), random(rng), output_remaining(remaining),
        shadow(model.dims(), state, device) { init(); }
    void upload(void *, const void *, size_t);
    void download(void *, const void *, size_t);
    void ple_input(LayerIO &, int position);
    void launch(Fragment & f, double & duration) {
        GpuTimer timer(stats, device.compute, duration); f.launch(device);
    }
    bool verify(const DraftRunner::Request &, DraftRunner::Emitted &);

    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    int32_t * host = nullptr;
    cudaEvent_t downloaded = nullptr;
    Arena arena;
    ggml_tensor * zero_record = nullptr, * zero_layer = nullptr;
    cudaStream_t download_stream = nullptr;
    cudaEvent_t computed = nullptr;
    struct Shape {
        int rows = 0;
        VerifyRows index_rows{};
        std::unique_ptr<LayerIO> io;
        std::unique_ptr<NativeLayerPlan> native, canonical;
        ggml_tensor * control = nullptr, *plan = nullptr, *ids = nullptr, *input = nullptr, *weights = nullptr;
        ggml_tensor * unique = nullptr, *mapping = nullptr, *pointers = nullptr;
        ggml_tensor *output = nullptr, *residuals = nullptr, *probabilities = nullptr, *parameters = nullptr, *tokens = nullptr;
        ggml_tensor *proposal = nullptr, *rejected_row = nullptr, *correction = nullptr, *residual_mass = nullptr;
        NativeGdnReplay gdn[fixed::layers]{};
        ggml_tensor * ple_input = nullptr;
        ggml_tensor *index_old[fixed::layers]{}, *index_new[fixed::layers]{};
        std::unique_ptr<Fragment> pre[fixed::layers], post[fixed::layers], expert[fixed::layers];
        std::unique_ptr<Fragment> snapshot, commit[fixed::verify_rows], probability, sample, rejection;
    } shapes[fixed::mtp_max_drafts];
    ~Impl() {
        SOURIS_CUDA(cudaStreamSynchronize(device.compute));
        SOURIS_CUDA(cudaStreamSynchronize(download_stream));
        for (auto & shape : shapes) {
            for (auto & f : shape.pre) { f.reset(); }
            for (auto & f : shape.post) { f.reset(); }
            for (auto & f : shape.expert) { f.reset(); }
            shape.snapshot.reset();
            for (auto & f : shape.commit) { f.reset(); }
            shape.probability.reset(); shape.sample.reset(); shape.rejection.reset();
        }
        ggml_backend_buffer_free(buffer); ggml_free(ctx);
        SOURIS_CUDA(cudaStreamDestroy(download_stream));
        SOURIS_CUDA(cudaEventDestroy(computed)); SOURIS_CUDA(cudaEventDestroy(downloaded));
        SOURIS_CUDA(cudaFreeHost(host));
    }
    void allocate(Shape & v, int rows) {
        auto & s = *this; const auto & d = s.model.dims();
        v.rows = rows; v.index_rows = {rows, d.index_dim};
        v.io = std::make_unique<LayerIO>(s.model, rows, s.device, nullptr);
        v.control = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, sizeof(Control));
        v.plan = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 2 + 4 * rows * fixed::topk);
        v.ids = v.io->route_ids;
        v.unique = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, fixed::topk * rows);
        v.mapping = ggml_dup_tensor(ctx, v.ids);
        v.pointers = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 3 * fixed::topk * rows);
        v.input = ggml_reshape_3d(ctx,v.io->expert_input,d.width,1,rows);
        v.weights = v.io->route_weights;
        v.output = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d.width, fixed::topk * rows);
        v.residuals = v.io->hidden;
        v.probabilities = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, s.model.tensor("output.weight")->ne[1], rows);
        v.parameters = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5);
        v.tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, rows);
        v.proposal = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, s.model.tensor("output.weight")->ne[1], fixed::mtp_max_drafts);
        v.rejected_row = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        v.correction = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        v.residual_mass = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
        v.ple_input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d.width, d.hc, rows);
        for (int layer = 0; layer < fixed::layers; ++layer) {
            if (d.recurrent[layer]) {
                auto & replay=v.gdn[layer];const int channels=d.state_dim*(2*d.key_heads+d.value_heads);
                replay.raw=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,channels,rows);
                replay.conv=ggml_dup_tensor(ctx,replay.raw);
                replay.q=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,d.state_dim,d.key_heads,rows);
                replay.k=ggml_dup_tensor(ctx,replay.q);
                replay.decay=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,d.value_heads,rows);
                replay.beta=ggml_dup_tensor(ctx,replay.decay);
            }
            else {
                v.index_old[layer] = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, d.index_dim, rows);
                v.index_new[layer] = ggml_dup_tensor(ctx, v.index_old[layer]);
            }
        }
    }
    static ggml_tensor * row_view(Frame & f, ggml_tensor * tensor, int row, int64_t n0, int64_t n1 = 1) {
        return ggml_view_2d(f.ctx, f.bind(tensor), n0, n1, n0 * sizeof(float), size_t(row) * n0 * n1 * sizeof(float));
    }
    template<int Rows> void build(Shape & v) {
        auto & s = *this; const auto & d = s.model.dims();
        v.native=std::make_unique<NativeLayerPlan>(s.model,shadow,s.device.backend);
        v.canonical=std::make_unique<NativeLayerPlan>(s.model,s.state,s.device.backend);
        NativeLayerKernels<Rows> graph(*v.native);auto & canonical=*v.canonical;
        v.snapshot = std::make_unique<Fragment>(2048);
        auto & initial = v.snapshot->frame;
        for (int layer = 0; layer < fixed::layers; ++layer) {
            if (!d.recurrent[layer]) { continue; }
            initial.expand(ggml_cpy(initial.ctx, initial.bind(s.state.layers[layer].conv), initial.bind(shadow.layers[layer].conv)));
            initial.expand(ggml_cpy(initial.ctx, initial.bind(s.state.layers[layer].recurrent), initial.bind(shadow.layers[layer].recurrent)));
        }
        initial.expand(ggml_cpy(initial.ctx, initial.bind(s.state.ple_history), initial.bind(shadow.ple_history)));
        for (int layer = 0; layer < fixed::layers; ++layer) {
            v.pre[layer] = std::make_unique<Fragment>(4096); v.post[layer] = std::make_unique<Fragment>(4096);
            auto & pre = v.pre[layer]->frame; auto & post = v.post[layer]->frame;
            if (!d.recurrent[layer]) {
                ggml_tensor * args[] = {pre.bind(s.state.layers[layer].index), pre.bind(v.control)};
                pre.expand(ggml_backend_cuda_custom_inplace(pre.ctx, pre.bind(v.index_old[layer]), args, 2, cuda_verify_index, &v.index_rows));
            }
            auto & io=*v.io;io.control=io.control_views[layer];
            io.gdn_replay=d.recurrent[layer]?&v.gdn[layer]:nullptr;
            io.ple_input=layer==d.ple_layer?v.ple_input:nullptr;
            graph.before(pre,io,layer);
            io.gdn_replay=nullptr;io.ple_input=nullptr;
            NativeLayerPlan::sum_experts(post,post.bind(v.output),io.routed);
            graph.after(post,io,layer,true);
            if (!d.recurrent[layer]) {
                ggml_tensor * args[] = {pre.bind(s.state.layers[layer].index), pre.bind(v.control)};
                pre.expand(ggml_backend_cuda_custom_inplace(pre.ctx, pre.bind(v.index_new[layer]), args, 2, cuda_verify_index, &v.index_rows));
            }
            ggml_tensor * args[] = {pre.bind(v.ids), pre.bind(s.pool.resident_table()), pre.bind(v.control),
                                   pre.bind(v.unique), pre.bind(v.mapping)};
            pre.expand(ggml_backend_cuda_custom_inplace(pre.ctx, pre.bind(v.plan), args, 5, cuda_verify_plan, nullptr));
            const int group = s.pool.group(layer);
            if (!v.expert[group]) {
                v.expert[group] = std::make_unique<Fragment>(); auto & f = v.expert[group]->frame;
                f.key = {-1, v.rows, fixed::topk * v.rows, s.model.store().layout().quant_signature(layer)};
                ggml_tensor * bind_args[] = {f.bind(s.pool.pointer_table()), f.bind(s.pool.resident_table()),
                    f.bind(v.unique), f.bind(v.control), f.bind(zero_record)};
                auto * resolved = ggml_backend_cuda_custom_inplace(f.ctx, f.bind(v.pointers), bind_args, 5,
                    cuda_verify_bind, const_cast<Layout *>(&s.pool.layout()));
                ggml_tensor * pointers[3], *weights[3];
                for (int role = 0; role < 3; ++role) {
                    pointers[role] = ggml_view_2d(f.ctx, resolved, fixed::topk * v.rows, 1,
                        3 * fixed::topk * v.rows * sizeof(void *), role * fixed::topk * v.rows * sizeof(void *));
                    weights[role] = f.bind(s.model.expert(layer, role));
                    weights[role]->data = pointers[role]->data; weights[role]->buffer = v.pointers->buffer;
                }
                auto * layer_id = f.bind(zero_layer);
                auto * hidden = ggml_backend_cuda_native_expert(f.ctx, s.device.backend, weights[1], f.bind(v.input), f.bind(v.io->native_expert_q8),
                    f.bind(v.mapping), pointers[1], layer_id, pointers[0]);
                auto * quantized=ggml_backend_cuda_native_quantize(f.ctx,s.device.backend,hidden);
                auto * output = ggml_backend_cuda_native_expert(f.ctx, s.device.backend, weights[2], hidden, quantized,
                    f.bind(v.mapping), pointers[2], layer_id, nullptr);
                ggml_tensor * weighted[] = {ggml_reshape_2d(f.ctx, output, d.width, v.rows * fixed::topk), f.bind(v.weights)};
                f.expand(ggml_backend_cuda_custom_inplace(f.ctx, f.bind(v.output), weighted, 2, cuda_verify_weight, nullptr));
            }
        }
        for (int accepted = 1; accepted <= v.rows; ++accepted) {
            v.commit[accepted - 1] = std::make_unique<Fragment>(16384);
            auto & f = v.commit[accepted - 1]->frame;
            for (int layer = 0; layer < fixed::layers; ++layer) {
                if (d.recurrent[layer]) {
                    canonical.replay_gdn(f,v.gdn[layer],v.control,layer,accepted);
                } else {
                    ggml_tensor * args[] = {f.bind(v.index_old[layer]), f.bind(v.index_new[layer]), f.bind(v.control)};
                    f.expand(ggml_backend_cuda_custom_inplace(f.ctx, f.bind(s.state.layers[layer].index), args, 3, cuda_verify_index, &v.index_rows));
                }
            }
            canonical.replay_ple(f,v.ple_input,v.control);
        }
        v.probability = std::make_unique<Fragment>(2048); v.sample = std::make_unique<Fragment>(2048);
        for (int mode = 0; mode < 2; ++mode) {
            auto & f = (mode ? v.sample : v.probability)->frame;
            for (int row = 0; row < v.rows; ++row) {
                auto * scores = row_view(f,v.io->logits,row,v.io->logits->ne[0]), *params = f.bind(v.parameters);
                auto * token = ggml_view_1d(f.ctx, f.bind(v.tokens), 1, row * sizeof(Token));
                if (!mode) { f.expand(ggml_cpy(f.ctx, ggml_argmax(f.ctx, scores), token)); continue; }
                auto * q = scores;
                auto * scratch = ggml_new_tensor_1d(f.ctx, GGML_TYPE_I8, sampling_scratch_bytes(q->ne[0]));
                ggml_tensor * args[] = {scores, params, token, scratch};
                q = ggml_backend_cuda_custom_4d(f.ctx, GGML_TYPE_F32, q->ne[0], 1, 1, 1, args, 4, cuda_sample, nullptr);
                f.expand(ggml_cpy(f.ctx, q, row_view(f, v.probabilities, row, q->ne[0])));
            }
        }
        v.rejection = std::make_unique<Fragment>();
        auto & f = v.rejection->frame;
        auto * p = ggml_get_rows(f.ctx, f.bind(v.probabilities), f.bind(v.rejected_row));
        auto * q = ggml_get_rows(f.ctx, f.bind(v.proposal), f.bind(v.rejected_row));
        auto * residual = ggml_clamp(f.ctx, ggml_sub(f.ctx, p, q), 0, INFINITY);
        auto * mass = ggml_sum_rows(f.ctx, residual);
        f.expand(ggml_cpy(f.ctx, mass, f.bind(v.residual_mass)));
        residual = ggml_div(f.ctx, residual, mass);
        residual = ggml_reshape_1d(f.ctx, residual, residual->ne[0]);
        auto * scratch = ggml_new_tensor_1d(f.ctx, GGML_TYPE_I8, sampling_scratch_bytes(residual->ne[0]));
        ggml_tensor * args[] = {residual, f.bind(v.parameters), scratch};
        f.expand(ggml_backend_cuda_custom_inplace(f.ctx, f.bind(v.correction), args, 3, cuda_sample, reinterpret_cast<void *>(1)));
    }
    void reserve(Fragment & f) { arena.reserve(f.frame); }
    void allocate(Fragment & f) { arena.allocate(f.frame); }
    void capture(Fragment & f) { f.capture(device); }
    enum class Action { Reserve, Allocate, Capture };
    void apply(Fragment & f, Action action) {
        switch (action) {
            case Action::Reserve: reserve(f); break;
            case Action::Allocate: allocate(f); break;
            case Action::Capture: capture(f); break;
        }
    }
    void each(Action action) {
        for (auto & v : shapes) {
            apply(*v.snapshot, action); apply(*v.probability, action); apply(*v.sample, action); apply(*v.rejection, action);
            for (int layer = 0; layer < fixed::layers; ++layer) { apply(*v.pre[layer], action); apply(*v.post[layer], action); }
            for (int group = 0; group < pool.group_count(); ++group) { apply(*v.expert[group], action); }
            for (int n = 0; n < v.rows; ++n) { apply(*v.commit[n], action); }
        }
    }
    void init() {
        pool.prepare_verify();
        ctx = ggml_init({ggml_tensor_overhead() * (fixed::layers*8+64)*fixed::mtp_max_drafts, nullptr, true}); GGML_ASSERT(ctx);
        SOURIS_CUDA(cudaHostAlloc(&host, mailbox_bytes, cudaHostAllocDefault));
        SOURIS_CUDA(cudaEventCreateWithFlags(&downloaded, cudaEventDisableTiming));
        SOURIS_CUDA(cudaStreamCreateWithFlags(&download_stream, cudaStreamNonBlocking));
        SOURIS_CUDA(cudaEventCreateWithFlags(&computed, cudaEventDisableTiming));
        zero_record = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, pool.layout().record_bytes);
        zero_layer = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        for (int t = 2; t <= fixed::verify_rows; ++t) { allocate(shapes[t - 2], t); }
        buffer = ggml_backend_alloc_ctx_tensors(ctx, device.backend); GGML_ASSERT(buffer);
        ggml_backend_buffer_clear(buffer, 0);
        build<2>(shapes[0]);build<3>(shapes[1]);build<4>(shapes[2]);build<5>(shapes[3]);
        each(Action::Reserve); each(Action::Allocate); arena.freeze();

    }
};

bool VerifyEngine::Impl::verify(const DraftRunner::Request & request, DraftRunner::Emitted & emitted) {
    auto & s = *this; auto & x = *this;
    auto & v = x.shapes[request.drafts - 1]; const int rows = v.rows;
    owner.transfer->drain();
    pool.begin_verify();
    const auto before = owner.transfer->counters().verify;
    const size_t saved_history = s.history.size();
    s.history.push_back(request.first_token);
    for (int row = 1; row < rows; ++row) { s.history.push_back(request.tokens[row - 1]); }
    v.snapshot->launch(s.device);
    Control controls[fixed::layers];
    for(int layer=0;layer<fixed::layers;++layer)controls[layer]={request.first_pos,rows,layer,0};
    s.upload(v.io->control_block->data,controls,sizeof(controls));
    s.upload(v.io->tokens->data,s.history.data()+request.first_pos,rows*sizeof(Token));
    for (int layer = 0; layer < fixed::layers; ++layer) {
        Control control{request.first_pos, rows, layer, 0}; s.upload(v.control->data, &control, sizeof(control));
        if (layer == s.model.dims().ple_layer) {
            s.ple_input(*v.io,request.first_pos);
        }
        s.launch(*v.pre[layer], s.stats.decode_pre_ms);
        SOURIS_CUDA(cudaEventRecord(s.computed, s.device.compute));
        SOURIS_CUDA(cudaStreamWaitEvent(s.download_stream, s.computed, 0));
        SOURIS_CUDA(cudaMemcpyAsync(x.host, v.plan->data, ggml_nbytes(v.plan), cudaMemcpyDeviceToHost, s.download_stream));
        SOURIS_CUDA(cudaEventRecord(x.downloaded, s.download_stream)); SOURIS_CUDA(cudaEventSynchronize(x.downloaded));
        const int hits = x.host[0], misses = x.host[1];
        GGML_ASSERT(hits >= 0 && misses >= 0 && hits + misses <= rows * fixed::topk);
        s.stats.decode_hits += hits; s.stats.decode_routes += hits + misses;
        const int entries = rows * fixed::topk;
        const int * union_keys = x.host + 2 + 2 * entries;
        const int * row_masks = x.host + 2 + 3 * entries;
        const int * missing = x.host + 2 + entries;
        // Pin every original hit before any provisional reservation can evict.
        s.pool.pin_verify(union_keys, row_masks, hits + misses);
        int missing_rows[fixed::verify_rows * fixed::topk];
        for (int i = 0; i < misses; ++i) {
            const int * found = std::find(union_keys, union_keys + hits + misses, missing[i]);
            GGML_ASSERT(found != union_keys + hits + misses);
            missing_rows[i] = row_masks[found - union_keys];
        }
        s.owner.transfer->demand_verify(missing, missing_rows, misses);
        TransferEngine::Lease last;
        for (int i = 0; i < misses; ++i) {
            auto lease = s.owner.transfer->acquire(missing[i]);
            if (lease.copy_sequence > last.copy_sequence) { last = lease; }
        }
        if (misses) { SOURIS_CUDA(cudaStreamWaitEvent(s.device.compute, s.pool.copied(last.slot), 0)); }
        s.launch(*v.expert[s.pool.group(layer)], s.stats.decode_hit_ms);
        s.launch(*v.post[layer], s.stats.decode_post_ms);
    }
    const float parameters[] = {s.temperature > 0 ? s.temperature : 1, s.top_p, float(s.top_k), s.min_p, std::generate_canonical<float, 24>(s.random)};
    s.upload(v.parameters->data, parameters, sizeof(parameters));
    (s.temperature > 0 ? v.sample : v.probability)->launch(s.device);
    Token predictions[fixed::verify_rows]; s.download(predictions, v.tokens->data, rows * sizeof(Token));
    int accepted = 0; bool accepted_eos = false;
    const size_t vocab = s.model.tensor("output.weight")->ne[1];
    for (; accepted < request.drafts; ++accepted) {
        const Token draft = request.tokens[accepted];
        if (s.temperature == 0) { if (predictions[accepted] != draft) { break; } }
        else {
            float p, q;
            s.download(&p, static_cast<float *>(v.probabilities->data) + accepted * vocab + draft, sizeof(float));
            s.download(&q, request.probabilities + accepted * vocab + draft, sizeof(float));
            if (!(q > 0) || !std::isfinite(p + q)) { GGML_ABORT("MTP rejection probabilities invalid"); }
            if (std::generate_canonical<double, 53>(s.random) >= std::min(1.0, double(p) / q)) { break; }
        }
        if (s.owner.runner.eos(draft)) { accepted_eos = true; break; }
    }
    int count = accepted + 1;
    if (s.output_remaining != SIZE_MAX) { count = std::min(size_t(count), s.output_remaining - 1); }
    count = std::min(count, s.config.context - request.first_pos);
    if (owner.scheduler->tool_request) {
        // first_token has already been observed. Leave the marker-completing
        // token pending so target Decode samples its body under the mask.
        count = std::min(count, 1 + owner.scheduler->grammar.text_prefix(request.tokens, count - 1));
    }
    GGML_ASSERT(count >= 1);
    emitted.count = count; emitted.tokens[0] = request.first_token;
    s.stats.mtp_accepted += std::min(accepted, count - 1);
    s.stats.mtp_committed += count; s.stats.mtp_verify_tokens += rows;
    for (int i = 1; i < count; ++i) { emitted.tokens[i] = request.tokens[i - 1]; }
    emitted.next_token = count <= accepted ? request.tokens[count - 1] : predictions[count - 1];
    if (accepted_eos && count == accepted + 1) { emitted.next_token = request.tokens[accepted]; }
    // Sampling rejection needs a sample from positive (p-q), never the original p.
    if (s.temperature > 0 && !accepted_eos && count == accepted + 1 && accepted < request.drafts) {
        const float residual_parameters[] = {1, 1, 0, 0, std::generate_canonical<float, 24>(s.random)};
        s.upload(v.parameters->data, residual_parameters, sizeof(residual_parameters));
        s.upload(v.rejected_row->data, &accepted, sizeof(accepted));
        SOURIS_CUDA(cudaMemcpyAsync(v.proposal->data, request.probabilities, request.drafts * vocab * sizeof(float),
                                   cudaMemcpyDeviceToDevice, s.device.compute));
        v.rejection->launch(s.device); s.download(&emitted.next_token, v.correction->data, sizeof(Token));
        float mass; s.download(&mass, v.residual_mass->data, sizeof(mass));
        if (!(mass > 0) || !std::isfinite(mass)) { GGML_ABORT("MTP rejection residual has no finite mass"); }
        GGML_ASSERT(emitted.next_token >= 0 && emitted.next_token < int(vocab));
    }
    const Control commit{request.first_pos, count, 0, 0}; s.upload(v.control->data, &commit, sizeof(commit));
    { GpuTimer timer(s.stats, s.device.compute, s.stats.mtp_commit_ms); v.commit[count - 1]->launch(s.device); }
    // Keep every hit/provisional lease until the accepted target state is committed.
    SOURIS_CUDA(cudaEventRecord(s.computed, s.device.compute));
    SOURIS_CUDA(cudaEventSynchronize(s.computed));
    s.history.resize(saved_history); s.history.insert(s.history.end(), emitted.tokens, emitted.tokens + count);
    s.state.position += count;
    emitted.residuals = v.residuals;
    const auto result = s.pool.finish_verify(count, s.device.compute);
    s.owner.transfer->drain();
    const auto after = s.owner.transfer->counters().verify;
    s.stats.mtp_provisional_experts += result.provisional;
    s.stats.mtp_provisional_peak = std::max(s.stats.mtp_provisional_peak, size_t(result.provisional));
    s.stats.mtp_committed_experts += result.committed;
    s.stats.mtp_rolled_back_experts += result.rolled_back;
    s.stats.mtp_accepted_hits += result.hits;
    s.stats.mtp_verify_read_bytes += after.read_bytes - before.read_bytes;
    s.stats.mtp_verify_h2d_bytes += after.h2d_bytes - before.h2d_bytes;
    s.stats.mtp_verify_ram_hits += after.ram_hits - before.ram_hits;
    s.stats.mtp_verify_ssd_miss += after.ssd_misses - before.ssd_misses;
    s.stats.mtp_verify_extents += after.extents - before.extents;
    s.stats.mtp_verify_read_ms += after.read_ms - before.read_ms;
    s.stats.mtp_verify_copy_submit_ms += after.copy_submit_ms - before.copy_submit_ms;
    s.stats.mtp_verify_wait_ms += after.wait_ms - before.wait_ms;
    s.stats.flush_gpu();
    return true;
}

void VerifyEngine::Impl::upload(void * target, const void * source, size_t bytes) {
    auto & ring = staging.decode_ring();
    GGML_ASSERT(bytes <= ring.control_bytes);
    auto & slot = ring.acquire(0);
    std::memcpy(slot.host, source, bytes);
    SOURIS_CUDA(cudaMemcpyAsync(target, slot.host, bytes, cudaMemcpyHostToDevice, device.compute));
    ring.record(0, device.compute);
    stats.decode_activation_h2d_bytes += bytes;
}
void VerifyEngine::Impl::download(void * target, const void * source, size_t bytes) {
    auto & ring = staging.decode_ring();
    GGML_ASSERT(bytes <= ring.control_bytes);
    auto & slot = ring.acquire(1);
    SOURIS_CUDA(cudaEventRecord(computed, device.compute));
    SOURIS_CUDA(cudaStreamWaitEvent(download_stream, computed, 0));
    SOURIS_CUDA(cudaMemcpyAsync(slot.host, source, bytes, cudaMemcpyDeviceToHost, download_stream));
    ring.record(1, download_stream); ring.wait(1);
    std::memcpy(target, slot.host, bytes);
    stats.decode_activation_d2h_bytes += bytes;
}
void VerifyEngine::Impl::ple_input(LayerIO & io, int position) {
    auto & ring = staging.decode_ring();
    auto & slot = ring.acquire(0);
    const size_t bytes = ggml_nbytes(io.ple);
    GGML_ASSERT(bytes <= ring.control_bytes);
    model.ple_rows(history, position, io.rows, static_cast<float *>(slot.host));
    SOURIS_CUDA(cudaMemcpyAsync(io.ple->data, slot.host, bytes, cudaMemcpyHostToDevice, device.compute));
    ring.record(0, device.compute);
    stats.decode_activation_h2d_bytes += bytes;
}
VerifyEngine::VerifyEngine(Engine::Impl & owner, Tokens & history, const float & temperature,
                           const float & top_p, const int & top_k, const float & min_p, std::mt19937_64 & random, const size_t & remaining) :
    impl_(std::make_unique<Impl>(owner, history, temperature, top_p, top_k, min_p, random, remaining)) {}
VerifyEngine::~VerifyEngine() = default;
void VerifyEngine::capture() { impl_->each(Impl::Action::Capture); }
bool VerifyEngine::verify(const DraftRunner::Request & request, DraftRunner::Emitted & emitted) {
    return impl_->verify(request, emitted);
}
bool VerifyEngine::dispatch(void * owner, const DraftRunner::Request & request, DraftRunner::Emitted & emitted) {
    return static_cast<VerifyEngine *>(owner)->verify(request, emitted);
}
}
