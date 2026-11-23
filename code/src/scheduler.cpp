#include "scheduler.h"
#include "engine_internal.h"
#include "verify_engine.h"
#include "ggml-cuda.h"
#include <algorithm>
#include <array>
#include <cstring>

namespace souris {
namespace {
void post_layer(Frame & frame, GraphBuilder & graph, LayerIO & io, int layer, bool head = true) {
    auto * values = ggml_reshape_3d(frame.ctx, frame.bind(io.expert_output), io.routed->ne[0], fixed::topk, io.rows);
    frame.expand(ggml_cpy(frame.ctx, GraphBuilder::sum_experts(frame, values), frame.bind(io.routed)));
    graph.build_layer(frame, io, layer, GraphBuilder::Part::Post, head);
}
}
Scheduler::Scheduler(Engine::Impl & r) : owner(r), config(r.config), device(r.device), model(r.model),
    state(*r.state), budget(r.budget), transfer(*r.transfer), staging(*r.staging), pool(*r.pool), stats(r.stats),
    graph(model, state, device.backend), native(model, state, device.backend), moe(model),
    decode_io(model, 1, device, nullptr),
    prefill_io(model, int(std::min(size_t(std::min(fixed::default_chunk, config.context)), r.hidden->tokens)), device,
               r.hidden->tensor),
    tile(std::min(fixed::default_chunk, config.context)) {
    SOURIS_CUDA(cudaEventCreateWithFlags(&computed, cudaEventDisableTiming));
    build_sampling();
    build_decode();
    if (!config.mtp.empty()) { build_draft(); }
}
Scheduler::~Scheduler() {
    SOURIS_CUDA(cudaStreamSynchronize(device.compute));
    SOURIS_CUDA(cudaStreamSynchronize(device.download));
    SOURIS_CUDA(cudaEventDestroy(computed));
}
Scheduler::SampleMailbox::SampleMailbox() {
    SOURIS_CUDA(cudaHostAlloc(&host, sizeof(Packet), cudaHostAllocDefault));
    SOURIS_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    SOURIS_CUDA(cudaEventCreateWithFlags(&sampled, cudaEventDisableTiming));
    SOURIS_CUDA(cudaEventCreateWithFlags(&copied, cudaEventDisableTiming));
    SOURIS_CUDA(cudaEventCreateWithFlags(&uploaded, cudaEventDisableTiming));
}
Scheduler::SampleMailbox::~SampleMailbox() {
    SOURIS_CUDA(cudaStreamSynchronize(stream));
    if (params_pending) { SOURIS_CUDA(cudaEventSynchronize(uploaded)); }
    SOURIS_CUDA(cudaEventDestroy(sampled)); SOURIS_CUDA(cudaEventDestroy(copied));
    SOURIS_CUDA(cudaEventDestroy(uploaded)); SOURIS_CUDA(cudaStreamDestroy(stream));
    SOURIS_CUDA(cudaFreeHost(host));
}
Token Scheduler::consume_sample(bool account) {
    if (sample_mailbox.result_pending) {
        const auto start = StatsClock::now();
        SOURIS_CUDA(cudaEventSynchronize(sample_mailbox.copied));
        // The GPU dependency has already been enqueued. Charge deferred CPU
        // consumption to the original phase rather than inflating tok/s.
        if (account && sample_mailbox.deferred) {
            (sample_mailbox.first ? stats.prefill_sample_ms : stats.decode_ms) += milliseconds(start);
        }
        sampled_token = sample_mailbox.host->token;
        sample_mailbox.result_pending = false;
        GGML_ASSERT(sampled_token >= 0 && sampled_token < decode_io.logits->ne[0]);
    }
    return sampled_token;
}
Scheduler::RouteSlot::RouteSlot() {
    SOURIS_CUDA(cudaHostAlloc(&host, fixed::route_slot_bytes, cudaHostAllocDefault));
    SOURIS_CUDA(cudaEventCreateWithFlags(&done, cudaEventDisableTiming));
    SOURIS_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
}
Scheduler::RouteSlot::~RouteSlot() {
    SOURIS_CUDA(cudaStreamSynchronize(stream));
    SOURIS_CUDA(cudaStreamDestroy(stream));
    SOURIS_CUDA(cudaEventDestroy(done));
    SOURIS_CUDA(cudaFreeHost(host));
}
const int32_t * Scheduler::RouteSlot::download(const void * source, cudaEvent_t ready, Statistics & stats) {
    SOURIS_CUDA(cudaStreamWaitEvent(stream, ready, 0));
    {
        GpuTimer timer(stats, stream, stats.decode_route_d2h_ms);
        SOURIS_CUDA(cudaMemcpyAsync(host, source, fixed::route_plan_bytes, cudaMemcpyDeviceToHost, stream));
    }
    SOURIS_CUDA(cudaEventRecord(done, stream));
    SOURIS_CUDA(cudaEventSynchronize(done));
    return host;
}
StagingRing::Ring & Scheduler::control_ring() {
    return prefilling ? staging.prefill_ring() : staging.decode_ring();
}
void Scheduler::upload(void * target, const void * source, size_t bytes) {
    auto & ring = control_ring();
    size_t offset = 0;
    while (offset < bytes) {
        auto & slot = ring.acquire(0);
        const size_t count = std::min(bytes - offset, ring.control_bytes);
        std::memcpy(slot.host, static_cast<const char *>(source) + offset, count);
        GpuTimer timer(stats, device.compute,
            prefilling ? stats.prefill_control_h2d_ms : stats.decode_activation_h2d_ms);
        if (!prefilling) { stats.decode_activation_h2d_bytes += count; }
        // Activation leaves are reused by ordered compute fragments. The record
        // copy stream remains independent and can upload subsequent experts.
        SOURIS_CUDA(cudaMemcpyAsync(static_cast<char *>(target) + offset, slot.host, count,
                                   cudaMemcpyHostToDevice, device.compute));
        ring.record(0, device.compute);
        offset += count;
    }
}
void Scheduler::control(LayerIO & io, int position, int layer) {
    auto & ring = control_ring();
    auto & slot = ring.acquire(0);
    const Control value{position, io.rows, layer};
    std::memcpy(slot.host, &value, sizeof(value));
    GpuTimer timer(stats, device.compute,
        prefilling ? stats.prefill_control_h2d_ms : stats.decode_activation_h2d_ms);
    if (!prefilling) { stats.decode_activation_h2d_bytes += sizeof(value); }
    SOURIS_CUDA(cudaMemcpyAsync(io.control->data, slot.host, sizeof(value), cudaMemcpyHostToDevice, device.compute));
    ring.record(0, device.compute);
}
void Scheduler::inputs(LayerIO & io, int offset, int count, int layer) {
    auto & ring = control_ring();
    if (layer == 0 && !(!prefilling && device_input_ready && &io == &decode_io)) {
        upload(io.tokens->data, history.data() + offset, size_t(count) * sizeof(Token));
        for (const auto & image : images) {
            const int first = std::max(offset, image.start), last = std::min(offset + count, image.start + image.count);
            if (first < last) { io.images.push_back({first - offset, last - first, first - image.start, image.rows}); }
        }
    }
    if (layer == model.dims().ple_layer) {
        const size_t row_bytes = size_t(model.dims().ple_heads) * model.dims().ple_dim * sizeof(float);
        for (int begin = 0; begin < count;) {
            auto & slot = ring.acquire(0);
            const int rows = std::min(size_t(count - begin), ring.control_bytes / row_bytes);
            GGML_ASSERT(rows > 0);
            model.ple_rows(history, offset + begin, rows, static_cast<float *>(slot.host));
            GpuTimer timer(stats, device.compute,
                prefilling ? stats.prefill_control_h2d_ms : stats.decode_activation_h2d_ms);
            if (!prefilling) { stats.decode_activation_h2d_bytes += size_t(rows) * row_bytes; }
            SOURIS_CUDA(cudaMemcpyAsync(static_cast<char *>(io.ple->data) + size_t(begin) * row_bytes,
                slot.host, size_t(rows) * row_bytes, cudaMemcpyHostToDevice, device.compute));
            ring.record(0, device.compute);
            begin += rows;
        }
    }
    if (prefilling) { control(io, offset, layer); }
}
void Scheduler::upload_controls(int position) {
    const auto start = StatsClock::now();
    Control controls[fixed::layers];
    for (int layer = 0; layer < fixed::layers; ++layer) { controls[layer] = {position, 1, layer, 0}; }
    upload(decode_io.control_block->data, controls, sizeof(controls));
    stats.decode_control_h2d_ms += milliseconds(start);
    stats.decode_control_bytes = sizeof(controls);
}
void Scheduler::download(void * target, const void * source, size_t bytes, double * copy_time) {
    auto & ring = control_ring();
    size_t offset = 0;
    while (offset < bytes) {
        auto & slot = ring.acquire(1);
        const size_t count = std::min(bytes - offset, ring.control_bytes);
        SOURIS_CUDA(cudaEventRecord(computed, device.compute));
        SOURIS_CUDA(cudaStreamWaitEvent(device.download, computed, 0));
        {
            double & duration = prefilling ? stats.prefill_control_d2h_ms : stats.decode_activation_d2h_ms;
            GpuTimer timer(stats, device.download, duration, copy_time);
            if (!prefilling) { stats.decode_activation_d2h_bytes += count; }
            SOURIS_CUDA(cudaMemcpyAsync(slot.host, static_cast<const char *>(source) + offset, count,
                                       cudaMemcpyDeviceToHost, device.download));
        }
        ring.record(1, device.download);
        ring.wait(1);
        std::memcpy(static_cast<char *>(target) + offset, slot.host, count);
        offset += count;
    }
}
void Scheduler::build_sampling() {
    auto & g = greedy_sample.frame;
    auto * scores = g.bind(decode_io.logits);
    auto *argmax = ggml_argmax(g.ctx, scores);
    argmax->data = decode_io.sample_token->data; argmax->buffer = decode_io.sample_token->buffer;
    g.expand(argmax);
    auto & f = stochastic_sample.frame;
    auto * parameters = f.bind(decode_io.sample_params);
    auto * scratch = ggml_new_tensor_1d(f.ctx, GGML_TYPE_I8, sampling_scratch_bytes(decode_io.logits->ne[0]));
    ggml_tensor * args[] = {f.bind(decode_io.logits), parameters, scratch};
    f.expand(ggml_backend_cuda_custom_inplace(f.ctx, f.bind(decode_io.sample_token), args, 3, cuda_sample, nullptr));
    sample_arena.reserve(g); sample_arena.reserve(f);
    sample_arena.allocate(g); sample_arena.allocate(f); sample_arena.freeze();
}
void Scheduler::read_logits(LayerIO & io) {
    if (io.logits->data != decode_io.logits->data) {
        SOURIS_CUDA(cudaMemcpyAsync(decode_io.logits->data, io.logits->data, ggml_nbytes(io.logits),
                                   cudaMemcpyDeviceToDevice, device.compute));
    }
    if (grammar.active()) {
        ++tool_mask_tokens;
        const auto mask = grammar.mask();
        if (mask.empty()) { stats.error = grammar.error(); sampled_token = -1; return; }
        cuda_token_mask(static_cast<float *>(decode_io.logits->data), mask.device, mask.device_sparse,
                        mask.count, decode_io.logits->ne[0], device.compute);
    }
    if (temperature > 0) {
        const float params[] = {temperature, top_p, float(top_k), min_p, std::generate_canonical<float, 24>(random)};
        if (sample_mailbox.params_pending) { SOURIS_CUDA(cudaEventSynchronize(sample_mailbox.uploaded)); }
        std::memcpy(sample_mailbox.host->params, params, sizeof(params));
        SOURIS_CUDA(cudaMemcpyAsync(decode_io.sample_params->data, sample_mailbox.host->params,
                                   sizeof(params), cudaMemcpyHostToDevice, device.compute));
        SOURIS_CUDA(cudaEventRecord(sample_mailbox.uploaded, device.compute));
        sample_mailbox.params_pending = true;
        stochastic_sample.launch(device);
    } else {
        greedy_sample.launch(device);
    }
    auto &mailbox = sample_mailbox;
    GGML_ASSERT(!mailbox.result_pending);
    SOURIS_CUDA(cudaEventRecord(mailbox.sampled, device.compute));
    SOURIS_CUDA(cudaStreamWaitEvent(mailbox.stream, mailbox.sampled, 0));
    SOURIS_CUDA(cudaMemcpyAsync(&mailbox.host->token, decode_io.sample_token->data, sizeof(Token),
                               cudaMemcpyDeviceToHost, mailbox.stream));
    SOURIS_CUDA(cudaEventRecord(mailbox.copied, mailbox.stream));
    mailbox.result_pending = true;
    mailbox.first = &io == &prefill_io;
    mailbox.deferred = !tool_request && !draft;
    if (mailbox.deferred) {
        // Prepare only state-free work. EOS/stop/cancel can discard it without
        // advancing KV/GDN/PLE or changing the committed prefix.
        SOURIS_CUDA(cudaMemcpyAsync(decode_io.tokens->data, decode_io.sample_token->data, sizeof(Token),
                                   cudaMemcpyDeviceToDevice, device.compute));
        input_prologue.launch(device);
        device_input_ready = true;
    } else {
        consume_sample(false);
        observe_tool(sampled_token);
        device_input_ready = false;
    }
}
void Scheduler::observe_tool(Token token) {
    if (tool_request && !model.tokenizer().eos(token)) {
        const bool before = grammar.active();
        if (grammar.active()) {
            if (!grammar.advance(token)) { stats.error = grammar.error(); }
            else if (grammar.accepting()) { grammar.finish_tool_call(); }
        } else {
            ++tool_text_tokens;
            grammar.observe(token);
            if (!grammar.error().empty()) { stats.error = grammar.error(); }
        }
        if (!before && grammar.active()) { ++tool_mtp_entries; }
    }
}
void Scheduler::eager(Frame & frame, Arena & arena) {
    arena.allocate(frame);
    GGML_ASSERT(ggml_backend_cuda_compute_eager(device.backend, frame.graph) == GGML_STATUS_SUCCESS);
}
Scheduler::Fragment::~Fragment() {
    if (executable) { SOURIS_CUDA(cudaGraphExecDestroy(executable)); }
    if (source) { SOURIS_CUDA(cudaGraphDestroy(source)); }
}
void Scheduler::Fragment::capture(Device & device) {
    key = frame.key;
    SOURIS_CUDA(cudaStreamBeginCapture(device.compute, cudaStreamCaptureModeRelaxed));
    GGML_ASSERT(ggml_backend_cuda_compute_eager(device.backend, frame.graph) == GGML_STATUS_SUCCESS);
    SOURIS_CUDA(cudaStreamEndCapture(device.compute, &source));
    SOURIS_CUDA(cudaGraphInstantiate(&executable, source, nullptr, nullptr, 0));
}
void Scheduler::Fragment::launch(Device & device) { GGML_ASSERT(key == frame.key); SOURIS_CUDA(cudaGraphLaunch(executable, device.compute)); }

size_t Scheduler::workspace_bytes() {
    auto & x = *this;
    auto & storage = prefill_io;
    storage.hidden->data = storage.residual->data;
    storage.hidden->buffer = storage.residual->buffer;
    Frame frame;
    Arena measure;
    size_t maximum = 0;
    for (int layer = 0; layer < fixed::layers; ++layer) {
        frame.reset(); x.graph.build_layer(frame, storage, layer, GraphBuilder::Part::Before, true);
        maximum = std::max(maximum, measure.measure(frame));
        frame.reset(); post_layer(frame, x.graph, storage, layer, true);
        maximum = std::max(maximum, measure.measure(frame));
    }
    if (model.has_vision()) {
        // Reserve the maximal segment count, with almost a full tile of text.
        // These are allocation-only aliases, never encoder inputs or execution.
        for (int i = 0; i < 4 && 2*i+1 < storage.rows; ++i) { storage.images.push_back({2*i+1, 1, 0, storage.expert_input}); }
        frame.reset(); x.graph.build_layer(frame, storage, 0, GraphBuilder::Part::Before, true);
        maximum = std::max(maximum, measure.measure(frame));
        storage.images.clear();
    }
    storage.hidden->data = nullptr; storage.hidden->buffer = nullptr;
    return maximum + moe.workspace_bytes_;
}
bool Scheduler::prefill(const Tokens & tokens, int start) {
    GGML_ASSERT(!sample_mailbox.result_pending);
    device_input_ready = false;
    auto & x = *this;
    x.moe.begin_mmq();
    x.prefilling = true;
    x.history = tokens;
    const int total = int(tokens.size());
    GGML_ASSERT(start >= 0 && start < total);
    if (prompt_checkpoint >= 0) { state.capture_checkpoint(prompt_checkpoint); }
    auto & storage = prefill_io;
    *storage.hidden = *owner.runner.hidden_view(0, storage.rows);
    Arena arena;
    Frame frame;
    for (int layer = 0; layer < fixed::layers; ++layer) {
        frame.reset(); x.graph.build_layer(frame, storage, layer, GraphBuilder::Part::Before, true); arena.reserve(frame);
        frame.reset(); post_layer(frame, x.graph, storage, layer, true); arena.reserve(frame);
    }
    if (model.has_vision()) {
        for (int i = 0; i < 4 && 2*i+1 < storage.rows; ++i) { storage.images.push_back({2*i+1, 1, 0, storage.expert_input}); }
        frame.reset(); x.graph.build_layer(frame, storage, 0, GraphBuilder::Part::Before, true); arena.reserve(frame);
        storage.images.clear();
    }
    if (arena.bytes() > x.budget.prefill_workspace) {
        GGML_ABORT("insufficient Prefill expert slots");
    }
    arena.freeze();
    x.stats.prefill_workspace_peak = arena.bytes();
    std::vector<int> order, missing;
    order.reserve(fixed::experts); missing.reserve(fixed::experts);
    std::array<int32_t, fixed::experts> counts{}, selected{};
    x.stats.prompt_tokens = total;
    x.stats.resources();
    bool completed = true;
    for (int layer = 0; layer < fixed::layers; ++layer) {
        const auto layer_start = StatsClock::now();
        
        std::array<bool, fixed::experts> seen{};
        std::array<int, fixed::experts> cumulative{};
        std::array<TransferEngine::Lease, fixed::experts> leases;
        const auto admitted_before = x.transfer.counters().prefill.admitted;
        int union_size = 0;
        for (int offset = start; offset < total;) {
            int count = std::min(tile, total - offset);
            // A prefix stores state before its last token, never hidden or logits.
            if (prompt_checkpoint >= 0 && offset < total - 1) { count = std::min(count, total - 1 - offset); }
            LayerIO io(x.model, count, x.device, owner.runner.hidden_view(offset - start, count), &storage);
            x.inputs(io, offset, count, layer);
            const auto before_start = StatsClock::now();
            frame.reset(); x.graph.build_layer(frame, io, layer, GraphBuilder::Part::Before, true);
            { GpuTimer timer(x.stats, x.device.compute, x.stats.prefill_before_gpu_ms); x.eager(frame, arena); }
            if (prompt_checkpoint >= 0 && offset + count == total - 1) {
                state.capture_checkpoint_layer(prompt_checkpoint, layer, layer == model.dims().ple_layer);
            }
            x.stats.prefill_before_ms += milliseconds(before_start);
            ++x.stats.prefill_eager_tiles;
            {
                GpuTimer timer(x.stats, x.device.compute, x.stats.prefill_group_ms);
                x.moe.group(io, x.device.compute);
                x.moe.select(io, x.device.compute);
            }
            x.download(counts.data(), io.moe_group->data, sizeof(counts), &x.stats.prefill_counts_d2h_ms);
            x.stats.prefill_counts_d2h_bytes += sizeof(counts);
            order.clear(); missing.clear();
            std::array<bool, fixed::experts> admission{};
            for (int expert = 0; expert < fixed::experts; ++expert) {
                if (!counts[expert]) { continue; }
                cumulative[expert] += counts[expert];
                const int key = layer * fixed::experts + expert;
                order.push_back(key);
                if (seen[expert]) { x.stats.prefill_hits += counts[expert]; continue; }
                seen[expert] = true;
                ++union_size;
                leases[expert] = {key, x.pool.pin(key)};
                if (leases[expert].slot < 0) { missing.push_back(key); }
                else { x.stats.prefill_hits += counts[expert]; }
            }
            if (union_size > x.budget.prefill_slots) { GGML_ABORT("insufficient Prefill expert slots"); }
            auto physical_order = [&](int a, int b) {
                return x.model.store().record_offset(a / fixed::experts, a % fixed::experts) < x.model.store().record_offset(b / fixed::experts, b % fixed::experts);
            };
            std::sort(order.begin(), order.end(), physical_order);
            std::sort(missing.begin(), missing.end(), physical_order);
            for (size_t i = 0; i < missing.size(); ++i) {
                admission[i] = x.budget.cache_records >= fixed::experts ||
                    cumulative[missing[i] % fixed::experts] >= fixed::admission_count;
            }
            // Reserve all missing final slots before readers consume pinned blocks.
            const auto demand_start = StatsClock::now();
            x.transfer.demand(missing.data(), missing.size(), admission.data());
            x.stats.prefill_demand_ms += milliseconds(demand_start);
            for (size_t i = 0; i < order.size(); ++i) { selected[i] = order[i] % fixed::experts; }

            for (size_t begin = 0; begin < order.size(); begin += config.mmq_group) {
                const size_t end = std::min(begin + config.mmq_group, order.size());
                int tasks = 0;
                TransferEngine::Lease copied;
                for (size_t i = begin; i < end; ++i) {
                    const int expert = selected[i];
                    tasks += (counts[expert] + moe_tile_rows - 1) / moe_tile_rows;
                    if (leases[expert].slot < 0) {
                        leases[expert] = x.transfer.acquire(order[i]);
                    }
                    if (leases[expert].copy_sequence > copied.copy_sequence) { copied = leases[expert]; }
                }
                // All H2D/bind operations share one ordered stream. Its last
                // event in this batch covers the batch, without waiting for later batches.
                if (copied.copy_sequence) {
                    SOURIS_CUDA(cudaStreamWaitEvent(x.device.compute, x.pool.copied(copied.slot), 0));
                }
                GpuTimer timer(x.stats, x.device.compute, x.stats.prefill_moe_ms);
                x.moe.prefill(io, x.pool.table(), layer, int(begin), int(end), tasks, x.device.compute);
                ++x.stats.prefill_moe_batches;
            }
            const bool last = offset + count == total;
            const auto post_start = StatsClock::now();
            frame.reset(); post_layer(frame, x.graph, io, layer, last);
            { GpuTimer timer(x.stats, x.device.compute, x.stats.prefill_post_gpu_ms); x.eager(frame, arena); }
            x.stats.prefill_post_ms += milliseconds(post_start);
            offset += count;
        }
        for (int expert = 0; expert < fixed::experts; ++expert) {
            if (seen[expert]) { x.transfer.release(leases[expert], x.device.compute); }
        }
        x.stats.prefill_union[layer] = union_size;
        x.stats.prefill_routes += size_t(total - start) * fixed::topk;
        x.stats.prefill_admit[layer] = x.transfer.counters().prefill.admitted - admitted_before;
        x.stats.prefill_layer_ms[layer] = milliseconds(layer_start);
        x.stats.resources();
        if (cancelled && cancelled()) { completed = false; break; }
    }
    x.transfer.drain();
    x.moe.end_mmq();
    x.state.position = total;
    storage.hidden->data = nullptr; storage.hidden->buffer = nullptr;
    x.prefilling = false;
    return completed;
}
void Scheduler::build_decode() {
    auto & x = *this;
    native.prepare_decode(input_prologue.frame, decode_io);
    decode_arena.reserve(input_prologue.frame);
    for (int layer = 0; layer < fixed::layers; ++layer) {
        pre_[layer] = std::make_unique<Fragment>(); post_[layer] = std::make_unique<Fragment>();
        decode_io.control = decode_io.control_views[layer];
        NativeLayerKernels<1>(x.native).before(pre_[layer]->frame,decode_io,layer);
        {
            auto & f = pre_[layer]->frame;
            ggml_tensor * args[] = {f.bind(decode_io.route_ids), f.bind(pool.resident_table()),
                f.bind(decode_io.control), f.bind(decode_io.moe_selected), f.bind(decode_io.moe_group), f.bind(decode_io.layer_id)};
            f.expand(ggml_backend_cuda_custom_inplace(f.ctx, f.bind(decode_io.decode_plan), args, 6, cuda_decode_plan, nullptr));
        }
        decode_io.control = decode_io.control_views[0];
        NativeLayerPlan::sum_experts(post_[layer]->frame,post_[layer]->frame.bind(decode_io.expert_output),decode_io.routed);
        NativeLayerKernels<1>(x.native).after(post_[layer]->frame,decode_io,layer);
        decode_arena.reserve(pre_[layer]->frame); decode_arena.reserve(post_[layer]->frame);
        const int group = x.pool.group(layer);
        if (!expert_[group]) {
            expert_[group] = std::make_unique<Fragment>();
            moe.native_expert(expert_[group]->frame, decode_io, x.pool.pointer_table(), device.backend, layer, 1, true);
            decode_arena.reserve(expert_[group]->frame);
            for (int count = 1; count <= fixed::topk; ++count) {
                hit_[group][count - 1] = std::make_unique<Fragment>();
                moe.native_expert(hit_[group][count - 1]->frame, decode_io, x.pool.pointer_table(), device.backend, layer, count, false);
                decode_arena.reserve(hit_[group][count - 1]->frame);
            }
        }
    }
    decode_arena.freeze();
    decode_arena.allocate(input_prologue.frame);
    for (int layer = 0; layer < fixed::layers; ++layer) {
        decode_arena.allocate(pre_[layer]->frame); decode_arena.allocate(post_[layer]->frame);
    }
    for (int group = 0; group < x.pool.group_count(); ++group) {
        decode_arena.allocate(expert_[group]->frame);
        for (int rank = 0; rank < fixed::topk; ++rank) {
            decode_arena.allocate(hit_[group][rank]->frame);
        }
    }
}
void Scheduler::capture_decode() {
    auto & x = *this;
    greedy_sample.capture(device);
    stochastic_sample.capture(device);
    input_prologue.capture(device);
    for (int layer = 0; layer < fixed::layers; ++layer) { pre_[layer]->capture(x.device); post_[layer]->capture(x.device); }
    for (int group = 0; group < x.pool.group_count(); ++group) {
        expert_[group]->capture(x.device);
        for (int rank = 0; rank < fixed::topk; ++rank) {
            capture_hit(group, rank + 1);
        }
    }
    capture_draft();
}
void Scheduler::launch(Fragment & graph) {
    GpuTimer timer(stats, device.compute, stats.decode_gpu_ms);
    graph.launch(device);
}
void Scheduler::launch(Fragment & graph, double & target) {
    GpuTimer timer(stats, device.compute, target);
    graph.launch(device);
}
void Scheduler::expert(int layer, TransferEngine::Lease lease) {
    SOURIS_CUDA(cudaStreamWaitEvent(device.compute, pool.copied(lease.slot), 0));
    upload(decode_io.miss_index->data, &lease.ordinal, sizeof(lease.ordinal));
    launch(*expert_[pool.group(layer)], stats.decode_miss_ms);
    ++stats.decode_miss_graph_launches;
    transfer.release(lease, device.compute);
}
bool Scheduler::decode(Token token) {
    if (draft && !grammar.active()) { return decode_draft(token); }
    const size_t drafts_before = stats.mtp_drafts;
    if (draft && tool_request) { draft_body = true; }
    auto & x = *this;
    x.history.push_back(token);
    const int position = x.state.position;
    x.upload_controls(position);
    for (int layer = 0; layer < fixed::layers; ++layer) {
        if (cancelled && cancelled()) { transfer.drain(); return false; }
        x.inputs(decode_io, position, 1, layer);
        if (layer == 0 && !device_input_ready) { input_prologue.launch(device); }
        const auto route_start = StatsClock::now();
        launch(*pre_[layer], x.stats.decode_pre_ms);
        SOURIS_CUDA(cudaEventRecord(computed, x.device.compute));
        const int32_t * plan = route.download(decode_io.decode_plan->data, computed, x.stats);
        x.stats.decode_route_ms += milliseconds(route_start);
        const int hits = plan[0];
        int misses = plan[1];
        GGML_ASSERT(hits >= 0 && misses >= 0 && hits + misses == fixed::topk);
        x.pool.pin_slots(layer, plan + 2, hits);
        x.stats.decode_hits += hits;
        x.stats.decode_routes += fixed::topk;
        x.stats.decode_plan_d2h_bytes += fixed::route_plan_bytes;
        x.transfer.demand(plan + 2 + hits, misses);
        if (hits) {
            launch(*hit_[x.pool.group(layer)][hits - 1], x.stats.decode_hit_ms);
            ++x.stats.decode_hit_graph_launches;
            x.stats.decode_hit_graph_experts += hits;
            ++x.stats.decode_hit_counts[hits - 1];
        }
        const auto release_start = StatsClock::now();
        x.transfer.release_batch(plan + 2, hits, x.device.compute);
        x.stats.decode_release_batch_ms += milliseconds(release_start);
        ++x.stats.decode_release_batches;
        x.stats.decode_release_experts += hits;
        while (misses) {
            TransferEngine::Lease lease;
            if (!x.transfer.try_acquire(lease)) { x.transfer.wait_ready(); continue; }
            expert(layer, lease);
            --misses;
        }
        launch(*post_[layer], x.stats.decode_post_ms);
    }
    x.read_logits(decode_io);
    if (tool_request) { tool_body_mtp_drafts += stats.mtp_drafts - drafts_before; }
    if (draft && draft_body && !grammar.active()) {
        // Body tokens were consumed only by target Decode. Restart the drafter
        // from the last real target cell, without retaining stale proposal KV.
        draft->reset();
        draft->append_prompt(decode_io.hidden, &sampled_token, 1, position);
        draft_body = false;
    }
    // Logits already fence this token. Retire debug events here so profiling
    // storage stays bounded independently of the requested Decode length.
    x.stats.flush_gpu();
    ++x.state.position;
    return true;
}
void Scheduler::capture_hit(int dtype, int count) {
    if (count < 1 || count > fixed::topk || dtype < 0 || dtype >= pool.group_count()) { GGML_ABORT("invalid hit count"); }
    hit_[dtype][count - 1]->capture(device);
}
}
