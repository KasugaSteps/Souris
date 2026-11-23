#include "engine_internal.h"
#include "scheduler.h"
#include "ggml-cuda.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace souris {
HiddenLease::HiddenLease(int width, int hc, Device &, size_t count) : tokens(count) {
    context = ggml_init({ggml_tensor_overhead() * 4, nullptr, true});
    GGML_ASSERT(context);
    // Hidden stays FP32 on every device; the compact F16 path is not supported.
    tensor = ggml_new_tensor_3d(context, GGML_TYPE_F32, width, hc, count);
    bytes = ggml_nbytes(tensor);
}
HiddenLease::~HiddenLease() { GGML_ASSERT(!tail.data); ggml_free(context); }
void HiddenLease::borrow(SlotPool & storage, int slots) {
    GGML_ASSERT(!tail.data);
    tail = storage.borrow_tail(slots);
    pool = &storage;
    if (bytes > size_t(slots) * storage.layout().record_bytes) { GGML_ABORT("insufficient hidden slots"); }
    tensor->data = tail.data; tensor->buffer = tail.buffer;
}
void HiddenLease::release() {
    GGML_ASSERT(tail.data && pool);
    tensor->data = nullptr; tensor->buffer = nullptr; view = {};
    pool->return_tail(tail); pool = nullptr;
}
ggml_tensor * HiddenLease::slice(size_t offset, int count) {
    GGML_ASSERT(tail.data && count > 0 && offset <= tokens && size_t(count) <= tokens - offset);
    view = *tensor;
    view.ne[2] = count;
    view.nb[3] = tensor->nb[2] * count;
    view.data = static_cast<char *>(tensor->data) + offset * tensor->nb[2];
    return &view;
}
Engine::Impl::Impl(Engine & owner, Model & source, Device & gpu, Config & settings) :
    runner(owner), config(settings), device(gpu), model(source), repository(source.store()) {
    const auto start = StatsClock::now();
    if (config.context < fixed::warmup_tokens + 1 || config.read_threads < 1 ||
        config.mmq_group < 1 || config.mmq_group > 64 ||
        config.slru_protected < 1 || config.slru_protected > 99) { GGML_ABORT("Invalid startup configuration"); }
    state = std::make_unique<KVCacheManager>(model.dims(), config.context + (config.mtp.empty() ? 0 : DraftRunner::max_drafts), device);
    state->prepare_checkpoints();
    cache = std::make_unique<ExpertCache>(repository.layout(), config.expert_cache_bytes, config.serve);
    budget.cache_records = cache->capacity();
    pool = std::make_unique<SlotPool>(repository.layout(), device);
    staging = std::make_unique<StagingRing>(config.staging_bytes, repository.layout());
    transfer = std::make_unique<TransferEngine>(repository, device, *pool, *cache, *staging, config);
    stats.load_ms = milliseconds(start);
}
Engine::Impl::~Impl() { if (transfer) { transfer->drain(); } }
void Engine::Impl::plan_budget(size_t max_prompt_tokens) {
    GGML_ASSERT(!hidden && max_prompt_tokens > 0 && max_prompt_tokens <= size_t(config.context));
    const auto start = StatsClock::now();
    actual_max_prompt = max_prompt_tokens;
    hidden = std::make_unique<HiddenLease>(model.dims().width, model.dims().hc, device, std::max(max_prompt_tokens, size_t(fixed::warmup_tokens)));
    scheduler = std::make_unique<Scheduler>(*this);
    ggml_backend_cuda_prepare(device.backend, fixed::kernel_scratch);
    budget.prefill_workspace = scheduler->workspace_bytes();
    size_t available, total;
    SOURIS_CUDA(cudaMemGetInfo(&available, &total));
    budget.startup_free = available;
    const size_t record_bytes = repository.layout().record_bytes;
    if (available <= fixed::headroom) { GGML_ABORT("insufficient GPU expert slots"); }
    budget.slots = int(std::min(available - fixed::headroom, config.gpu_expert_bytes) / record_bytes);
    budget.workspace_slots = int((budget.prefill_workspace + record_bytes - 1) / record_bytes);
    const int hidden_records = int((hidden->bytes + record_bytes - 1) / record_bytes);
    budget.hidden_slots = (hidden_records + fixed::slot_chunk - 1) / fixed::slot_chunk * fixed::slot_chunk;
    budget.prefill_slots = budget.slots - budget.workspace_slots - budget.hidden_slots;
    if (budget.prefill_slots < fixed::experts) { GGML_ABORT("insufficient GPU expert slots"); }
    budget.capture_free_before = available;
    const size_t growth = ggml_backend_cuda_scratch_growth(device.backend, false);
    scheduler->capture_decode();
    budget.capture_scratch_allocations = ggml_backend_cuda_scratch_growth(device.backend, false) - growth;
    GGML_ASSERT(budget.capture_scratch_allocations == 0);
    SOURIS_CUDA(cudaMemGetInfo(&budget.capture_free_after, &total));
    const int64_t capture_bytes = int64_t(budget.capture_free_before) - int64_t(budget.capture_free_after);
    if (budget.capture_free_after < fixed::headroom || capture_bytes >= int64_t(fixed::headroom)) {
        GGML_ABORT("CUDA graph driver metadata exhausted the 1 GiB reserve");
    }
    pool->allocate(budget.slots, config.slru_protected, budget.prefill_slots, budget.hidden_slots);
    stats.load_ms += milliseconds(start);
}
void Engine::Impl::reset_request() {
    GGML_ASSERT(scheduler);
    transfer->drain();
    state->reset();
    scheduler->history.clear();
    scheduler->images.clear();
    scheduler->reset_draft();
    transfer->begin_request();
    stats.begin_request();
}
bool Engine::Impl::run_prefill(const Tokens & tokens, int offset, bool reset) {
    if (reset) { reset_request(); }
    if (tokens.empty() || tokens.size() > actual_max_prompt) {
        stats.ok = false; stats.error = "invalid prompt length"; return false;
    }
    const size_t vocabulary = model.tensor("token_embd.weight")->ne[1];
    for (Token token : tokens) {
        if (token < 0 || size_t(token) >= vocabulary) {
            stats.ok = false; stats.error = "token ID out of range"; return false;
        }
    }
    hidden->borrow(*pool, budget.hidden_slots);
    pool->borrow_workspace(budget.workspace_slots);
    transfer->configure(TransferEngine::Phase::Prefill);
    const auto start = StatsClock::now();
    const bool completed = scheduler->prefill(tokens, offset);
    transfer->drain();
    stats.prefill_ms = milliseconds(start);
    if (completed) {
        const auto sample_start = StatsClock::now();
        scheduler->read_logits(scheduler->prefill_io);
        stats.prefill_sample_ms = milliseconds(sample_start);
    }
    if (completed && scheduler->draft && !scheduler->grammar.active()) {
        const auto prompt_start = StatsClock::now();
        scheduler->append_draft_prompt(tokens, offset);
        stats.mtp_prompt_wall_ms = milliseconds(prompt_start);
    }
    pool->return_workspace();
    hidden->release();
    GGML_ASSERT(pool->capacity() == budget.slots);
    transfer->configure(TransferEngine::Phase::Decode);
    stats.resources();
    return completed;
}
bool Engine::Impl::run_decode(Token token, size_t remaining) {
    GGML_ASSERT(!hidden->tail.data && pool->capacity() == budget.slots);
    if (state->position >= config.context) {
        stats.ok = false; stats.error = "context length exceeded"; return false;
    }
    const auto start = StatsClock::now();
    scheduler->output_remaining = remaining;
    if (!scheduler->decode(token)) { return false; }
    stats.decode_ms += milliseconds(start);
    ++stats.generated_tokens;
    return true;
}
Token Engine::Impl::next() const {
    return scheduler->consume_sample();
}
void Engine::Impl::warmup() {
    const auto start = StatsClock::now();
    transfer->drain(); cache->clear(); pool->reset();
    const Tokens source = model.tokenizer().encode("The quick brown fox jumps over the lazy dog.\n");
    GGML_ASSERT(!source.empty());
    Tokens prompt(fixed::warmup_tokens);
    for (size_t i = 0; i < prompt.size(); ++i) { prompt[i] = source[i % source.size()]; }
    // Warmup uses the allocation floor without changing the real request limit.
    const size_t actual = actual_max_prompt;
    actual_max_prompt = hidden->tokens;
    if (!run_prefill(prompt) || !run_decode(next())) { GGML_ABORT("Warmup failed: %s", stats.error.c_str()); }
    actual_max_prompt = actual;
    scheduler->consume_sample();
    transfer->drain();
    stats.flush_gpu();
    if (process_swap()) { GGML_ABORT("Warmup used swap"); }
    size_t free, total;
    SOURIS_CUDA(cudaMemGetInfo(&free, &total));
    const double prefill_ms = stats.prefill_ms, decode_ms = stats.decode_ms;
    const double peak = double(stats.vram_peak) / fixed::mib;
    const double workspace = double(stats.prefill_workspace_peak) / fixed::mib;
    cache->clear(); pool->reset();
    reset_request();
    stats.warmup_ms = milliseconds(start);
    stats.warmup_prefill_ms = prefill_ms;
    stats.warmup_decode_ms = decode_ms;
    stats.warmup_free_vram_mib = double(free) / fixed::mib;
    stats.warmup_peak_vram_mib = peak;
    stats.warmup_prefill_workspace_mib = workspace;
}
void Engine::Impl::preload() {
    GGML_ASSERT(scheduler && !preload_started);
    preload_started = true;
    if (cache->layers() == 0) { return; }
    cache->begin_preload();
    const auto start = StatsClock::now();
    constexpr size_t records_per_read = fixed::extent_records;
    for (size_t layer = 0; layer < cache->layers(); ++layer) {
        for (size_t e = 0; e < fixed::experts; e += records_per_read) {
            const size_t first = layer * fixed::experts + e;
            repository.read(cache->preload_target(first), first * repository.layout().record_bytes,
                             records_per_read * repository.layout().record_bytes);
            cache->publish_preload(first, records_per_read);
        }
        preload_layers = layer + 1;
    }
    preload_ms = milliseconds(start);
}
Engine::Engine(Model & model, Device & device, Config & config) :
    impl_(std::make_unique<Impl>(*this, model, device, config)) {}
Engine::~Engine() = default;
Engine::Budget Engine::plan_budget(size_t tokens) {
    impl_->plan_budget(tokens);
    return {impl_->budget.slots, impl_->budget.prefill_slots, impl_->budget.slots, impl_->budget.workspace_slots, impl_->budget.hidden_slots};
}
void Engine::warmup() { impl_->warmup(); }
void Engine::finish_request() {
    auto & e = *impl_;
    e.transfer->drain();
    e.scheduler->consume_sample(false);
    SOURIS_CUDA(cudaStreamSynchronize(e.device.compute));
    SOURIS_CUDA(cudaStreamSynchronize(e.device.download));
    e.stats.flush_gpu();
    e.scheduler->cancelled = {};
    e.scheduler->prompt_checkpoint = -1;
}
void Engine::reset() { impl_->reset_request(); }
void Engine::preload() { impl_->preload(); }
void Engine::sampling(float temperature, float top_p, uint64_t seed, int top_k, float min_p) {
    GGML_ASSERT(std::isfinite(temperature) && temperature >= 0 && std::isfinite(top_p) && top_p > 0 && top_p <= 1);
    GGML_ASSERT(top_k >= 0 && std::isfinite(min_p) && min_p >= 0 && min_p <= 1);
    impl_->scheduler->temperature = temperature;
    impl_->scheduler->top_p = top_p;
    impl_->scheduler->top_k = top_k;
    impl_->scheduler->min_p = min_p;
    impl_->scheduler->random.seed(seed);
}
ggml_tensor * Engine::hidden_view(size_t offset, int count) { return impl_->hidden->slice(offset, count); }
Tokens Engine::tokenize(const std::string & text) const { return impl_->model.tokenizer().encode(text); }
std::string Engine::piece(Token token) const { return impl_->model.tokenizer().piece(token); }
bool Engine::eos(Token token) const { return impl_->model.tokenizer().eos(token); }
Token Engine::next() const { return impl_->next(); }

std::string Engine::startup() const {
    const auto & x = *impl_;
    return nlohmann::json{{"event", "startup"}, {"model", x.config.model.srs1_path},
        {"max_prompt_tokens", x.actual_max_prompt}, {"hidden_tokens", x.hidden->tokens},
        {"D_total", x.budget.slots}, {"D_prefill", x.budget.prefill_slots}, {"D_decode", x.budget.slots},
        {"hidden_slots", x.budget.hidden_slots}, {"workspace_slots", x.budget.workspace_slots},
        {"record_bytes", x.repository.layout().record_bytes}, {"cache_bytes", x.cache->bytes()},
        {"cache_layers", x.cache->layers()}, {"preload_layers", size_t(x.preload_layers)},
        {"preload_ms", double(x.preload_ms)}, {"mmq_group", x.config.mmq_group}, {"kv_type", "q8_0"}}.dump();
}
}
