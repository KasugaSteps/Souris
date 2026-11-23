#include "scheduler.h"
#include "engine_internal.h"
#include "verify_engine.h"
#include "ggml-cuda.h"
#include <algorithm>

namespace souris {
void Scheduler::build_draft() {
    emitted_queue.reserve(DraftRunner::max_drafts);
    history.reserve(size_t(config.context) + DraftRunner::max_drafts);
    ggml_backend_cuda_prepare(device.backend, fixed::kernel_scratch);
    draft = std::make_unique<DraftRunner>(model, device, stats, temperature, top_p, top_k, min_p,
        [](void * owner) -> uint64_t { return static_cast<Scheduler *>(owner)->random(); }, this);
    draft->init(config.mtp, config.mtp_max_drafts, config.context);
    verify = std::make_unique<VerifyEngine>(owner, history, temperature, top_p, top_k, min_p, random, output_remaining);
}
void Scheduler::capture_draft() {
    if (draft) { draft->capture(); verify->capture(); }
}
void Scheduler::reset_draft() {
    emitted_queue.clear(); emitted_cursor = 0; pending_token = -1;
    if (draft) { draft->reset(); }
    draft_body = false;
}
void Scheduler::append_draft_prompt(const Tokens & tokens, int start) {
    const int total = tokens.size();
    const int first = std::max(start, total - fixed::mtp_window);
    for (int begin = first; begin < total - 1; begin += 256) {
        const int rows = std::min(256, total - 1 - begin);
        draft->append_prompt(owner.runner.hidden_view(begin - start, rows), tokens.data() + begin + 1, rows, begin);
    }
    auto * residual = owner.runner.hidden_view(total - 1 - start, 1);
    SOURIS_CUDA(cudaMemcpyAsync(decode_io.hidden->data, residual->data, ggml_nbytes(decode_io.hidden),
                               cudaMemcpyDeviceToDevice, device.compute));
    draft->append_prompt(residual, &sampled_token, 1, total - 1);
    // The borrowed target hidden must remain valid through the final prompt cell.
    SOURIS_CUDA(cudaStreamSynchronize(device.compute));
    stats.mtp_prompt_tokens = total - first;
}
bool Scheduler::decode_draft(Token token) {
    if (emitted_cursor < emitted_queue.size()) {
        GGML_ASSERT(token == emitted_queue[emitted_cursor]);
        ++emitted_cursor;
        sampled_token = emitted_cursor < emitted_queue.size() ? emitted_queue[emitted_cursor] : pending_token;
        observe_tool(sampled_token);
        return true;
    }
    if (output_remaining <= 1) { return true; }
    DraftRunner::Emitted emitted;
    draft->step(token, decode_io.hidden, state.position, VerifyEngine::dispatch, verify.get(), emitted);
    SOURIS_CUDA(cudaMemcpyAsync(decode_io.hidden->data,
        static_cast<char *>(emitted.residuals->data) + (emitted.count - 1) * emitted.residuals->nb[2],
        ggml_nbytes(decode_io.hidden), cudaMemcpyDeviceToDevice, device.compute));
    emitted_queue.assign(emitted.tokens + 1, emitted.tokens + emitted.count); emitted_cursor = 0;
    sampled_token = emitted_queue.empty() ? emitted.next_token : emitted_queue.front();
    pending_token = emitted.next_token;
    observe_tool(sampled_token);
    return true;
}
}
