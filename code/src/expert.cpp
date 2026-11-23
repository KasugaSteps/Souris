#include "expert.h"
#include "ggml-cuda.h"
#include "ops.h"
#include <algorithm>

namespace souris {
SlotPool::SlotPool(const Layout & layout, Device & device) :
    layout_(layout), device_(device) {
    context_ = ggml_init({ggml_tensor_overhead() * 16, nullptr, true});
    GGML_ASSERT(context_);
    table_ = ggml_new_tensor_3d(context_, GGML_TYPE_I64, fixed::experts, 3, fixed::layers);
    resident_ = ggml_new_tensor_2d(context_, GGML_TYPE_I32, fixed::experts, fixed::layers);
    table_buffer_ = ggml_backend_alloc_ctx_tensors(context_, device.backend);
    GGML_ASSERT(table_buffer_);
    ggml_backend_buffer_set_usage(table_buffer_, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    for (int layer = 0; layer < fixed::layers; ++layer) {
        int previous = 0;
        for (; previous < layer; ++previous) {
            const bool same = layout.quant_signature(previous) == layout.quant_signature(layer);
            if (same) { break; }
        }
        if (previous < layer) { groups_[layer] = groups_[previous]; }
        else {
            groups_[layer] = group_count_++;
        }
    }
    reset();
}
SlotPool::~SlotPool() {
    SOURIS_CUDA(cudaStreamSynchronize(device_.compute));
    SOURIS_CUDA(cudaStreamSynchronize(device_.upload));
    for (auto & slot : slots_) {
        SOURIS_CUDA(cudaEventDestroy(slot.copied));
        SOURIS_CUDA(cudaEventDestroy(slot.used));
    }
    for (auto & chunk : chunks_) { if (chunk.buffer) { ggml_backend_buffer_free(chunk.buffer); } }
    if (verify_done_) { SOURIS_CUDA(cudaEventDestroy(verify_done_)); }
    ggml_backend_buffer_free(verify_buffer_);
    ggml_backend_buffer_free(table_buffer_);
    ggml_free(context_);
}
void SlotPool::allocate_chunk(Chunk & chunk) {
    chunk.buffer = ggml_backend_alloc_buffer(device_.backend, size_t(chunk.count) * layout_.record_bytes);
    if (!chunk.buffer) { GGML_ABORT("insufficient GPU expert slots"); }
    ggml_backend_buffer_set_usage(chunk.buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    chunk.data = ggml_backend_buffer_get_base(chunk.buffer);
    for (int i = 0; i < chunk.count; ++i) {
        addresses_[chunk.begin + i] = static_cast<char *>(chunk.data) + size_t(i) * layout_.record_bytes;
    }
}
void SlotPool::allocate(int count, int protected_percent, int prefill_slots, int hidden_slots) {
    GGML_ASSERT(slots_.empty() && count >= prefill_slots && prefill_slots >= fixed::experts);
    protected_percent_ = protected_percent;
    slots_.resize(count);
    addresses_.resize(count);
    // Split the allocation at the locked loan boundary. Tail records can then
    // be returned exactly, while every retained prefix address stays stable.
    for (int begin = 0; begin < count;) {
        int end = std::min(begin + fixed::slot_chunk, count);
        if (begin < prefill_slots) { end = std::min(end, prefill_slots); }
        else if (begin < count - hidden_slots) { end = std::min(end, count - hidden_slots); }
        else { end = count; }
        chunks_.push_back({nullptr, begin, end - begin});
        allocate_chunk(chunks_.back());
        begin = end;
    }
    for (auto & slot : slots_) {
        SOURIS_CUDA(cudaEventCreateWithFlags(&slot.copied, cudaEventDisableTiming));
        SOURIS_CUDA(cudaEventCreateWithFlags(&slot.used, cudaEventDisableTiming));
    }
    active_ = count;
    reset();
}
void * SlotPool::address(int slot) const {
    GGML_ASSERT(slot >= 0 && slot < active_);
    return addresses_[slot];
}
void SlotPool::retire_tail(int begin) {
    std::lock_guard<std::mutex> lock(mutex_);
    free_.erase(std::remove_if(free_.begin(), free_.end(), [begin](int id) { return id >= begin; }), free_.end());
    for (int id = begin; id < active_; ++id) {
        auto & slot = slots_[id];
        GGML_ASSERT(!slot.owned && (slot.state == SlotState::Resident || slot.state == SlotState::Free));
        if (slot.key >= 0) {
            placement_[slot.key] = -1;
            unlink(id);
            cuda_expert_invalidate(table_->data, static_cast<int32_t *>(resident_->data), slot.key, id, device_.upload);
        }
        slot.key = slot.published = -1;
        slot.state = SlotState::Free;
    }
    active_ = begin;
    SOURIS_CUDA(cudaStreamSynchronize(device_.upload));
}
SlotPool::Tail SlotPool::borrow_tail(int count) {
    if (hidden_borrowed_ || !idle()) { GGML_ABORT("hidden lease still in use"); }
    auto & chunk = chunks_.back();
    if (chunk.count != count || chunk.begin != active_ - count || !chunk.buffer) { GGML_ABORT("insufficient hidden slots"); }
    retire_tail(chunk.begin);
    hidden_borrowed_ = true;
    return {chunk.data, chunk.buffer, chunk.begin, count};
}
void SlotPool::borrow_workspace(int count) {
    GGML_ASSERT(hidden_borrowed_ && !workspace_end_ && idle());
    workspace_end_ = active_;
    retire_tail(active_ - count);
    for (auto & chunk : chunks_) {
        if (chunk.begin < active_ || chunk.begin >= workspace_end_) { continue; }
        ggml_backend_buffer_free(chunk.buffer);
        chunk.buffer = nullptr; chunk.data = nullptr;
        std::fill(addresses_.begin() + chunk.begin, addresses_.begin() + chunk.begin + chunk.count, nullptr);
    }
}
void SlotPool::return_workspace() {
    GGML_ASSERT(workspace_end_ && idle());
    for (auto & chunk : chunks_) {
        if (chunk.begin >= active_ && chunk.begin < workspace_end_) { allocate_chunk(chunk); }
    }
    std::lock_guard<std::mutex> lock(mutex_);
    for (int slot = active_; slot < workspace_end_; ++slot) { free_.push_back(slot); }
    active_ = workspace_end_; workspace_end_ = 0;
}
void SlotPool::return_tail(Tail & tail) {
    if (!hidden_borrowed_ || workspace_end_ || !idle() || active_ != tail.begin) { GGML_ABORT("hidden lease still in use"); }
    std::lock_guard<std::mutex> lock(mutex_);
    for (int slot = tail.begin; slot < tail.begin + tail.count; ++slot) { free_.push_back(slot); }
    active_ += tail.count;
    GGML_ASSERT(active_ == int(slots_.size()));
    hidden_borrowed_ = false;
    tail = {};
}
void SlotPool::reset() {
    GGML_ASSERT(!hidden_borrowed_ && !workspace_end_ && !verify_open_);
    std::lock_guard<std::mutex> lock(mutex_);
    free_.clear();
    for (int s = 0; s < 2; ++s) { lru_[s] = mru_[s] = -1; sizes_[s] = 0; }
    for (int i = 0; i < int(slots_.size()); ++i) {
        auto & slot = slots_[i];
        GGML_ASSERT(!slot.owned && (slot.state == SlotState::Free || slot.state == SlotState::Resident));
        slot.key = slot.published = slot.previous = slot.next = -1;
        slot.state = SlotState::Free;
        slot.protected_segment = false;
        if (i < active_) { free_.push_back(i); }
    }
    std::fill(std::begin(placement_), std::end(placement_), -1);
    SOURIS_CUDA(cudaMemsetAsync(resident_->data, 0xff, ggml_nbytes(resident_), device_.upload));
    SOURIS_CUDA(cudaMemsetAsync(table_->data, 0, ggml_nbytes(table_), device_.upload));
}
void SlotPool::unlink(int id) {
    auto & slot = slots_[id];
    int segment = slot.protected_segment;
    if (slot.previous >= 0) { slots_[slot.previous].next = slot.next; }
    else { lru_[segment] = slot.next; }
    if (slot.next >= 0) { slots_[slot.next].previous = slot.previous; }
    else { mru_[segment] = slot.previous; }
    --sizes_[segment];
    slot.previous = slot.next = -1;
}
void SlotPool::touch(int id) {
    auto & slot = slots_[id];
    int segment = slot.protected_segment;
    slot.previous = mru_[segment];
    slot.next = -1;
    if (mru_[segment] >= 0) { slots_[mru_[segment]].next = id; }
    else { lru_[segment] = id; }
    mru_[segment] = id;
    ++sizes_[segment];
}
int SlotPool::pin(int key) {
    std::lock_guard<std::mutex> lock(mutex_);
    int id = placement_[key];
    if (id < 0) { return -1; }
    pin_slot(id);
    return id;
}
void SlotPool::pin_slots(int layer, const int * ids, int count) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (int i = 0; i < count; ++i) {
        const int id = ids[i];
        if (id < 0 || id >= active_ || slots_[id].key / fixed::experts != layer ||
            placement_[slots_[id].key] != id || slots_[id].published != slots_[id].key) {
            GGML_ABORT("invalid decode plan");
        }
        pin_slot(id);
    }
}
void SlotPool::pin_slot(int id) {
    auto & slot = slots_[id];
    GGML_ASSERT(!slot.owned && (slot.state == SlotState::Resident || slot.state == SlotState::Active));
    slot.owned = true;
    slot.state = SlotState::Active;
    promote(id);
}
void SlotPool::promote(int id) {
    auto & slot = slots_[id];
    unlink(id);
    slot.protected_segment = true;
    touch(id);
    if (sizes_[1] > active_ * protected_percent_ / 100) {
        int demoted = lru_[1];
        unlink(demoted);
        slots_[demoted].protected_segment = false;
        touch(demoted);
    }
}
int SlotPool::reserve(int key) {
    std::lock_guard<std::mutex> lock(mutex_);
    GGML_ASSERT(!verify_open_);
    return reserve_slot(key, false);
}
int SlotPool::reserve_slot(int key, bool provisional) {
    GGML_ASSERT(placement_[key] < 0);
    int id = -1;
    if (!free_.empty()) { id = free_.front(); free_.pop_front(); }
    else {
        for (int segment = 0; segment < (provisional ? 1 : 2) && id < 0; ++segment) {
            for (int candidate = lru_[segment]; candidate >= 0; candidate = slots_[candidate].next) {
                const auto & slot = slots_[candidate];
                if (slot.state == SlotState::Resident && !slot.owned) { id = candidate; break; }
            }
        }
        if (id < 0) { return -1; }
        placement_[slots_[id].key] = -1;
        unlink(id);
    }
    auto & slot = slots_[id];
    slot.key = key;
    slot.state = SlotState::Reserved;
    slot.owned = true;
    slot.protected_segment = false;
    slot.provisional = provisional;
    placement_[key] = id;
    if (!provisional) { touch(id); }
    return id;
}
void SlotPool::reading(int id) {
    std::lock_guard<std::mutex> lock(mutex_);
    GGML_ASSERT(slots_[id].state == SlotState::Reserved);
    slots_[id].state = SlotState::Inflight;
}
void SlotPool::activate(int id) {
    std::lock_guard<std::mutex> lock(mutex_);
    GGML_ASSERT(slots_[id].state == SlotState::Inflight && slots_[id].owned);
    slots_[id].state = slots_[id].provisional ? SlotState::Provisional : SlotState::Active;
}
int SlotPool::previous_key(int id) const { return slots_[id].published; }
void SlotPool::publish(int id, int key) { slots_[id].published = key; }
uint64_t SlotPool::release(int id, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto & slot = slots_[id];
    GGML_ASSERT(slot.owned && slot.state == SlotState::Active);
    SOURIS_CUDA(cudaEventRecord(slot.used, stream));
    slot.owned = false;
    return ++slot.retirement;
}
void SlotPool::release_batch(const int * ids, int count, cudaStream_t stream, uint64_t * tickets) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (int i = 0; i < count; ++i) {
        GGML_ASSERT(ids[i] >= 0 && ids[i] < active_);
        auto & slot = slots_[ids[i]];
        GGML_ASSERT(slot.owned && slot.state == SlotState::Active);
        SOURIS_CUDA(cudaEventRecord(slot.used, stream));
        slot.owned = false;
        tickets[i] = ++slot.retirement;
    }
}
bool SlotPool::reclaim(int id, uint64_t ticket) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto & slot = slots_[id];
    if (slot.retirement != ticket) { return true; }
    if (slot.owned || slot.state != SlotState::Active) { return false; }
    auto status = cudaEventQuery(slot.used);
    if (status == cudaErrorNotReady) { return false; }
    SOURIS_CUDA(status);
    slot.state = SlotState::Resident;
    return true;
}
bool SlotPool::idle() {
    std::lock_guard<std::mutex> lock(mutex_);
    return std::all_of(slots_.begin(), slots_.begin() + active_, [](const Slot & slot) {
        return !slot.owned && (slot.state == SlotState::Free || slot.state == SlotState::Resident);
    });
}
void SlotPool::prepare_verify() {
    GGML_ASSERT(!verify_buffer_);
    verify_entries_.reserve(fixed::verify_experts);
    verify_metadata_.resize(3 * fixed::verify_experts);
    verify_buffer_ = ggml_backend_alloc_buffer(device_.backend, verify_metadata_.size() * sizeof(int32_t));
    GGML_ASSERT(verify_buffer_);
    SOURIS_CUDA(cudaEventCreateWithFlags(&verify_done_, cudaEventDisableTiming));
}
void SlotPool::begin_verify() {
    std::lock_guard<std::mutex> lock(mutex_);
    GGML_ASSERT(verify_buffer_ && !verify_open_ && verify_entries_.empty());
    verify_open_ = true;
}
void SlotPool::pin_verify(const int * keys, const int * rows, int count) {
    std::lock_guard<std::mutex> lock(mutex_);
    GGML_ASSERT(verify_open_);
    for (int i = 0; i < count; ++i) {
        const int id = placement_[keys[i]];
        if (id < 0) { continue; }
        auto & slot = slots_[id];
        GGML_ASSERT(!slot.owned && !slot.provisional && slot.state == SlotState::Resident);
        slot.owned = true;
        slot.state = SlotState::Active;
        verify_entries_.push_back({keys[i], id, rows[i], false});
    }
}
int SlotPool::reserve_verify(int key, int rows) {
    std::lock_guard<std::mutex> lock(mutex_);
    GGML_ASSERT(verify_open_ && verify_entries_.size() < size_t(fixed::verify_experts));
    const int id = reserve_slot(key, true);
    if (id < 0) { GGML_ABORT("insufficient free/probation slots for verify transaction"); }
    verify_entries_.push_back({key, id, rows, true});
    return id;
}
SlotPool::VerifyResult SlotPool::finish_verify(int accepted, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(mutex_);
    GGML_ASSERT(verify_open_ && accepted >= 1 && accepted <= fixed::verify_rows);
    VerifyResult result;
    const int mask = (1 << accepted) - 1;
    for (const auto & entry : verify_entries_) {
        const auto & slot = slots_[entry.slot];
        GGML_ASSERT(slot.owned && slot.key == entry.key && slot.published == entry.key);
        if (!entry.provisional) { continue; }
        GGML_ASSERT(slot.state == SlotState::Provisional);
        int32_t * values = verify_metadata_.data() + 3 * result.provisional++;
        values[0] = entry.key;
        values[1] = entry.slot;
        values[2] = (entry.rows & mask) != 0;
    }
    // Only metadata is copied at commit. Expert records stay in their original
    // provisional slots. Complete table publication before making slots reusable.
    if (result.provisional) {
        void * metadata = ggml_backend_buffer_get_base(verify_buffer_);
        SOURIS_CUDA(cudaMemcpyAsync(metadata, verify_metadata_.data(),
            3 * result.provisional * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        cuda_expert_finish(table_->data, static_cast<int32_t *>(resident_->data),
            static_cast<const int32_t *>(metadata), result.provisional, stream);
    }
    SOURIS_CUDA(cudaEventRecord(verify_done_, stream));
    SOURIS_CUDA(cudaEventSynchronize(verify_done_));
    for (const auto & entry : verify_entries_) {
        auto & slot = slots_[entry.slot];
        const bool keep = (entry.rows & mask) != 0;
        slot.owned = false;
        slot.provisional = false;
        if (entry.provisional && !keep) {
            placement_[entry.key] = -1;
            slot.key = slot.published = -1;
            slot.state = SlotState::Free;
            free_.push_back(entry.slot);
            ++result.rolled_back;
            continue;
        }
        slot.state = SlotState::Resident;
        if (entry.provisional) { touch(entry.slot); ++result.committed; }
        else if (keep) { promote(entry.slot); ++result.hits; }
    }
    verify_entries_.clear();
    verify_open_ = false;
    return result;
}
}
