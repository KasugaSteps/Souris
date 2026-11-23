#include "transfer.h"
#include "ops.h"
#include "stats.h"
#include <algorithm>
#include <cstring>

namespace souris {
TransferEngine::TransferEngine(const ExpertRepository & repository, Device & device, SlotPool & pool,
                               ExpertCache & cache, StagingRing & staging, const Config & config) :
    repository_(repository), layout_(pool.layout()), device_(device), pool_(pool), cache_(cache),
    blocks_(staging.prefill_ring().expert_bytes / layout_.record_bytes) {
    if (blocks_.size() < size_t(fixed::staging_min_records)) {
        GGML_ABORT("Staging must hold at least 64 * record_bytes in pinned host memory");
    }
    for (size_t i = 0; i < blocks_.size(); ++i) {
        auto & block = blocks_[i];
        block.data = static_cast<char *>(staging.prefill_ring().data) + i * layout_.record_bytes;
        SOURIS_CUDA(cudaEventCreateWithFlags(&block.copied, cudaEventDisableTiming));
    }
    for (int i = 0; i < config.read_threads; ++i) { readers_.emplace_back(&TransferEngine::read, this, false); }
    ram_stager_ = std::thread(&TransferEngine::read, this, true);
    copier_ = std::thread(&TransferEngine::copy, this);
}
TransferEngine::~TransferEngine() {
    drain();
    { std::lock_guard<std::mutex> lock(mutex_); stopping_ = true; }
    changed_.notify_all();
    for (auto & reader : readers_) { reader.join(); }
    ram_stager_.join();
    copier_.join();
    for (auto & block : blocks_) { SOURIS_CUDA(cudaEventDestroy(block.copied)); }
}
TransferEngine::PhaseCounters & TransferEngine::phase_counters(bool provisional) {
    if (provisional) { return counters_.verify; }
    return phase_ == Phase::Prefill ? counters_.prefill : counters_.decode;
}
void TransferEngine::configure(Phase phase) {
    drain();
    std::lock_guard<std::mutex> lock(mutex_);
    phase_ = phase;
}
void TransferEngine::begin_request() {
    std::lock_guard<std::mutex> lock(mutex_);
    counters_ = {};
}
void TransferEngine::wait_read_idle() {
    std::unique_lock<std::mutex> lock(mutex_);
    changed_.wait(lock, [&] { return jobs_.empty() && readers_active_ == 0; });
}
TransferEngine::Counters TransferEngine::counters() {
    std::lock_guard<std::mutex> lock(mutex_);
    return counters_;
}
void TransferEngine::demand(const int * keys, int count, const bool * admit) {
    enqueue(keys, count, admit, nullptr);
}
void TransferEngine::demand_verify(const int * keys, const int * rows, int count) {
    enqueue(keys, count, nullptr, rows);
}
void TransferEngine::enqueue(const int * keys, int count, const bool * admit, const int * rows) {
    std::unique_lock<std::mutex> lock(mutex_);
    std::vector<Record> records;
    records.reserve(count);
    for (int i = 0; i < count; ++i) {
        int slot;
        if (rows) { slot = pool_.reserve_verify(keys[i], rows[i]); }
        else {
            while ((slot = pool_.reserve(keys[i])) < 0) {
                changed_.wait_for(lock, std::chrono::microseconds(fixed::transfer_poll_us));
            }
        }
        records.push_back({keys[i], slot, !rows && (!admit || admit[i]), i, rows != nullptr});
    }
    std::sort(records.begin(), records.end(), [&](const Record & a, const Record & b) {
        return repository_.record_offset(a.key / fixed::experts, a.key % fixed::experts) < repository_.record_offset(b.key / fixed::experts, b.key % fixed::experts);
    });
    for (size_t i = 0; i < records.size();) {
        Job job;
        job.records[0] = records[i++];
        job.count = 1;
        job.source = rows ? cache_.peek(job.records[0].key) : cache_.get(job.records[0].key);
        if (!job.source.data) {
            // Merge adjacent SSD records only. A cached neighbor retains its lease.
            while (i < records.size() && job.count < std::min(fixed::extent_records, capacity()) &&
                   records[i].key == job.records[job.count - 1].key + 1) {
                auto cached = rows ? cache_.peek(records[i].key) : cache_.get(records[i].key);
                if (cached.data) { cache_.release(cached); break; }
                job.records[job.count++] = records[i++];
            }
        }
        if (job.source.data) { ram_ready_.push_back(job); }
        else { jobs_.push_back(job); }
    }
    changed_.notify_all();
}
int TransferEngine::free_run(int count) const {
    int run = 0;
    for (int i = 0; i < int(blocks_.size()); ++i) {
        run = blocks_[i].free ? run + 1 : 0;
        if (run == count) { return i - count + 1; }
    }
    return -1;
}
void TransferEngine::read(bool cached) {
    auto & queue = cached ? ram_ready_ : jobs_;
    for (;;) {
        std::unique_lock<std::mutex> lock(mutex_);
        const auto wait_start = StatsClock::now();
        changed_.wait(lock, [&] {
            return stopping_ || (!queue.empty() && (cached || ram_ready_.empty()) && free_run(queue.front().count) >= 0);
        });
        if (stopping_) { return; }
        const Job job = queue.front();
        phase_counters(job.records[0].provisional).staging_wait_ms += milliseconds(wait_start);
        queue.pop_front();
        const int first = free_run(job.count);
        GGML_ASSERT(first >= 0);
        for (int i = 0; i < job.count; ++i) {
            blocks_[first + i].free = false;
            blocks_[first + i].record = job.records[i];
            pool_.reading(job.records[i].slot);
        }
        ++readers_active_;
        const size_t occupied = std::count_if(blocks_.begin(), blocks_.end(), [](const Block & b) { return !b.free; });
        counters_.staging_peak = std::max(counters_.staging_peak, occupied * layout_.record_bytes);
        lock.unlock();
        const auto read_start = StatsClock::now();
        if (job.source.data) {
            std::memcpy(blocks_[first].data, job.source.data, layout_.record_bytes);
            cache_.release(job.source);
        } else {
            repository_.read(blocks_[first].data, size_t(job.records[0].key) * layout_.record_bytes, size_t(job.count) * layout_.record_bytes);
        }
        uint64_t admitted = 0;
        if (!job.source.data) {
            for (int i = 0; i < job.count; ++i) {
                const auto & record = job.records[i];
                if (record.admit && cache_.put(record.key, blocks_[first + i].data)) { ++admitted; }
            }
        }
        lock.lock();
        auto & counters = phase_counters(job.records[0].provisional);
        counters.read_ms += milliseconds(read_start);
        if (job.source.data) { ++counters.ram_hits; }
        else {
            counters.ssd_misses += job.count;
            counters.read_bytes += size_t(job.count) * layout_.record_bytes;
            ++counters.extents;
            counters.admitted += admitted;
        }
        --readers_active_;
        for (int i = 0; i < job.count; ++i) {
            if (cached) { ready_.push_front(first + i); }
            else { ready_.push_back(first + i); }
        }
        lock.unlock();
        changed_.notify_all();
    }
}
void TransferEngine::copy() {
    std::unique_lock<std::mutex> lock(mutex_);
    for (;;) {
        changed_.wait(lock, [&] { return stopping_ || !ready_.empty() || !copied_.empty() || !retired_.empty(); });
        if (stopping_) { return; }
        for (auto i = copied_.begin(); i != copied_.end();) {
            auto status = cudaEventQuery(blocks_[*i].copied);
            if (status == cudaErrorNotReady) { ++i; continue; }
            SOURIS_CUDA(status);
            blocks_[*i].free = true;
            i = copied_.erase(i);
            changed_.notify_all();
        }
        for (auto i = retired_.begin(); i != retired_.end();) {
            if (pool_.reclaim(i->slot, i->ticket)) { i = retired_.erase(i); changed_.notify_all(); }
            else { ++i; }
        }
        if (ready_.empty()) {
            changed_.wait_for(lock, std::chrono::microseconds(fixed::transfer_poll_us));
            continue;
        }
        const int id = ready_.front();
        ready_.pop_front();
        auto & block = blocks_[id];
        const Record record = block.record;
        lock.unlock();
        const auto copy_start = StatsClock::now();
        void * destination = pool_.address(record.slot);
        SOURIS_CUDA(cudaMemcpyAsync(destination, block.data, layout_.record_bytes, cudaMemcpyHostToDevice, device_.upload));
        cuda_expert_bind(pool_.table(), static_cast<int32_t *>(pool_.resident_table()->data), record.slot, record.key, pool_.previous_key(record.slot), destination, layout_, device_.upload, record.provisional);
        pool_.publish(record.slot, record.key);
        SOURIS_CUDA(cudaEventRecord(pool_.copied(record.slot), device_.upload));
        SOURIS_CUDA(cudaEventRecord(block.copied, device_.upload));
        lock.lock();
        copied_.push_back(id);
        uploaded_.push_back({record.key, record.slot, ++copy_sequence_, record.ordinal, record.provisional});
        phase_counters(record.provisional).h2d_bytes += layout_.record_bytes;
        phase_counters(record.provisional).copy_submit_ms += milliseconds(copy_start);
        changed_.notify_all();
    }
}
bool TransferEngine::acquire_ready(Lease & result, int key) {
    for (auto i = uploaded_.begin(); i != uploaded_.end(); ++i) {
        if (key >= 0 && i->key != key) { continue; }
        // Publication follows copied-event recording. The consumer queues a
        // stream wait and can submit compute while DMA is still in flight.
        result = *i;
        uploaded_.erase(i);
        pool_.activate(result.slot);
        return true;
    }
    return false;
}
TransferEngine::Lease TransferEngine::acquire(int key) {
    auto start = StatsClock::now();
    std::unique_lock<std::mutex> lock(mutex_);
    Lease result;
    while (!acquire_ready(result, key)) { changed_.wait_for(lock, std::chrono::microseconds(fixed::transfer_poll_us)); }
    phase_counters(result.provisional).wait_ms += milliseconds(start);
    return result;
}
bool TransferEngine::try_acquire(Lease & lease) {
    std::lock_guard<std::mutex> lock(mutex_);
    return acquire_ready(lease, -1);
}
void TransferEngine::wait_ready() {
    auto start = StatsClock::now();
    std::unique_lock<std::mutex> lock(mutex_);
    changed_.wait_for(lock, std::chrono::microseconds(fixed::transfer_poll_us));
    phase_counters().wait_ms += milliseconds(start);
}
void TransferEngine::release(Lease lease, cudaStream_t stream) {
    const auto ticket = pool_.release(lease.slot, stream);
    { std::lock_guard<std::mutex> lock(mutex_); retired_.push_back({lease.slot, ticket}); }
    changed_.notify_all();
}
void TransferEngine::release_batch(const int * slots, int count, cudaStream_t stream) {
    GGML_ASSERT(count >= 0 && count <= fixed::verify_rows * fixed::topk);
    uint64_t tickets[fixed::verify_rows * fixed::topk];
    pool_.release_batch(slots, count, stream, tickets);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (int i = 0; i < count; ++i) { retired_.push_back({slots[i], tickets[i]}); }
    }
    changed_.notify_all();
}
void TransferEngine::drain() {
    std::unique_lock<std::mutex> lock(mutex_);
    changed_.wait(lock, [&] {
        return jobs_.empty() && ram_ready_.empty() && !readers_active_ && ready_.empty() && copied_.empty() &&
            uploaded_.empty() && retired_.empty() && pool_.idle();
    });
}
}
