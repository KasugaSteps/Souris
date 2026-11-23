#pragma once
#include "expert.h"
#include "expert_cache.h"
#include "staging_ring.h"
#include <array>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>

namespace souris {
/** Direct record reads and event-owned H2D, shared by both schedulers. */
class TransferEngine {
public:
    enum class Phase { Prefill, Decode };
    struct Lease {
        int key = -1, slot = -1;
        /** Submission order on the single copy stream; zero denotes a cache hit. */
        uint64_t copy_sequence = 0;
        int ordinal = -1;
        bool provisional = false;
    };
    struct PhaseCounters {
        uint64_t read_bytes = 0, h2d_bytes = 0, ram_hits = 0, ssd_misses = 0;
        uint64_t extents = 0, admitted = 0;
        double wait_ms = 0;
        double read_ms = 0, staging_wait_ms = 0, copy_submit_ms = 0;
    };
    struct Counters { PhaseCounters prefill, decode, verify; size_t staging_peak = 0; };
    TransferEngine(const ExpertRepository &, Device &, SlotPool &, ExpertCache &, StagingRing &, const Config &);
    ~TransferEngine();
    void configure(Phase);
    void begin_request();
    /** All hits must be pinned. Reserves every final GPU slot before staging. */
    void demand(const int * keys, int count, const bool * admit = nullptr);
    /** One speculative upload per key, with read-only RAM lookup and no protected eviction. */
    void demand_verify(const int * keys, const int * rows, int count);
    Lease acquire(int key);
    bool try_acquire(Lease &);
    void wait_ready();
    void release(Lease, cudaStream_t);
    /** Retire one layer's hit slots with separate pool and transfer lock scopes. */
    void release_batch(const int * slots, int count, cudaStream_t);
    void drain();
    Counters counters();
    /** Yield background preload while demand readers have pending work. */
    void wait_read_idle();
    int capacity() const { return int(blocks_.size()); }
    size_t bytes() const { return blocks_.size() * layout_.record_bytes; }
private:
    struct Record { int key = -1, slot = -1; bool admit = true; int ordinal = -1; bool provisional = false; };
    struct Job {
        Record records[fixed::extent_records];
        int count = 0;
        ExpertCache::Lease source;
    };
    struct Block {
        Record record;
        void * data = nullptr;
        cudaEvent_t copied = nullptr;
        bool free = true;
    };
    struct Retirement { int slot; uint64_t ticket; };
    const ExpertRepository & repository_;
    const Layout & layout_;
    Device & device_;
    SlotPool & pool_;
    ExpertCache & cache_;
    std::vector<Block> blocks_;
    Phase phase_ = Phase::Prefill;
    Counters counters_;
    bool stopping_ = false;
    uint64_t copy_sequence_ = 0;
    int readers_active_ = 0;
    std::deque<Job> jobs_, ram_ready_;
    std::deque<int> ready_, copied_;
    std::deque<Lease> uploaded_;
    std::deque<Retirement> retired_;
    std::mutex mutex_;
    std::condition_variable changed_;
    std::vector<std::thread> readers_;
    std::thread ram_stager_;
    std::thread copier_;
    int free_run(int count) const;
    bool acquire_ready(Lease &, int key);
    PhaseCounters & phase_counters(bool provisional = false);
    void enqueue(const int * keys, int count, const bool * admit, const int * rows);
    void read(bool cached);
    void copy();
};
}
