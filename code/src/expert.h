#pragma once
#include "device.h"
#include <deque>
#include <mutex>

namespace souris {
struct ExpertTable {
    const void * gate[fixed::experts];
    const void * up[fixed::experts];
    const void * down[fixed::experts];
};
enum class SlotState : uint8_t { Free, Reserved, Inflight, Resident, Active, Provisional };

/** Stable pointer table with byte-budgeted GPU records and SLRU ownership. */
class SlotPool {
public:
    SlotPool(const Layout &, Device &);
    ~SlotPool();
    /** Allocate the locked startup capacity after Decode workspace preparation. */
    void allocate(int count, int protected_percent, int prefill_slots, int hidden_slots);
    /** Exclusive contiguous tail. The GPU allocation remains owned by SlotPool. */
    struct Tail { void * data = nullptr; ggml_backend_buffer_t buffer = nullptr; int begin = 0, count = 0; };
    Tail borrow_tail(int count);
    void return_tail(Tail &);
    /** Temporarily return the eager workspace chunks to the CUDA allocator. */
    void borrow_workspace(int slots);
    void return_workspace();
    void reset();
    int capacity() const { return active_; }
    const Layout & layout() const { return layout_; }
    int group(int layer) const { return groups_[layer]; }
    int group_count() const { return group_count_; }
    void * table() const { return table_->data; }
    ggml_tensor * resident_table() const { return resident_; }
    /** Pin only GPU-planned hit slots before any demand can evict. */
    void pin_slots(int layer, const int * slots, int count);
    ggml_tensor * pointer_table() const { return table_; }
    void * address(int slot) const;
    int previous_key(int slot) const;
    void publish(int slot, int key);
    /** Pin every actual hit before any missing record may select a victim. */
    int pin(int key);
    /** Allocate transaction metadata before the startup VRAM query. */
    void prepare_verify();
    /** Start one speculative transaction after ordinary transfers retire. */
    void begin_verify();
    /** Hold existing hits without changing their SLRU segment or recency. */
    void pin_verify(const int * keys, const int * rows, int count);
    /** Reserve from free/probation only; exhaustion is a transaction failure. */
    int reserve_verify(int key, int rows);
    struct VerifyResult { int provisional = 0, committed = 0, rolled_back = 0, hits = 0; };
    /** Commit accepted keys in place and release rejected keys after state commit. */
    VerifyResult finish_verify(int accepted, cudaStream_t);
    /** Reserve the final destination. Returns -1 while all victims have owners. */
    int reserve(int key);
    void reading(int slot);
    void activate(int slot);
    uint64_t release(int slot, cudaStream_t stream);
    /** Record each hit's used event under one lock, returning independent tickets. */
    void release_batch(const int * slots, int count, cudaStream_t stream, uint64_t * tickets);
    bool reclaim(int slot, uint64_t ticket);
    bool idle();
    cudaEvent_t copied(int slot) const { return slots_[slot].copied; }
private:
    struct Slot {
        int key = -1, published = -1, previous = -1, next = -1;
        SlotState state = SlotState::Free;
        bool owned = false, protected_segment = false, provisional = false;
        cudaEvent_t copied = nullptr, used = nullptr;
        uint64_t retirement = 0;
    };
    const Layout & layout_;
    Device & device_;
    ggml_context * context_;
    ggml_tensor * table_, *resident_;
    ggml_backend_buffer_t table_buffer_;
    int groups_[fixed::layers]{}, group_count_ = 0;
    struct Chunk { void * data = nullptr; int begin, count; ggml_backend_buffer_t buffer = nullptr; };
    std::vector<Chunk> chunks_;
    std::vector<void *> addresses_;
    std::vector<Slot> slots_;
    std::deque<int> free_;
    int placement_[fixed::layers * fixed::experts];
    int workspace_end_ = 0;
    bool hidden_borrowed_ = false;
    int active_ = 0, protected_percent_ = fixed::protected_percent;
    int lru_[2] = {-1,-1}, mru_[2] = {-1,-1}, sizes_[2] = {};
    std::mutex mutex_;
    struct VerifyEntry { int key, slot, rows; bool provisional; };
    std::vector<VerifyEntry> verify_entries_;
    std::vector<int32_t> verify_metadata_;
    ggml_backend_buffer_t verify_buffer_ = nullptr;
    cudaEvent_t verify_done_ = nullptr;
    bool verify_open_ = false;
    int reserve_slot(int key, bool provisional);
    void retire_tail(int begin);
    void pin_slot(int slot);
    void promote(int slot);
    void allocate_chunk(Chunk &);
    void unlink(int slot);
    void touch(int slot);
};
}
