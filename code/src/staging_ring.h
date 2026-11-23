#pragma once
#include "device.h"

namespace souris {
/** Two pinned allocations. Expert storage is shared by both serial phases. */
class StagingRing {
public:
    struct Ring {
        struct Slot { void * host = nullptr; cudaEvent_t copied = nullptr; bool pending = false; };
        /** Expert prefix, followed by two independent control directions. */
        void * data = nullptr;
        size_t expert_bytes = 0, control_bytes = 0;
        Ring(size_t expert_bytes, size_t control_bytes);
        ~Ring();
        Ring(const Ring &) = delete;
        Ring & operator=(const Ring &) = delete;
        Slot & acquire(int direction);
        void record(int direction, cudaStream_t stream);
        void wait(int direction);
    private:
        Slot slots_[fixed::control_slots];
    };
    StagingRing(size_t budget_bytes, const Layout &);
    Ring & prefill_ring() { return prefill_; }
    Ring & decode_ring() { return decode_; }
private:
    Ring prefill_, decode_;
};
}
