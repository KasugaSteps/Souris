#include "staging_ring.h"

namespace souris {
StagingRing::Ring::Ring(size_t expert, size_t control) : expert_bytes(expert), control_bytes(control) {
    SOURIS_CUDA(cudaHostAlloc(&data, expert + fixed::control_slots * control, cudaHostAllocDefault));
    for (int i = 0; i < fixed::control_slots; ++i) {
        slots_[i].host = static_cast<char *>(data) + expert + i * control;
        SOURIS_CUDA(cudaEventCreateWithFlags(&slots_[i].copied, cudaEventDisableTiming));
    }
}
StagingRing::Ring::~Ring() {
    for (int i = 0; i < fixed::control_slots; ++i) { wait(i); SOURIS_CUDA(cudaEventDestroy(slots_[i].copied)); }
    SOURIS_CUDA(cudaFreeHost(data));
}
StagingRing::Ring::Slot & StagingRing::Ring::acquire(int direction) { wait(direction); return slots_[direction]; }
void StagingRing::Ring::record(int direction, cudaStream_t stream) {
    auto & slot = slots_[direction];
    SOURIS_CUDA(cudaEventRecord(slot.copied, stream)); slot.pending = true;
}
void StagingRing::Ring::wait(int direction) {
    auto & slot = slots_[direction];
    if (slot.pending) { SOURIS_CUDA(cudaEventSynchronize(slot.copied)); slot.pending = false; }
}
StagingRing::StagingRing(size_t budget, const Layout & layout) :
    prefill_(budget / layout.record_bytes * layout.record_bytes, fixed::prefill_control_slot),
    decode_(0, fixed::decode_ring_bytes / fixed::control_slots) {
    const size_t record = layout.record_bytes;
    if (prefill_.expert_bytes < fixed::staging_min_records * record) { GGML_ABORT("Staging must hold at least 64 * record_bytes"); }
}
}
