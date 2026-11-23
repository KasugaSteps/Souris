#include "expert_cache.h"
#include "repository.h"
#include "ggml.h"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <sstream>
#include <sys/mman.h>

namespace souris {
static size_t proc_bytes(const char * path, const char * name) {
    std::ifstream file(path);
    if (!file) { GGML_ABORT("Cannot read %s", path); }
    std::string line, key;
    size_t value;
    while (std::getline(file, line)) {
        std::istringstream row(line);
        if (row >> key >> value && key == name) { return value * 1024; }
    }
    GGML_ABORT("Missing %s in %s", name, path);
}
size_t available_ram() { return proc_bytes("/proc/meminfo", "MemAvailable:"); }
size_t process_swap() { return proc_bytes("/proc/self/status", "VmSwap:"); }

ExpertCache::ExpertCache(const Layout & layout, size_t budget, bool server) : layout_(layout) {
    const size_t record_bytes = layout.record_bytes;
    const size_t total = size_t(fixed::layers) * fixed::experts * record_bytes;
    const size_t available = available_ram();
    if (budget == SIZE_MAX) {
        const size_t reserve = available > fixed::host_reserve ? available - fixed::host_reserve : 0;
        budget = server ? std::min(total, reserve) : std::min({total, available * fixed::ram_cache_percent / 100, reserve});
        if (budget < record_bytes) { GGML_ABORT("RAM budget requires explicit host_cache_mib=0"); }
    }
    bytes_ = std::min(total, budget / record_bytes * record_bytes);
    if (budget && !bytes_) { GGML_ABORT("ExpertCache budget is smaller than one record; use host_cache_mib=0"); }
    if (bytes_ > available) { GGML_ABORT("ExpertCache exceeds available RAM"); }
    entries_.resize(bytes_ / record_bytes);
    placement_.resize(fixed::layers * fixed::experts, -1);
    if (bytes_) {
        arena_ = mmap(nullptr, bytes_, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        if (arena_ == MAP_FAILED) { GGML_ABORT("Cannot allocate pageable ExpertCache"); }
    }
    clear();
}
ExpertCache::~ExpertCache() {
    if (arena_) { GGML_ASSERT(munmap(arena_, bytes_) == 0); }
}
void ExpertCache::unlink(int slot) {
    auto & entry = entries_[slot];
    if (entry.previous >= 0) { entries_[entry.previous].next = entry.next; }
    else { lru_ = entry.next; }
    if (entry.next >= 0) { entries_[entry.next].previous = entry.previous; }
    else { mru_ = entry.previous; }
}
void ExpertCache::touch(int slot) {
    auto & entry = entries_[slot];
    entry.previous = mru_;
    entry.next = -1;
    if (mru_ >= 0) { entries_[mru_].next = slot; }
    else { lru_ = slot; }
    mru_ = slot;
}
ExpertCache::Lease ExpertCache::get(int key) {
    std::lock_guard<std::mutex> lock(mutex_);
    const int slot = placement_[key];
    if (slot < 0 || !entries_[slot].ready) { return {}; }
    ++entries_[slot].references;
    unlink(slot);
    touch(slot);
    return {static_cast<char *>(arena_) + size_t(slot) * layout_.record_bytes, slot};
}
void ExpertCache::release(Lease lease) {
    if (lease.slot < 0) { return; }
    std::lock_guard<std::mutex> lock(mutex_);
    GGML_ASSERT(entries_[lease.slot].references > 0);
    --entries_[lease.slot].references;
}
ExpertCache::Lease ExpertCache::peek(int key) {
    std::lock_guard<std::mutex> lock(mutex_);
    const int slot = placement_[key];
    if (slot < 0 || !entries_[slot].ready) { return {}; }
    ++entries_[slot].references;
    return {static_cast<char *>(arena_) + size_t(slot) * layout_.record_bytes, slot};
}
std::vector<int> ExpertCache::keys() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<int> result;
    result.reserve(entries_.size());
    for (int key = 0; key < int(placement_.size()); ++key) {
        const int slot = placement_[key];
        if (slot >= 0 && entries_[slot].ready) { result.push_back(key); }
    }
    return result;
}
bool ExpertCache::put(int key, const void * record) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (entries_.empty()) { return false; }
    if (placement_[key] >= 0) { return entries_[placement_[key]].ready; }
    int slot;
    if (!free_.empty()) {
        slot = free_.back();
        free_.pop_back();
    } else {
        slot = lru_;
        while (slot >= 0 && entries_[slot].references) { slot = entries_[slot].next; }
        if (slot < 0) { return false; }
        placement_[entries_[slot].key] = -1;
        unlink(slot);
    }
    auto & entry = entries_[slot];
    entry = {key, -1, -1, 1, false};
    placement_[key] = slot;
    touch(slot);
    lock.unlock();
    std::memcpy(static_cast<char *>(arena_) + size_t(slot) * layout_.record_bytes, record, layout_.record_bytes);
    lock.lock();
    entry.ready = true;
    entry.references = 0;
    return true;
}
void ExpertCache::begin_preload() {
    std::lock_guard<std::mutex> lock(mutex_);
    GGML_ASSERT(lru_ == -1);
    const size_t count = layers() * fixed::experts;
    for (size_t key = 0; key < count; ++key) {
        GGML_ASSERT(free_.back() == int(key));
        free_.pop_back();
        entries_[key] = {int(key), -1, -1, 1, false};
        placement_[key] = int(key);
        touch(int(key));
    }
}
void * ExpertCache::preload_target(size_t first) {
    return static_cast<char *>(arena_) + first * layout_.record_bytes;
}
void ExpertCache::publish_preload(size_t first, size_t count) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (size_t i = first; i < first + count; ++i) {
        auto & entry = entries_[i];
        GGML_ASSERT(entry.references == 1 && !entry.ready && entry.key == int(i));
        entry.ready = true;
        entry.references = 0;
    }
}
void ExpertCache::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    free_.clear();
    lru_ = mru_ = -1;
    std::fill(placement_.begin(), placement_.end(), -1);
    for (int slot = int(entries_.size()) - 1; slot >= 0; --slot) {
        GGML_ASSERT(entries_[slot].references == 0);
        entries_[slot] = {};
        free_.push_back(slot);
    }
}
}
