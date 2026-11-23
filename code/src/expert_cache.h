#pragma once
#include "souris/constants.h"
#include <mutex>
#include <vector>

namespace souris {
struct Layout;
/** Pageable immutable-record LRU shared by Prefill and Decode. */
class ExpertCache {
public:
    struct Lease {
        const void * data = nullptr;
        int slot = -1;
    };
    ExpertCache(const Layout &, size_t budget, bool server);
    ~ExpertCache();
    ExpertCache(const ExpertCache &) = delete;
    ExpertCache & operator=(const ExpertCache &) = delete;
    size_t capacity() const { return entries_.size(); }
    /** Snapshot resident keys for offline trace only. */
    std::vector<int> keys() const;
    size_t bytes() const { return bytes_; }
    /** Touch the LRU and protect the returned address until release(). */
    Lease get(int key);
    /** Borrow an existing immutable record without changing its LRU position. */
    Lease peek(int key);
    void release(Lease);
    /** Immutable insertion. Returns false when every candidate has a consumer. */
    bool put(int key, const void * record);
    void clear();
    size_t layers() const { return capacity() / fixed::experts; }
    /** Reserve complete startup layers; direct reads target this same pageable arena. */
    void begin_preload();
    void * preload_target(size_t first);
    void publish_preload(size_t first, size_t count);
private:
    struct Entry {
        int key = -1, previous = -1, next = -1;
        int references = 0;
        bool ready = false;
    };
    void * arena_ = nullptr;
    const Layout & layout_;
    size_t bytes_;
    std::vector<Entry> entries_;
    std::vector<int> placement_, free_;
    int lru_ = -1, mru_ = -1;
    mutable std::mutex mutex_;
    void unlink(int slot);
    void touch(int slot);
};
/** Current OS memory measurements, expressed in bytes. */
size_t available_ram();
size_t process_swap();
}
