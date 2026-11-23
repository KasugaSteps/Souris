#pragma once
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace souris {
/** Runtime-wide ownership of a completion, including delivery and deferred settlement. */
class RequestGate {
public:
    using Clock = std::chrono::steady_clock;
    using Cancel = std::function<bool()>;
    /** Four queued completions leave HTTP workers available for finalization RPCs. */
    static constexpr size_t max_waiters = 4;
    /** Admission only: normal cold requests can exceed the independent 30s pending TTL. */
    static constexpr std::chrono::seconds admission_wait{120};
    class Lease {
    public:
        ~Lease();
        /** Complete this request only after action and engine drain have finished. */
        void finish(const std::function<void()> & action);
        /** Transfer this same lease to deferred ownership; no new admission occurs. */
        void defer(const std::string & commit_token, Clock::time_point staged_at);
    private:
        friend class RequestGate;
        Lease() = default;
        RequestGate * gate_ = nullptr;
        uint64_t id_ = 0;
    };
    RequestGate(std::function<void()> drain, std::function<void()> abort);
    ~RequestGate();
    std::shared_ptr<Lease> acquire(Clock::time_point deadline, const Cancel &, std::string & error,
                                   const char ** code = nullptr);
    bool owns(const Lease &) const;
    /** Re-enter the same logical request after a retry abort, without admission. */
    std::shared_ptr<Lease> resume(const std::string & token,
        const std::function<bool()> & validate, std::string & error);
    /** Serialize a receipt RPC with the active lease, never acquire another lease. */
    bool finalize(const std::string & token, Clock::time_point deadline, const Cancel &,
                  const std::function<bool()> & action, std::string & error, bool retry = false);
private:
    enum class State { Idle, Generating, Deferred, Finalizing };
    mutable std::mutex mutex_;
    std::condition_variable changed_;
    std::deque<uint64_t> waiting_;
    uint64_t next_ = 0, owner_ = 0;
    State state_ = State::Idle;
    std::string token_;
    Clock::time_point expires_;
    bool stopping_ = false;
    std::function<void()> drain_, abort_;
    std::thread expiry_;
    void finish(uint64_t id, const std::function<void()> & action);
    void defer(uint64_t id, const std::string & token, Clock::time_point staged_at);
    void settle(const std::function<void()> & action);
    void release();
    void expire();
};
}
