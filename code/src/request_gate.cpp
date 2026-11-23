#include "souris/request_gate.h"
#include "ggml.h"
#include <algorithm>

namespace souris {
RequestGate::RequestGate(std::function<void()> drain, std::function<void()> abort)
    : drain_(std::move(drain)), abort_(std::move(abort)), expiry_([this] { expire(); }) {}
RequestGate::~RequestGate() {
    { std::lock_guard<std::mutex> lock(mutex_); stopping_ = true; }
    changed_.notify_all();
    expiry_.join();
    GGML_ASSERT(state_ == State::Idle);
}
RequestGate::Lease::~Lease() {
    if (gate_) { gate_->finish(id_, gate_->abort_); }
}
void RequestGate::Lease::finish(const std::function<void()> & action) {
    GGML_ASSERT(gate_);
    auto * gate = gate_;
    gate_ = nullptr;
    gate->finish(id_, action);
}
void RequestGate::Lease::defer(const std::string & token, Clock::time_point staged_at) {
    GGML_ASSERT(gate_);
    gate_->defer(id_, token, staged_at);
    gate_ = nullptr;
}
std::shared_ptr<RequestGate::Lease> RequestGate::acquire(Clock::time_point deadline,
    const Cancel & cancelled, std::string & error, const char ** code) {
    if (code) { *code = nullptr; }
    auto lease = std::shared_ptr<Lease>(new Lease);
    std::unique_lock<std::mutex> lock(mutex_);
    if (waiting_.size() >= max_waiters) {
        if (code) { *code = "model_busy"; }
        error = "request admission queue is full"; return {};
    }
    const uint64_t id = ++next_;
    waiting_.push_back(id);
    for (;;) {
        if (stopping_ || (cancelled && cancelled()) || Clock::now() >= deadline) {
            error = stopping_ ? "runtime is stopping" : cancelled && cancelled() ?
                "request admission cancelled" : "request admission timed out";
            if (code && !stopping_ && !(cancelled && cancelled()) && Clock::now() >= deadline) {
                *code = "queue_wait_timeout";
            }
            waiting_.erase(std::find(waiting_.begin(), waiting_.end(), id));
            lock.unlock(); changed_.notify_all(); return {};
        }
        if (state_ == State::Idle && waiting_.front() == id) {
            waiting_.pop_front(); owner_ = id; state_ = State::Generating;
            lease->gate_ = this; lease->id_ = id;
            return lease;
        }
        // A socket-close callback has no notification handle. Timed CV checks
        // cancellation; ownership is released only by the lifecycle transition.
        changed_.wait_until(lock, std::min(deadline, Clock::now() + std::chrono::milliseconds(50)));
    }
}
std::shared_ptr<RequestGate::Lease> RequestGate::resume(const std::string & token,
    const std::function<bool()> & validate, std::string & error) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (state_ != State::Deferred || token_ != token || Clock::now() >= expires_) {
        error = "retry lease is unavailable"; return {};
    }
    state_ = State::Finalizing;
    lock.unlock();
    const bool ok = validate();
    lock.lock();
    if (!ok) {
        state_ = State::Deferred;
        error = "retry credentials do not match"; changed_.notify_all(); return {};
    }
    auto lease = std::shared_ptr<Lease>(new Lease);
    lease->gate_ = this; lease->id_ = owner_;
    state_ = State::Generating; token_.clear();
    return lease;
}
bool RequestGate::owns(const Lease & lease) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return lease.gate_ == this && owner_ == lease.id_ && state_ == State::Generating;
}
void RequestGate::release() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        owner_ = 0; token_.clear(); state_ = State::Idle;
    }
    changed_.notify_all();
}
void RequestGate::finish(uint64_t id, const std::function<void()> & action) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        GGML_ASSERT(owner_ == id && (state_ == State::Generating || state_ == State::Deferred));
        state_ = State::Finalizing;
    }
    settle(action);
}
void RequestGate::settle(const std::function<void()> & action) {
    try {
        drain_(); action(); drain_();
    } catch (...) {
        abort_(); drain_(); release(); throw;
    }
    release();
}
void RequestGate::defer(uint64_t id, const std::string & token, Clock::time_point staged_at) {
    // Still GENERATING while the producer and response-delivery owner drain.
    drain_();
    {
        std::lock_guard<std::mutex> lock(mutex_);
        GGML_ASSERT(owner_ == id && state_ == State::Generating && !token.empty());
        token_ = token; expires_ = staged_at + std::chrono::seconds(30);
        state_ = State::Deferred;
    }
    changed_.notify_all();
}
bool RequestGate::finalize(const std::string & token, Clock::time_point deadline,
    const Cancel & cancelled, const std::function<bool()> & action, std::string & error, bool retry) {
    std::unique_lock<std::mutex> lock(mutex_);
    for (;;) {
        if (state_ == State::Deferred && Clock::now() >= expires_) {
            state_ = State::Finalizing;
            lock.unlock(); settle(abort_); lock.lock(); continue;
        }
        if (state_ != State::Generating && state_ != State::Finalizing) { break; }
        if (stopping_ || (cancelled && cancelled()) || Clock::now() >= deadline) {
            error = "request finalization wait cancelled or timed out"; return false;
        }
        changed_.wait_until(lock, std::min(deadline, Clock::now() + std::chrono::milliseconds(50)));
    }
    const State previous = state_;
    const bool current = previous == State::Deferred && token_ == token;
    if (retry && !current) { error = "retry lease is unavailable"; return false; }
    state_ = State::Finalizing;
    lock.unlock();
    bool ok;
    try {
        if (current) { drain_(); }
        ok = action();
        if (current && ok) { drain_(); }
    } catch (...) {
        if (current) { abort_(); drain_(); release(); }
        else { lock.lock(); state_ = previous; lock.unlock(); changed_.notify_all(); }
        throw;
    }
    if (current && ok && retry) {
        lock.lock();
        state_ = State::Deferred; expires_ = Clock::now() + std::chrono::seconds(30);
        lock.unlock(); changed_.notify_all();
    } else if (current && ok) { release(); }
    else {
        lock.lock(); state_ = previous; lock.unlock(); changed_.notify_all();
    }
    return ok;
}
void RequestGate::expire() {
    std::unique_lock<std::mutex> lock(mutex_);
    for (;;) {
        changed_.wait(lock, [&] { return stopping_ || state_ == State::Deferred; });
        if (stopping_ && state_ == State::Idle) { return; }
        if (state_ != State::Deferred) {
            GGML_ASSERT(!stopping_); continue;
        }
        if (!stopping_ && Clock::now() < expires_) {
            changed_.wait_until(lock, expires_); continue;
        }
        state_ = State::Finalizing;
        lock.unlock(); settle(abort_); lock.lock();
    }
}
}
