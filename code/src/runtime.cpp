#include "souris/runtime.h"
#include "souris/engine.h"
#include "souris/session_manager.h"
#include "device.h"

namespace souris {
std::unique_ptr<Runtime> Runtime::build(const Config & config) {
    return std::unique_ptr<Runtime>(new Runtime(config));
}
Runtime::Runtime(const Config & config) : config_(config), device_(std::make_unique<Device>()),
    model_(Model::load(config.model, device_->backend)),
    engine_(std::make_unique<Engine>(*model_, *device_, config_)) {
    engine_->plan_budget(config_.max_prompt_tokens);
    engine_->warmup();
    engine_->preload();
    session_ = std::make_unique<SessionManager>(*engine_);
    requests_ = std::make_unique<RequestGate>([this] { engine_->finish_request(); },
        [this] { session_->abandon(); });
}
SessionManager & Runtime::session(const RequestGate::Lease & lease) const {
    GGML_ASSERT(requests_->owns(lease));
    return *session_;
}
std::shared_ptr<RequestGate::Lease> Runtime::resume(const std::string & session,
    const std::string & request, const std::string & token, std::string & error) {
    return requests_->resume(token, [&] {
        return session_->active_id() == session && !session_->has_staged() &&
            session_->staged_matches(session, request, token);
    }, error);
}
bool Runtime::finalize(const std::string & token, RequestGate::Clock::time_point deadline,
    const RequestGate::Cancel & cancel, const std::function<bool(SessionManager &)> & action, std::string & error, bool retry) {
    return requests_->finalize(token, deadline, cancel, [&] { return action(*session_); }, error, retry);
}
Runtime::~Runtime() = default;
}
