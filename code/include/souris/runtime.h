#pragma once
#include "types.h"
#include "request_gate.h"
#include <memory>

namespace souris {
class Device;
class Model;
class Engine;
class SessionManager;

/** Composition root. Request ownership drains before session, engine, model and device. */
class Runtime {
public:
    static std::unique_ptr<Runtime> build(const Config &);
    ~Runtime();
    Device & device() const { return *device_; }
    Model & model() const { return *model_; }
    Engine & engine() const { return *engine_; }
    SessionManager & session(const RequestGate::Lease &) const;
    RequestGate & requests() const { return *requests_; }
    /** Continue an aborted tool attempt using the same authenticated owner. */
    std::shared_ptr<RequestGate::Lease> resume(const std::string & session,
        const std::string & request, const std::string & token, std::string & error);
    bool finalize(const std::string & token, RequestGate::Clock::time_point deadline,
        const RequestGate::Cancel &, const std::function<bool(SessionManager &)> &, std::string & error, bool retry = false);
    const Config & config() const { return config_; }
private:
    explicit Runtime(const Config &);
    Config config_;
    std::unique_ptr<Device> device_;
    std::unique_ptr<Model> model_;
    std::unique_ptr<Engine> engine_;
    std::unique_ptr<SessionManager> session_;
    std::unique_ptr<RequestGate> requests_;
};
}
