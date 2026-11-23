#pragma once
#include "draft_runner.h"
#include "souris/engine.h"
#include <random>

namespace souris {
/** Internal fixed-T verification over a target SlotPool transaction. */
class VerifyEngine {
public:
    VerifyEngine(Engine::Impl &, Tokens & history, const float & temperature,
                 const float & top_p, const int & top_k, const float & min_p, std::mt19937_64 &, const size_t & remaining);
    ~VerifyEngine();
    /** Capture the prepared T=2..5 graphs after target budgeting. */
    void capture();
    /** Verify provisionally and commit only the accepted prefix in place. */
    bool verify(const DraftRunner::Request &, DraftRunner::Emitted &);
    static bool dispatch(void *, const DraftRunner::Request &, DraftRunner::Emitted &);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
