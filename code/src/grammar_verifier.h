#pragma once
#include "souris/types.h"
#include <memory>
#include <string>

namespace souris {
class Tokenizer;
/** Serialized, validated raw tool constraint; no executable grammar input. */
struct ToolConstraint { std::string json; };
/** Cached device bitset and its sparse host representation. */
struct TokenMask {
    const uint32_t * device = nullptr;
    const uint32_t * bits = nullptr;
    const Token * allowed = nullptr;
    size_t count = 0, words = 0;
    const Token * device_sparse = nullptr;
    bool empty() const { return count == 0; }
    bool allows(Token token) const { return token >= 0 && size_t(token / 32) < words && (bits[token / 32] & (1u << (token % 32))); }
};
/** Request-local tool automaton; compiled schemas, token trie and masks stay internal. */
class GrammarVerifier {
public:
    GrammarVerifier();
    ~GrammarVerifier();
    bool configure(const ToolConstraint &, Tokenizer &);
    bool observe(Token);
    bool begin_tool_call();
    bool finish_tool_call();
    int call_count() const;
    bool active() const;
    TokenMask mask();
    bool advance(Token);
    bool accepting() const;
    void reset();
    const std::string & error() const;
private:
    friend class VerifyEngine;
    /** Number of speculative TEXT tokens before a marker-completing token. */
    int text_prefix(const Token *, int) const;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
