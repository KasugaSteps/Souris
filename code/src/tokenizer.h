#pragma once
#include "model.h"
#include <unordered_map>

namespace souris {
/** GGUF byte-level BPE, using the vendored llama.cpp Unicode pretokenizer. */
class Tokenizer {
public:
    explicit Tokenizer(const Model & model);
    Tokens encode(const std::string & text) const;
    std::string piece(Token token) const;
    bool eos(Token token) const;
    int vocab_size() const { return int(pieces_.size()); }
    Token special_token(const std::string & name) const;
    Token image_pad() const { return special_token("<|image_pad|>"); }
    /** Validate the fixed Qwen vision vocabulary and its BPE boundaries. */
    void validate_vision() const;
private:
    std::vector<std::string> pieces_, patterns_;
    std::vector<Token> special_;
    std::unordered_map<std::string,Token> ids_;
    std::unordered_map<std::string,int> merges_;
    Token eos_=-1, bos_=-1;
    bool add_bos_=false;
    void ordinary(const std::string & text, Tokens & output) const;
};
}
