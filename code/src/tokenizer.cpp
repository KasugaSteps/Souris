#include "tokenizer.h"
#include "unicode.h"
#include <algorithm>
#include <climits>
#include <cstdint>
#include <queue>

namespace souris {
Tokenizer::Tokenizer(const Model & model) {
    auto * meta=model.metadata(); const auto vocab=gguf_find_key(meta,"tokenizer.ggml.tokens"); GGML_ASSERT(vocab>=0);
    const auto types=gguf_find_key(meta,"tokenizer.ggml.token_type");
    const auto * token_types=types>=0 ? static_cast<const int32_t *>(gguf_get_arr_data(meta,types)) : nullptr;
    for (size_t i=0; i<gguf_get_arr_n(meta,vocab); ++i) {
        pieces_.emplace_back(gguf_get_arr_str(meta,vocab,i)); ids_.emplace(pieces_.back(),i);
        if (token_types && (token_types[i]==3 || token_types[i]==4)) { special_.push_back(i); }
    }
    const auto merges=gguf_find_key(meta,"tokenizer.ggml.merges"); GGML_ASSERT(merges>=0);
    for (size_t i=0; i<gguf_get_arr_n(meta,merges); ++i) { merges_.emplace(gguf_get_arr_str(meta,merges,i),i); }
    const auto pre=gguf_find_key(meta,"tokenizer.ggml.pre"); GGML_ASSERT(pre>=0);
    const std::string kind=gguf_get_val_str(meta,pre);
    if (kind=="qwen35") {
        patterns_={"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"};
    } else {
        GGML_ASSERT(kind=="qwen2" || kind=="deepseek-r1-qwen");
        patterns_={"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"};
    }
    eos_=model.integer("tokenizer.ggml.eos_token_id",0); bos_=model.integer("tokenizer.ggml.bos_token_id",0);
    add_bos_=model.integer("tokenizer.ggml.add_bos_token",0)!=0;
}
void Tokenizer::ordinary(const std::string & text, Tokens & output) const {
    for (const auto & word : unicode_regex_split(text,patterns_)) {
        struct Node {
            std::string text;
            int prev = -1, next = -1;
            uint64_t version = 0;
            bool active = true;
        };
        struct Candidate {
            int rank, left, right;
            uint64_t left_version, right_version;
        };
        struct Later {
            bool operator()(const Candidate & a, const Candidate & b) const {
                if (a.rank != b.rank) { return a.rank > b.rank; }
                return a.left > b.left;
            }
        };
        std::vector<Node> nodes;
        for (size_t offset=0; offset<word.size();) {
            const size_t bytes=unicode_len_utf8(word[offset]);
            Node node; node.text = word.substr(offset,bytes);
            if (!nodes.empty()) { node.prev = int(nodes.size()) - 1; nodes.back().next = int(nodes.size()); }
            nodes.push_back(std::move(node)); offset += bytes;
        }
        std::priority_queue<Candidate, std::vector<Candidate>, Later> queue;
        auto push = [&](int left) {
            if (left < 0) { return; }
            Node & node = nodes[left];
            const int right = node.next;
            if (right < 0) { return; }
            const auto found = merges_.find(node.text + " " + nodes[right].text);
            if (found != merges_.end()) {
                queue.push({found->second, left, right, node.version, nodes[right].version});
            }
        };
        for (size_t i = 0; i + 1 < nodes.size(); ++i) { push(int(i)); }
        while (!queue.empty()) {
            const Candidate candidate = queue.top(); queue.pop();
            if (candidate.left < 0 || candidate.right < 0) { continue; }
            Node & left = nodes[candidate.left];
            Node & right = nodes[candidate.right];
            if (!left.active || !right.active || left.next != candidate.right ||
                left.version != candidate.left_version || right.version != candidate.right_version) { continue; }
            left.text += right.text;
            left.next = right.next;
            if (right.next >= 0) { nodes[right.next].prev = candidate.left; }
            ++left.version;
            right.active = false;
            push(left.prev);
            push(candidate.left);
        }
        for (int i = 0; i >= 0 && i < int(nodes.size()); i = nodes[i].next) {
            if (!nodes[i].active) { continue; }
            const auto found = ids_.find(nodes[i].text);
            GGML_ASSERT(found != ids_.end());
            output.push_back(found->second);
        }
    }
}
Tokens Tokenizer::encode(const std::string & text) const {
    Tokens result; if (add_bos_) { result.push_back(bos_); }
    size_t cursor=0;
    while (cursor<text.size()) {
        size_t next=text.size(); Token token=-1;
        for (Token id : special_) {
            const size_t position=text.find(pieces_[id],cursor);
            if (position<next || (position==next && token>=0 && pieces_[id].size()>pieces_[token].size())) { next=position; token=id; }
        }
        ordinary(text.substr(cursor,next-cursor),result);
        if (token<0) { break; }
        result.push_back(token); cursor=next+pieces_[token].size();
    }
    return result;
}
std::string Tokenizer::piece(Token token) const {
    GGML_ASSERT(token>=0 && size_t(token)<pieces_.size());
    if (std::find(special_.begin(),special_.end(),token)!=special_.end()) { return pieces_[token]; }
    std::string result; for (auto cpt : unicode_cpts_from_utf8(pieces_[token])) { result.push_back(char(unicode_cpt_to_byte(cpt))); } return result;
}
bool Tokenizer::eos(Token token) const {
    GGML_ASSERT(token>=0 && size_t(token)<pieces_.size());
    return token==eos_ || pieces_[token]=="<|im_end|>" || pieces_[token]=="<|endoftext|>";
}
Token Tokenizer::special_token(const std::string & name) const {
    const auto found = ids_.find(name);
    return found == ids_.end() ? -1 : found->second;
}
void Tokenizer::validate_vision() const {
    const char * names[] = {"<|vision_start|>", "<|vision_end|>", "<|vision_pad|>", "<|image_pad|>"};
    for (int i = 0; i < 4; ++i) {
        const auto encoded = encode(names[i]);
        if (special_token(names[i]) != 248053 + i || encoded.size() != 1 || encoded[0] != 248053 + i) {
            GGML_ABORT("invalid vision tokenizer: %s must be token %d", names[i], 248053 + i);
        }
    }
    std::string pads;
    for (int i = 0; i < 576; ++i) { pads += names[3]; }
    if (encode(pads).size() != 576 || encode(std::string(names[0]) + pads + names[1]).size() != 578) {
        GGML_ABORT("invalid vision tokenizer: expected 576 image tokens");
    }
}

}
