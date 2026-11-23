#include "grammar_verifier.h"
#include "tokenizer.h"
#include "device.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <map>
#include <unordered_map>
#include <stdexcept>
#include <functional>
#include <cmath>

namespace souris {
namespace {
using Json = nlohmann::json;
using Stack = std::vector<int>;
using StateSet = std::vector<Stack>;
bool valid_constant(const Json & v, const Json & s) {
    const std::string type = s.value("type", std::string());
    if ((!type.empty() && type == "object" && !v.is_object()) || (type == "array" && !v.is_array()) ||
        (type == "string" && !v.is_string()) || (type == "boolean" && !v.is_boolean()) ||
        (type == "null" && !v.is_null()) || (type == "number" && !v.is_number()) ||
        (type == "integer" && (!v.is_number() || std::floor(v.get<double>()) != v.get<double>()))) { return false; }
    if (v.is_object()) {
        const auto props = s.value("properties", Json::object());
        for (const auto & name : s.value("required", Json::array())) { if (!name.is_string() || !v.contains(name.get<std::string>())) { return false; } }
        for (const auto & item : v.items()) {
            if (props.contains(item.key())) { if (!valid_constant(item.value(), props[item.key()])) { return false; } }
            else if (!s.value("additionalProperties", true)) { return false; }
        }
    }
    if (v.is_array() && s.contains("items")) { for (const auto & item : v) { if (!valid_constant(item, s["items"])) { return false; } } }
    if (s.contains("const") && v != s["const"]) { return false; }
    if (s.contains("enum") && std::find(s["enum"].begin(), s["enum"].end(), v) == s["enum"].end()) { return false; }
    return true;
}
struct Bytes {
    uint64_t bits[4]{};
    void add(unsigned char c) { bits[c / 64] |= uint64_t(1) << (c % 64); }
    bool has(unsigned char c) const { return bits[c / 64] & (uint64_t(1) << (c % 64)); }
};
/** A byte CFG is expanded to canonical terminal-front stacks. State masks are
 * cached by these stacks, so free-form string contents do not create new states. */
struct Grammar {
    std::vector<std::vector<Stack>> rules;
    std::vector<Bytes> terminals;
    std::unordered_map<std::string,int> literals;
    int ws, string, number, integer, value, body;
    int rule() {
        if (rules.size() >= 65536) { throw std::runtime_error("tool schema compilation limit"); }
        rules.emplace_back(); return int(rules.size()) - 1;
    }
    int chars(const std::string & text) {
        Bytes bytes; for (unsigned char c : text) { bytes.add(c); }
        terminals.push_back(bytes); return -int(terminals.size());
    }
    int range(int first, int last, const std::string & except = {}) {
        std::string text; for (int c = first; c <= last; ++c) { if (except.find(char(c)) == std::string::npos) { text += char(c); } }
        return chars(text);
    }
    Stack literal(const std::string & text) { Stack seq; for (char c : text) { seq.push_back(chars(std::string(1, c))); } return seq; }
    int lit(const std::string & text) {
        const auto found = literals.find(text); if (found != literals.end()) { return found->second; }
        int r = rule(); rules[r].push_back(literal(text)); literals.emplace(text,r); return r;
    }
    void add(int r, Stack seq) { rules[r].push_back(std::move(seq)); }
    int alternatives(const std::vector<int> & choices) { int r = rule(); for (int c : choices) { add(r, {c}); } return r; }
    int optional(int r) { int out = rule(); add(out, {}); add(out, {r}); return out; }
    Grammar() {
        ws = rule(); add(ws, {}); add(ws, {chars(" \t\r\n"), ws});
        const int digit = range('0', '9'), nonzero = range('1', '9');
        const int digits = rule(); add(digits, {}); add(digits, {digit, digits});
        const int whole = rule(); add(whole, {chars("0")}); add(whole, {nonzero, digits});
        integer = rule(); add(integer, {optional(lit("-")), whole});
        int fraction = rule(); add(fraction, {lit("."), digit, digits});
        int exponent = rule(); add(exponent, {chars("eE"), optional(alternatives({lit("+"), lit("-")})), digit, digits});
        number = rule(); add(number, {integer, optional(fraction), optional(exponent)});
        const int hex = chars("0123456789abcdefABCDEF");
        int escaped = rule(); add(escaped, {chars("\"\\/bfnrt")});
        add(escaped, {lit("u"), chars("0123456789abcefABCEF"), hex, hex, hex});
        add(escaped, {lit("u"), chars("dD"), chars("01234567"), hex, hex});
        add(escaped, {lit("u"), chars("dD"), chars("89abAB"), hex, hex, lit("\\u"), chars("dD"), chars("cdefCDEF"), hex, hex});
        int unicode = rule(), continuation = range(0x80, 0xbf);
        add(unicode, {range(0xc2,0xdf),continuation});
        add(unicode, {chars(std::string(1,char(0xe0))),range(0xa0,0xbf),continuation});
        add(unicode, {range(0xe1,0xec),continuation,continuation});
        add(unicode, {chars(std::string(1,char(0xed))),range(0x80,0x9f),continuation});
        add(unicode, {range(0xee,0xef),continuation,continuation});
        add(unicode, {chars(std::string(1,char(0xf0))),range(0x90,0xbf),continuation,continuation});
        add(unicode, {range(0xf1,0xf3),continuation,continuation,continuation});
        add(unicode, {chars(std::string(1,char(0xf4))),range(0x80,0x8f),continuation,continuation});
        int tail = rule(); add(tail, {}); add(tail, {range(32,127,"\"\\"), tail}); add(tail,{unicode,tail}); add(tail, {lit("\\"), escaped, tail});
        string = rule(); add(string, {lit("\""), tail, lit("\"")});
        value = rule();
        int object = schema(Json{{"type", "object"}}), array = schema(Json{{"type", "array"}});
        for (int r : {string, number, lit("true"), lit("false"), lit("null"), object, array}) { add(value, {r}); }
    }
    // Unknown property keys must not alias a declared property. Build a small
    // exclusion trie; escaped property names are intentionally not emitted.
    int unknown_key(const std::vector<std::string> & names) {
        struct Node { std::map<unsigned char,int> edges; bool end = false; };
        std::vector<Node> trie(1);
        for (const auto & name : names) {
            int at = 0; for (unsigned char c : name) {
                auto found = trie[at].edges.find(c);
                if (found == trie[at].edges.end()) { int next = trie.size(); trie[at].edges[c] = next; trie.emplace_back(); at = next; }
                else { at = found->second; }
            } trie[at].end = true;
        }
        int rest = rule(); add(rest, {lit("\"")}); add(rest, {range(32,126,"\"\\"), rest});
        std::vector<int> refs; for (size_t i=0;i<trie.size();++i) { refs.push_back(rule()); }
        for (size_t i=0;i<trie.size();++i) {
            if (!trie[i].end) { add(refs[i], {lit("\"")}); }
            std::string exclude = "\"\\";
            for (auto edge : trie[i].edges) { exclude += char(edge.first); add(refs[i], {chars(std::string(1,char(edge.first))), refs[edge.second]}); }
            add(refs[i], {range(32,126,exclude), rest});
        }
        int result=rule(); add(result, {lit("\""), refs[0]}); return result;
    }
    int schema(const Json & s, int depth = 0) {
        if (!s.is_object() || depth > 24) { throw std::runtime_error("unsupported tool schema: object/depth"); }
        for (auto & item : s.items()) {
            if (item.key() != "type" && item.key() != "properties" && item.key() != "required" && item.key() != "additionalProperties" &&
                item.key() != "items" && item.key() != "enum" && item.key() != "const" && item.key() != "description" && item.key() != "default") {
                throw std::runtime_error("unsupported tool schema: " + item.key());
            }
        }
        // Validate nested schemas even when const/enum narrows the language.
        if (s.contains("properties")) { if (!s["properties"].is_object()) { throw std::runtime_error("invalid properties"); } }
        int result = rule();
        if (s.contains("const") || s.contains("enum")) {
            Json values=s.contains("const") ? Json::array({s["const"]}) : s["enum"];
            if (!values.is_array() || values.empty()) { throw std::runtime_error("invalid schema enum"); }
            Json base=s; base.erase("const"); base.erase("enum"); schema(base,depth+1);
            for (const auto & v:values) { if(valid_constant(v,s)){add(result, {lit(v.dump())});} }
            if(rules[result].empty()){throw std::runtime_error("unsatisfiable tool const/enum");}return result;
        }
        if (!s.contains("type")) {
            for (auto & item : s.items()) { if (item.key()!="description" && item.key()!="default") { throw std::runtime_error("tool schema requires type"); } }
            return value;
        }
        if (!s["type"].is_string()) { throw std::runtime_error("unsupported tool schema type union"); }
        const std::string type=s["type"];
        if (type=="string") { return string; }
        if (type=="number") { return number; }
        if (type=="integer") { return integer; }
        if (type=="boolean") { return alternatives({lit("true"),lit("false")}); }
        if (type=="null") { return lit("null"); }
        if (type=="array") {
            int item=s.contains("items") ? schema(s["items"],depth+1) : value;
            int more=rule(); add(more,{}); add(more,{lit(","),ws,item,ws,more});
            add(result,{lit("["),ws,lit("]")}); add(result,{lit("["),ws,item,ws,more,lit("]")}); return result;
        }
        if (type!="object") { throw std::runtime_error("unsupported tool schema type: "+type); }
        Json props=s.value("properties",Json::object());
        if (props.size()>12) { throw std::runtime_error("tool schema object exceeds 12 declared properties"); }
        std::vector<std::string> names; std::vector<int> values;
        for (auto & p:props.items()) {
            if (std::any_of(p.key().begin(),p.key().end(),[](unsigned char c){return c<32||c>126||c=='\"'||c=='\\';})) {
                throw std::runtime_error("unsupported escaped/non-ASCII property name");
            }
            names.push_back(p.key()); values.push_back(schema(p.value(),depth+1));
        }
        unsigned required=0;
        if (s.contains("required")) {
            if (!s["required"].is_array()) { throw std::runtime_error("invalid required"); }
            for (const auto & name:s["required"]) {
                if (!name.is_string()) { throw std::runtime_error("invalid required name"); }
                auto it=std::find(names.begin(),names.end(),name.get<std::string>());
                if (it==names.end()) { throw std::runtime_error("required name missing from properties"); }
                required |= 1u << (it-names.begin());
            }
        }
        if (s.contains("additionalProperties") && !s["additionalProperties"].is_boolean()) { throw std::runtime_error("unsupported additionalProperties schema"); }
        bool extra=s.value("additionalProperties",true);
        int unknown=extra ? unknown_key(names) : -1;
        std::map<std::pair<unsigned,bool>,int> states;
        std::function<int(unsigned,bool)> contents=[&](unsigned seen,bool first) {
            auto key=std::make_pair(seen,first); auto found=states.find(key); if(found!=states.end()) { return found->second; }
            int r=rule();states[key]=r;
            if ((seen & required)==required) { add(r,{lit("}")}); }
            int comma=first ? lit("") : lit(",");
            for(size_t i=0;i<names.size();++i) { if (!(seen & (1u<<i))) {
                int next=contents(seen|(1u<<i),false);
                add(r,{comma,ws,lit(Json(names[i]).dump()),ws,lit(":"),ws,values[i],ws,next});
            } }
            if(extra) { int next=first ? contents(seen,false) : r; add(r,{comma,ws,unknown,ws,lit(":"),ws,value,ws,next}); }
            return r;
        };
        int start=contents(0,true);add(result,{lit("{"),ws,start});return result;
    }
    StateSet expand(Stack input) const {
        StateSet pending{std::move(input)}, out;
        while(!pending.empty()) {
            Stack stack=std::move(pending.back());pending.pop_back();
            if(stack.empty() || stack.back()<0) { out.push_back(std::move(stack));continue; }
            int r=stack.back();stack.pop_back();
            for(const auto & production:rules[r]) { Stack next=stack; next.insert(next.end(),production.rbegin(),production.rend()); pending.push_back(std::move(next)); }
            if(pending.size()+out.size()>32768) { throw std::runtime_error("tool grammar state overflow"); }
        }
        std::sort(out.begin(),out.end());out.erase(std::unique(out.begin(),out.end()),out.end());return out;
    }
    bool consume(StateSet & state,unsigned char byte) const {
        StateSet next;
        for(auto stack:state) { if(!stack.empty() && terminals[-stack.back()-1].has(byte)) {
            stack.pop_back();auto expanded=expand(std::move(stack));next.insert(next.end(),std::make_move_iterator(expanded.begin()),std::make_move_iterator(expanded.end()));
        } }
        std::sort(next.begin(),next.end());next.erase(std::unique(next.begin(),next.end()),next.end());state=std::move(next);return !state.empty();
    }
};
struct Trie {
    struct Node { std::vector<std::pair<unsigned char,int>> edges; Tokens tokens; };
    std::vector<Node> nodes{1}; std::vector<std::string> pieces; Tokens eos;
    explicit Trie(Tokenizer & tokenizer) {
        pieces.reserve(tokenizer.vocab_size());
        for(int token=0;token<tokenizer.vocab_size();++token) {
            pieces.push_back(tokenizer.piece(token));if(tokenizer.eos(token)) { eos.push_back(token);continue; }
            int at=0;
            for(unsigned char byte:pieces.back()) {
                auto & edges=nodes[at].edges; auto it=std::find_if(edges.begin(),edges.end(),[&](auto edge){return edge.first==byte;});
                if(it!=edges.end()) { at=it->second; } else { int next=nodes.size();edges.push_back({byte,next});nodes.emplace_back();at=next; }
            }
            if(!pieces.back().empty()) { nodes[at].tokens.push_back(token); }
        }
    }
};
struct MaskEntry {
    std::vector<uint32_t> bits;Tokens allowed; uint32_t * gpu=nullptr;
    ~MaskEntry(){if(gpu){cudaFree(gpu);}}
};
struct Compiled {
    Grammar grammar;
    std::unordered_map<std::string,std::unique_ptr<MaskEntry>> masks;
};
enum Mode { Text, Open, Body, Close, Between };
struct Parser { Mode mode=Text;int offset=0,calls=0;StateSet stacks;std::string rolling; };
bool complete(const Parser & p) { return std::any_of(p.stacks.begin(),p.stacks.end(),[](const Stack & s){return s.empty();}); }
const std::string opening="<tool_call>", closing="</tool_call>";
std::string state_key(const Parser & s) {
    std::string key;auto append=[&](int n){key.append(reinterpret_cast<const char *>(&n),sizeof(n));};
    append(s.mode);append(s.offset);append(s.calls);for(const auto & stack:s.stacks){append(stack.size());for(int n:stack){append(n);}}return key;
}
}
struct GrammarVerifier::Impl {
    Tokenizer * tokenizer=nullptr; std::shared_ptr<Trie> trie;
    std::shared_ptr<Compiled> compiled;
    std::unordered_map<std::string,std::shared_ptr<Compiled>> schemas;
    Parser state;std::string failure;bool automatic=false;int limit=4;
    bool next(Parser & p,unsigned char c) const {
        if(p.mode==Text){
            if(!automatic){return true;}
            p.rolling+=char(c);if(p.rolling.size()>opening.size()){p.rolling.erase(0,p.rolling.size()-opening.size());}
            if(p.rolling==opening){
                if(p.calls>=limit){return false;}
                p.mode=Body;p.offset=0;p.rolling.clear();p.stacks=compiled->grammar.expand({compiled->grammar.body});
            }
            return true;
        }
        if(p.mode==Between){if(c==' '||c=='\n'||c=='\r'||c=='\t'){return true;}if(p.calls>=limit){return false;}p.mode=Open;p.offset=0;}
        if(p.mode==Open){
            if(p.offset==0 && (c==' '||c=='\n'||c=='\r'||c=='\t')){return true;}
            if(c!=opening[p.offset]){return false;}
            if(++p.offset==int(opening.size())){p.mode=Body;p.offset=0;p.stacks=compiled->grammar.expand({compiled->grammar.body});}return true;
        }
        if(p.mode==Body){
            StateSet next=p.stacks;
            if(compiled->grammar.consume(next,c)){p.stacks=std::move(next);return true;}
            if(!complete(p)){return false;}p.mode=Close;p.offset=0;p.stacks.clear();
        }
        if(p.mode==Close){
            if(c!=closing[p.offset]){return false;}
            if(++p.offset==int(closing.size())){++p.calls;p.offset=0;p.mode=automatic?Text:Between;}return true;
        }
        return false;
    }
    bool eof(const Parser & p) const{return p.mode==Text || (p.mode==Between && p.calls>0);}
};
GrammarVerifier::GrammarVerifier():impl_(std::make_unique<Impl>()){}
GrammarVerifier::~GrammarVerifier()=default;
bool GrammarVerifier::configure(const ToolConstraint & constraint,Tokenizer & tokenizer){
    auto & x=*impl_;reset();
    if(constraint.json.empty()){x.compiled.reset();return true;}
    try{
        Json j=Json::parse(constraint.json);
        if(!j.is_object()||j.value("kind",std::string())!="tool_call"){throw std::runtime_error("invalid tool constraint");}
        for(auto & item:j.items()){if(item.key()!="kind"&&item.key()!="choice"&&item.key()!="tools"&&item.key()!="max_calls"&&item.key()!="parallel_tool_calls"){throw std::runtime_error("unsupported tool constraint field");}}
        if(j.contains("max_calls")&&(!j["max_calls"].is_number_integer()||j["max_calls"]<1||j["max_calls"]>8)){throw std::runtime_error("invalid max_calls");}
        if(j.contains("parallel_tool_calls")&&!j["parallel_tool_calls"].is_boolean()){throw std::runtime_error("invalid parallel_tool_calls");}
        x.limit=j.value("parallel_tool_calls",true)?j.value("max_calls",4):1;
        Json choice=j.value("choice",Json("auto"));std::string named;
        x.automatic=choice=="auto";
        if(choice.is_object()&&choice.size()==1&&choice.contains("name")&&choice["name"].is_string()){named=choice["name"];}
        else if(choice!="auto"&&choice!="required"){throw std::runtime_error("invalid tool choice");}
        if(!j.contains("tools")||!j["tools"].is_array()||j["tools"].empty()){throw std::runtime_error("tool constraint needs tools");}
        if(x.tokenizer!=&tokenizer){x.schemas.clear();x.trie=std::make_shared<Trie>(tokenizer);x.tokenizer=&tokenizer;}
        // Mode and call limits affect state masks, not schema compilation.
        const std::string key=Json{{"tools",j["tools"]},{"name",named}}.dump();auto found=x.schemas.find(key);
        if(found!=x.schemas.end()){x.compiled=found->second;}
        else{
            auto compiled=std::make_shared<Compiled>();auto & g=compiled->grammar;g.body=g.rule();std::vector<std::string> names;
            for(const auto & tool:j["tools"]){
                if(!tool.is_object()||!tool.contains("name")||!tool["name"].is_string()||!tool.contains("parameters")||!tool["parameters"].is_object()||tool["parameters"].value("type",std::string())!="object"){throw std::runtime_error("invalid tool schema");}
                std::string name=tool["name"];if(name.empty()||name.size()>64||std::find(names.begin(),names.end(),name)!=names.end()){throw std::runtime_error("invalid tool name");}names.push_back(name);
                int args=g.schema(tool["parameters"]);if(!named.empty()&&name!=named){continue;}
                g.add(g.body,{g.ws,g.lit("{"),g.ws,g.lit("\"name\""),g.ws,g.lit(":"),g.ws,g.lit(Json(name).dump()),g.ws,g.lit(","),g.ws,g.lit("\"arguments\""),g.ws,g.lit(":"),g.ws,args,g.ws,g.lit("}"),g.ws});
            }
            if(g.rules[g.body].empty()){throw std::runtime_error("named tool is unavailable");}
            if(x.schemas.size()>=16){x.schemas.erase(x.schemas.begin());}x.schemas.emplace(key,compiled);x.compiled=std::move(compiled);
        }
        return true;
    }catch(const std::exception & e){x.failure=e.what();x.compiled.reset();return false;}
}
bool GrammarVerifier::begin_tool_call(){auto & x=*impl_;if(!x.compiled||x.state.calls>=x.limit){return false;}x.state.mode=x.automatic?Body:Open;x.state.offset=0;x.state.rolling.clear();if(x.automatic){x.state.stacks=x.compiled->grammar.expand({x.compiled->grammar.body});}return true;}
bool GrammarVerifier::finish_tool_call(){return impl_->state.calls>0 && accepting();}
int GrammarVerifier::call_count()const{return impl_->state.calls;}
bool GrammarVerifier::active()const{return impl_->compiled&&(impl_->state.mode!=Text||impl_->automatic);}
bool GrammarVerifier::accepting()const{return impl_->eof(impl_->state);}
void GrammarVerifier::reset(){impl_->state={};impl_->failure.clear();}
const std::string & GrammarVerifier::error()const{return impl_->failure;}
bool GrammarVerifier::observe(Token token){
    auto & x=*impl_;if(!x.compiled||!x.automatic){return false;}
    if(!advance(token)){return false;}
    return active();
}
int GrammarVerifier::text_prefix(const Token * tokens,int count) const {
    const auto & x=*impl_;
    if(!x.compiled || !x.automatic){return count;}
    Parser probe=x.state;
    for(int i=0;i<count;++i){
        for(unsigned char byte:x.trie->pieces[tokens[i]]){
            if(!x.next(probe,byte) || probe.mode!=Text){return i;}
        }
    }
    return count;
}
bool GrammarVerifier::advance(Token token){
    auto & x=*impl_;
    try{for(unsigned char byte:x.trie->pieces[token]){if(!x.next(x.state,byte)){x.failure="internal tool grammar error";return false;}}return true;}
    catch(const std::exception & e){x.failure=e.what();return false;}
}
TokenMask GrammarVerifier::mask(){
    auto & x=*impl_;if(!active()){return {};}
    try{
        const std::string key=std::to_string(x.limit)+(x.automatic?"A":"R")+state_key(x.state);
        auto & cache=x.compiled->masks;auto found=cache.find(key);
        if(found==cache.end()){
            if(cache.size()>=1024){throw std::runtime_error("tool grammar cache overflow");}
            auto entry=std::make_unique<MaskEntry>();entry->bits.resize((x.trie->pieces.size()+31)/32);
            auto allow=[&](Token token){entry->bits[token/32]|=1u<<(token%32);entry->allowed.push_back(token);};
            // At the automatic-call limit a token may complete <tool_call> from a
            // trie node; walk() adds node tokens without replaying their bytes.
            auto allowed=[&](const Parser & state,Token token){
                if(!x.automatic||state.mode!=Text||state.calls<x.limit){return true;}
                Parser probe=state;
                for(unsigned char byte:x.trie->pieces[token]){if(!x.next(probe,byte)){return false;}}
                return true;
            };
            std::function<void(int,const Parser &)> walk=[&](int node,const Parser & state){
                for(Token token:x.trie->nodes[node].tokens){if(allowed(state,token)){allow(token);}}
                for(auto edge:x.trie->nodes[node].edges){Parser next=state;if(x.next(next,edge.first)){walk(edge.second,next);}}
            };
            if (x.state.mode != Between || x.state.calls < x.limit) { walk(0,x.state); }
            if(x.eof(x.state)){for(Token token:x.trie->eos){allow(token);}}
            std::sort(entry->allowed.begin(),entry->allowed.end());
            if(entry->allowed.empty()){throw std::runtime_error("empty tool grammar mask");}
            const size_t sparse=entry->allowed.size()<=64 ? entry->allowed.size() : 0;
            SOURIS_CUDA(cudaMalloc(&entry->gpu,(entry->bits.size()+sparse)*sizeof(uint32_t)));
            SOURIS_CUDA(cudaMemcpy(entry->gpu,entry->bits.data(),entry->bits.size()*sizeof(uint32_t),cudaMemcpyHostToDevice));
            if(sparse){SOURIS_CUDA(cudaMemcpy(entry->gpu+entry->bits.size(),entry->allowed.data(),sparse*sizeof(Token),cudaMemcpyHostToDevice));}
            found=cache.emplace(key,std::move(entry)).first;
        }
        auto & e=*found->second;return {e.gpu,e.bits.data(),e.allowed.data(),e.allowed.size(),e.bits.size(),e.allowed.size()<=64?reinterpret_cast<Token *>(e.gpu+e.bits.size()):nullptr};
    }catch(const std::exception & e){x.failure=e.what();return {};}
}
}
