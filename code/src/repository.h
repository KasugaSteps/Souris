#pragma once
#include "souris/types.h"
#include "ggml-backend.h"
#include "gguf.h"
#include <array>
#include <memory>
#include <string>
#include <vector>

namespace souris {
struct Srs1Header;
/** Runtime record geometry derived once from the SRS1 v1 model header. */
struct Layout {
    uint64_t record_bytes = 0, expert_bytes = 0;
    uint64_t role_offset[3]{}, role_bytes[3]{};
    QuantDesc quant[fixed::layers][3]{};
    uint64_t quant_signature(int layer) const {
        const QuantType types[] = {QuantType(quant[layer][0].type), QuantType(quant[layer][1].type), QuantType(quant[layer][2].type)};
        return souris::quant_signature(types);
    }
};

/** Read-only mapping of one dense shard range. */
struct MappedFile {
    void * data=nullptr;
    size_t bytes=0;
    ggml_backend_buffer_t cpu_buffer=nullptr;
    MappedFile(const std::string & name,size_t offset,size_t bytes);
    ~MappedFile();
    MappedFile(const MappedFile &)=delete;
    MappedFile & operator=(const MappedFile &)=delete;
    static void prefetch(const void * address, size_t bytes);
private:
    void * mapping_=nullptr;
};
/** One mapped dense shard and its immutable GGUF tensor directory. */
struct MappedShard {
    MappedFile file;
    gguf_context * metadata = nullptr;
    ggml_context * tensors = nullptr;
    MappedShard(const std::string &, uint64_t offset, uint64_t bytes);
    ~MappedShard();
};

/** SRS1 metadata and direct expert reader; dense shards map separately. */
class ExpertRepository {
public:
    struct Segment { uint64_t offset,bytes; };
    struct Span { uint64_t offset; size_t bytes; ggml_type type; };
    const Layout & layout() const { return layout_; }
    const Srs1Header & header() const { return *header_; }
    Span role(int layer, int expert, int role) const;
    const std::array<int, fixed::layers> & compress_ratios() const { return ratios_; }
    const std::vector<std::unique_ptr<MappedShard>> & mappings() const { return shards_; }
    gguf_context * metadata() const { return shards_.front()->metadata; }
    explicit ExpertRepository(const std::string & path);
    ~ExpertRepository();
    ExpertRepository(const ExpertRepository &)=delete;
    ExpertRepository & operator=(const ExpertRepository &)=delete;
    /** Read aligned expert bytes directly into caller-owned storage; short reads are fatal. */
    void read(void * target,size_t offset,size_t bytes) const;
    /** Absolute file offset of one expert record. */
    uint64_t record_offset(int layer, int expert) const { return expert_offset_ + uint64_t(layer * fixed::experts + expert) * layout_.record_bytes; }
private:
    std::unique_ptr<Srs1Header> header_;
    Layout layout_{};
    std::array<int, fixed::layers> ratios_{};
    std::vector<std::unique_ptr<MappedShard>> shards_;
    int open_direct() const;
    int direct_=-1;
    std::string path_;
    uint64_t expert_offset_=0;
};
}
