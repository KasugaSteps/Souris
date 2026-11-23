#include "repository.h"
#include "model.h"
#include "ggml.h"
#include <cstring>
#include <cerrno>
#include <filesystem>
#include "ggml-cpu.h"
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace souris {
MappedFile::MappedFile(const std::string & path,size_t offset,size_t count) : bytes(count) {
#ifdef _WIN32
    const auto wide=std::filesystem::u8path(path);
    HANDLE file=CreateFileW(wide.c_str(),GENERIC_READ,FILE_SHARE_READ,nullptr,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,nullptr);
    GGML_ASSERT(file!=INVALID_HANDLE_VALUE);
    SYSTEM_INFO system;
    GetSystemInfo(&system);
    const size_t base=offset/system.dwAllocationGranularity*system.dwAllocationGranularity;
    HANDLE mapping=CreateFileMappingW(file,nullptr,PAGE_READONLY,0,0,nullptr);
    GGML_ASSERT(mapping);
    mapping_=MapViewOfFile(mapping,FILE_MAP_READ,DWORD(uint64_t(base)>>32),DWORD(base),bytes+offset-base);
    CloseHandle(mapping);
    CloseHandle(file);
    GGML_ASSERT(mapping_);
    data=static_cast<char *>(mapping_)+offset-base;
#else
    const int file=open(path.c_str(),O_RDONLY|O_CLOEXEC);
    GGML_ASSERT(file>=0);
    mapping_=mmap(nullptr,bytes,PROT_READ,MAP_PRIVATE,file,off_t(offset));
    close(file);
    GGML_ASSERT(mapping_!=MAP_FAILED);
    data=mapping_;
#endif
    cpu_buffer=ggml_backend_cpu_buffer_from_ptr(data,bytes);
    GGML_ASSERT(cpu_buffer);
    ggml_backend_buffer_set_usage(cpu_buffer,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
}
MappedFile::~MappedFile() {
    ggml_backend_buffer_free(cpu_buffer);
#ifdef _WIN32
    UnmapViewOfFile(mapping_);
#else
    munmap(mapping_,bytes);
#endif
}
void MappedFile::prefetch(const void * address,size_t bytes) {
    if (!bytes) { return; }
#ifdef _WIN32
    WIN32_MEMORY_RANGE_ENTRY range{const_cast<void *>(address),bytes};
    GGML_ASSERT(PrefetchVirtualMemory(GetCurrentProcess(),1,&range,0));
#else
    const size_t page=sysconf(_SC_PAGE_SIZE);
    const uintptr_t begin=uintptr_t(address)&~uintptr_t(page-1);
    GGML_ASSERT(madvise(reinterpret_cast<void *>(begin),bytes+uintptr_t(address)-begin,MADV_WILLNEED)==0);
#endif
}
MappedShard::MappedShard(const std::string & path,uint64_t offset,uint64_t count) : file(path,offset,count) {
    const void * data=file.data;
    const size_t bytes=file.bytes;
    metadata=gguf_init_from_buffer(data,bytes,{true,&tensors});
    GGML_ASSERT(metadata && tensors);
    for (int64_t i=0; i<gguf_get_n_tensors(metadata); ++i) {
        auto * tensor=ggml_get_tensor(tensors,gguf_get_tensor_name(metadata,i));
        if (std::string(ggml_get_name(tensor)).find("_exps.weight") != std::string::npos) { continue; }
        const size_t offset=gguf_get_data_offset(metadata)+gguf_get_tensor_offset(metadata,i);
        GGML_ASSERT(offset<=bytes && ggml_nbytes(tensor)<=bytes-offset);
        tensor->data=const_cast<char *>(static_cast<const char *>(data))+offset;
        tensor->buffer=file.cpu_buffer;
    }
}
MappedShard::~MappedShard() { ggml_free(tensors); gguf_free(metadata); }

ExpertRepository::ExpertRepository(const std::string & path) : path_(path) {
    const uint64_t bytes=std::filesystem::file_size(std::filesystem::u8path(path));
    GGML_ASSERT(bytes>=fixed::srs_header_size);
    MappedFile prefix(path,0,fixed::srs_header_size);
    const auto * buffer=static_cast<const char *>(prefix.data);
    header_ = std::make_unique<Srs1Header>();
    auto & header = *header_;
    std::memcpy(&header,buffer,sizeof(header));
    if (std::memcmp(header.magic,"SRS1",4)!=0 || header.version!=1) { GGML_ABORT("unsupported SRS1 version"); }
    GGML_ASSERT(header.header_size==fixed::srs_header_size && header.shard_count>0 && header.shard_count<=fixed::srs_max_shards);
    GGML_ASSERT(header.shard_table_offset==fixed::srs_shard_table);
    GGML_ASSERT(header.layers==fixed::layers && header.experts==fixed::experts);
    GGML_ASSERT(header.expert_count==uint64_t(fixed::layers)*fixed::experts);
    if (header.role_count != 3) { GGML_ABORT("invalid SRS1 role count"); }
    if (header.topk != fixed::topk) { GGML_ABORT("invalid SRS1 top-k"); }
    uint64_t role_end = 0;
    for (int role = 0; role < 3; ++role) {
        if (header.role_offset[role] < role_end || header.role_offset[role] % fixed::expert_alignment ||
            !header.role_bytes[role] || header.role_bytes[role] > UINT64_MAX - header.role_offset[role]) {
            GGML_ABORT("invalid SRS1 layout");
        }
        layout_.role_offset[role] = header.role_offset[role];
        layout_.role_bytes[role] = header.role_bytes[role];
        role_end = header.role_offset[role] + header.role_bytes[role];
    }
    if (role_end > UINT64_MAX - fixed::staging_alignment + 1) { GGML_ABORT("invalid SRS1 layout"); }
    layout_.record_bytes = GGML_PAD(role_end, fixed::staging_alignment);
    layout_.expert_bytes = layout_.record_bytes;
    std::memcpy(layout_.quant, header.quant, sizeof(header.quant));
    for (int layer = 0; layer < fixed::layers; ++layer) {
        for (int role = 0; role < 3; ++role) { quant_traits(QuantType(layout_.quant[layer][role].type)); }
        if (layout_.quant[layer][0].type != layout_.quant[layer][1].type) {
            GGML_ABORT("gate/up quantization and row layout must match");
        }
    }
    if (layout_.record_bytes > UINT64_MAX / header.expert_count) { GGML_ABORT("invalid SRS1 layout"); }
    GGML_ASSERT(header.expert_size==header.expert_count*layout_.record_bytes);
    GGML_ASSERT(header.expert_offset<=bytes && header.expert_size==bytes-header.expert_offset);
    GGML_ASSERT(header.expert_offset%fixed::staging_alignment==0);
    const uint64_t expert_offset=header.expert_offset;
    expert_offset_=expert_offset;
    std::vector<Segment> shards(header.shard_count);
    std::memcpy(shards.data(),buffer+header.shard_table_offset,shards.size()*sizeof(Segment));
    uint64_t end=header.header_size;
    for (const auto & shard : shards) {
        GGML_ASSERT(shard.offset>=end && shard.offset%fixed::staging_alignment==0);
        GGML_ASSERT(shard.offset<=expert_offset && shard.bytes<=expert_offset-shard.offset);
        end=shard.offset+shard.bytes;
    }
    for (const auto & shard : shards) {
        shards_.push_back(std::make_unique<MappedShard>(path, shard.offset, shard.bytes));
    }
    auto * meta = metadata();
    const int index = gguf_find_key(meta, "qwen4exp.attention.compress_ratios");
    if (index < 0) { GGML_ABORT("unsupported model metadata: qwen4exp.attention.compress_ratios"); }
    if (gguf_get_kv_type(meta, index) != GGUF_TYPE_ARRAY || gguf_get_arr_n(meta, index) != fixed::layers) {
        GGML_ABORT("invalid metadata: compress_ratios");
    }
    const void * values = gguf_get_arr_data(meta, index);
    for (int i = 0; i < fixed::layers; ++i) {
        uint64_t ratio;
        switch (gguf_get_arr_type(meta, index)) {
            case GGUF_TYPE_UINT32: ratio = static_cast<const uint32_t *>(values)[i]; break;
            case GGUF_TYPE_INT32: ratio = static_cast<const int32_t *>(values)[i]; break;
            case GGUF_TYPE_UINT64: ratio = static_cast<const uint64_t *>(values)[i]; break;
            case GGUF_TYPE_INT64: ratio = static_cast<const int64_t *>(values)[i]; break;
            default: GGML_ABORT("invalid metadata: compress_ratios type");
        }
        if (ratio != uint64_t(i % 4 == 3 ? 4 : 0)) { GGML_ABORT("invalid metadata: compress_ratios[%d]", i); }
        ratios_[i] = int(ratio);
    }
    direct_=open_direct();
}
ExpertRepository::Span ExpertRepository::role(int layer, int expert, int role) const {
    GGML_ASSERT(layer >= 0 && layer < fixed::layers && expert >= 0 && expert < fixed::experts && role >= 0 && role < 3);
    const char * names[] = {"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"};
    const auto name = "blk." + std::to_string(layer) + "." + names[role];
    ggml_tensor * shape = nullptr;
    for (const auto & shard : shards_) {
        if ((shape = ggml_get_tensor(shard->tensors, name.c_str()))) { break; }
    }
    if (!shape) { GGML_ABORT("missing expert shape: %s", name.c_str()); }
    const auto type = QuantType(layout_.quant[layer][role].type);
    const auto bytes = quant_traits(type).row_bytes(shape->ne[0]) * shape->ne[1];
    if (bytes > layout_.role_bytes[role]) { GGML_ABORT("invalid SRS1 role capacity"); }
    return {record_offset(layer, expert) + layout_.role_offset[role], bytes, type};
}
ExpertRepository::~ExpertRepository() { if (direct_>=0) { GGML_ASSERT(close(direct_)==0); } }
int ExpertRepository::open_direct() const {
#ifdef _WIN32
    GGML_ABORT("O_DIRECT ExpertCache loading requires Linux");
#else
    const int fd=open(path_.c_str(),O_RDONLY|O_DIRECT|O_CLOEXEC);
    if (fd<0) { GGML_ABORT("ExpertCache open(O_DIRECT) failed for %s: %s",path_.c_str(),std::strerror(errno)); }
    return fd;
#endif
}
void ExpertRepository::read(void * target,size_t offset,size_t bytes) const {
    GGML_ASSERT(uintptr_t(target)%fixed::staging_alignment==0 && offset%fixed::staging_alignment==0 && bytes%fixed::staging_alignment==0);
    GGML_ASSERT(offset<=size_t(fixed::layers)*fixed::experts*layout_.record_bytes &&
                bytes<=size_t(fixed::layers)*fixed::experts*layout_.record_bytes-offset);
    ssize_t result;
    do { result=pread(direct_,target,bytes,expert_offset_+offset); } while (result<0 && errno==EINTR);
    if (result<0) { GGML_ABORT("Expert read at %zu failed: %s",offset,std::strerror(errno)); }
    if (size_t(result)!=bytes) { GGML_ABORT("Expert short read at %zu: %zd of %zu",offset,result,bytes); }
}
}
