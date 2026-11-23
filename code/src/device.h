#pragma once
#include "model.h"
#include "ggml-cuda-custom.h"
#include "ggml-alloc.h"
#include "moe.h"
#include <cuda_runtime.h>

#define SOURIS_CUDA(call) do { const cudaError_t error_=(call); if (error_!=cudaSuccess) { GGML_ABORT("%s: %s",#call,cudaGetErrorString(error_)); } } while (0)

namespace souris {
/** Per-frame CUDA metadata; token positions, IDs and sequences remain device data. */
struct DeviceOp {
    int tokens=0, capacity=0, pool=0;
    size_t positions=0, write_ids=0, read_ids=0, pool_mask=0, tail_ids=0, pool_positions=0;
};

/** Immutable identity of a captured layer fragment or a shared hit graph. */
struct GraphKey {
    int layer = -1, rows = 0, count = 0;
    uint64_t quant_signature = 0;
    bool operator==(const GraphKey & other) const {
        return layer == other.layer && rows == other.rows && count == other.count &&
               quant_signature == other.quant_signature;
    }
};

/** Backend execution stream and independent record/control copies. */
struct Device {
    ggml_backend_t backend;
    cudaStream_t compute, upload, download;
    size_t total;
    Device();
    ~Device();
};
/** Stable metadata for one eager fragment or one retained capture. */
struct Frame {
    ggml_context * ctx;
    ggml_cgraph * graph;
    GraphKey key;
    explicit Frame(size_t nodes=1024);
    ~Frame();
    Frame(const Frame &)=delete;
    Frame & operator=(const Frame &)=delete;
    /** Borrow allocated storage without importing another graph's producer nodes. */
    ggml_tensor * bind(ggml_tensor * tensor);
    void expand(ggml_tensor * tensor);
    /** Reuse metadata only on the uncaptured eager path. */
    void reset();
    DeviceOp & add_op();
private:
    static constexpr int max_ops=8;
    DeviceOp operations_[max_ops];
    int operations_count_=0;
    size_t nodes_;
};

/** GGML allocator reused by ordered fragments; reserve all shapes before capture. */
struct Arena {
    ggml_gallocr_t allocator;
    Arena();
    ~Arena();
    size_t measure(Frame & frame);
    void reserve(Frame & frame);
    void allocate(Frame & frame);
    size_t bytes() const;
    /** Preserve captured tensor addresses after all supported shapes are reserved. */
    void freeze() { frozen_ = true; }
private:
    bool frozen_ = false;
};

/** Create a typed view of byte storage with fixed external addresses. */
ggml_tensor * storage_view(ggml_context * ctx, ggml_tensor * storage, ggml_type type,
                          int64_t n0, int64_t n1, int64_t n2, size_t offset, size_t row_stride=0, size_t plane_stride=0);
}
