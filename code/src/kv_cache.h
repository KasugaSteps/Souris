#pragma once
#include "device.h"

namespace souris {
/** Direct Q8_0 K/V tensors and separate QSA/recurrent histories. */
class KVCacheManager {
public:
    struct Layer { ggml_tensor * key=nullptr, * value=nullptr, * index=nullptr, * conv=nullptr, * recurrent=nullptr; };
    Layer layers[fixed::layers];
    ggml_tensor * ple_history=nullptr;
    int capacity, position=0;
    size_t kv_bytes=0, fixed_bytes=0;
    KVCacheManager(const Dimensions & dim, int context, Device & device);
    /** Verify shares append-only K/V and index rows, while recurrent/PLE state is private. */
    KVCacheManager(const Dimensions &, KVCacheManager & canonical, Device &);
    ~KVCacheManager();
    void reset();
    /** Allocate eight reusable recurrent/PLE checkpoints before budget locking. */
    void prepare_checkpoints();
    /** Copy only state preceding a prefix boundary; append-only K/V and index stay in place. */
    void capture_checkpoint(int slot);
    void capture_checkpoint_layer(int slot, int layer, bool ple);
    void restore_checkpoint(int slot, int position);
    ggml_type type() const { return GGML_TYPE_Q8_0; }
    struct Views { ggml_tensor * key; ggml_tensor * value; };
    /** Quantize appended head blocks into the persistent Q8_0 tensors. */
    Views append(Frame &, int layer, ggml_tensor * key, ggml_tensor * value, ggml_tensor * rows);

private:
    void allocate_q8_0(const Dimensions &);
    ggml_context * context_ = nullptr, * kv_context_ = nullptr;
    ggml_backend_buffer_t kv_buffer_ = nullptr, fixed_buffer_ = nullptr;
    Device & device_;
    ggml_backend_buffer_t checkpoints_ = nullptr;
    size_t checkpoint_stride_ = 0;
    struct CheckpointTensor { ggml_tensor * tensor; size_t offset; };
    std::vector<CheckpointTensor> checkpoint_tensors_;
};

}
