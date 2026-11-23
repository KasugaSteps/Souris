#pragma once
#include <cstdint>
/** Optional device indirection for quantized MoE weights. */
struct ggml_cuda_expert_ptrs {
    const void * const * data=nullptr;
    const int32_t * layer=nullptr;
    int64_t layer_stride=0;
    __device__ const void * get(int expert) const { return data[int64_t(*layer)*layer_stride+expert]; }
};
