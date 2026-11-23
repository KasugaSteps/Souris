#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#ifdef GGML_USE_HIP
#define GGML_CUDA_NAME "ROCm"
#define GGML_CUBLAS_NAME "hipBLAS"
#elif defined(GGML_USE_MUSA)
#define GGML_CUDA_NAME "MUSA"
#define GGML_CUBLAS_NAME "muBLAS"
#else
#define GGML_CUDA_NAME "CUDA"
#define GGML_CUBLAS_NAME "cuBLAS"
#endif
#define GGML_CUDA_MAX_DEVICES       16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_cuda_init(int device);

GGML_BACKEND_API bool ggml_backend_is_cuda(ggml_backend_t backend);

/** Query actual CUDA graph capture, replay and direct submission counts. */
GGML_BACKEND_API void ggml_backend_cuda_get_graph_stats(ggml_backend_t backend, uint64_t * captures, uint64_t * replays, uint64_t * direct);
/** Set graph inactivity eviction in microseconds; zero retains graphs until backend teardown. */
GGML_BACKEND_API void ggml_backend_cuda_set_graph_cache_timeout(ggml_backend_t backend, int64_t microseconds);
/** Drop cached captures after synchronizing this backend, before replacing graph allocations. */
GGML_BACKEND_API void ggml_backend_cuda_clear_graph_cache(ggml_backend_t backend);
/** Bound retained captures and retire captures belonging to discarded graph metadata. */
GGML_BACKEND_API void ggml_backend_cuda_set_graph_cache_limit(ggml_backend_t backend, size_t limit);
GGML_BACKEND_API void ggml_backend_cuda_forget_graph(ggml_backend_t backend, const struct ggml_cgraph * graph);
GGML_BACKEND_API void ggml_backend_cuda_get_graph_cache_stats(ggml_backend_t backend, uint64_t * peak, uint64_t * evictions);
/** Current retained entries and host metadata bytes; driver execution objects are opaque. */
GGML_BACKEND_API void ggml_backend_cuda_get_graph_cache_usage(ggml_backend_t backend, size_t * entries, size_t * host_bytes);
/** Nonblocking completion query for an event created by the CUDA backend. */
GGML_BACKEND_API bool ggml_backend_cuda_event_query(ggml_backend_event_t event);

// device buffer
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device);

// conduct allreduce operation between devices
GGML_BACKEND_API bool ggml_backend_cuda_allreduce_tensor(ggml_backend_t * backends, struct ggml_tensor ** tensors, size_t n_backends);

// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type(void);

GGML_BACKEND_API int  ggml_backend_cuda_get_device_count(void);
GGML_BACKEND_API void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size);
GGML_BACKEND_API void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total);

GGML_BACKEND_API bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size);
GGML_BACKEND_API void ggml_backend_cuda_unregister_host_buffer(void * buffer);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_cuda_reg(void);

#ifdef  __cplusplus
}
#endif
