#include "device.h"
#include "ggml-cuda.h"
#include <cstring>
#include <algorithm>

namespace souris {
Device::Device() {
    backend=ggml_backend_cuda_init(0); GGML_ASSERT(backend);
    int major, minor;
    SOURIS_CUDA(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0));
    SOURIS_CUDA(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0));
    if (major * 10 + minor < 80) { GGML_ABORT("unsupported CUDA configuration: cp.async/int8 tensor core"); }
    compute=static_cast<cudaStream_t>(ggml_backend_cuda_get_stream(backend));
    SOURIS_CUDA(cudaStreamCreateWithFlags(&upload,cudaStreamNonBlocking));
    SOURIS_CUDA(cudaStreamCreateWithFlags(&download,cudaStreamNonBlocking));
    size_t free; SOURIS_CUDA(cudaMemGetInfo(&free,&total));
}
Device::~Device() {
    SOURIS_CUDA(cudaStreamSynchronize(upload)); SOURIS_CUDA(cudaStreamSynchronize(download));
    SOURIS_CUDA(cudaStreamSynchronize(compute)); SOURIS_CUDA(cudaStreamDestroy(upload)); SOURIS_CUDA(cudaStreamDestroy(download));
    ggml_backend_free(backend);
}
Frame::Frame(size_t nodes) : nodes_(nodes) {
    ctx=ggml_init({ggml_tensor_overhead()*nodes+ggml_graph_overhead_custom(nodes,false),nullptr,true}); GGML_ASSERT(ctx);
    graph=ggml_new_graph_custom(ctx,nodes,false);
}
Frame::~Frame() { ggml_free(ctx); }
void Frame::reset() { key = {}; ggml_reset(ctx); graph=ggml_new_graph_custom(ctx,nodes_,false); operations_count_=0; }
ggml_tensor * Frame::bind(ggml_tensor * source) {
    GGML_ASSERT(source);
    auto * target=ggml_new_tensor(ctx,source->type,GGML_MAX_DIMS,source->ne);
    std::memcpy(target->nb,source->nb,sizeof(target->nb)); target->data=source->data; target->buffer=source->buffer;
    return target;
}
DeviceOp & Frame::add_op() { GGML_ASSERT(operations_count_<max_ops); return operations_[operations_count_++]; }
void Frame::expand(ggml_tensor * tensor) { ggml_set_output(tensor); ggml_build_forward_expand(graph,tensor); }
Arena::Arena() : allocator(ggml_gallocr_new(ggml_backend_cuda_buffer_type(0))) { GGML_ASSERT(allocator); }
Arena::~Arena() { ggml_gallocr_free(allocator); }
size_t Arena::measure(Frame & frame) { size_t size=0; ggml_gallocr_reserve_n_size(allocator,frame.graph,nullptr,nullptr,&size); return size; }
void Arena::reserve(Frame & frame) { GGML_ASSERT(ggml_gallocr_reserve(allocator,frame.graph)); }
void Arena::allocate(Frame & frame) {
    const size_t capacity = bytes();
    GGML_ASSERT(ggml_gallocr_alloc_graph(allocator, frame.graph));
    if (frozen_ && bytes() != capacity) { GGML_ABORT("GGML arena exceeded its locked workspace budget"); }
}
size_t Arena::bytes() const { return ggml_gallocr_get_buffer_size(allocator,0); }
ggml_tensor * storage_view(ggml_context * ctx, ggml_tensor * storage, ggml_type type,
                          int64_t n0, int64_t n1, int64_t n2, size_t offset, size_t row_stride, size_t plane_stride) {
    auto * view=ggml_new_tensor_3d(ctx,type,n0,n1,n2);
    if (row_stride) { view->nb[1]=row_stride; }
    if (plane_stride) { view->nb[2]=plane_stride; } else { view->nb[2]=view->nb[1]*n1; }
    view->nb[3]=view->nb[2]*n2; view->view_src=storage; view->view_offs=offset;
    if (storage->buffer) { ggml_backend_view_init(view); }
    return view;
}
}
