#include "prefill_ops.h"

namespace souris {
ggml_tensor * prefill_hc_down(Frame & f, ggml_tensor * input, int hc) {
    ggml_tensor * args[] = {input};
    return ggml_backend_cuda_custom_4d(f.ctx, GGML_TYPE_F32, input->ne[0], input->ne[1], 1, 1,
        args, 1, cuda_prefill_hc_down, reinterpret_cast<void *>(uintptr_t(hc)));
}
ggml_tensor * prefill_hc_up(Frame & f, ggml_tensor * gate, ggml_tensor * input, int hc) {
    ggml_tensor * args[] = {gate, input};
    return ggml_backend_cuda_custom_4d(f.ctx, GGML_TYPE_F32, input->ne[0] / hc, input->ne[1], 1, 1,
        args, 2, cuda_prefill_hc_up, reinterpret_cast<void *>(uintptr_t(hc)));
}
ggml_tensor * prefill_combine(Frame & f, ggml_tensor * residual, ggml_tensor * output,
                               ggml_tensor * inject, ggml_tensor * target) {
    ggml_tensor * args[] = {residual, output, inject};
    return ggml_backend_cuda_custom_inplace(f.ctx, f.bind(target), args, 3, cuda_prefill_combine, nullptr);
}
ggml_tensor * prefill_shared_combine(Frame & f, ggml_tensor * residual, ggml_tensor * routed,
                                      ggml_tensor * shared, ggml_tensor * gate, ggml_tensor * inject,
                                      ggml_tensor * target) {
    ggml_tensor * args[] = {residual, routed, shared, gate, inject};
    return ggml_backend_cuda_custom_inplace(f.ctx, f.bind(target), args, 5, cuda_prefill_shared_combine, nullptr);
}
}
