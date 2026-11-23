#include "stats.h"
#include "expert_cache.h"
#include <algorithm>
#include <fstream>
#include <sstream>

namespace souris {
double milliseconds(StatsClock::time_point start) {
    return std::chrono::duration<double, std::milli>(StatsClock::now() - start).count();
}
void Statistics::begin_request() {
    flush_gpu();
    Statistics fresh;
    fresh.load_ms = load_ms;
    fresh.warmup_ms = warmup_ms;
    fresh.warmup_prefill_ms = warmup_prefill_ms;
    fresh.warmup_decode_ms = warmup_decode_ms;
    fresh.warmup_free_vram_mib = warmup_free_vram_mib;
    fresh.warmup_peak_vram_mib = warmup_peak_vram_mib;
    fresh.warmup_prefill_workspace_mib = warmup_prefill_workspace_mib;
    *this = std::move(fresh);
}
void Statistics::resources() {
    flush_gpu();
    size_t free, total;
    SOURIS_CUDA(cudaMemGetInfo(&free, &total));
    vram_peak = std::max(vram_peak, total - free);
    std::ifstream status("/proc/self/status");
    std::string line, key;
    size_t value;
    while (std::getline(status, line)) {
        std::istringstream row(line);
        if (row >> key >> value && key == "VmHWM:") { ram_peak = std::max(ram_peak, value * 1024); }
    }
    swap_peak = std::max(swap_peak, process_swap());
}
void Statistics::flush_gpu() {
    for (const auto & sample : gpu_samples) {
        SOURIS_CUDA(cudaEventSynchronize(sample.end));
        float elapsed;
        SOURIS_CUDA(cudaEventElapsedTime(&elapsed, sample.begin, sample.end));
        *sample.target += elapsed;
        if (sample.second) { *sample.second += elapsed; }
        SOURIS_CUDA(cudaEventDestroy(sample.begin));
        SOURIS_CUDA(cudaEventDestroy(sample.end));
    }
    gpu_samples.clear();
}
GpuTimer::GpuTimer(Statistics & stats, cudaStream_t stream, double & target, double * second) :
    stats_(stats), stream_(stream), target_(target), second_(second) {
    if (&target != &stats.prefill_moe_ms) { return; }
    SOURIS_CUDA(cudaEventCreate(&begin_));
    SOURIS_CUDA(cudaEventCreate(&end_));
    SOURIS_CUDA(cudaEventRecord(begin_, stream_));
}
GpuTimer::~GpuTimer() {
    if (!begin_) { return; }
    SOURIS_CUDA(cudaEventRecord(end_, stream_));
    stats_.gpu_samples.push_back({begin_, end_, &target_, second_});
}
}
