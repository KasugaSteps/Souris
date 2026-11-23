#include "ops.h"
#include <cuda_fp16.h>
#include <math_constants.h>
#include <algorithm>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>

namespace souris {
static __device__ void expert_bind(uint64_t * table,int key,int previous,uint64_t record,int role,const uint64_t * offsets) {
    if (previous>=0) {
        auto * old=table+(previous/fixed::experts*3+role)*fixed::experts+previous%fixed::experts;
        atomicCAS(reinterpret_cast<unsigned long long *>(old),record+offsets[role],0ULL);
    }
    auto * entry=table+(key/fixed::experts*3+role)*fixed::experts+key%fixed::experts;
    atomicExch(reinterpret_cast<unsigned long long *>(entry),record+offsets[role]);
}
static __global__ void expert_bind_kernel(uint64_t * table,int32_t * resident,int slot,int key,int previous,uint64_t record,uint64_t gate,uint64_t up,uint64_t down,bool provisional) {
    const int role=threadIdx.x;
    if (role>=3) { return; }
    const uint64_t offsets[] = {gate, up, down};
    expert_bind(table,key,previous,record,role,offsets);
    if (!role) {
        if (previous >= 0) { atomicCAS(resident + previous, slot, -1); }
        resident[key] = provisional ? -slot - 2 : slot;
    }
}
void cuda_expert_bind(void * table,int32_t * resident,int slot,int key,int previous,void * record,const Layout & layout,cudaStream_t stream,bool provisional) {
    expert_bind_kernel<<<1,3,0,stream>>>(static_cast<uint64_t *>(table),resident,slot,key,previous,reinterpret_cast<uint64_t>(record),layout.role_offset[0],layout.role_offset[1],layout.role_offset[2],provisional);
    SOURIS_CUDA(cudaGetLastError());
}
static __global__ void expert_finish_kernel(uint64_t * table, int32_t * resident, const int * metadata, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 3 * count) { return; }
    const int * entry = metadata + 3 * (i / 3);
    const int key = entry[0], slot = entry[1], keep = entry[2], role = i % 3;
    if (!keep) { table[(key / fixed::experts * 3 + role) * fixed::experts + key % fixed::experts] = 0; }
    if (!role) { resident[key] = keep ? slot : -1; }
}
void cuda_expert_finish(void * table, int32_t * resident, const int32_t * metadata, int count, cudaStream_t stream) {
    expert_finish_kernel<<<(3 * count + 255) / 256, 256, 0, stream>>>(
        static_cast<uint64_t *>(table), resident, metadata, count);
    SOURIS_CUDA(cudaGetLastError());
}

static __global__ void invalidate_kernel(uint64_t * table, int32_t * resident, int key, int slot) {
    if (threadIdx.x < 3) { table[(key / fixed::experts * 3 + threadIdx.x) * fixed::experts + key % fixed::experts] = 0; }
    if (!threadIdx.x) { atomicCAS(resident + key, slot, -1); }
}
void cuda_expert_invalidate(void * table, int32_t * resident, int key, int slot, cudaStream_t stream) {
    invalidate_kernel<<<1, 3, 0, stream>>>(static_cast<uint64_t *>(table), resident, key, slot);
    SOURIS_CUDA(cudaGetLastError());
}
static __global__ void decode_plan_kernel(const int * ids, const int * resident, const Control * control,
                                         int * plan, int * selected, int * rank_map, int * layer_id) {
    const int rank = threadIdx.x;
    const int expert = rank < fixed::topk ? ids[rank] : 0;
    const int key = control->layer * fixed::experts + expert;
    const int slot = rank < fixed::topk ? resident[key] : -1;
    const unsigned hits = __ballot_sync(0xffffffff, rank < fixed::topk && slot >= 0);
    const unsigned misses = __ballot_sync(0xffffffff, rank < fixed::topk && slot < 0);
    const int nh = __popc(hits);
    if (!rank) { plan[0] = nh; plan[1] = fixed::topk - nh; plan[2 + fixed::topk] = hits; *layer_id = control->layer; }
    if (rank >= fixed::topk) { return; }
    rank_map[expert] = rank;
    const unsigned before = (1u << rank) - 1;
    if (slot >= 0) { const int i = __popc(hits & before); selected[i] = expert; plan[2 + i] = slot; }
    else { plan[2 + nh + __popc(misses & before)] = key; }
}
void cuda_decode_plan(ggml_tensor * dst, void * stream, void *) {
    decode_plan_kernel<<<1, 32, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const int *>(dst->src[0]->data), static_cast<const int *>(dst->src[1]->data),
        static_cast<const Control *>(dst->src[2]->data), static_cast<int *>(dst->data),
        static_cast<int *>(dst->src[3]->data), static_cast<int *>(dst->src[4]->data), static_cast<int *>(dst->src[5]->data));
    SOURIS_CUDA(cudaGetLastError());
}
static __global__ void decode_select_kernel(const int * plan, const int * miss_index, int * selected) {
    selected[0] = plan[2 + plan[0] + *miss_index] % fixed::experts;
}
void cuda_decode_select(ggml_tensor * dst, void * stream, void *) {
    decode_select_kernel<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const int *>(dst->src[0]->data), static_cast<const int *>(dst->src[1]->data), static_cast<int *>(dst->data));
    SOURIS_CUDA(cudaGetLastError());
}
static __global__ void decode_scatter_kernel(const float * values, const int * selected, const int * rank_map,
                                            const float * weights, float * output, int width, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width * count) { return; }
    const int rank = rank_map[selected[i / width]];
    output[size_t(rank) * width + i % width] = values[i] * weights[rank];
}
void cuda_decode_scatter(ggml_tensor * dst, void * stream, void *) {
    const int width = dst->src[0]->ne[0], count = dst->src[0]->ne[1];
    decode_scatter_kernel<<<(width * count + 255) / 256, 256, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const float *>(dst->src[0]->data), static_cast<const int *>(dst->src[1]->data),
        static_cast<const int *>(dst->src[2]->data), static_cast<const float *>(dst->src[3]->data),
        static_cast<float *>(dst->data), width, count);
    SOURIS_CUDA(cudaGetLastError());
}

using SampleKey = unsigned long long;
constexpr int sample_threads = 256, sample_items = 1024;
struct SampleState {
    SampleKey prefix, cutoff;
    double remaining, mass, target;
    float maximum, total;
    int selected, candidates;
};
struct SampleScratch {
    SampleState * state;
    SampleKey * candidates; // Packed float weight + token ID, capacity n; live entries <= k.
    float * weights;
    double * cdf, * partial, * histogram;
};
size_t sampling_scratch_bytes(int n) {
    const int blocks = (n + sample_items - 1) / sample_items;
    return ((256 + size_t(n) * (sizeof(float) + sizeof(double) + sizeof(SampleKey)) + 7) & ~size_t(7)) +
        size_t(blocks) * 257 * sizeof(double);
}
static SampleScratch sample_scratch(void * data, int n) {
    auto * bytes = static_cast<char *>(data);
    SampleScratch s{};
    s.state = reinterpret_cast<SampleState *>(bytes);
    s.candidates = reinterpret_cast<SampleKey *>(bytes + 256);
    s.cdf = reinterpret_cast<double *>(s.candidates + n);
    s.weights = reinterpret_cast<float *>(s.cdf + n);
    // Align double arrays after a vocabulary whose size need not be even.
    size_t offset = (256 + size_t(n) * 20 + 7) & ~size_t(7);
    s.partial = reinterpret_cast<double *>(bytes + offset);
    s.histogram = s.partial + (n + sample_items - 1) / sample_items;
    return s;
}
static __device__ SampleKey sample_key(float weight, int id) {
    return (SampleKey(__float_as_uint(weight)) << 32) | (0xffffffffu - unsigned(id));
}
static __global__ void sample_max(const float * input, const float * params, SampleScratch s, int n, bool residual) {
    using Reduce = cub::BlockReduce<float, sample_threads>;
    __shared__ typename Reduce::TempStorage temp;
    float maximum = -CUDART_INF_F;
    for (int i = blockIdx.x * sample_items + threadIdx.x; i < min(n, (blockIdx.x + 1) * sample_items); i += sample_threads) {
        maximum = fmaxf(maximum, residual ? input[i] : input[i] / params[0]);
    }
    maximum = Reduce(temp).Reduce(maximum, cub::Max());
    if (!threadIdx.x) { s.partial[blockIdx.x] = maximum; }
}
static __global__ void sample_max_reduce(SampleScratch s, int blocks) {
    if (threadIdx.x) { return; }
    float maximum = -CUDART_INF_F;
    for (int i = 0; i < blocks; ++i) { maximum = fmaxf(maximum, float(s.partial[i])); }
    s.state->maximum = maximum;
}
static __global__ void sample_weights(const float * input, const float * params, SampleScratch s, int n, bool residual) {
    using Reduce = cub::BlockReduce<double, sample_threads>;
    __shared__ typename Reduce::TempStorage temp;
    double sum = 0;
    for (int i = blockIdx.x * sample_items + threadIdx.x; i < min(n, (blockIdx.x + 1) * sample_items); i += sample_threads) {
        const float w = residual ? input[i] : expf(input[i] / params[0] - s.state->maximum);
        s.weights[i] = w; sum += w;
    }
    sum = Reduce(temp).Sum(sum);
    if (!threadIdx.x) { s.partial[blockIdx.x] = sum; }
}
static __global__ void sample_init(const float * params, SampleScratch s, int blocks, bool residual) {
    if (threadIdx.x) { return; }
    double total = 0;
    for (int i = 0; i < blocks; ++i) { total += s.partial[i]; }
    s.state->total = total;
    // Weights share the softmax denominator: ratio/mass cutoffs can stay unnormalized.
    const float min_weight = params[2] > 0 ? 0.f : params[3] * (residual ? s.state->maximum : 1.f);
    s.state->cutoff = SampleKey(__float_as_uint(min_weight)) << 32; s.state->prefix = 0;
    s.state->remaining = params[2];
    s.state->candidates = 0;
}

constexpr int split_items = 4096, split_limit = 64;
/** Each warp keeps its exact top-k in registers, then the block merges four lists. */
static __global__ void sample_split_part(const float *params, SampleScratch s, int n) {
    const int k = min(int(params[2]), n);
    if (params[2] <= 0 || params[2] > split_limit) { return; }
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    SampleKey values[32];
#pragma unroll
    for (int j = 0; j < 32; ++j) {
        const int i = blockIdx.x * split_items + warp * 1024 + lane + j * 32;
        values[j] = i < n && s.weights[i] > 0 ? sample_key(s.weights[i], i) : 0;
    }
    __shared__ SampleKey lists[4][split_limit];
    for (int rank = 0; rank < k; ++rank) {
        SampleKey best = 0;
#pragma unroll
        for (int j = 0; j < 32; ++j) { best = max(best, values[j]); }
        for (int d = 16; d; d >>= 1) { best = max(best, __shfl_down_sync(0xffffffff, best, d)); }
        best = __shfl_sync(0xffffffff, best, 0);
        if (!lane) { lists[warp][rank] = best; }
#pragma unroll
        for (int j = 0; j < 32; ++j) { if (values[j] == best) { values[j] = 0; } }
    }
    __syncthreads();
    if (!threadIdx.x) {
        int cursor[4] = {};
        auto *parts = reinterpret_cast<SampleKey *>(s.histogram) + blockIdx.x * split_limit;
        for (int rank = 0; rank < k; ++rank) {
            SampleKey best = 0; int chosen = 0;
            for (int w = 0; w < 4; ++w) {
                const SampleKey key = cursor[w] < k ? lists[w][cursor[w]] : 0;
                if (key > best) { best = key; chosen = w; }
            }
            parts[rank] = best; ++cursor[chosen];
        }
    }
}
/** Merge sorted block lists, truncate, then sample in token-ID order for seeded parity. */
static __global__ void sample_split_tail(const float *params, SampleScratch s, int n, int *token, float *q) {
    const int k = min(int(params[2]), n);
    if (params[2] <= 0 || params[2] > split_limit) { return; }
    extern __shared__ SampleKey memory[];
    const int blocks = (n + split_items - 1) / split_items;
    auto *heads = reinterpret_cast<int *>(memory + blocks * split_limit);
    for (int i = threadIdx.x; i < blocks * k; i += blockDim.x) {
        memory[i] = reinterpret_cast<SampleKey *>(s.histogram)[i / k * split_limit + i % k];
    }
    for (int b = threadIdx.x; b < blocks; b += blockDim.x) { heads[b] = 0; }
    __syncthreads();
    if (threadIdx.x >= 32) { return; }
    const int lane = threadIdx.x;
    for (int rank = 0; rank < k; ++rank) {
        SampleKey best = 0; int chosen = -1;
        for (int b = lane; b < blocks; b += 32) {
            const SampleKey key = heads[b] < k ? memory[b * k + heads[b]] : 0;
            if (key > best) { best = key; chosen = b; }
        }
        const SampleKey local = best;
        for (int d = 16; d; d >>= 1) { best = max(best, __shfl_down_sync(0xffffffff, best, d)); }
        best = __shfl_sync(0xffffffff, best, 0);
        if (!lane) { s.candidates[rank] = best; }
        if (chosen >= 0 && local == best) { ++heads[chosen]; }
        __syncwarp();
    }
    if (lane) { return; }
    const float minimum = params[3];
    int count = 0; double mass = 0;
    for (int i = 0; i < k; ++i) {
        const SampleKey key = s.candidates[i];
        const float w = __uint_as_float(unsigned(key >> 32));
        if (w > 0 && w >= minimum) { s.candidates[count++] = key; mass += w; }
    }
    if (params[1] < 1) {
        const double threshold = double(params[1]) * mass;
        double prefix = 0; int keep = 0;
        while (keep < count) {
            prefix += __uint_as_float(unsigned(s.candidates[keep++] >> 32));
            if (prefix >= threshold) { break; }
        }
        count = keep; mass = prefix;
    }
    // Only k records are sorted; preserve the original token-index CDF and tie rule.
    for (int i = 1; i < count; ++i) {
        const SampleKey key = s.candidates[i]; int j = i;
        while (j && unsigned(s.candidates[j - 1]) < unsigned(key)) {
            s.candidates[j] = s.candidates[j - 1]; --j;
        }
        s.candidates[j] = key;
    }
    const double target = fminf(fmaxf(params[4], 0), 0x1.fffffep-1f) * mass;
    double prefix = 0; int selected = -1;
    for (int i = 0; i < count; ++i) {
        const SampleKey key = s.candidates[i];
        const int id = int(0xffffffffu - unsigned(key));
        const float w = __uint_as_float(unsigned(key >> 32));
        prefix += w; s.cdf[i] = prefix;
        if (selected < 0 && prefix > target) { selected = id; }
        if (q) { q[id] = float(w / mass); }
    }
    if (selected < 0 && count) { selected = int(0xffffffffu - unsigned(s.candidates[count - 1])); }
    token[0] = selected; s.state->candidates = count; s.state->mass = mass;
}
/** Large k: compact in stable token order without sorting the vocabulary. */
static __global__ void sample_collect(const float *params, SampleScratch s, int n) {
    if (params[2] <= split_limit) { return; }
    using Scan = cub::BlockScan<int, sample_threads>;
    __shared__ typename Scan::TempStorage temp;
    const int first = blockIdx.x * sample_items + threadIdx.x * 4;
    SampleKey keys[4]; int count = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int i = first + j; const float w = i < n ? s.weights[i] : 0;
        const SampleKey key = sample_key(w, i);
        keys[j] = i < n && w > 0 && key >= s.state->cutoff ? key : 0;
        count += keys[j] != 0;
    }
    int prefix, total;
    Scan(temp).ExclusiveSum(count, prefix, total);
    // Reuse the CDF arena until stable compaction has finished.
    auto *temporary = reinterpret_cast<SampleKey *>(s.cdf);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        if (keys[j]) { temporary[blockIdx.x * sample_items + prefix++] = keys[j]; }
    }
    if (!threadIdx.x) { s.partial[blockIdx.x] = total; }
}
static __global__ void sample_collect_prefix(const float *params, SampleScratch s, int blocks) {
    if (params[2] <= split_limit || threadIdx.x) { return; }
    int count = 0;
    for (int b = 0; b < blocks; ++b) { s.histogram[b] = count; count += int(s.partial[b]); }
    s.state->candidates = count;
}
static __global__ void sample_compact(const float *params, SampleScratch s) {
    if (params[2] <= split_limit) { return; }
    const int count = int(s.partial[blockIdx.x]), first = int(s.histogram[blockIdx.x]);
    const auto *temporary = reinterpret_cast<const SampleKey *>(s.cdf);
    for (int i = threadIdx.x; i < count; i += blockDim.x) {
        s.candidates[first + i] = temporary[blockIdx.x * sample_items + i];
    }
}
static __device__ bool sample_split(const float *params) {
    return params[2] > 0 && params[2] <= split_limit;
}
/** Min-p and mass visit only compact candidates when k is finite. */
static __global__ void sample_candidate_mass(const float *params, SampleScratch s, int n) {
    if (sample_split(params)) { return; }
    const bool compact = params[2] > 0;
    const int length = compact ? s.state->candidates : n;
    if (blockIdx.x * sample_items >= length) { return; }
    using Reduce = cub::BlockReduce<double, sample_threads>;
    __shared__ typename Reduce::TempStorage temp;
    double sum = 0;
    for (int i = blockIdx.x * sample_items + threadIdx.x; i < min(length, (blockIdx.x + 1) * sample_items); i += blockDim.x) {
        const SampleKey key = compact ? s.candidates[i] : sample_key(s.weights[i], i);
        const float w = __uint_as_float(unsigned(key >> 32));
        if (w >= params[3] && key >= s.state->cutoff) { sum += w; }
        else if (compact) { s.candidates[i] = 0; }
    }
    sum = Reduce(temp).Sum(sum);
    if (!threadIdx.x) { s.partial[blockIdx.x] = sum; }
}
/** Each block builds one radix histogram. Grid reductions have a fixed order. */
static __global__ void sample_histogram(const float * params, SampleScratch s, int n, int shift, bool mass) {
    if (sample_split(params) || (mass ? params[1] >= 1 : params[2] <= 0 || params[2] >= n)) { return; }
    const bool compact = mass && params[2] > 0;
    const int length = compact ? s.state->candidates : n;
    if (blockIdx.x * sample_items >= length) { return; }
    __shared__ double hist[8][256];
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    for (int i = tid; i < 2048; i += sample_threads) { (&hist[0][0])[i] = 0; }
    __syncthreads();
    const SampleKey prefix_mask = shift == 56 ? 0 : (~0ull << (shift + 8));
    for (int base = blockIdx.x * sample_items; base < min(length, (blockIdx.x + 1) * sample_items); base += sample_threads) {
        const int i = base + tid;
        const SampleKey key = i < length ? (compact ? s.candidates[i] : sample_key(s.weights[i], i)) : 0;
        const float w = __uint_as_float(unsigned(key >> 32));
        const bool valid = i < length && w > 0 && (key & prefix_mask) == s.state->prefix &&
            (!mass || key >= s.state->cutoff);
        const unsigned active = __ballot_sync(0xffffffff, valid);
        if (valid) {
            const int digit = (key >> shift) & 255;
            const unsigned peers = __match_any_sync(active, digit);
            if (!mass) {
                if (lane == __ffs(peers) - 1) { atomicAdd(&hist[warp][digit], double(__popc(peers))); }
            } else {
                // Four values per lane; warp-local histograms bound atomic contention.
                atomicAdd(&hist[warp][digit], double(w));
            }
        }
    }
    __syncthreads();
    double sum = 0;
    for (int w = 0; w < 8; ++w) { sum += hist[w][tid]; }
    s.histogram[blockIdx.x * 256 + tid] = sum;
}
static __global__ void sample_radix_reduce(const float * params, SampleScratch s, int n, int shift, bool mass, int blocks) {
    if (sample_split(params) || (mass ? params[1] >= 1 : params[2] <= 0 || params[2] >= n)) { return; }
    __shared__ double hist[256];
    double sum = 0;
    const int active_blocks = mass && params[2] > 0 ? (s.state->candidates + sample_items - 1) / sample_items : blocks;
    for (int b = 0; b < active_blocks; ++b) { sum += s.histogram[b * 256 + threadIdx.x]; }
    hist[threadIdx.x] = sum;
    __syncthreads();
    if (threadIdx.x) { return; }
    int chosen = 0;
    for (int d = 255; d >= 0; --d) {
        if (hist[d] <= 0) { continue; }
        chosen = d;
        if (s.state->remaining <= hist[d]) { break; }
        s.state->remaining -= hist[d];
    }
    s.state->prefix |= SampleKey(chosen) << shift;
    if (!shift) { s.state->cutoff = max(s.state->cutoff, s.state->prefix); }
}
static __global__ void sample_support(const float * params, SampleScratch s, int n) {
    if (sample_split(params)) { return; }
    const bool compact = params[2] > 0;
    const int length = compact ? s.state->candidates : n;
    if (blockIdx.x * sample_items >= length) { return; }
    using Scan = cub::BlockScan<double, sample_threads>;
    __shared__ typename Scan::TempStorage temp;
    const int first = blockIdx.x * sample_items + threadIdx.x * 4;
    double values[4], sum = 0;
    for (int j = 0; j < 4; ++j) {
        const int i = first + j;
        const SampleKey key = i < length ? (compact ? s.candidates[i] : sample_key(s.weights[i], i)) : 0;
        const float w = __uint_as_float(unsigned(key >> 32));
        values[j] = i < length && w > 0 && key >= s.state->cutoff ? w : 0;
        sum += values[j];
    }
    double prefix, aggregate;
    Scan(temp).ExclusiveSum(sum, prefix, aggregate);
    for (int j = 0; j < 4; ++j) {
        prefix += values[j];
        if (first + j < length) { s.cdf[first + j] = prefix; }
    }
    if (!threadIdx.x) { s.partial[blockIdx.x] = aggregate; }
}
static __global__ void sample_mass_reduce(const float * params, SampleScratch s, int blocks, bool initial, int * token) {
    if (sample_split(params) || threadIdx.x) { return; }
    if (params[2] > 0) { blocks = (s.state->candidates + sample_items - 1) / sample_items; }
    double sum = 0;
    for (int b = 0; b < blocks; ++b) { sum += s.partial[b]; }
    if (initial) {
        s.state->remaining = double(params[1]) * sum;
        s.state->prefix = 0;
        return;
    }
    s.state->mass = sum;
    double target = fminf(fmaxf(params[4], 0), 0x1.fffffep-1f) * sum;
    s.state->selected = -1; token[0] = -1;
    for (int b = 0; b < blocks; ++b) {
        const double mass = s.partial[b];
        if (mass <= 0) { continue; }
        s.state->selected = b; s.state->target = target;
        if (target < mass) { break; }
        target -= mass;
    }
}
static __global__ void sample_finish(const float *params, SampleScratch s, int *token, float *q, int n) {
    if (sample_split(params)) { return; }
    const bool compact = params[2] > 0;
    const int length = compact ? s.state->candidates : n;
    const int begin = blockIdx.x * sample_items, end = min(length, begin + sample_items);
    if (begin >= length) { return; }
    if (q) {
        for (int i = begin + threadIdx.x; i < end; i += sample_threads) {
            const SampleKey key = compact ? s.candidates[i] : sample_key(s.weights[i], i);
            const int id = compact ? int(0xffffffffu - unsigned(key)) : i;
            if (key && key >= s.state->cutoff && s.state->mass > 0) {
                q[id] = float(__uint_as_float(unsigned(key >> 32)) / s.state->mass);
            }
        }
    }
    if (!threadIdx.x && blockIdx.x == s.state->selected) {
        int lo = begin, hi = end - 1;
        while (lo < hi) {
            const int mid = (lo + hi) / 2;
            if (s.cdf[mid] <= s.state->target) { lo = mid + 1; } else { hi = mid; }
        }
        token[0] = compact ? int(0xffffffffu - unsigned(s.candidates[lo])) : lo;
    }
}
void cuda_sample(ggml_tensor * dst, void * stream, void * userdata) {
    const bool distribution = dst->type == GGML_TYPE_F32, residual = userdata != nullptr;
    const int n = dst->src[0]->ne[0], blocks = (n + sample_items - 1) / sample_items;
    auto * scratch = dst->src[distribution ? 3 : 2];
    GGML_ASSERT(ggml_nbytes(scratch) >= sampling_scratch_bytes(n));
    const auto s = sample_scratch(scratch->data, n);
    const auto cuda = static_cast<cudaStream_t>(stream);
    const auto * input = static_cast<const float *>(dst->src[0]->data);
    const auto * params = static_cast<const float *>(dst->src[1]->data);
    auto * token = static_cast<int *>(distribution ? dst->src[2]->data : dst->data);
    // Dense q is an output representation: zero-fill, then scatter only support entries.
    if (distribution) { SOURIS_CUDA(cudaMemsetAsync(dst->data, 0, size_t(n) * sizeof(float), cuda)); }
    sample_max<<<blocks,256,0,cuda>>>(input,params,s,n,residual);
    sample_max_reduce<<<1,1,0,cuda>>>(s,blocks);
    sample_weights<<<blocks,256,0,cuda>>>(input,params,s,n,residual);
    sample_init<<<1,1,0,cuda>>>(params,s,blocks,residual);
    const int parts = (n + split_items - 1) / split_items;
    sample_split_part<<<parts,128,0,cuda>>>(params,s,n);
    const size_t shared = size_t(parts) * (split_limit * sizeof(SampleKey) + sizeof(int));
    sample_split_tail<<<1,128,shared,cuda>>>(params,s,n,token,distribution?static_cast<float *>(dst->data):nullptr);
    for (int shift=56; shift>=0; shift-=8) {
        sample_histogram<<<blocks,256,0,cuda>>>(params,s,n,shift,false);
        sample_radix_reduce<<<1,256,0,cuda>>>(params,s,n,shift,false,blocks);
    }
    sample_collect<<<blocks,256,0,cuda>>>(params,s,n);
    sample_collect_prefix<<<1,1,0,cuda>>>(params,s,blocks);
    sample_compact<<<blocks,256,0,cuda>>>(params,s);
    sample_candidate_mass<<<blocks,256,0,cuda>>>(params,s,n);
    sample_mass_reduce<<<1,1,0,cuda>>>(params,s,blocks,true,token);
    for (int shift=56; shift>=0; shift-=8) {
        sample_histogram<<<blocks,256,0,cuda>>>(params,s,n,shift,true);
        sample_radix_reduce<<<1,256,0,cuda>>>(params,s,n,shift,true,blocks);
    }
    sample_support<<<blocks,256,0,cuda>>>(params,s,n);
    sample_mass_reduce<<<1,1,0,cuda>>>(params,s,blocks,false,token);
    sample_finish<<<blocks,256,0,cuda>>>(params,s,token,distribution?static_cast<float *>(dst->data):nullptr,n);
    SOURIS_CUDA(cudaGetLastError());
}
static __global__ void token_mask_kernel(float * logits, const uint32_t * bits, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && !(bits[i / 32] & (1u << (i % 32)))) { logits[i] = -CUDART_INF_F; }
}
static __global__ void sparse_mask_kernel(float * logits, const int * allowed, int size, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) { return; }
    bool valid = false;
    if (i >= allowed[0] && i <= allowed[size - 1]) {
        int lo = 0, hi = size;
        while (lo < hi) { const int mid = (lo + hi) / 2; if (allowed[mid] < i) { lo = mid + 1; } else { hi = mid; } }
        valid = lo < size && allowed[lo] == i;
    }
    if (!valid) { logits[i] = -CUDART_INF_F; }
}
void cuda_token_mask(float * logits, const uint32_t * bits, const Token * sparse, int allowed, int count, cudaStream_t stream) {
    if (sparse) { sparse_mask_kernel<<<(count + 255) / 256, 256, 0, stream>>>(logits, sparse, allowed, count); }
    else { token_mask_kernel<<<(count + 255) / 256, 256, 0, stream>>>(logits, bits, count); }
    SOURIS_CUDA(cudaGetLastError());
}

constexpr int threads=256;
static __global__ void auxiliary_kernel(char * output, const Control * control, DeviceOp op) {
    const int index=blockIdx.x*blockDim.x+threadIdx.x;
    const int pools=(op.capacity+op.pool-1)/op.pool;
    auto * positions=reinterpret_cast<int32_t *>(output+op.positions);
    auto * writes=reinterpret_cast<int64_t *>(output+op.write_ids);
    auto * reads=reinterpret_cast<int32_t *>(output+op.read_ids);
    auto * mask=reinterpret_cast<half *>(output+op.pool_mask);
    auto * tails=reinterpret_cast<int32_t *>(output+op.tail_ids);
    auto * pool_positions=reinterpret_cast<int32_t *>(output+op.pool_positions);
    if (index<4*op.tokens) { positions[index]=control->position+index%op.tokens; }
    if (index<op.tokens) { writes[index]=control->position+index; }
    // Gather a written row for masked future positions, keeping attention finite.
    if (index<op.capacity) { reads[index]=index<control->position+op.tokens ? index : 0; }
    if (index<pools*op.tokens) {
        const int position=control->position+index/pools;
        mask[index]=__float2half((index%pools+1)*op.pool<=position+1 ? 0.0f : -CUDART_INF_F);
    }
    if (index<(op.pool-1)*op.tokens) {
        const int token=index/(op.pool-1), member=index%(op.pool-1), position=control->position+token;
        const int tail=(position+1)%op.pool;
        tails[index]=member<tail ? (position+1)/op.pool*op.pool+member : op.capacity;
    }
    if (index<4*pools) { pool_positions[index]=index%pools*op.pool; }
}
void cuda_auxiliary(ggml_tensor * dst, void * stream, void * userdata) {
    const auto & op=*static_cast<const DeviceOp *>(userdata); const int pools=(op.capacity+op.pool-1)/op.pool;
    const int count=std::max({4*op.tokens,op.capacity,pools*op.tokens,(op.pool-1)*op.tokens,4*pools});
    auxiliary_kernel<<<(count+threads-1)/threads,threads,0,static_cast<cudaStream_t>(stream)>>>(static_cast<char *>(dst->data),static_cast<const Control *>(dst->src[0]->data),op);
    SOURIS_CUDA(cudaGetLastError());
}
}
