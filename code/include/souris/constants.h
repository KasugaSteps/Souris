#pragma once
#include <cstddef>
#include <cstdint>

namespace souris::fixed {
constexpr int layers = 48;
constexpr int experts = 512;
constexpr int topk = 10;
constexpr int mtp_max_drafts = 4;
constexpr int mtp_window = 8192;
constexpr int verify_rows = mtp_max_drafts + 1;
constexpr int verify_experts = layers * verify_rows * topk;
constexpr int moe_batch = 32;
constexpr int staging_min_records = 64;
constexpr int default_chunk = 1024;
constexpr int default_read_threads = 8;
constexpr int protected_percent = 60;
constexpr int ram_cache_percent = 60;
constexpr int transfer_poll_us = 50;
constexpr int slot_chunk = 32;
constexpr int extent_records = 4;
constexpr int admission_count = 4;
constexpr int warmup_tokens = 128;
constexpr size_t mib = size_t(1024) * 1024;
constexpr size_t headroom = 1024 * mib;
constexpr size_t host_reserve = 8192 * mib;
constexpr size_t staging_bytes = 1024 * mib;
constexpr size_t prefill_control_slot = 64 * 1024;
constexpr size_t decode_ring_bytes = 8 * mib;
constexpr size_t route_slot_bytes = 64;
constexpr size_t route_plan_bytes = (topk + 2) * sizeof(int32_t);
constexpr size_t staging_alignment = 4096;
constexpr size_t expert_alignment = 256;
constexpr int control_slots = 2;
constexpr size_t srs_header_size = 4096;
constexpr size_t srs_shard_table = 2048;
constexpr int srs_max_shards = 8;
constexpr int minimum_context = warmup_tokens + 1;
constexpr int kv_token_alignment = 256;
constexpr size_t kernel_scratch = 128 * mib;
constexpr size_t compact_hidden_vram = size_t(8) * 1024 * mib;
}
