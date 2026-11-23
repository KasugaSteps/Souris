#pragma once
#include "constants.h"
#include "ggml.h"
#include <string>
#include <vector>

namespace souris {
/** Quantization IDs retain the GGML wire values used by the model directory. */
using QuantType = ggml_type;
struct QuantTraits {
    QuantType type;
    int block_values;
    size_t block_bytes;
    /** GGML row geometry, including block alignment validation. */
    size_t row_bytes(int64_t columns) const;
};
QuantTraits quant_traits(QuantType type);
/** Stable ordered signature for gate, up and down graph dispatch. */
uint64_t quant_signature(const QuantType types[3]);


/** Model-owned on-disk quantization descriptor. */
struct QuantDesc { uint32_t type; };

using Token = int32_t;
using Tokens = std::vector<Token>;

/** Immutable text/vision artifact selection, loaded together by Model. */
struct ModelConfig {
    std::string srs1_path;
    std::string vision_mmproj;
    int max_images = 4;
    int image_cache_entries = 8;
};

/** Startup budgets and single-request execution options. */
struct Config {
    ModelConfig model;
    int context = 4608;
    /** Zero selects context after parsing; the resolved limit is locked at startup. */
    int max_prompt_tokens = 0;
    int port = 8000;
    std::string log_file;
    std::string log_level = "info";
    /** Byte budget rounded down to complete expert records at startup. */
    size_t staging_bytes = fixed::staging_bytes;
    int read_threads = fixed::default_read_threads;
    int mmq_group = 16;
    int slru_protected = fixed::protected_percent;
    std::string mtp;
    int mtp_max_drafts = fixed::mtp_max_drafts;
    /** SIZE_MAX selects the free-VRAM budget with a fixed 1 GiB reserve. */
    size_t gpu_expert_bytes = SIZE_MAX;
    /** SIZE_MAX selects the automatic pageable budget. Zero is explicit no-cache. */
    size_t expert_cache_bytes = SIZE_MAX;
    bool serve = false;
};
}
