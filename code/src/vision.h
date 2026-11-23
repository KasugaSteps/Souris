#pragma once
#include "souris/image.h"
#include "ggml-backend.h"
#include <memory>

namespace souris {
/** Fixed Qwen3VL image encoder. All weights, graph storage and cached rows are GPU resident. */
class VisionEncoder {
public:
    static std::unique_ptr<VisionEncoder> load(const std::string & path, ggml_backend_t, int cache_entries);
    ~VisionEncoder();
    /** Index selects one of four request outputs; rows survive until the next request. */
    ImageEmbedding encode_rgb8(const uint8_t *, int width, int height, int index = 0);
    int image_tokens() const { return image::tokens; }
    int output_width() const { return image::width; }
    size_t resident_bytes() const;
private:
    VisionEncoder(const std::string &, ggml_backend_t, int);
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
