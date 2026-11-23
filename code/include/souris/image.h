#pragma once
#include <cstdint>
#include <string>
#include <vector>

struct ggml_tensor;
namespace souris {
namespace image {
constexpr int size = 768, tokens = 576, width = 2560, max_count = 4;
constexpr size_t rgb_bytes = size_t(size) * size * 3;
constexpr const char * preprocess_version = "souris-letterbox-rgb8-v1";
}
/** RGB8 request data. Hash and token offset are computed by the backend. */
struct Image {
    int width = image::size, height = image::size;
    std::vector<uint8_t> rgb;
    std::string hash;
    int token_start = -1;
};
/** Prefix identity excludes RGB storage and GPU embedding lifetimes. */
struct ImageIdentity {
    std::string hash;
    int start = -1, count = image::tokens;
    bool operator==(const ImageIdentity & other) const {
        return hash == other.hash && start == other.start && count == other.count;
    }
};
/** Device rows owned by VisionEncoder until the next request. */
struct ImageEmbedding { ggml_tensor * rows = nullptr; };
std::string image_hash(const uint8_t * rgb, size_t bytes);
/** Strict base64 decoder; raw images have one fixed byte size. */
bool decode_rgb8(const std::string & encoded, std::vector<uint8_t> & rgb);
}
