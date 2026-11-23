#include "souris/image.h"
#include "ggml.h"
#include <openssl/evp.h>
#include <cstring>

namespace souris {
std::string image_hash(const uint8_t * rgb, size_t bytes) {
    auto * context = EVP_MD_CTX_new();
    GGML_ASSERT(context);
    unsigned char digest[EVP_MAX_MD_SIZE]; unsigned int length = 0;
    GGML_ASSERT(EVP_DigestInit_ex(context, EVP_sha256(), nullptr) == 1);
    GGML_ASSERT(EVP_DigestUpdate(context, rgb, bytes) == 1);
    GGML_ASSERT(EVP_DigestUpdate(context, image::preprocess_version, std::strlen(image::preprocess_version)) == 1);
    GGML_ASSERT(EVP_DigestFinal_ex(context, digest, &length) == 1);
    EVP_MD_CTX_free(context);
    const char hex[] = "0123456789abcdef";
    std::string hash(length * 2, '0');
    for (unsigned int i = 0; i < length; ++i) { hash[2*i] = hex[digest[i] >> 4]; hash[2*i+1] = hex[digest[i] & 15]; }
    return hash;
}
bool decode_rgb8(const std::string & encoded, std::vector<uint8_t> & rgb) {
    // 768*768*3 is divisible by three, so its canonical base64 has no padding.
    if (encoded.size() != image::rgb_bytes / 3 * 4) { return false; }
    for (unsigned char ch : encoded) {
        if (!((ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z') ||
              (ch >= '0' && ch <= '9') || ch == '+' || ch == '/')) { return false; }
    }
    rgb.resize(image::rgb_bytes);
    return EVP_DecodeBlock(rgb.data(), reinterpret_cast<const unsigned char *>(encoded.data()), int(encoded.size())) == int(rgb.size());
}
}
