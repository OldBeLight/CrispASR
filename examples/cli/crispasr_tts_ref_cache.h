// crispasr_tts_ref_cache.h — on-disk cache for expensive TTS reference
// conditioning (voice cloning). Encoding a reference — a codec encoder, a
// Conformer/Perceiver, or an ASR pass over the clip — is slow and produces a
// small, reusable blob. Cache it next to the voice file so repeated runs skip
// the encode entirely (#221).
//
// Format (little-endian):
//   magic "CRC1" | u32 version | u32 tag_len | tag bytes |
//   u32 nshape | nshape * u32 shape | u32 payload_bytes | payload bytes
//
// The tag identifies the backend + representation (e.g. "irodori-latent",
// "indextts-cond", "f5-reftext"); a mismatching tag rejects the cache so two
// backends never read each other's blob. The cache is invalidated when the
// voice file is newer than it. Payload is opaque bytes — callers interpret it
// as float32 latents, UTF-8 text, etc.
//
// Header-only; no backend dependencies.

#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace crispasr_ref_cache {

inline const char kMagic[4] = {'C', 'R', 'C', '1'};
inline constexpr uint32_t kVersion = 1;

// Globally disable via CRISPASR_TTS_REF_CACHE=0.
inline bool disabled() {
    const char* e = std::getenv("CRISPASR_TTS_REF_CACHE");
    return e && std::strcmp(e, "0") == 0;
}

// "<voice><suffix>", e.g. suffix ".iro32latent".
inline std::string path_for(const std::string& voice, const char* suffix) {
    return voice + suffix;
}

inline bool mtime(const std::string& path, time_t& out) {
    struct stat st;
    if (stat(path.c_str(), &st) != 0)
        return false;
    out = st.st_mtime;
    return true;
}

// Load a cache if present, well-formed, tagged `tag`, and at least as new as
// `voice_path`. Fills `shape` + `payload` and returns true on success.
inline bool load(const std::string& cache_path, const std::string& voice_path, const char* tag,
                 std::vector<uint32_t>& shape, std::vector<uint8_t>& payload) {
    time_t ct = 0, vt = 0;
    if (!mtime(cache_path, ct))
        return false;
    if (mtime(voice_path, vt) && ct < vt)
        return false; // stale
    FILE* f = std::fopen(cache_path.c_str(), "rb");
    if (!f)
        return false;
    auto rd = [&](void* p, size_t n) { return std::fread(p, 1, n, f) == n; };
    auto rdu32 = [&](uint32_t& v) { return rd(&v, sizeof(uint32_t)); };
    char magic[4];
    uint32_t ver = 0, tag_len = 0, nshape = 0, pbytes = 0;
    bool ok = rd(magic, 4) && std::memcmp(magic, kMagic, 4) == 0 && rdu32(ver) && ver == kVersion && rdu32(tag_len) &&
              tag_len < 256;
    if (ok) {
        std::string got(tag_len, '\0');
        ok = (tag_len == 0 || rd(&got[0], tag_len)) && got == tag && rdu32(nshape) && nshape < 16;
    }
    if (ok) {
        shape.resize(nshape);
        for (uint32_t i = 0; ok && i < nshape; i++)
            ok = rdu32(shape[i]);
        ok = ok && rdu32(pbytes) && pbytes <= (256u << 20); // 256 MB sanity cap
    }
    if (ok) {
        payload.resize(pbytes);
        ok = (pbytes == 0 || rd(payload.data(), pbytes));
    }
    std::fclose(f);
    return ok;
}

inline void save(const std::string& cache_path, const char* tag, const std::vector<uint32_t>& shape,
                 const void* payload, size_t nbytes) {
    FILE* f = std::fopen(cache_path.c_str(), "wb");
    if (!f)
        return;
    auto wr = [&](const void* p, size_t n) { std::fwrite(p, 1, n, f); };
    auto wru32 = [&](uint32_t v) { wr(&v, sizeof(uint32_t)); };
    uint32_t tag_len = (uint32_t)std::strlen(tag);
    wr(kMagic, 4);
    wru32(kVersion);
    wru32(tag_len);
    wr(tag, tag_len);
    wru32((uint32_t)shape.size());
    for (uint32_t d : shape)
        wru32(d);
    wru32((uint32_t)nbytes);
    wr(payload, nbytes);
    std::fclose(f);
}

// ── float-blob convenience (latents / conditioning) ──
inline bool load_floats(const std::string& cache_path, const std::string& voice_path, const char* tag,
                        std::vector<uint32_t>& shape, std::vector<float>& out) {
    std::vector<uint8_t> payload;
    if (!load(cache_path, voice_path, tag, shape, payload) || (payload.size() % sizeof(float)) != 0)
        return false;
    out.resize(payload.size() / sizeof(float));
    std::memcpy(out.data(), payload.data(), payload.size());
    return true;
}

inline void save_floats(const std::string& cache_path, const char* tag, const std::vector<uint32_t>& shape,
                        const float* data, size_t count) {
    save(cache_path, tag, shape, data, count * sizeof(float));
}

} // namespace crispasr_ref_cache
