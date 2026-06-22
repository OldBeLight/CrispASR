// Minimal audio file decoder for the language wrappers.
//
// libwhisper callers (Dart, Python, Rust wrappers) need cross-platform
// decoding of common audio formats so they can hand
// `crispasr_session_transcribe` a clean 16-kHz mono float32 buffer
// regardless of the original input format — without an ffmpeg dependency.
//
// miniaudio (MIT-0) handles WAV/MP3/FLAC (+ AIFF/W64/RF64 via its dr_wav) out
// of the box and does resampling + channel down-mix internally via its
// `ma_decoder` stream. Ogg Vorbis is handled by stb_vorbis (include it
// header-only before miniaudio so MA_HAS_VORBIS is auto-defined). Opus is added
// as a miniaudio custom backend (libopus/opusfile, CRISPASR_HAVE_OPUS; see
// below), and AAC/M4A/ALAC/CAF fall back to AudioToolbox on Apple. All
// permissive-licensed; ffmpeg stays an optional dynamic fallback only.

// stb_vorbis lives in examples/ — use relative path from src/
#define STB_VORBIS_HEADER_ONLY
#include "../examples/stb_vorbis.c"

#define MINIAUDIO_IMPLEMENTATION
// Device IO (capture mode) is needed for `crispasr_mic_*` (PLAN #62d);
// MA_NO_DEVICE_IO would strip the ma_device_* symbols. Threading
// follows from device IO. MA_NO_GENERATION (no oscillators / synth
// helpers) is still safe to keep.
#define MA_NO_GENERATION

// On iOS / tvOS / watchOS / visionOS, miniaudio's CoreAudio backend
// pulls in AVFoundation Objective-C headers from a .cpp TU, which the
// C++ front-end can't parse (NSString, etc., need Objective-C++ →
// .mm), and visionOS additionally has no AVAudioSession at all. The
// static-lib build artifact for those platforms is consumed by host
// apps that handle mic capture themselves, so we drop device IO here.
// The crispasr_mic_* C ABI is stubbed out to return failure on the
// same platforms (see crispasr_mic.cpp). macOS keeps device IO.
//
// TARGET_OS_IPHONE is the catch-all for the iOS-family platforms
// (iOS, tvOS, watchOS, visionOS); macOS leaves it as 0.
#if defined(__APPLE__)
#include <TargetConditionals.h>
#if TARGET_OS_IPHONE
#define MA_NO_DEVICE_IO
#endif
#endif

#include "miniaudio.h"

#undef STB_VORBIS_HEADER_ONLY
#include "../examples/stb_vorbis.c"

#include <cstdlib>
#include <cstring>

// Optional .opus (Ogg/Opus) support via libopus + opusfile (BSD-3-Clause),
// wired in as a miniaudio custom decoding backend so .opus flows through the
// same ma_decoder resample-to-16k + downmix + chunked-read path as WAV / MP3 /
// FLAC / OGG-Vorbis — no separate decode path, and no ffmpeg. The backend
// vtable (`ma_decoding_backend_libopus`) is defined in the vendored
// examples/miniaudio_libopus.c, compiled as a separate C TU and linked against
// this TU's MINIAUDIO_IMPLEMENTATION (see src/CMakeLists.txt). Gated on
// CRISPASR_HAVE_OPUS, which CMake sets when opusfile is found (pkg-config) or
// built statically (FetchContent fallback for platforms without system libs).
#if defined(CRISPASR_HAVE_OPUS)
#include "miniaudio_libopus.h"
namespace {
// ma_decoding_backend_libopus is already a `ma_decoding_backend_vtable*`.
ma_decoding_backend_vtable* g_crispasr_opus_backends[] = {ma_decoding_backend_libopus};
} // namespace
#define CRISPASR_OPUS_DECODER_CONFIG(cfg)                                                                              \
    do {                                                                                                               \
        (cfg).ppCustomBackendVTables = g_crispasr_opus_backends;                                                       \
        (cfg).customBackendCount = 1;                                                                                  \
    } while (0)
#else
#define CRISPASR_OPUS_DECODER_CONFIG(cfg)                                                                              \
    do {                                                                                                               \
    } while (0)
#endif

#ifdef _WIN32
#define CA_EXPORT extern "C" __declspec(dllexport)
#else
#define CA_EXPORT extern "C" __attribute__((visibility("default")))
#endif

// Forward declarations — satisfies -Wmissing-declarations without
// pulling in the full crispasr.h header (which conflicts with
// miniaudio's implementation-mode defines in this TU).
CA_EXPORT int crispasr_audio_load(const char*, float**, int*, int*);
CA_EXPORT int crispasr_audio_load_stereo(const char*, float**, float**, int*, int*, int*);
CA_EXPORT void crispasr_audio_free(float*);

namespace {
constexpr int kTargetSampleRate = 16000;
constexpr int kTargetChannels = 1;
} // namespace

// Apple-platform fallback for formats the permissive miniaudio path can't
// decode — notably AAC / M4A / ALAC / CAF. AudioToolbox's ExtAudioFile is a
// system framework (no ffmpeg, no GPL) that decodes + resamples + remixes to a
// requested client format in one pass. On non-Apple platforms the equivalent
// is Media Foundation (Windows) / MediaCodec (Android), wired separately; Linux
// has no system AAC decoder (fdk-aac or the optional ffmpeg fallback). Decodes
// to interleaved f32 @ 16 kHz with `want_channels` channels (1..2; 0 = native,
// capped at 2). malloc-owned buffer; 0 on success.
#if defined(__APPLE__)
#include <AudioToolbox/AudioToolbox.h>
#include <CoreFoundation/CoreFoundation.h>
namespace {
int crispasr_at_decode(const char* path, int want_channels, float** out_interleaved, int* out_frames,
                       int* out_channels) {
    CFStringRef cfpath = CFStringCreateWithCString(nullptr, path, kCFStringEncodingUTF8);
    if (!cfpath)
        return -2;
    CFURLRef url = CFURLCreateWithFileSystemPath(nullptr, cfpath, kCFURLPOSIXPathStyle, false);
    CFRelease(cfpath);
    if (!url)
        return -2;
    ExtAudioFileRef af = nullptr;
    OSStatus st = ExtAudioFileOpenURL(url, &af);
    CFRelease(url);
    if (st != noErr || !af)
        return -2;

    AudioStreamBasicDescription fileFmt;
    UInt32 sz = sizeof(fileFmt);
    if (ExtAudioFileGetProperty(af, kExtAudioFileProperty_FileDataFormat, &sz, &fileFmt) != noErr) {
        ExtAudioFileDispose(af);
        return -2;
    }
    int native_ch = (int)fileFmt.mChannelsPerFrame;
    if (native_ch < 1)
        native_ch = 1;
    int ch = want_channels > 0 ? want_channels : (native_ch >= 2 ? 2 : 1);
    if (ch < 1)
        ch = 1;
    if (ch > 2)
        ch = 2;

    AudioStreamBasicDescription cli;
    std::memset(&cli, 0, sizeof(cli));
    cli.mSampleRate = kTargetSampleRate;
    cli.mFormatID = kAudioFormatLinearPCM;
    cli.mFormatFlags = kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked; // interleaved f32
    cli.mBitsPerChannel = 32;
    cli.mChannelsPerFrame = (UInt32)ch;
    cli.mFramesPerPacket = 1;
    cli.mBytesPerFrame = (UInt32)(sizeof(float) * ch);
    cli.mBytesPerPacket = cli.mBytesPerFrame;
    if (ExtAudioFileSetProperty(af, kExtAudioFileProperty_ClientDataFormat, sizeof(cli), &cli) != noErr) {
        ExtAudioFileDispose(af);
        return -2;
    }

    const UInt32 chunkFrames = (UInt32)kTargetSampleRate; // 1 s
    float* buf = nullptr;
    size_t cap = 0, used = 0; // frames
    for (;;) {
        if (cap - used < chunkFrames) {
            const size_t newcap = cap ? cap * 2 : (size_t)chunkFrames * 8;
            float* nb = (float*)std::realloc(buf, newcap * (size_t)ch * sizeof(float));
            if (!nb) {
                std::free(buf);
                ExtAudioFileDispose(af);
                return -3;
            }
            buf = nb;
            cap = newcap;
        }
        AudioBufferList abl;
        abl.mNumberBuffers = 1;
        abl.mBuffers[0].mNumberChannels = (UInt32)ch;
        abl.mBuffers[0].mDataByteSize = (UInt32)(chunkFrames * (UInt32)ch * sizeof(float));
        abl.mBuffers[0].mData = buf + used * (size_t)ch;
        UInt32 n = chunkFrames;
        if (ExtAudioFileRead(af, &n, &abl) != noErr) {
            std::free(buf);
            ExtAudioFileDispose(af);
            return -4;
        }
        if (n == 0)
            break;
        used += n;
    }
    ExtAudioFileDispose(af);
    if (used == 0) {
        std::free(buf);
        return -2;
    }
    *out_interleaved = buf;
    *out_frames = (int)used;
    *out_channels = ch;
    return 0;
}
} // namespace
#endif // __APPLE__

/// Decode an audio file into float32 mono PCM at 16 kHz. Supports WAV / MP3 /
/// FLAC / AIFF / W64 / RF64 (miniaudio), OGG Vorbis (stb_vorbis), .opus
/// (libopus/opusfile, when CRISPASR_HAVE_OPUS), and AAC / M4A / ALAC / CAF on
/// Apple (AudioToolbox fallback). The returned buffer is malloc-owned and must
/// be released with `crispasr_audio_free`.
///
/// Returns 0 on success and writes:
///   *out_pcm         → float * of `*out_samples` elements (mono)
///   *out_samples     → number of samples written
///   *out_sample_rate → 16000 (we always resample to this)
///
/// Negative return codes:
///   -1 bad args
///   -2 decoder init failed (unsupported format or read error)
///   -3 allocation failed
///   -4 decode of a chunk failed mid-stream
CA_EXPORT int crispasr_audio_load(const char* path, float** out_pcm, int* out_samples, int* out_sample_rate) {
    if (!path || !out_pcm || !out_samples)
        return -1;
    *out_pcm = nullptr;
    *out_samples = 0;
    if (out_sample_rate)
        *out_sample_rate = 0;

    ma_decoder_config cfg = ma_decoder_config_init(ma_format_f32, kTargetChannels, kTargetSampleRate);
    CRISPASR_OPUS_DECODER_CONFIG(cfg);
    ma_decoder decoder;
    if (ma_decoder_init_file(path, &cfg, &decoder) != MA_SUCCESS) {
#if defined(__APPLE__)
        // Format miniaudio can't decode (AAC / M4A / ALAC / CAF …) — try the
        // OS-native AudioToolbox decoder. Requesting 1 channel gives mono
        // directly (AudioConverter downmixes).
        float* itl = nullptr;
        int fr = 0, ch = 0;
        if (crispasr_at_decode(path, 1, &itl, &fr, &ch) == 0) {
            *out_pcm = itl;
            *out_samples = fr;
            if (out_sample_rate)
                *out_sample_rate = kTargetSampleRate;
            return 0;
        }
#endif
        return -2;
    }

    // Decode in 1-second chunks. ma_decoder_get_length_in_pcm_frames can
    // fail on MP3 / streaming sources; chunked-read is what the CLI uses
    // and sidesteps that. The total allocation grows geometrically so we
    // don't re-alloc every chunk.
    constexpr ma_uint64 kChunkFrames = (ma_uint64)kTargetSampleRate; // 1 s
    float* buf = nullptr;
    size_t capacity = 0;
    size_t used = 0;

    for (;;) {
        if (capacity - used < kChunkFrames) {
            const size_t new_cap = capacity ? capacity * 2 : kChunkFrames * 8;
            float* nb = (float*)std::realloc(buf, new_cap * sizeof(float));
            if (!nb) {
                if (buf)
                    std::free(buf);
                ma_decoder_uninit(&decoder);
                return -3;
            }
            buf = nb;
            capacity = new_cap;
        }

        ma_uint64 frames_read = 0;
        const ma_result rc = ma_decoder_read_pcm_frames(&decoder, buf + used, kChunkFrames, &frames_read);
        used += (size_t)frames_read;

        if (rc == MA_AT_END || frames_read == 0)
            break;
        if (rc != MA_SUCCESS) {
            std::free(buf);
            ma_decoder_uninit(&decoder);
            return -4;
        }
    }
    ma_decoder_uninit(&decoder);

    // Trim trailing capacity we didn't fill — keeps the allocation tight.
    if (used < capacity) {
        float* tb = (float*)std::realloc(buf, used * sizeof(float));
        if (tb)
            buf = tb;
    }

    *out_pcm = buf;
    *out_samples = (int)used;
    if (out_sample_rate)
        *out_sample_rate = kTargetSampleRate;
    return 0;
}

/// Decode an audio file into stereo (2-channel) float32 PCM at 16 kHz.
/// If the source is mono, both left and right receive the same data and
/// `*out_channels` is set to 1. If stereo, the interleaved samples are
/// deinterleaved into separate left and right buffers and `*out_channels`
/// is set to 2. Each output buffer is malloc-owned and must be released
/// with `crispasr_audio_free`.
///
/// Returns 0 on success, negative on error (same codes as
/// `crispasr_audio_load`).
CA_EXPORT int crispasr_audio_load_stereo(const char* path, float** out_left, float** out_right, int* out_samples,
                                         int* out_sample_rate, int* out_channels) {
    if (!path || !out_left || !out_right || !out_samples || !out_channels)
        return -1;
    *out_left = nullptr;
    *out_right = nullptr;
    *out_samples = 0;
    *out_channels = 0;
    if (out_sample_rate)
        *out_sample_rate = 0;

    // Detect native channel count (channels = 0 → native).
    ma_decoder_config probe_cfg = ma_decoder_config_init(ma_format_f32, 0, kTargetSampleRate);
    CRISPASR_OPUS_DECODER_CONFIG(probe_cfg);
    ma_decoder probe;
    if (ma_decoder_init_file(path, &probe_cfg, &probe) != MA_SUCCESS) {
#if defined(__APPLE__)
        // AAC / M4A / ALAC / CAF … via AudioToolbox. Decode native (≤2 ch)
        // interleaved, then split into the per-channel L/R outputs.
        float* itl = nullptr;
        int fr = 0, ch = 0;
        if (crispasr_at_decode(path, 0, &itl, &fr, &ch) == 0) {
            float* left = (float*)std::malloc((size_t)fr * sizeof(float));
            float* right = (float*)std::malloc((size_t)fr * sizeof(float));
            if (!left || !right) {
                std::free(itl);
                std::free(left);
                std::free(right);
                return -3;
            }
            if (ch >= 2) {
                for (int i = 0; i < fr; ++i) {
                    left[i] = itl[(size_t)i * 2];
                    right[i] = itl[(size_t)i * 2 + 1];
                }
                *out_channels = 2;
            } else {
                std::memcpy(left, itl, (size_t)fr * sizeof(float));
                std::memcpy(right, itl, (size_t)fr * sizeof(float));
                *out_channels = 1;
            }
            std::free(itl);
            *out_left = left;
            *out_right = right;
            *out_samples = fr;
            if (out_sample_rate)
                *out_sample_rate = kTargetSampleRate;
            return 0;
        }
#endif
        return -2;
    }
    const int native_channels = (int)probe.outputChannels;
    ma_decoder_uninit(&probe);

    // Re-open with the target channel count (1 or 2) and 16 kHz.
    const int decode_channels = (native_channels >= 2) ? 2 : 1;
    ma_decoder_config cfg = ma_decoder_config_init(ma_format_f32, (ma_uint32)decode_channels, kTargetSampleRate);
    CRISPASR_OPUS_DECODER_CONFIG(cfg);
    ma_decoder decoder;
    if (ma_decoder_init_file(path, &cfg, &decoder) != MA_SUCCESS)
        return -2;

    constexpr ma_uint64 kChunkFrames = (ma_uint64)kTargetSampleRate; // 1 s
    float* buf = nullptr;
    size_t capacity = 0; // in frames
    size_t used = 0;     // in frames

    for (;;) {
        if (capacity - used < kChunkFrames) {
            const size_t new_cap = capacity ? capacity * 2 : kChunkFrames * 8;
            float* nb = (float*)std::realloc(buf, new_cap * (size_t)decode_channels * sizeof(float));
            if (!nb) {
                if (buf)
                    std::free(buf);
                ma_decoder_uninit(&decoder);
                return -3;
            }
            buf = nb;
            capacity = new_cap;
        }

        ma_uint64 frames_read = 0;
        const ma_result rc =
            ma_decoder_read_pcm_frames(&decoder, buf + used * (size_t)decode_channels, kChunkFrames, &frames_read);
        used += (size_t)frames_read;

        if (rc == MA_AT_END || frames_read == 0)
            break;
        if (rc != MA_SUCCESS) {
            std::free(buf);
            ma_decoder_uninit(&decoder);
            return -4;
        }
    }
    ma_decoder_uninit(&decoder);

    // Allocate per-channel output buffers.
    float* left = (float*)std::malloc(used * sizeof(float));
    float* right = (float*)std::malloc(used * sizeof(float));
    if (!left || !right) {
        std::free(buf);
        std::free(left);
        std::free(right);
        return -3;
    }

    if (decode_channels == 1) {
        // Mono: copy same data to both channels.
        std::memcpy(left, buf, used * sizeof(float));
        std::memcpy(right, buf, used * sizeof(float));
        *out_channels = 1;
    } else {
        // Stereo: deinterleave [L0 R0 L1 R1 ...] into separate buffers.
        for (size_t i = 0; i < used; ++i) {
            left[i] = buf[i * 2];
            right[i] = buf[i * 2 + 1];
        }
        *out_channels = 2;
    }
    std::free(buf);

    *out_left = left;
    *out_right = right;
    *out_samples = (int)used;
    if (out_sample_rate)
        *out_sample_rate = kTargetSampleRate;
    return 0;
}

/// Release a buffer allocated by `crispasr_audio_load`.
CA_EXPORT void crispasr_audio_free(float* pcm) {
    if (pcm)
        std::free(pcm);
}
