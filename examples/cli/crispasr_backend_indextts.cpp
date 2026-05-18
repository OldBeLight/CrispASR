// crispasr_backend_indextts.cpp -- adapter for IndexTTS-1.5 TTS.
//
// Two-GGUF runtime: the GPT AR model (loaded from --model) and the
// BigVGAN vocoder (loaded via --codec-model, or auto-discovered as
// a sibling file).

#include "crispasr_backend.h"
#include "crispasr_backend_utils.h"
#include "crispasr_model_mgr_cli.h"
#include "crispasr_model_registry.h"
#include "whisper_params.h"

#include "indextts.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace {

static bool file_exists(const std::string& path) {
    struct stat st;
    return stat(path.c_str(), &st) == 0;
}

// Look for a sibling BigVGAN vocoder file next to the GPT model.
static std::string discover_vocoder(const std::string& model_path) {
    auto dir_of = [](const std::string& p) -> std::string {
        auto sep = p.find_last_of("/\\");
        return (sep == std::string::npos) ? std::string(".") : p.substr(0, sep);
    };
    const std::string dir = dir_of(model_path);
    static const char* candidates[] = {
        "indextts-bigvgan.gguf",
        "indextts-bigvgan-f16.gguf",
        "indextts-bigvgan-q8_0.gguf",
    };
    for (const char* name : candidates) {
        std::string p = dir + "/" + name;
        if (file_exists(p)) {
            return p;
        }
    }
    return "";
}

// Minimal WAV reader (mono float32 at any sample rate — caller resamples).
static bool read_wav_mono(const std::string& path, std::vector<float>& pcm, int& sample_rate) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        return false;
    }

    char riff[4];
    if (fread(riff, 1, 4, f) != 4 || memcmp(riff, "RIFF", 4) != 0) {
        fclose(f);
        return false;
    }
    fseek(f, 4, SEEK_CUR); // skip chunk size
    char wave[4];
    if (fread(wave, 1, 4, f) != 4 || memcmp(wave, "WAVE", 4) != 0) {
        fclose(f);
        return false;
    }

    // Find fmt chunk
    int16_t audio_format = 0;
    int16_t n_channels = 0;
    int32_t sr = 0;
    int16_t bits_per_sample = 0;
    bool found_fmt = false;
    bool found_data = false;
    int32_t data_size = 0;

    while (!feof(f)) {
        char chunk_id[4];
        int32_t chunk_size;
        if (fread(chunk_id, 1, 4, f) != 4) {
            break;
        }
        if (fread(&chunk_size, 4, 1, f) != 1) {
            break;
        }

        if (memcmp(chunk_id, "fmt ", 4) == 0) {
            if (fread(&audio_format, 2, 1, f) != 1) {
                break;
            }
            if (fread(&n_channels, 2, 1, f) != 1) {
                break;
            }
            if (fread(&sr, 4, 1, f) != 1) {
                break;
            }
            fseek(f, 6, SEEK_CUR); // byte_rate + block_align
            if (fread(&bits_per_sample, 2, 1, f) != 1) {
                break;
            }
            if (chunk_size > 16) {
                fseek(f, chunk_size - 16, SEEK_CUR);
            }
            found_fmt = true;
        } else if (memcmp(chunk_id, "data", 4) == 0) {
            data_size = chunk_size;
            found_data = true;
            break;
        } else {
            fseek(f, chunk_size, SEEK_CUR);
        }
    }

    if (!found_fmt || !found_data || audio_format != 1 || bits_per_sample != 16) {
        fclose(f);
        return false;
    }

    sample_rate = sr;
    int n_samples_total = data_size / (n_channels * (bits_per_sample / 8));
    pcm.resize(n_samples_total);

    std::vector<int16_t> raw(n_samples_total * n_channels);
    size_t read_count = fread(raw.data(), sizeof(int16_t), raw.size(), f);
    fclose(f);

    int n_read = (int)(read_count / n_channels);
    for (int i = 0; i < n_read; i++) {
        float sum = 0.0f;
        for (int ch = 0; ch < n_channels; ch++) {
            sum += (float)raw[i * n_channels + ch] / 32768.0f;
        }
        pcm[i] = sum / n_channels;
    }
    pcm.resize(n_read);
    return true;
}

class IndexttsBackend : public CrispasrBackend {
public:
    IndexttsBackend() = default;
    ~IndexttsBackend() override { IndexttsBackend::shutdown(); }

    const char* name() const override { return "indextts"; }

    uint32_t capabilities() const override {
        return CAP_TTS | CAP_AUTO_DOWNLOAD | CAP_TEMPERATURE | CAP_FLASH_ATTN | CAP_VOICE_CLONING;
    }

    std::vector<crispasr_segment> transcribe(const float* /*samples*/, int /*n_samples*/, int64_t /*t_offset_cs*/,
                                             const whisper_params& /*params*/) override {
        fprintf(stderr, "crispasr[indextts]: transcription is not supported by this backend\n");
        return {};
    }

    bool init(const whisper_params& p) override {
        indextts_context_params cp = indextts_context_default_params();
        cp.n_threads = p.n_threads;
        cp.verbosity = p.no_prints ? 0 : 1;
        cp.use_gpu = crispasr_backend_should_use_gpu(p);
        cp.temperature = p.temperature; // 0 = greedy argmax, >0 = sampling
        cp.seed = p.seed;
        ctx_ = indextts_init_from_file(p.model.c_str(), cp);
        if (!ctx_) {
            fprintf(stderr, "crispasr[indextts]: failed to load GPT model '%s'\n", p.model.c_str());
            return false;
        }

        // BigVGAN vocoder
        std::string vocoder_path = p.tts_codec_model;
        if (!vocoder_path.empty() && vocoder_path != "auto" && vocoder_path != "default") {
            vocoder_path = crispasr_resolve_model_cli(vocoder_path, p.backend, p.no_prints, p.cache_dir,
                                                      p.auto_download, p.tts_codec_quant);
        } else {
            vocoder_path.clear();
        }
        if (vocoder_path.empty()) {
            vocoder_path = discover_vocoder(p.model);
        }
        if (vocoder_path.empty()) {
            CrispasrRegistryEntry entry;
            if (crispasr_registry_lookup(p.backend, entry, p.tts_codec_quant) && !entry.companion_filename.empty()) {
                vocoder_path = crispasr_resolve_model_cli(entry.companion_filename, p.backend, p.no_prints, p.cache_dir,
                                                          p.auto_download, p.tts_codec_quant);
            }
        }
        if (!vocoder_path.empty()) {
            indextts_set_vocoder_path(ctx_, vocoder_path.c_str());
            if (!p.no_prints) {
                fprintf(stderr, "crispasr[indextts]: vocoder path = '%s'\n", vocoder_path.c_str());
            }
        } else if (!p.no_prints) {
            fprintf(stderr, "crispasr[indextts]: no BigVGAN vocoder found. Pass --codec-model PATH or place "
                            "indextts-bigvgan.gguf next to the GPT model.\n");
        }

        // Load reference voice if provided
        if (!p.tts_voice.empty()) {
            voice_path_ = p.tts_voice;
        }

        return true;
    }

    std::vector<float> synthesize(const std::string& text, const whisper_params& params) override {
        if (!ctx_ || text.empty()) {
            return {};
        }

        // Load reference audio if specified
        const float* ref_pcm = nullptr;
        int ref_n_samples = 0;
        std::vector<float> ref_audio;

        if (!voice_path_.empty()) {
            int sr = 0;
            if (read_wav_mono(voice_path_, ref_audio, sr)) {
                // IndexTTS API expects 24kHz mono float32 PCM. Resample if needed.
                if (sr != 24000 && sr > 0) {
                    double ratio = 24000.0 / (double)sr;
                    int out_len = (int)(ref_audio.size() * ratio);
                    std::vector<float> resampled(out_len);
                    for (int i = 0; i < out_len; i++) {
                        double src = i / ratio;
                        int idx = (int)src;
                        double frac = src - idx;
                        if (idx + 1 < (int)ref_audio.size())
                            resampled[i] = (float)((1.0 - frac) * ref_audio[idx] + frac * ref_audio[idx + 1]);
                        else if (idx < (int)ref_audio.size())
                            resampled[i] = ref_audio[idx];
                    }
                    ref_audio = std::move(resampled);
                    if (!params.no_prints) {
                        fprintf(stderr, "crispasr[indextts]: resampled reference audio from %d Hz to 24000 Hz\n", sr);
                    }
                    sr = 24000;
                }
                ref_pcm = ref_audio.data();
                ref_n_samples = (int)ref_audio.size();
                if (!params.no_prints) {
                    fprintf(stderr, "crispasr[indextts]: loaded reference audio '%s' (%d samples, %d Hz)\n",
                            voice_path_.c_str(), ref_n_samples, sr);
                }
            } else {
                fprintf(stderr, "crispasr[indextts]: failed to load reference audio '%s'\n", voice_path_.c_str());
            }
        }

        int n = 0;
        float* pcm = indextts_synthesize(ctx_, text.c_str(), ref_pcm, ref_n_samples, &n);
        if (!pcm || n <= 0) {
            fprintf(stderr, "crispasr[indextts]: synthesis failed\n");
            return {};
        }
        std::vector<float> out(pcm, pcm + n);
        indextts_pcm_free(pcm);
        return out;
    }

    void shutdown() override {
        if (ctx_) {
            indextts_free(ctx_);
            ctx_ = nullptr;
        }
    }

private:
    indextts_context* ctx_ = nullptr;
    std::string voice_path_;
};

} // namespace

std::unique_ptr<CrispasrBackend> crispasr_make_indextts_backend() {
    return std::unique_ptr<CrispasrBackend>(new IndexttsBackend());
}
