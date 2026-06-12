// crispasr_backend_nemotron.cpp — adapter for nvidia/nemotron-3.5-asr-streaming-0.6b.
//
// Wraps nemotron_init_from_file + nemotron_transcribe_ex and converts the
// native nemotron_result into std::vector<crispasr_segment>.

#include "crispasr_backend.h"
#include "crispasr_backend_utils.h"
#include "whisper_params.h"

#include "nemotron.h"

#include <cstdio>
#include <cstring>

namespace {

class NemotronBackend : public CrispasrBackend {
public:
    NemotronBackend() = default;
    ~NemotronBackend() override { NemotronBackend::shutdown(); }

    const char* name() const override { return "nemotron"; }

    uint32_t capabilities() const override {
        return CAP_TIMESTAMPS_NATIVE | CAP_WORD_TIMESTAMPS | CAP_TOKEN_CONFIDENCE | CAP_FLASH_ATTN |
               CAP_PUNCTUATION_NATIVE | CAP_TEMPERATURE | CAP_BEAM_SEARCH | CAP_DIARIZE | CAP_AUTO_DOWNLOAD |
               CAP_UNBOUNDED_INPUT;
    }

    bool init(const whisper_params& p) override {
        nemotron_context_params cp = nemotron_context_default_params();
        cp.n_threads = p.n_threads;
        cp.use_flash = p.flash_attn;
        cp.verbosity = p.no_prints ? 0 : 1;
        cp.use_gpu = crispasr_backend_should_use_gpu(p);

        ctx_ = nemotron_init_from_file(p.model.c_str(), cp);
        if (!ctx_) {
            fprintf(stderr, "crispasr[nemotron]: failed to load model '%s'\n", p.model.c_str());
            return false;
        }

        // Set language if specified
        if (!p.language.empty()) {
            nemotron_set_language(ctx_, p.language.c_str());
        }

        return true;
    }

    void warmup() override {
        if (!ctx_)
            return;
        std::vector<float> silence(8000, 0.0f);
        nemotron_result* r = nemotron_transcribe_ex(ctx_, silence.data(), (int)silence.size(), 0);
        if (r)
            nemotron_result_free(r);
    }

    std::vector<crispasr_segment> transcribe(const float* samples, int n_samples, int64_t t_offset_cs,
                                             const whisper_params& params) override {
        std::vector<crispasr_segment> out;
        if (!ctx_)
            return out;

        nemotron_set_temperature(ctx_, params.temperature, params.seed);
        nemotron_set_beam_size(ctx_, params.beam_size > 0 ? params.beam_size : 1);

        nemotron_result* r = nemotron_transcribe_ex(ctx_, samples, n_samples, t_offset_cs);
        if (!r)
            return out;

        crispasr_segment seg;
        seg.t0 = t_offset_cs;
        seg.t1 = t_offset_cs;
        seg.text = r->text ? r->text : "";

        // Words
        seg.words.reserve(r->n_words);
        for (int i = 0; i < r->n_words; i++) {
            const auto& w = r->words[i];
            crispasr_word cw;
            cw.text = w.text;
            cw.t0 = w.t0;
            cw.t1 = w.t1;
            seg.words.push_back(std::move(cw));
        }

        // Tokens
        seg.tokens.reserve(r->n_tokens);
        for (int i = 0; i < r->n_tokens; i++) {
            const auto& t = r->tokens[i];
            crispasr_token ct;
            ct.text = t.text;
            ct.id = t.id;
            ct.t0 = t.t0;
            ct.t1 = t.t1;
            ct.confidence = t.p;
            seg.tokens.push_back(std::move(ct));
        }

        if (!seg.words.empty()) {
            seg.t0 = seg.words.front().t0;
            seg.t1 = seg.words.back().t1;
        } else if (!seg.tokens.empty()) {
            seg.t0 = seg.tokens.front().t0;
            seg.t1 = seg.tokens.back().t1;
        }

        nemotron_result_free(r);
        out.push_back(std::move(seg));
        return out;
    }

    void shutdown() override {
        if (ctx_) {
            nemotron_free(ctx_);
            ctx_ = nullptr;
        }
    }

private:
    nemotron_context* ctx_ = nullptr;
};

} // namespace

std::unique_ptr<CrispasrBackend> crispasr_make_nemotron_backend() {
    return std::unique_ptr<CrispasrBackend>(new NemotronBackend());
}
