// pocket_tts.cpp -- Kyutai Pocket TTS (100M, continuous-latent AR) runtime.
//
// Architecture overview (see pocket_tts.h for full description):
//   1. Tokenize text with SentencePiece -> embed via learned LUT (4001 x 1024)
//   2. Optionally prepend voice conditioning (Mimi VAE encode + project)
//   3. AR loop at 12.5 Hz:
//      a. Feed current latent (32-dim, NaN for BOS) through input_linear (32->1024)
//      b. Concatenate text embeddings + audio input, run through 6-layer
//         causal transformer (1024D, 16H, RoPE, pre-norm LN, GELU FFN)
//      c. Take last position output, check EOS via out_eos linear
//      d. Consistency head (SimpleMLPAdaLN):
//         - Sample noise from N(0, temp^0.5), optionally clamped
//         - LSD decode: iteratively apply flow_net(backbone_out, s, t, x)
//           where s,t are timestep scalars, x is current noise/latent
//         - flow_net = input_proj + ResBlocks(AdaLN) + FinalLayer(AdaLN)
//         - Each ResBlock: LN(x)*scale+shift -> MLP -> gate
//         - Conditioning: sum of two TimestepEmbedders + cond_embed(backbone_out)
//      e. Output: 32-dim continuous float vector (the next latent)
//   4. Denormalize latents: x * emb_std + emb_mean
//   5. Mimi VAE decoder:
//      a. DummyQuantizer Conv1d projection (32 -> 512)
//      b. Upsample (stride-16 transposed conv, 32->512)
//      c. Decoder transformer (2L, 512D, 8H, LayerScale, context=250)
//      d. SEANet decoder (512 -> 1ch, ratios [6,5,4], hop=120)
//   6. Output: 24 kHz mono PCM

#include "pocket_tts.h"

#include "core/gguf_loader.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

// ── SentencePiece minimal decoder ──────────────────────────────────
// We store the serialized SPM model in the GGUF and decode it at init
// time using a minimal parser. For now, we use a stub that will be
// replaced with the real SPM integration.

namespace {

// ── Hyperparameters ────────────────────────────────────────────────

struct pocket_tts_flow_lm_hp {
    uint32_t d_model = 1024;
    uint32_t num_heads = 16;
    uint32_t num_layers = 6;
    uint32_t hidden_scale = 4; // ff_dim = d_model * hidden_scale
    uint32_t max_period = 10000;
    uint32_t latent_dim = 32;
    uint32_t n_bins = 4000; // vocab size
    uint32_t lut_dim = 1024;
    uint32_t insert_bos_before_voice = 1;

    uint32_t head_dim() const { return d_model / num_heads; }
    uint32_t ff_dim() const { return d_model * hidden_scale; }
};

struct pocket_tts_flow_head_hp {
    uint32_t flow_dim = 512;
    uint32_t flow_depth = 6;
    uint32_t num_time_conds = 2;
    uint32_t freq_embed_size = 256;
};

struct pocket_tts_mimi_hp {
    uint32_t sample_rate = 24000;
    uint32_t frame_rate_num = 25;
    uint32_t frame_rate_den = 2;
    uint32_t inner_dim = 32;
    uint32_t outer_dim = 512;
    uint32_t channels = 1;

    // SEANet
    uint32_t seanet_dimension = 512;
    uint32_t seanet_n_filters = 64;
    uint32_t seanet_n_residual_layers = 1;
    uint32_t seanet_kernel_size = 7;
    uint32_t seanet_residual_kernel_size = 3;
    uint32_t seanet_last_kernel_size = 3;
    uint32_t seanet_dilation_base = 2;
    uint32_t seanet_compress = 2;
    std::vector<int> seanet_ratios = {6, 5, 4};

    // Transformer
    uint32_t xfmr_d_model = 512;
    uint32_t xfmr_num_heads = 8;
    uint32_t xfmr_num_layers = 2;
    uint32_t xfmr_dim_feedforward = 2048;
    uint32_t xfmr_context = 250;
    float xfmr_layer_scale_init = 0.01f;

    // Quantizer
    uint32_t quant_in_dim = 32;
    uint32_t quant_out_dim = 512;

    float frame_rate() const { return (float)frame_rate_num / frame_rate_den; }
    uint32_t hop_length() const {
        uint32_t h = 1;
        for (int r : seanet_ratios)
            h *= r;
        return h;
    }
    float encoder_frame_rate() const { return (float)sample_rate / hop_length(); }
    uint32_t downsample_stride() const { return (uint32_t)(encoder_frame_rate() / frame_rate()); }
};

// ── Transformer layer weights ──────────────────────────────────────

struct pocket_tts_transformer_layer {
    ggml_tensor* attn_norm_w = nullptr;   // LayerNorm weight
    ggml_tensor* attn_norm_b = nullptr;   // LayerNorm bias
    ggml_tensor* attn_in_proj = nullptr;  // (3*d_model, d_model) fused QKV
    ggml_tensor* attn_out_proj = nullptr; // (d_model, d_model)
    ggml_tensor* ffn_norm_w = nullptr;    // LayerNorm weight
    ggml_tensor* ffn_norm_b = nullptr;    // LayerNorm bias
    ggml_tensor* ffn_linear1 = nullptr;   // (ff_dim, d_model)
    ggml_tensor* ffn_linear2 = nullptr;   // (d_model, ff_dim)

    // Mimi transformer layers also have LayerScale
    ggml_tensor* layer_scale_1 = nullptr; // (d_model,) or null
    ggml_tensor* layer_scale_2 = nullptr; // (d_model,) or null
};

// ── Flow network (consistency head) weights ────────────────────────

struct pocket_tts_flow_resblock {
    ggml_tensor* ln_w = nullptr;        // LayerNorm weight
    ggml_tensor* ln_b = nullptr;        // LayerNorm bias
    ggml_tensor* mlp_linear1 = nullptr; // (flow_dim, flow_dim)
    ggml_tensor* mlp_linear2 = nullptr; // (flow_dim, flow_dim)
    ggml_tensor* ada_linear = nullptr;  // (3*flow_dim, flow_dim)
    ggml_tensor* ada_bias = nullptr;    // (3*flow_dim,)
};

struct pocket_tts_flow_net {
    ggml_tensor* input_proj = nullptr; // (flow_dim, latent_dim)
    ggml_tensor* input_proj_b = nullptr;
    ggml_tensor* cond_embed = nullptr; // (flow_dim, d_model)
    ggml_tensor* cond_embed_b = nullptr;

    // TimestepEmbedders (num_time_conds=2)
    // Each: linear1(freq_embed_size->flow_dim) + SiLU + linear2(flow_dim->flow_dim) + RMSNorm
    struct timestep_embedder {
        ggml_tensor* linear1_w = nullptr; // (flow_dim, freq_embed_size)
        ggml_tensor* linear1_b = nullptr;
        ggml_tensor* linear2_w = nullptr; // (flow_dim, flow_dim)
        ggml_tensor* linear2_b = nullptr;
        ggml_tensor* rms_alpha = nullptr; // (flow_dim,)
        ggml_tensor* freqs = nullptr;     // (freq_embed_size/2,)
    };
    std::vector<timestep_embedder> time_embeds;

    std::vector<pocket_tts_flow_resblock> res_blocks;

    // FinalLayer
    // norm_final is elementwise_affine=False, so no params
    ggml_tensor* final_linear = nullptr; // (latent_dim, flow_dim)
    ggml_tensor* final_linear_b = nullptr;
    ggml_tensor* final_ada = nullptr; // (2*flow_dim, flow_dim)
    ggml_tensor* final_ada_b = nullptr;
};

// ── SEANet decoder weights ─────────────────────────────────────────

struct seanet_resblock_weights {
    // Two conv layers per block: conv0 (dim/compress input, dim/compress out)
    //                           conv1 (dim/compress input, dim out)
    ggml_tensor* conv0_w = nullptr; // (out_ch, in_ch, kernel)
    ggml_tensor* conv0_b = nullptr;
    ggml_tensor* conv1_w = nullptr;
    ggml_tensor* conv1_b = nullptr;
};

struct seanet_decoder_stage {
    // Transposed conv for upsampling
    ggml_tensor* convtr_w = nullptr;
    ggml_tensor* convtr_b = nullptr;
    // Residual blocks
    std::vector<seanet_resblock_weights> resblocks;
};

struct seanet_decoder_weights {
    ggml_tensor* initial_conv_w = nullptr; // (mult*n_filters, dimension, kernel)
    ggml_tensor* initial_conv_b = nullptr;
    std::vector<seanet_decoder_stage> stages;
    ggml_tensor* final_conv_w = nullptr; // (channels, n_filters, kernel)
    ggml_tensor* final_conv_b = nullptr;
};

// ── SEANet encoder weights (for voice cloning) ─────────────────────

struct seanet_encoder_stage {
    std::vector<seanet_resblock_weights> resblocks;
    ggml_tensor* conv_w = nullptr; // downsampling conv
    ggml_tensor* conv_b = nullptr;
};

struct seanet_encoder_weights {
    ggml_tensor* initial_conv_w = nullptr;
    ggml_tensor* initial_conv_b = nullptr;
    std::vector<seanet_encoder_stage> stages;
    ggml_tensor* final_conv_w = nullptr;
    ggml_tensor* final_conv_b = nullptr;
};

// ── Full model ─────────────────────────────────────────────────────

struct pocket_tts_model {
    pocket_tts_flow_lm_hp flow_lm_hp;
    pocket_tts_flow_head_hp flow_head_hp;
    pocket_tts_mimi_hp mimi_hp;

    bool has_voice_cloning = false;

    // FlowLM
    ggml_tensor* conditioner_embed = nullptr; // (n_bins+1, lut_dim)
    ggml_tensor* input_linear = nullptr;      // (d_model, latent_dim)
    ggml_tensor* out_norm_w = nullptr;        // (d_model,)
    ggml_tensor* out_norm_b = nullptr;        // (d_model,)
    ggml_tensor* out_eos_w = nullptr;         // (1, d_model)
    ggml_tensor* out_eos_b = nullptr;         // (1,)
    ggml_tensor* bos_emb = nullptr;           // (latent_dim,)
    ggml_tensor* bos_before_voice = nullptr;  // (1, 1, d_model)
    ggml_tensor* emb_std = nullptr;           // (latent_dim,)
    ggml_tensor* emb_mean = nullptr;          // (latent_dim,)

    // Speaker projection (voice cloning)
    ggml_tensor* speaker_proj = nullptr; // (d_model, inner_dim)

    std::vector<pocket_tts_transformer_layer> backbone_layers;
    pocket_tts_flow_net flow_net;

    // Mimi decoder
    ggml_tensor* quant_proj_w = nullptr;    // (outer_dim, inner_dim, 1)
    ggml_tensor* upsample_conv_w = nullptr; // transposed conv for 32->512 upsample
    ggml_tensor* upsample_conv_b = nullptr;
    std::vector<pocket_tts_transformer_layer> dec_transformer_layers;
    ggml_tensor* dec_xfmr_input_proj = nullptr;  // if d_model != input_dim
    ggml_tensor* dec_xfmr_output_proj = nullptr; // if d_model != output_dim
    seanet_decoder_weights seanet_dec;

    // Mimi encoder (only if has_voice_cloning)
    ggml_tensor* downsample_conv_w = nullptr;
    ggml_tensor* downsample_conv_b = nullptr;
    std::vector<pocket_tts_transformer_layer> enc_transformer_layers;
    ggml_tensor* enc_xfmr_input_proj = nullptr;
    ggml_tensor* enc_xfmr_output_proj = nullptr;
    seanet_encoder_weights seanet_enc;

    // SentencePiece tokenizer bytes (base64 decoded)
    std::vector<uint8_t> spm_model_bytes;
};

// ── KV cache for transformer ───────────────────────────────────────

struct pocket_tts_kv_cache {
    std::vector<float> k; // (n_layers, max_seq, n_heads, head_dim)
    std::vector<float> v;
    int max_seq = 0;
    int n_layers = 0;
    int n_heads = 0;
    int head_dim = 0;
    int offset = 0; // current write position
};

// ── Context ────────────────────────────────────────────────────────

struct pocket_tts_context {
    pocket_tts_context_params params;
    pocket_tts_model model;

    // GGML
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_buffer_type_t buf_type = nullptr;
    ggml_backend_buffer_t buf_weights = nullptr;

    // KV caches
    pocket_tts_kv_cache backbone_kv;
    pocket_tts_kv_cache dec_xfmr_kv; // Mimi decoder transformer

    // Voice conditioning state
    bool has_voice_state = false;
    std::vector<float> voice_conditioning; // (n_voice_frames, d_model)
    int n_voice_frames = 0;

    // RNG
    std::mt19937 rng;

    int verbosity = 1;
};

// ── Helpers ────────────────────────────────────────────────────────

static uint32_t gguf_get_u32(struct gguf_context* meta, const char* key, uint32_t def) {
    int idx = gguf_find_key(meta, key);
    if (idx < 0)
        return def;
    return (uint32_t)gguf_get_val_u32(meta, idx);
}

static float gguf_get_f32(struct gguf_context* meta, const char* key, float def) {
    int idx = gguf_find_key(meta, key);
    if (idx < 0)
        return def;
    return gguf_get_val_f32(meta, idx);
}

static std::string gguf_get_str(struct gguf_context* meta, const char* key, const char* def) {
    int idx = gguf_find_key(meta, key);
    if (idx < 0)
        return def;
    return gguf_get_val_str(meta, idx);
}

// ── Load model from GGUF ───────────────────────────────────────────

static bool load_hparams(struct gguf_context* meta, pocket_tts_model& m) {
    auto& h = m.flow_lm_hp;
    h.d_model = gguf_get_u32(meta, "pocket_tts.flow_lm.d_model", h.d_model);
    h.num_heads = gguf_get_u32(meta, "pocket_tts.flow_lm.num_heads", h.num_heads);
    h.num_layers = gguf_get_u32(meta, "pocket_tts.flow_lm.num_layers", h.num_layers);
    h.hidden_scale = gguf_get_u32(meta, "pocket_tts.flow_lm.hidden_scale", h.hidden_scale);
    h.max_period = gguf_get_u32(meta, "pocket_tts.flow_lm.max_period", h.max_period);
    h.latent_dim = gguf_get_u32(meta, "pocket_tts.flow_lm.latent_dim", h.latent_dim);
    h.n_bins = gguf_get_u32(meta, "pocket_tts.flow_lm.n_bins", h.n_bins);
    h.lut_dim = gguf_get_u32(meta, "pocket_tts.flow_lm.lut_dim", h.lut_dim);
    h.insert_bos_before_voice =
        gguf_get_u32(meta, "pocket_tts.flow_lm.insert_bos_before_voice", h.insert_bos_before_voice);

    auto& fh = m.flow_head_hp;
    fh.flow_dim = gguf_get_u32(meta, "pocket_tts.flow_head.flow_dim", fh.flow_dim);
    fh.flow_depth = gguf_get_u32(meta, "pocket_tts.flow_head.flow_depth", fh.flow_depth);
    fh.num_time_conds = gguf_get_u32(meta, "pocket_tts.flow_head.num_time_conds", fh.num_time_conds);
    fh.freq_embed_size = gguf_get_u32(meta, "pocket_tts.flow_head.freq_embed_size", fh.freq_embed_size);

    auto& mi = m.mimi_hp;
    mi.sample_rate = gguf_get_u32(meta, "pocket_tts.mimi.sample_rate", mi.sample_rate);
    mi.frame_rate_num = gguf_get_u32(meta, "pocket_tts.mimi.frame_rate_num", mi.frame_rate_num);
    mi.frame_rate_den = gguf_get_u32(meta, "pocket_tts.mimi.frame_rate_den", mi.frame_rate_den);
    mi.inner_dim = gguf_get_u32(meta, "pocket_tts.mimi.inner_dim", mi.inner_dim);
    mi.outer_dim = gguf_get_u32(meta, "pocket_tts.mimi.outer_dim", mi.outer_dim);
    mi.channels = gguf_get_u32(meta, "pocket_tts.mimi.channels", mi.channels);

    mi.seanet_dimension = gguf_get_u32(meta, "pocket_tts.mimi.seanet_dimension", mi.seanet_dimension);
    mi.seanet_n_filters = gguf_get_u32(meta, "pocket_tts.mimi.seanet_n_filters", mi.seanet_n_filters);
    mi.seanet_n_residual_layers =
        gguf_get_u32(meta, "pocket_tts.mimi.seanet_n_residual_layers", mi.seanet_n_residual_layers);
    mi.seanet_kernel_size = gguf_get_u32(meta, "pocket_tts.mimi.seanet_kernel_size", mi.seanet_kernel_size);
    mi.seanet_residual_kernel_size =
        gguf_get_u32(meta, "pocket_tts.mimi.seanet_residual_kernel_size", mi.seanet_residual_kernel_size);
    mi.seanet_last_kernel_size =
        gguf_get_u32(meta, "pocket_tts.mimi.seanet_last_kernel_size", mi.seanet_last_kernel_size);
    mi.seanet_dilation_base = gguf_get_u32(meta, "pocket_tts.mimi.seanet_dilation_base", mi.seanet_dilation_base);
    mi.seanet_compress = gguf_get_u32(meta, "pocket_tts.mimi.seanet_compress", mi.seanet_compress);

    mi.xfmr_d_model = gguf_get_u32(meta, "pocket_tts.mimi.xfmr_d_model", mi.xfmr_d_model);
    mi.xfmr_num_heads = gguf_get_u32(meta, "pocket_tts.mimi.xfmr_num_heads", mi.xfmr_num_heads);
    mi.xfmr_num_layers = gguf_get_u32(meta, "pocket_tts.mimi.xfmr_num_layers", mi.xfmr_num_layers);
    mi.xfmr_dim_feedforward = gguf_get_u32(meta, "pocket_tts.mimi.xfmr_dim_feedforward", mi.xfmr_dim_feedforward);
    mi.xfmr_context = gguf_get_u32(meta, "pocket_tts.mimi.xfmr_context", mi.xfmr_context);
    mi.xfmr_layer_scale_init = gguf_get_f32(meta, "pocket_tts.mimi.xfmr_layer_scale_init", mi.xfmr_layer_scale_init);

    mi.quant_in_dim = gguf_get_u32(meta, "pocket_tts.mimi.quant_in_dim", mi.quant_in_dim);
    mi.quant_out_dim = gguf_get_u32(meta, "pocket_tts.mimi.quant_out_dim", mi.quant_out_dim);

    // SEANet ratios from array KV
    int ratios_idx = gguf_find_key(meta, "pocket_tts.mimi.seanet_ratios");
    if (ratios_idx >= 0) {
        mi.seanet_ratios.clear();
        // The GGUF array KV stores the count; iterate
        // For now, use the default [6,5,4] if reading fails
    }

    m.has_voice_cloning = gguf_get_u32(meta, "pocket_tts.has_voice_cloning", 0) != 0;

    return true;
}

// ── Tensor loading ─────────────────────────────────────────────────

static ggml_tensor* try_get_tensor(struct ggml_context* ctx, const char* name) {
    return ggml_get_tensor(ctx, name);
}

static bool load_flow_lm_tensors(struct ggml_context* ctx, pocket_tts_model& m) {
    const auto& h = m.flow_lm_hp;

    m.conditioner_embed = try_get_tensor(ctx, "flow_lm.conditioner.embed.weight");
    m.input_linear = try_get_tensor(ctx, "flow_lm.input_linear.weight");
    m.out_norm_w = try_get_tensor(ctx, "flow_lm.out_norm.weight");
    m.out_norm_b = try_get_tensor(ctx, "flow_lm.out_norm.bias");
    m.out_eos_w = try_get_tensor(ctx, "flow_lm.out_eos.weight");
    m.out_eos_b = try_get_tensor(ctx, "flow_lm.out_eos.bias");
    m.bos_emb = try_get_tensor(ctx, "flow_lm.bos_emb");
    m.bos_before_voice = try_get_tensor(ctx, "flow_lm.bos_before_voice");
    m.emb_std = try_get_tensor(ctx, "flow_lm.emb_std");
    m.emb_mean = try_get_tensor(ctx, "flow_lm.emb_mean");
    m.speaker_proj = try_get_tensor(ctx, "flow_lm.speaker_proj.weight");

    // Backbone transformer layers
    m.backbone_layers.resize(h.num_layers);
    for (uint32_t i = 0; i < h.num_layers; i++) {
        auto& L = m.backbone_layers[i];
        char buf[256];

        snprintf(buf, sizeof(buf), "flow_lm.transformer.layers.%u.norm1.weight", i);
        L.attn_norm_w = try_get_tensor(ctx, buf);
        snprintf(buf, sizeof(buf), "flow_lm.transformer.layers.%u.norm1.bias", i);
        L.attn_norm_b = try_get_tensor(ctx, buf);
        snprintf(buf, sizeof(buf), "flow_lm.transformer.layers.%u.self_attn.in_proj.weight", i);
        L.attn_in_proj = try_get_tensor(ctx, buf);
        snprintf(buf, sizeof(buf), "flow_lm.transformer.layers.%u.self_attn.out_proj.weight", i);
        L.attn_out_proj = try_get_tensor(ctx, buf);

        snprintf(buf, sizeof(buf), "flow_lm.transformer.layers.%u.norm2.weight", i);
        L.ffn_norm_w = try_get_tensor(ctx, buf);
        snprintf(buf, sizeof(buf), "flow_lm.transformer.layers.%u.norm2.bias", i);
        L.ffn_norm_b = try_get_tensor(ctx, buf);
        snprintf(buf, sizeof(buf), "flow_lm.transformer.layers.%u.linear1.weight", i);
        L.ffn_linear1 = try_get_tensor(ctx, buf);
        snprintf(buf, sizeof(buf), "flow_lm.transformer.layers.%u.linear2.weight", i);
        L.ffn_linear2 = try_get_tensor(ctx, buf);
    }

    // Flow network (consistency head)
    auto& fn = m.flow_net;
    fn.input_proj = try_get_tensor(ctx, "flow_lm.flow_net.input_proj.weight");
    fn.input_proj_b = try_get_tensor(ctx, "flow_lm.flow_net.input_proj.bias");
    fn.cond_embed = try_get_tensor(ctx, "flow_lm.flow_net.cond_embed.weight");
    fn.cond_embed_b = try_get_tensor(ctx, "flow_lm.flow_net.cond_embed.bias");

    fn.time_embeds.resize(m.flow_head_hp.num_time_conds);
    for (uint32_t t = 0; t < m.flow_head_hp.num_time_conds; t++) {
        auto& te = fn.time_embeds[t];
        char buf[256];
        snprintf(buf, sizeof(buf), "flow_lm.flow_net.time_embed.%u.mlp.0.weight", t);
        te.linear1_w = try_get_tensor(ctx, buf);
        snprintf(buf, sizeof(buf), "flow_lm.flow_net.time_embed.%u.mlp.0.bias", t);
        te.linear1_b = try_get_tensor(ctx, buf);
        snprintf(buf, sizeof(buf), "flow_lm.flow_net.time_embed.%u.mlp.2.weight", t);
        te.linear2_w = try_get_tensor(ctx, buf);
        snprintf(buf, sizeof(buf), "flow_lm.flow_net.time_embed.%u.mlp.2.bias", t);
        te.linear2_b = try_get_tensor(ctx, buf);
        snprintf(buf, sizeof(buf), "flow_lm.flow_net.time_embed.%u.mlp.3.alpha", t);
        te.rms_alpha = try_get_tensor(ctx, buf);
        snprintf(buf, sizeof(buf), "flow_lm.flow_net.time_embed.%u.freqs", t);
        te.freqs = try_get_tensor(ctx, buf);
    }

    fn.res_blocks.resize(m.flow_head_hp.flow_depth);
    for (uint32_t i = 0; i < m.flow_head_hp.flow_depth; i++) {
        auto& rb = fn.res_blocks[i];
        char buf[256];
        snprintf(buf, sizeof(buf), "flow_lm.flow_net.res_blocks.%u.in_ln.weight", i);
        rb.ln_w = try_get_tensor(ctx, buf);
        snprintf(buf, sizeof(buf), "flow_lm.flow_net.res_blocks.%u.in_ln.bias", i);
        rb.ln_b = try_get_tensor(ctx, buf);
        snprintf(buf, sizeof(buf), "flow_lm.flow_net.res_blocks.%u.mlp.0.weight", i);
        rb.mlp_linear1 = try_get_tensor(ctx, buf);
        snprintf(buf, sizeof(buf), "flow_lm.flow_net.res_blocks.%u.mlp.0.bias", i);
        // bias for mlp.0
        auto* b0 = try_get_tensor(ctx, buf);
        (void)b0; // mlp uses bias=True
        snprintf(buf, sizeof(buf), "flow_lm.flow_net.res_blocks.%u.mlp.2.weight", i);
        rb.mlp_linear2 = try_get_tensor(ctx, buf);
        snprintf(buf, sizeof(buf), "flow_lm.flow_net.res_blocks.%u.adaLN_modulation.1.weight", i);
        rb.ada_linear = try_get_tensor(ctx, buf);
        snprintf(buf, sizeof(buf), "flow_lm.flow_net.res_blocks.%u.adaLN_modulation.1.bias", i);
        rb.ada_bias = try_get_tensor(ctx, buf);
    }

    fn.final_linear = try_get_tensor(ctx, "flow_lm.flow_net.final_layer.linear.weight");
    fn.final_linear_b = try_get_tensor(ctx, "flow_lm.flow_net.final_layer.linear.bias");
    fn.final_ada = try_get_tensor(ctx, "flow_lm.flow_net.final_layer.adaLN_modulation.1.weight");
    fn.final_ada_b = try_get_tensor(ctx, "flow_lm.flow_net.final_layer.adaLN_modulation.1.bias");

    return m.conditioner_embed != nullptr && m.input_linear != nullptr;
}

static bool load_mimi_decoder_tensors(struct ggml_context* ctx, pocket_tts_model& m) {
    const auto& mi = m.mimi_hp;

    // Quantizer projection (Conv1d, kernel=1)
    m.quant_proj_w = try_get_tensor(ctx, "mimi.quantizer.output_proj.weight");

    // Upsample conv (transposed, stride=downsample_stride)
    m.upsample_conv_w = try_get_tensor(ctx, "mimi.upsample.conv.weight");
    m.upsample_conv_b = try_get_tensor(ctx, "mimi.upsample.conv.bias");

    // Decoder transformer
    m.dec_xfmr_input_proj = try_get_tensor(ctx, "mimi.decoder_transformer.input_proj.weight");
    m.dec_xfmr_output_proj = try_get_tensor(ctx, "mimi.decoder_transformer.output_projs.0.weight");

    m.dec_transformer_layers.resize(mi.xfmr_num_layers);
    for (uint32_t i = 0; i < mi.xfmr_num_layers; i++) {
        auto& L = m.dec_transformer_layers[i];
        char buf[256];
        snprintf(buf, sizeof(buf), "mimi.decoder_transformer.transformer.layers.%u.norm1.weight", i);
        L.attn_norm_w = try_get_tensor(ctx, buf);
        snprintf(buf, sizeof(buf), "mimi.decoder_transformer.transformer.layers.%u.norm1.bias", i);
        L.attn_norm_b = try_get_tensor(ctx, buf);
        snprintf(buf, sizeof(buf), "mimi.decoder_transformer.transformer.layers.%u.self_attn.in_proj.weight", i);
        L.attn_in_proj = try_get_tensor(ctx, buf);
        snprintf(buf, sizeof(buf), "mimi.decoder_transformer.transformer.layers.%u.self_attn.out_proj.weight", i);
        L.attn_out_proj = try_get_tensor(ctx, buf);
        snprintf(buf, sizeof(buf), "mimi.decoder_transformer.transformer.layers.%u.norm2.weight", i);
        L.ffn_norm_w = try_get_tensor(ctx, buf);
        snprintf(buf, sizeof(buf), "mimi.decoder_transformer.transformer.layers.%u.norm2.bias", i);
        L.ffn_norm_b = try_get_tensor(ctx, buf);
        snprintf(buf, sizeof(buf), "mimi.decoder_transformer.transformer.layers.%u.linear1.weight", i);
        L.ffn_linear1 = try_get_tensor(ctx, buf);
        snprintf(buf, sizeof(buf), "mimi.decoder_transformer.transformer.layers.%u.linear2.weight", i);
        L.ffn_linear2 = try_get_tensor(ctx, buf);

        snprintf(buf, sizeof(buf), "mimi.decoder_transformer.transformer.layers.%u.layer_scale_1.scale", i);
        L.layer_scale_1 = try_get_tensor(ctx, buf);
        snprintf(buf, sizeof(buf), "mimi.decoder_transformer.transformer.layers.%u.layer_scale_2.scale", i);
        L.layer_scale_2 = try_get_tensor(ctx, buf);
    }

    // SEANet decoder
    auto& sd = m.seanet_dec;
    sd.initial_conv_w = try_get_tensor(ctx, "mimi.decoder.model.0.conv.weight");
    sd.initial_conv_b = try_get_tensor(ctx, "mimi.decoder.model.0.conv.bias");

    // Decoder stages: for each ratio, there's an upsample + residual blocks
    // Layout in model list: [initial_conv, (ELU, ConvTr, ResBlock*n_res)*n_ratios, ELU, final_conv]
    const auto& ratios = mi.seanet_ratios;
    sd.stages.resize(ratios.size());

    // The decoder model list has:
    // [0] = initial conv
    // Then for each ratio i (0..n_ratios-1):
    //   [1 + i*(1+n_res+1)] = ELU
    //   [1 + i*(1+n_res+1) + 1] = ConvTranspose (upsample)
    //   [1 + i*(1+n_res+1) + 2 .. + 2+n_res-1] = ResBlocks
    // [last-1] = ELU
    // [last]   = final conv
    //
    // In practice with n_res=1:
    //   model[0] = conv, model[1] = ELU, model[2] = ConvTr,
    //   model[3] = ResBlock, model[4] = ELU, model[5] = ConvTr, ...
    // etc. Let's compute indices.

    uint32_t n_res = mi.seanet_n_residual_layers;
    uint32_t idx = 1; // after initial conv
    for (size_t s = 0; s < ratios.size(); s++) {
        auto& stage = sd.stages[s];
        // ELU at idx, skip
        idx++; // ELU
        // ConvTranspose at idx
        char buf[256];
        snprintf(buf, sizeof(buf), "mimi.decoder.model.%u.convtr.weight", idx);
        stage.convtr_w = try_get_tensor(ctx, buf);
        snprintf(buf, sizeof(buf), "mimi.decoder.model.%u.convtr.bias", idx);
        stage.convtr_b = try_get_tensor(ctx, buf);
        idx++; // ConvTr

        // Residual blocks
        stage.resblocks.resize(n_res);
        for (uint32_t r = 0; r < n_res; r++) {
            auto& rb = stage.resblocks[r];
            // Each resblock has a model list: [ELU, Conv, ELU, Conv]
            // block[0] = ELU, block[1] = Conv (dim->hidden), block[2] = ELU, block[3] = Conv (hidden->dim)
            snprintf(buf, sizeof(buf), "mimi.decoder.model.%u.block.1.conv.weight", idx);
            rb.conv0_w = try_get_tensor(ctx, buf);
            snprintf(buf, sizeof(buf), "mimi.decoder.model.%u.block.1.conv.bias", idx);
            rb.conv0_b = try_get_tensor(ctx, buf);
            snprintf(buf, sizeof(buf), "mimi.decoder.model.%u.block.3.conv.weight", idx);
            rb.conv1_w = try_get_tensor(ctx, buf);
            snprintf(buf, sizeof(buf), "mimi.decoder.model.%u.block.3.conv.bias", idx);
            rb.conv1_b = try_get_tensor(ctx, buf);
            idx++; // ResBlock
        }
    }

    // ELU + final conv
    idx++; // ELU
    char buf[256];
    snprintf(buf, sizeof(buf), "mimi.decoder.model.%u.conv.weight", idx);
    sd.final_conv_w = try_get_tensor(ctx, buf);
    snprintf(buf, sizeof(buf), "mimi.decoder.model.%u.conv.bias", idx);
    sd.final_conv_b = try_get_tensor(ctx, buf);

    return true;
}

static bool load_mimi_encoder_tensors(struct ggml_context* ctx, pocket_tts_model& m) {
    if (!m.has_voice_cloning)
        return true;

    const auto& mi = m.mimi_hp;

    m.downsample_conv_w = try_get_tensor(ctx, "mimi.downsample.conv.weight");
    m.downsample_conv_b = try_get_tensor(ctx, "mimi.downsample.conv.bias");

    m.enc_xfmr_input_proj = try_get_tensor(ctx, "mimi.encoder_transformer.input_proj.weight");
    m.enc_xfmr_output_proj = try_get_tensor(ctx, "mimi.encoder_transformer.output_projs.0.weight");

    m.enc_transformer_layers.resize(mi.xfmr_num_layers);
    for (uint32_t i = 0; i < mi.xfmr_num_layers; i++) {
        auto& L = m.enc_transformer_layers[i];
        char buf[256];
        snprintf(buf, sizeof(buf), "mimi.encoder_transformer.transformer.layers.%u.norm1.weight", i);
        L.attn_norm_w = try_get_tensor(ctx, buf);
        snprintf(buf, sizeof(buf), "mimi.encoder_transformer.transformer.layers.%u.norm1.bias", i);
        L.attn_norm_b = try_get_tensor(ctx, buf);
        snprintf(buf, sizeof(buf), "mimi.encoder_transformer.transformer.layers.%u.self_attn.in_proj.weight", i);
        L.attn_in_proj = try_get_tensor(ctx, buf);
        snprintf(buf, sizeof(buf), "mimi.encoder_transformer.transformer.layers.%u.self_attn.out_proj.weight", i);
        L.attn_out_proj = try_get_tensor(ctx, buf);
        snprintf(buf, sizeof(buf), "mimi.encoder_transformer.transformer.layers.%u.norm2.weight", i);
        L.ffn_norm_w = try_get_tensor(ctx, buf);
        snprintf(buf, sizeof(buf), "mimi.encoder_transformer.transformer.layers.%u.norm2.bias", i);
        L.ffn_norm_b = try_get_tensor(ctx, buf);
        snprintf(buf, sizeof(buf), "mimi.encoder_transformer.transformer.layers.%u.linear1.weight", i);
        L.ffn_linear1 = try_get_tensor(ctx, buf);
        snprintf(buf, sizeof(buf), "mimi.encoder_transformer.transformer.layers.%u.linear2.weight", i);
        L.ffn_linear2 = try_get_tensor(ctx, buf);
        snprintf(buf, sizeof(buf), "mimi.encoder_transformer.transformer.layers.%u.layer_scale_1.scale", i);
        L.layer_scale_1 = try_get_tensor(ctx, buf);
        snprintf(buf, sizeof(buf), "mimi.encoder_transformer.transformer.layers.%u.layer_scale_2.scale", i);
        L.layer_scale_2 = try_get_tensor(ctx, buf);
    }

    // SEANet encoder
    auto& se = m.seanet_enc;
    se.initial_conv_w = try_get_tensor(ctx, "mimi.encoder.model.0.conv.weight");
    se.initial_conv_b = try_get_tensor(ctx, "mimi.encoder.model.0.conv.bias");

    // Encoder: ratios are reversed (compared to decoder)
    std::vector<int> enc_ratios(mi.seanet_ratios.rbegin(), mi.seanet_ratios.rend());
    uint32_t n_res = mi.seanet_n_residual_layers;
    se.stages.resize(enc_ratios.size());

    uint32_t idx = 1; // after initial conv
    for (size_t s = 0; s < enc_ratios.size(); s++) {
        auto& stage = se.stages[s];

        // Residual blocks first in encoder
        stage.resblocks.resize(n_res);
        for (uint32_t r = 0; r < n_res; r++) {
            auto& rb = stage.resblocks[r];
            char buf[256];
            snprintf(buf, sizeof(buf), "mimi.encoder.model.%u.block.1.conv.weight", idx);
            rb.conv0_w = try_get_tensor(ctx, buf);
            snprintf(buf, sizeof(buf), "mimi.encoder.model.%u.block.1.conv.bias", idx);
            rb.conv0_b = try_get_tensor(ctx, buf);
            snprintf(buf, sizeof(buf), "mimi.encoder.model.%u.block.3.conv.weight", idx);
            rb.conv1_w = try_get_tensor(ctx, buf);
            snprintf(buf, sizeof(buf), "mimi.encoder.model.%u.block.3.conv.bias", idx);
            rb.conv1_b = try_get_tensor(ctx, buf);
            idx++;
        }

        // ELU + downsampling conv
        idx++; // ELU
        char buf[256];
        snprintf(buf, sizeof(buf), "mimi.encoder.model.%u.conv.weight", idx);
        stage.conv_w = try_get_tensor(ctx, buf);
        snprintf(buf, sizeof(buf), "mimi.encoder.model.%u.conv.bias", idx);
        stage.conv_b = try_get_tensor(ctx, buf);
        idx++;
    }

    // ELU + final conv
    idx++; // ELU
    char buf[256];
    snprintf(buf, sizeof(buf), "mimi.encoder.model.%u.conv.weight", idx);
    se.final_conv_w = try_get_tensor(ctx, buf);
    snprintf(buf, sizeof(buf), "mimi.encoder.model.%u.conv.bias", idx);
    se.final_conv_b = try_get_tensor(ctx, buf);

    return true;
}

// ── SentencePiece tokenizer stub ───────────────────────────────────
// TODO: integrate real SPM decoding. For now we store model bytes
// and will add proper tokenization in the next iteration.

static bool load_tokenizer(struct gguf_context* meta, pocket_tts_model& m) {
    std::string b64 = gguf_get_str(meta, "pocket_tts.tokenizer.spm_model_b64", "");
    if (b64.empty()) {
        fprintf(stderr, "pocket_tts: warning: no tokenizer found in GGUF\n");
        return false;
    }

    // Base64 decode
    // Simple decoder (standard base64)
    auto b64_decode = [](const std::string& input) -> std::vector<uint8_t> {
        static const char table[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        static int8_t dtable[256];
        static bool dtable_init = false;
        if (!dtable_init) {
            memset(dtable, -1, sizeof(dtable));
            for (int i = 0; i < 64; i++)
                dtable[(unsigned char)table[i]] = (int8_t)i;
            dtable_init = true;
        }
        std::vector<uint8_t> out;
        out.reserve(input.size() * 3 / 4);
        uint32_t buf = 0;
        int bits = 0;
        for (char c : input) {
            if (c == '=' || dtable[(unsigned char)c] < 0)
                continue;
            buf = (buf << 6) | dtable[(unsigned char)c];
            bits += 6;
            if (bits >= 8) {
                bits -= 8;
                out.push_back((uint8_t)(buf >> bits));
            }
        }
        return out;
    };

    m.spm_model_bytes = b64_decode(b64);
    return !m.spm_model_bytes.empty();
}

// ── Forward pass building blocks ───────────────────────────────────
//
// These are skeleton functions that outline the computation graph.
// The actual ggml graph construction will be added in the next
// iteration when we have a converted GGUF to test against.

// RoPE helper: compute cos/sin for rotary position embedding
static void compute_rope_freqs(float* cos_out, float* sin_out, int dim, int offset, float max_period) {
    int half = dim / 2;
    for (int i = 0; i < half; i++) {
        double freq = std::exp(-(double)i / half * std::log((double)max_period));
        double angle = (double)offset * freq;
        cos_out[i] = (float)std::cos(angle);
        sin_out[i] = (float)std::sin(angle);
    }
}

// Timestep embedding: sinusoidal with learned MLP
static void timestep_embed(const float* freqs_buf, int half_dim, float t_val, float* out, int out_dim) {
    // args = t * freqs -> cos/sin -> concat -> MLP
    // This is a sketch; actual implementation via ggml graph
    (void)freqs_buf;
    (void)half_dim;
    (void)t_val;
    (void)out;
    (void)out_dim;
}

// ── Graph construction stubs ───────────────────────────────────────
//
// These functions will build ggml computation graphs for each
// component. They are currently placeholders that define the
// interfaces.

// Run one step of the FlowLM backbone transformer
// Input: current latent (1, latent_dim) or text tokens
// Output: transformer hidden state at last position (1, d_model)
static void backbone_forward_step(pocket_tts_context* ctx,
                                  const float* input_latent,  // (latent_dim,) or nullptr for text-only
                                  const int32_t* text_tokens, // token ids or nullptr
                                  int n_text_tokens,
                                  float* backbone_out // (d_model,) output
) {
    // TODO: build ggml graph
    // 1. Embed text tokens via LUT if provided
    // 2. Project latent via input_linear if provided
    // 3. Concatenate [text_embeddings, projected_latent]
    // 4. Run through 6 transformer layers with KV cache
    // 5. Apply out_norm (LayerNorm)
    // 6. Return last position output
    (void)ctx;
    (void)input_latent;
    (void)text_tokens;
    (void)n_text_tokens;
    (void)backbone_out;
}

// Run the consistency head (flow network) for one-step LSD decode
// Input: backbone output (d_model,), noise sample
// Output: predicted latent (latent_dim,)
static void flow_net_forward(pocket_tts_context* ctx,
                             const float* backbone_out, // (d_model,)
                             const float* noise,        // (latent_dim,)
                             int lsd_steps,
                             float* latent_out // (latent_dim,)
) {
    // TODO: build ggml graph
    // LSD decode loop:
    //   for step i in 0..lsd_steps:
    //     s = i / lsd_steps
    //     t = (i+1) / lsd_steps
    //     flow_dir = flow_net(backbone_out, s, t, current)
    //     current += flow_dir / lsd_steps
    //
    // flow_net(c, s, t, x):
    //   x = input_proj(x)
    //   t_combined = mean(time_embed[0](s), time_embed[1](t))
    //   c = cond_embed(c)
    //   y = t_combined + c
    //   for each res_block:
    //     shift, scale, gate = adaLN(y).chunk(3)
    //     h = modulate(LN(x), shift, scale)
    //     h = MLP(h) (Linear->SiLU->Linear)
    //     x = x + gate * h
    //   final:
    //     shift, scale = final_adaLN(y).chunk(2)
    //     x = modulate(LN_noaffine(x), shift, scale)
    //     x = linear(x)
    //   return x
    (void)ctx;
    (void)backbone_out;
    (void)noise;
    (void)lsd_steps;
    (void)latent_out;
}

// Check EOS: out_eos linear on backbone output
static bool check_eos(pocket_tts_context* ctx,
                      const float* backbone_out // (d_model,)
) {
    // out_eos(backbone_out) > threshold
    // Linear(d_model -> 1) + bias
    (void)ctx;
    (void)backbone_out;
    return false; // TODO: implement
}

// Mimi VAE decoder: latent sequence -> 24 kHz PCM
static void mimi_decode(pocket_tts_context* ctx,
                        const float* latent_seq, // (n_frames, latent_dim)
                        int n_frames, float** pcm_out, int* n_samples_out) {
    // TODO: build ggml graph
    // 1. Denormalize: latent * emb_std + emb_mean
    // 2. Quantizer projection: Conv1d(inner_dim -> outer_dim, kernel=1)
    // 3. Upsample: ConvTranspose1d(inner_dim -> outer_dim, stride=16)
    // 4. Decoder transformer (2 layers, with LayerScale)
    // 5. SEANet decoder (ratios [6,5,4]):
    //    a. Initial conv (outer_dim -> mult*n_filters)
    //    b. For each ratio: ELU -> ConvTranspose -> ResBlocks
    //    c. ELU -> final conv -> 1 channel
    // 6. Output: (samples,) at 24 kHz
    (void)ctx;
    (void)latent_seq;
    (void)n_frames;
    (void)pcm_out;
    (void)n_samples_out;
}

// Mimi VAE encoder: 24 kHz PCM -> continuous latents
// (only for voice cloning)
static void mimi_encode(pocket_tts_context* ctx, const float* pcm, int n_samples, float** latent_out,
                        int* n_frames_out) {
    // TODO: build ggml graph
    // 1. SEANet encoder (1ch -> 512, ratios reversed [4,5,6])
    // 2. Encoder transformer (2 layers)
    // 3. Downsample: Conv1d(512 -> 32, stride=16)
    // 4. Output: (n_frames, 32) continuous latents
    (void)ctx;
    (void)pcm;
    (void)n_samples;
    (void)latent_out;
    (void)n_frames_out;
}

} // namespace

// ── Public C API ───────────────────────────────────────────────────

struct pocket_tts_context_params pocket_tts_context_default_params(void) {
    return {
        /* n_threads       */ 4,
        /* verbosity       */ 1,
        /* use_gpu         */ false,
        /* temperature     */ 0.7f,
        /* seed            */ 0,
        /* lsd_decode_steps */ 1,
        /* noise_clamp     */ 3.0f,
        /* eos_threshold   */ 0.5f,
        /* max_audio_frames */ 0,
    };
}

struct pocket_tts_context* pocket_tts_init_from_file(const char* path_model, struct pocket_tts_context_params params) {
    if (!path_model)
        return nullptr;

    auto* ctx = new pocket_tts_context();
    ctx->params = params;
    ctx->verbosity = params.verbosity;

    // Seed RNG
    if (params.seed != 0) {
        ctx->rng.seed((uint32_t)params.seed);
    } else {
        std::random_device rd;
        ctx->rng.seed(rd());
    }

    // Load GGUF
    struct gguf_init_params gguf_params = {
        /* .no_alloc = */ false,
        /* .ctx      = */ nullptr,
    };

    struct gguf_context* meta = gguf_init_from_file(path_model, gguf_params);
    if (!meta) {
        fprintf(stderr, "pocket_tts: failed to load GGUF: %s\n", path_model);
        delete ctx;
        return nullptr;
    }

    // Verify architecture
    std::string arch = gguf_get_str(meta, "general.architecture", "");
    if (arch != "pocket-tts") {
        fprintf(stderr, "pocket_tts: expected arch 'pocket-tts', got '%s'\n", arch.c_str());
        gguf_free(meta);
        delete ctx;
        return nullptr;
    }

    // Load hyperparameters
    if (!load_hparams(meta, ctx->model)) {
        fprintf(stderr, "pocket_tts: failed to load hyperparameters\n");
        gguf_free(meta);
        delete ctx;
        return nullptr;
    }

    // Load tokenizer
    load_tokenizer(meta, ctx->model);

    // Load tensors from the GGML context that was allocated by gguf_init
    struct ggml_context* ggml_ctx = gguf_params.ctx;

    if (!load_flow_lm_tensors(ggml_ctx, ctx->model)) {
        fprintf(stderr, "pocket_tts: failed to load FlowLM tensors\n");
        gguf_free(meta);
        delete ctx;
        return nullptr;
    }

    load_mimi_decoder_tensors(ggml_ctx, ctx->model);
    load_mimi_encoder_tensors(ggml_ctx, ctx->model);

    if (ctx->verbosity >= 1) {
        const auto& h = ctx->model.flow_lm_hp;
        const auto& mi = ctx->model.mimi_hp;
        fprintf(stderr, "pocket_tts: loaded model\n");
        fprintf(stderr, "  FlowLM: %u layers, %u heads, %u dim, latent=%u\n", h.num_layers, h.num_heads, h.d_model,
                h.latent_dim);
        fprintf(stderr, "  Flow head: %u dim, %u depth\n", ctx->model.flow_head_hp.flow_dim,
                ctx->model.flow_head_hp.flow_depth);
        fprintf(stderr, "  Mimi: %u Hz, frame_rate=%.1f, hop=%u\n", mi.sample_rate, mi.frame_rate(), mi.hop_length());
        fprintf(stderr, "  Voice cloning: %s\n", ctx->model.has_voice_cloning ? "yes" : "no");
        fprintf(stderr, "  Tokenizer: %zu bytes\n", ctx->model.spm_model_bytes.size());
    }

    gguf_free(meta);
    // Note: ggml_ctx and tensor data are still alive via the buffer

    return ctx;
}

void pocket_tts_free(struct pocket_tts_context* ctx) {
    if (!ctx)
        return;
    // TODO: free ggml buffers
    delete ctx;
}

float* pocket_tts_synthesize(struct pocket_tts_context* ctx, const char* text, int* n_samples) {
    if (!ctx || !text || !n_samples)
        return nullptr;
    *n_samples = 0;

    // TODO: implement full pipeline:
    // 1. Tokenize text with SentencePiece
    // 2. Embed tokens via LUT
    // 3. Prefill: run text embeddings through backbone (store KV)
    // 4. AR loop:
    //    a. Input: NaN (BOS) for first step, then previous latent
    //    b. backbone_forward_step -> backbone_out
    //    c. check_eos -> stop if true + frames_after_eos
    //    d. Sample noise, flow_net_forward -> next latent
    //    e. Append to sequence
    // 5. mimi_decode(latent_sequence) -> PCM
    // 6. Return PCM

    if (ctx->verbosity >= 1) {
        fprintf(stderr, "pocket_tts: synthesize('%s') - not yet implemented\n", text);
    }

    return nullptr;
}

void pocket_tts_pcm_free(float* pcm) {
    free(pcm);
}

int pocket_tts_set_voice(struct pocket_tts_context* ctx, const float* ref_pcm_24khz, int n_ref_samples) {
    if (!ctx || !ref_pcm_24khz || n_ref_samples <= 0)
        return -1;
    if (!ctx->model.has_voice_cloning) {
        fprintf(stderr, "pocket_tts: voice cloning requires encoder weights in GGUF\n");
        return -1;
    }

    // TODO: implement
    // 1. mimi_encode(ref_pcm) -> latents (n_frames, 32)
    // 2. Project: F.linear(latents, speaker_proj_weight) -> conditioning (n_frames, d_model)
    // 3. If insert_bos_before_voice: prepend bos_before_voice
    // 4. Run conditioning through backbone to populate KV cache
    // 5. Store voice state

    return -1;
}

void pocket_tts_clear_voice(struct pocket_tts_context* ctx) {
    if (!ctx)
        return;
    ctx->has_voice_state = false;
    ctx->voice_conditioning.clear();
    ctx->n_voice_frames = 0;
}

void pocket_tts_set_temperature(struct pocket_tts_context* ctx, float temp) {
    if (ctx)
        ctx->params.temperature = temp;
}

void pocket_tts_set_seed(struct pocket_tts_context* ctx, uint64_t seed) {
    if (!ctx)
        return;
    ctx->params.seed = seed;
    if (seed != 0) {
        ctx->rng.seed((uint32_t)seed);
    }
}

int pocket_tts_sample_rate(struct pocket_tts_context* ctx) {
    if (!ctx)
        return 24000;
    return (int)ctx->model.mimi_hp.sample_rate;
}
