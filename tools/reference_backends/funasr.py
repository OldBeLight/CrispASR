"""FunAudioLLM/Fun-ASR-Nano-2512 (and Fun-ASR-MLT-Nano-2512) reference dump.

CTC-path only: hooks `WavFrontend` → `SenseVoiceEncoderSmall` (70 SANM blocks)
→ `Transformer` adaptor used as `ctc_decoder` (5 blocks) → `CTC.log_softmax`,
captures every architectural boundary. Stage 2 (the Qwen3-0.6B LLM-decoder)
is intentionally NOT captured here — it goes on a follow-on branch.

Stages exposed (all optional; absent → silent skip):

  raw_audio                  (N,)              F32 PCM
  mel_features               (T_lfr, 560)      F32 — post-WavFrontend post-LFR
                                               (80 mels × lfr_m=7 stacked)
  encoder_layer_K   K=0..69  (T_lfr, 512)      F32 — after SANM block K
                                               (encoders0[0..0] + encoders[0..48]
                                                + tp_encoders[0..19] flattened
                                                in forward order)
  encoder_main_out           (T_lfr, 512)      F32 — after `after_norm`, i.e.
                                               between the 50-block stack and
                                               the 20-block tp stack
  encoder_output             (T_lfr, 512)      F32 — after final `tp_norm`
  ctc_decoder_layer_K K=0..4 (T_lfr, 512)      F32 — after CTC-decoder block K
  ctc_decoder_output         (T_lfr, 512)      F32 — after 5 CTC-decoder blocks
  ctc_logits                 (T_lfr, 8749)     F32 — log_softmax probabilities
                                               (NOTE: log-prob, not raw logits)
  ctc_text                   (string, written  decoded transcript via
                              as ascii bytes)  SenseVoiceTokenizer

Usage:

  HF_HOME=/Volumes/backups/ai/huggingface-hub \\
  HUGGINGFACE_HUB_CACHE=/Volumes/backups/ai/huggingface-hub \\
  python tools/dump_reference.py --backend funasr \\
      --model-dir FunAudioLLM/Fun-ASR-Nano-2512 \\
      --audio samples/jfk.wav \\
      --output /Volumes/backups/ai/huggingface-hub/funasr/funasr-ref.gguf

`--model-dir` accepts either a HuggingFace repo id (downloads to HF cache)
or a local path containing `model.pt`, `multilingual.tiktoken`, and
`Qwen3-0.6B/` (the LLM weights — not used by the CTC path but loaded for
state-dict completeness).

Acceptance bar (all elementwise vs C++):
  mel_features      cos >= 0.999 (frontend should be bit-near-exact)
  encoder_layer_K   cos >= 0.99  (F16 weights, 70 SANM blocks accumulate)
  ctc_logits        cos >= 0.98
  greedy CTC text   byte-identical
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Set

import numpy as np


DEFAULT_STAGES = [
    "raw_audio",
    "mel_features",
    "encoder_main_out",
    "encoder_output",
    "ctc_decoder_output",
    "ctc_logits",
    "ctc_text",
] + [f"encoder_layer_{i}" for i in range(70)] + [f"ctc_decoder_layer_{i}" for i in range(5)]


def _ensure_fun_asr_nano_importable() -> None:
    """funasr 1.3.1's fun_asr_nano/model.py uses `from ctc import CTC` instead
    of the relative form. Without this shim the import fails with
    `ModuleNotFoundError: No module named 'ctc'`. Prepending the package
    directory to sys.path makes the (mis-spelt) absolute import resolve to
    the same ctc.py we'd want anyway. Upstream packaging bug; remove this
    workaround when funasr ships a relative import.
    """
    import funasr

    fan = Path(funasr.__file__).parent / "models" / "fun_asr_nano"
    s = str(fan)
    if s not in sys.path:
        sys.path.insert(0, s)


def dump(*, model_dir: Path, audio: np.ndarray, stages: Set[str],
         max_new_tokens: int) -> Dict[str, np.ndarray]:
    """Load Fun-ASR-Nano-2512 (or MLT variant), run the CTC path on `audio`,
    return captured activations as a dict[str, np.ndarray].

    The LLM half (Qwen3-0.6B autoregressive decode) is loaded into memory
    by `from_pretrained` but never invoked here — this dumper is the
    ground-truth reference for stage 1 (CTC) only.
    """
    import torch

    _ensure_fun_asr_nano_importable()
    from funasr.models.fun_asr_nano.model import FunASRNano
    from funasr.frontends.wav_frontend import WavFrontend

    pretrained = str(model_dir)
    print(f"  loading FunASRNano from {pretrained}")
    m, kwargs = FunASRNano.from_pretrained(model=pretrained, device="cpu")
    m.eval()

    # Build a deterministic WavFrontend matching upstream config.yaml. The
    # one the model loaded came from frontend_conf, which has dither=1.0 by
    # default — bad for reference dumping. Override to dither=0.
    fe = WavFrontend(
        cmvn_file=None,
        fs=16000,
        window="hamming",
        n_mels=80,
        frame_length=25,
        frame_shift=10,
        lfr_m=7,
        lfr_n=6,
        dither=0.0,
        upsacle_samples=True,  # sic — upstream typo, multiplies waveform by 32768
        snip_edges=True,
    )
    fe.eval()

    sig = torch.from_numpy(audio.astype(np.float32))[None, :]
    sig_len = torch.tensor([audio.shape[0]])

    out: Dict[str, np.ndarray] = {}
    if "raw_audio" in stages:
        out["raw_audio"] = audio.astype(np.float32)

    # Hook plumbing
    from . import _hooks
    captured: Dict = {}
    stage_modules = []

    enc = m.audio_encoder
    # Flatten the (encoders0 + encoders) chain so layer indexing is linear
    # forward-order. encoders0 has exactly 1 block (the in_size != size
    # entry block with NO residual on the attn branch — see EncoderLayerSANM
    # forward). encoders has num_blocks - 1 blocks. After those, after_norm
    # fires, then tp_encoders (tp_blocks blocks), then tp_norm.
    base_layers = list(enc.encoders0) + list(enc.encoders)
    for i, layer in enumerate(base_layers):
        n = f"encoder_layer_{i}"
        if n in stages:
            stage_modules.append((n, layer))
    n_base = len(base_layers)
    tp_layers = list(getattr(enc, "tp_encoders", []))
    for j, layer in enumerate(tp_layers):
        n = f"encoder_layer_{n_base + j}"
        if n in stages:
            stage_modules.append((n, layer))
    # after_norm / tp_norm: capture their outputs as named stages.
    if "encoder_main_out" in stages and hasattr(enc, "after_norm"):
        stage_modules.append(("encoder_main_out", enc.after_norm))

    cd = getattr(m, "ctc_decoder", None)
    if cd is not None and cd.blocks is not None:
        for i, b in enumerate(cd.blocks):
            n = f"ctc_decoder_layer_{i}"
            if n in stages:
                stage_modules.append((n, b))

    handles = _hooks.capture_modules(captured, stage_modules)

    with torch.no_grad():
        feats, feats_lens = fe(sig, sig_len)
        # feats: (1, T_lfr, 560)
        T_lfr = int(feats_lens.item())
        if "mel_features" in stages:
            out["mel_features"] = feats[0, :T_lfr].cpu().float().numpy()

        encoder_out, encoder_out_lens = m.audio_encoder(feats, feats_lens)
        T_enc = int(encoder_out_lens.item())
        if "encoder_output" in stages:
            out["encoder_output"] = encoder_out[0, :T_enc].cpu().float().numpy()

        if cd is not None:
            decoder_out, decoder_out_lens = m.ctc_decoder(encoder_out, encoder_out_lens)
            T_dec = int(decoder_out_lens.item())
            if "ctc_decoder_output" in stages:
                out["ctc_decoder_output"] = decoder_out[0, :T_dec].cpu().float().numpy()

            ctc_logits = m.ctc.log_softmax(decoder_out)
            if "ctc_logits" in stages:
                out["ctc_logits"] = ctc_logits[0, :T_dec].cpu().float().numpy()

            if "ctc_text" in stages:
                yseq = ctc_logits[0, :T_dec].argmax(dim=-1)
                yseq = torch.unique_consecutive(yseq, dim=-1)
                blank_id = getattr(m, "blank_id", m.ctc.blank_id)
                ids = yseq[yseq != blank_id].tolist()
                txt = m.ctc_tokenizer.decode(ids) if hasattr(m, "ctc_tokenizer") else ""
                # GGUF doesn't have a string-tensor type — encode as
                # uint8 ascii bytes so the diff harness can read it
                # alongside the float tensors.
                out["ctc_text"] = np.frombuffer(txt.encode("utf-8"), dtype=np.uint8).copy()

    _hooks.drop_hooks(handles)
    out.update(_hooks.finalize(captured, T_max=T_lfr))
    return out
