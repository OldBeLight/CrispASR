"""Canary-Qwen (nvidia/canary-qwen-2.5b) reference backend.

SALM architecture:
  * FastConformer encoder (32L, d=1024, 8 heads) from canary-1b-flash
  * Linear projection (1024 → 2048)
  * Qwen3-1.7B LLM decoder with merged LoRA (28L, d=2048, GQA 16/8)

This reference module uses NeMo's SALM API to capture encoder activations
and run the full generate() for the greedy decode path.

Requirements: nemo_toolkit[asr,tts] >= 2.5.0
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Set

import numpy as np

DEFAULT_STAGES = [
    "raw_audio",
    "mel_spectrogram",
    "encoder_output",      # FastConformer encoder output (T_enc, 1024)
    "projected",           # after linear projection (T_enc, 2048)
    "llm_argmax",          # greedy decoded token IDs
    "generated_text",      # decoded transcript
]


def dump(
    model_id: str,
    audio_path: str,
    output_path: str,
    stages: Set[str] | None = None,
    max_enc_layers: int = -1,
    device: str = "cpu",
) -> Dict[str, np.ndarray]:
    """Run canary-qwen inference and dump per-stage intermediates to GGUF.

    This is a stub — full implementation requires NeMo >= 2.5.0 with the
    speechlm2 module. The architecture is straightforward (FastConformer +
    linear proj + Qwen3 LLM) and hooks follow the same pattern as other
    NeMo-based reference backends.
    """
    import sys
    print("canary_qwen reference backend: requires NeMo >= 2.5.0", file=sys.stderr)
    print("Full implementation pending model availability on this VPS", file=sys.stderr)

    if stages is None:
        stages = set(DEFAULT_STAGES)

    results: Dict[str, np.ndarray] = {}

    try:
        import torch
        import soundfile as sf

        # Load audio
        audio, sr = sf.read(audio_path)
        if sr != 16000:
            import torchaudio
            audio = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
            audio = torchaudio.functional.resample(audio, sr, 16000).squeeze(0).numpy()
        if audio.ndim > 1:
            audio = audio.mean(axis=-1)
        audio = audio.astype(np.float32)

        if "raw_audio" in stages:
            results["raw_audio"] = audio

        # Try loading with NeMo SALM
        from nemo.collections.speechlm2.models import SALM
        model = SALM.from_pretrained(model_id)
        model.eval()

        # Run inference
        answer_ids = model.generate(
            prompts=[
                [{"role": "user",
                  "content": f"Transcribe the following: {model.audio_locator_tag}",
                  "audio": [audio_path]}]
            ],
            max_new_tokens=256,
        )

        if "llm_argmax" in stages:
            results["llm_argmax"] = answer_ids[0].cpu().numpy().astype(np.int32)

        if "generated_text" in stages:
            text = model.tokenizer.ids_to_text(answer_ids[0].cpu())
            # Store as uint8 bytes
            results["generated_text"] = np.frombuffer(text.encode("utf-8"), dtype=np.uint8)

    except ImportError as e:
        print(f"canary_qwen reference: missing dependency: {e}", file=sys.stderr)
        print("Install with: pip install 'nemo_toolkit[asr,tts]'", file=sys.stderr)

    return results
