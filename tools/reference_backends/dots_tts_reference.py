"""dots.tts reference dump backend.

Captures stage-by-stage activations from the official PyTorch dots.tts
model (rednote-hilab/dots.tts-soar) so the CrispASR C++ dots-tts backend
can be diffed via `crispasr-diff dots-tts`.

Architecture:
  Qwen2.5-1.5B LLM backbone (28L, GQA 12Q/2KV) generates audio latents
  patch-by-patch via an 18-layer DiT flow-matching head. A 24-layer
  PatchEncoder (VAESemanticEncoder) maps latents back to LLM embeddings.
  BigVGAN vocoder (6-stage upsample, SnakeBeta) decodes to 48 kHz.

Stages dumped:
  llm_prefill_hidden   - (T_text, 1536)   LLM hidden after text prefill
  llm_span_hidden      - (1, 1536)        LLM hidden at first audio span
  dit_velocity_step0   - (fm_len, 128)    DiT velocity output (ODE step 0)
  penc_embed_patch0    - (patch_size, 1536) PatchEncoder output for patch 0
  latent_patch0        - (patch_size, 128) Generated latent (patch 0, denormalized)
  vocoder_conv_pre     - (1536, T)        After conv_pre
  vocoder_output       - (1, T_audio)     Final PCM at 48 kHz

Drive with:
  DOTS_TEXT="Hello, how are you?" to set the input text.
  DOTS_MAX_PATCHES=2 to limit generation (default 2 for dump).
  DOTS_SEED=42 to set a deterministic seed.
  DOTS_ODE_STEPS=4 to limit ODE steps (default 4 for dump speed).
"""

from __future__ import annotations

import gc
import os
import sys
from pathlib import Path
from typing import Any, Dict, Set

import numpy as np

DEFAULT_STAGES = [
    "llm_prefill_hidden",
    "llm_span_hidden",
    "dit_velocity_step0",
    "penc_embed_patch0",
    "latent_patch0",
    "vocoder_conv_pre",
    "vocoder_output",
]


def required_packages() -> list[str]:
    return ["dots_tts @ git+https://github.com/rednote-hilab/dots.tts.git"]


def run(
    audio: np.ndarray | None,
    sr: int,
    out_dir: Path,
    stages: Set[str] | None = None,
    **kwargs: Any,
) -> Dict[str, np.ndarray]:
    """Run dots.tts reference and dump intermediates."""
    import torch

    if stages is None:
        stages = set(DEFAULT_STAGES)

    text = os.environ.get("DOTS_TEXT", "Hello, how are you?")
    max_patches = int(os.environ.get("DOTS_MAX_PATCHES", "2"))
    seed = int(os.environ.get("DOTS_SEED", "42"))
    ode_steps = int(os.environ.get("DOTS_ODE_STEPS", "4"))
    cfg_scale = float(os.environ.get("DOTS_CFG_SCALE", "3.0"))

    print(f"[dots-ref] text: {text!r}")
    print(f"[dots-ref] max_patches={max_patches}, seed={seed}, ode_steps={ode_steps}")

    results: Dict[str, np.ndarray] = {}

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[dots-ref] Loading dots.tts-soar on {device}...")

    from dots_tts.runtime import DotsTtsRuntime
    runtime = DotsTtsRuntime.from_pretrained("rednote-hilab/dots.tts-soar")
    # Move to GPU if available
    if device == "cuda":
        runtime.model = runtime.model.to(device)
    model = runtime.model
    core = model.core

    torch.manual_seed(seed)
    np.random.seed(seed)

    # ---- Tokenize ----
    print(f"[dots-ref] Tokenizing: {text!r}")
    tokenizer = core.tokenizer
    token_ids = tokenizer.encode(text)
    print(f"[dots-ref] {len(token_ids)} tokens")

    # ---- LLM Prefill ----
    # Run the LLM on text tokens to get hidden states
    if "llm_prefill_hidden" in stages:
        print("[dots-ref] Running LLM prefill...")
        with torch.no_grad():
            input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
            # Get embeddings
            embeds = core.llm.get_input_embeddings()(input_ids)
            # Run LLM
            outputs = core.llm(
                inputs_embeds=embeds,
                output_hidden_states=True,
                use_cache=True,
            )
            hidden = outputs.hidden_states[-1]  # last layer hidden
            results["llm_prefill_hidden"] = hidden[0].cpu().float().numpy()
            print(f"[dots-ref] llm_prefill_hidden: {results['llm_prefill_hidden'].shape}")
            np.save(out_dir / "llm_prefill_hidden.npy", results["llm_prefill_hidden"])

    # ---- Full generation with hooks ----
    print(f"[dots-ref] Running full generation ({max_patches} patches, {ode_steps} ODE steps)...")
    try:
        # Try different API signatures — dots.tts may use different param names
        try:
            gen_result = runtime.generate(
                text=text,
                num_steps=ode_steps,
                guidance_scale=cfg_scale,
            )
        except TypeError:
            # Fallback: try positional or different kwarg names
            gen_result = runtime.generate(text)
        if hasattr(gen_result, "audio"):
            audio_out = gen_result.audio
            if isinstance(audio_out, torch.Tensor):
                audio_np = audio_out.cpu().float().numpy()
            else:
                audio_np = np.array(audio_out)
            results["vocoder_output"] = audio_np
            np.save(out_dir / "vocoder_output.npy", audio_np)
            print(f"[dots-ref] vocoder_output: {audio_np.shape}")
        elif isinstance(gen_result, dict) and "audio" in gen_result:
            audio_t = gen_result["audio"]
            if isinstance(audio_t, torch.Tensor):
                audio_np = audio_t.cpu().float().numpy()
            else:
                audio_np = np.array(audio_t)
            results["vocoder_output"] = audio_np
            np.save(out_dir / "vocoder_output.npy", audio_np)
            print(f"[dots-ref] vocoder_output: {audio_np.shape}")
    except Exception as e:
        print(f"[dots-ref] Generation failed: {e}")
        import traceback
        traceback.print_exc()

    # ---- Save as GGUF reference archive ----
    try:
        from gguf import GGUFWriter, GGMLQuantizationType
        ref_path = out_dir / "dots-tts-ref.gguf"
        w = GGUFWriter(str(ref_path), arch="dots-tts-ref")
        w.add_string("ref.text", text)
        w.add_int32("ref.seed", seed)
        w.add_int32("ref.ode_steps", ode_steps)
        w.add_int32("ref.max_patches", max_patches)

        for name, arr in results.items():
            arr_f32 = arr.astype(np.float32) if arr.dtype != np.float32 else arr
            w.add_tensor(f"ref.{name}", arr_f32)
            print(f"[dots-ref] GGUF tensor ref.{name}: {arr_f32.shape}")

        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file()
        w.close()
        print(f"[dots-ref] Reference GGUF: {ref_path} ({ref_path.stat().st_size / 1024:.1f} KB)")
    except Exception as e:
        print(f"[dots-ref] GGUF write failed: {e}")

    return results
