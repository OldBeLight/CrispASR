#!/usr/bin/env python3
"""
Parakeet-Unified-EN-0.6B: survey architecture + attempt GGUF conversion.

This kernel:
1. Loads the NeMo 2.x model via ASRModel.from_pretrained
2. Dumps the config (hparams, layer shapes, vocab)
3. Extracts state_dict keys + shapes
4. Runs inference on JFK to capture the reference transcript
5. Attempts conversion using the existing parakeet converter
"""
import json, os, subprocess, sys, time, traceback
from pathlib import Path

WORK = Path("/kaggle/working")
results = {}

def log(msg):
    print(msg, flush=True)
    try:
        with open(WORK / "progress.txt", "a") as f:
            f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
    except Exception:
        pass

def save():
    try:
        (WORK / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    except Exception:
        pass

def main():
    global results
    save()
    log("=== Parakeet-Unified survey ===")

    # HF token
    for p in ["/kaggle/input/crispasr-hf-token/hf_token.txt",
              "/kaggle/input/datasets/chr1s4/crispasr-hf-token/hf_token.txt"]:
        if os.path.exists(p):
            os.environ["HF_TOKEN"] = open(p).read().strip()
            break

    # Install NeMo if needed
    try:
        import nemo
        log(f"NeMo version: {nemo.__version__}")
    except ImportError:
        log("Installing NeMo...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "nemo_toolkit[asr]"], check=False)

    # Step 1: Load model
    log("\n[1] Loading parakeet-unified-en-0.6b via NeMo...")
    try:
        import torch
        import nemo.collections.asr as nemo_asr

        model = nemo_asr.models.ASRModel.from_pretrained(
            "nvidia/parakeet-unified-en-0.6b", map_location="cpu"
        )
        model.eval()
        log(f"  Model class: {type(model).__name__}")
        log(f"  Model type: {type(model).__mro__}")

        # Step 2: Dump config
        log("\n[2] Model config:")
        if hasattr(model, 'cfg'):
            cfg = model.cfg
            results["config"] = {}
            for key in ["encoder", "decoder", "joint", "preprocessor", "tokenizer"]:
                if hasattr(cfg, key):
                    try:
                        results["config"][key] = str(getattr(cfg, key))[:500]
                        log(f"  {key}: {str(getattr(cfg, key))[:200]}")
                    except Exception:
                        pass

            # Key architecture params
            if hasattr(cfg, "encoder"):
                enc = cfg.encoder
                for attr in ["d_model", "n_heads", "n_layers", "ff_expansion_factor",
                             "subsampling_factor", "subsampling", "conv_kernel_size",
                             "att_context_size", "att_context_style"]:
                    if hasattr(enc, attr):
                        val = getattr(enc, attr)
                        results.setdefault("encoder_params", {})[attr] = str(val)
                        log(f"  encoder.{attr} = {val}")

        # Step 3: State dict keys + shapes
        log("\n[3] State dict:")
        sd = model.state_dict()
        results["state_dict_keys"] = {}
        for i, (k, v) in enumerate(sorted(sd.items())):
            shape_str = str(list(v.shape)) if hasattr(v, 'shape') else "?"
            dtype_str = str(v.dtype) if hasattr(v, 'dtype') else "?"
            results["state_dict_keys"][k] = f"{shape_str} {dtype_str}"
            if i < 30 or "embed" in k or "joint" in k or "pred" in k:
                log(f"  {k}: {shape_str} {dtype_str}")
        log(f"  ... {len(sd)} total keys")
        results["n_keys"] = len(sd)
        save()

        # Step 4: Vocab
        log("\n[4] Vocab:")
        if hasattr(model, 'tokenizer') and hasattr(model.tokenizer, 'vocab_size'):
            results["vocab_size"] = model.tokenizer.vocab_size
            log(f"  vocab_size = {model.tokenizer.vocab_size}")

        # Step 5: Run inference
        log("\n[5] NeMo inference on JFK...")
        # Clone CrispASR for the JFK sample
        cdir = WORK / "CrispASR"
        if not cdir.exists():
            subprocess.check_call(["git", "clone", "--depth", "1",
                "https://github.com/CrispStrobe/CrispASR.git", str(cdir)])
        jfk = str(cdir / "samples" / "jfk.wav")

        import soundfile as sf
        audio, sr = sf.read(jfk)
        if sr != 16000:
            import torchaudio
            audio_t = torch.from_numpy(audio).unsqueeze(0).float()
            audio = torchaudio.functional.resample(audio_t, sr, 16000).squeeze(0).numpy()

        transcripts = model.transcribe([jfk])
        if isinstance(transcripts, list) and transcripts:
            t = transcripts[0] if isinstance(transcripts[0], str) else str(transcripts[0])
            results["nemo_transcript"] = t
            log(f"  Transcript: {t}")
        save()

        # Step 6: Try existing converter
        log("\n[6] Attempting GGUF conversion...")
        converter = str(cdir / "models" / "convert-parakeet-to-gguf.py")
        out_gguf = str(WORK / "parakeet-unified-en-0.6b-f16.gguf")
        r = subprocess.run([sys.executable, converter,
                           "--nemo", "nvidia/parakeet-unified-en-0.6b",
                           "--output", out_gguf],
                          capture_output=True, text=True, timeout=600)
        results["converter_rc"] = r.returncode
        results["converter_stderr"] = r.stderr[-1000:]
        if r.returncode == 0:
            log(f"  Conversion OK: {out_gguf}")
            results["gguf_size_mb"] = os.path.getsize(out_gguf) / (1024*1024)
        else:
            log(f"  Conversion failed rc={r.returncode}")
            log(f"  stderr: {r.stderr[-500:]}")

    except Exception as e:
        results["error"] = str(e)
        results["traceback"] = traceback.format_exc()
        log(f"ERROR: {e}")
        traceback.print_exc()

    save()
    log("\nDONE")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        results["_crash"] = str(e)
        save()
        traceback.print_exc()
