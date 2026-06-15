#!/usr/bin/env python3
"""
Parakeet-Unified-EN-0.6B: convert to GGUF via monkey-patched NeMo.

The model uses `att_chunk_context_size` which NeMo 2.7.x doesn't support.
We monkey-patch ConformerEncoder.__init__ to accept+ignore unknown kwargs,
then use the existing CrispASR converter.
"""
import json, os, subprocess, sys, time, traceback, shutil
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
    log("=== Parakeet-Unified conversion (monkey-patch) ===")

    # HF token
    for p in ["/kaggle/input/crispasr-hf-token/hf_token.txt",
              "/kaggle/input/datasets/chr1s4/crispasr-hf-token/hf_token.txt"]:
        if os.path.exists(p):
            tok = open(p).read().strip()
            os.environ["HF_TOKEN"] = tok
            os.environ["HUGGING_FACE_HUB_TOKEN"] = tok
            break

    # Clone CrispASR (for converter + JFK sample)
    cdir = Path("/tmp/CrispASR")
    if not cdir.exists():
        subprocess.check_call(["git", "clone", "--depth", "1",
            "https://github.com/CrispStrobe/CrispASR.git", str(cdir)])

    # Build CrispASR (for testing the GGUF)
    log("Building CrispASR...")
    if not shutil.which("cmake"):
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "cmake", "ninja"], check=False)
    subprocess.run("apt-get update -qq && apt-get install -y cmake ninja-build g++ 2>/dev/null || true",
                   shell=True, capture_output=True)
    bdir = cdir / "build"
    cmake_args = ["-DCMAKE_BUILD_TYPE=Release"]
    if shutil.which("ninja"): cmake_args += ["-G", "Ninja"]
    subprocess.check_call(["cmake", "-B", str(bdir)] + cmake_args, cwd=str(cdir))
    subprocess.check_call(["cmake", "--build", str(bdir), "-j2"], cwd=str(cdir))
    log("Build OK")

    # Install NeMo + deps
    log("Installing NeMo...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "nemo_toolkit[asr]", "gguf", "soundfile"], check=False)

    # Monkey-patch ConformerEncoder to accept unknown kwargs
    log("Monkey-patching ConformerEncoder...")
    import nemo.collections.asr.modules.conformer_encoder as ce
    _orig_init = ce.ConformerEncoder.__init__

    def _patched_init(self, *args, **kwargs):
        # Remove unknown kwargs that older NeMo doesn't support
        unknown = []
        import inspect
        valid = set(inspect.signature(_orig_init).parameters.keys())
        for k in list(kwargs.keys()):
            if k not in valid and k != "self":
                unknown.append((k, kwargs.pop(k)))
        if unknown:
            log(f"  Stripped unknown ConformerEncoder kwargs: {[k for k,v in unknown]}")
        return _orig_init(self, *args, **kwargs)

    ce.ConformerEncoder.__init__ = _patched_init

    # Load model
    log("Loading parakeet-unified-en-0.6b...")
    import torch
    import nemo.collections.asr as nemo_asr

    # Try concrete class first, then generic
    HybridCls = getattr(nemo_asr.models, "EncDecRNNTBPEModel", None)
    model = None
    for cls in [nemo_asr.models.ASRModel, HybridCls]:
        if cls is None:
            continue
        try:
            model = cls.from_pretrained("nvidia/parakeet-unified-en-0.6b", map_location="cpu")
            log(f"  Loaded via {cls.__name__}")
            break
        except Exception as e:
            log(f"  {cls.__name__} failed: {str(e)[:200]}")

    if model is None:
        # Last resort: restore_from with the .nemo file
        from huggingface_hub import hf_hub_download
        nemo_path = hf_hub_download("nvidia/parakeet-unified-en-0.6b",
                                     "parakeet-unified-en-0.6b.nemo",
                                     cache_dir=str(WORK / ".hf"))
        for cls in [HybridCls, nemo_asr.models.ASRModel]:
            if cls is None:
                continue
            try:
                model = cls.restore_from(nemo_path, map_location="cpu")
                log(f"  Restored via {cls.__name__}")
                break
            except Exception as e:
                log(f"  restore {cls.__name__} failed: {str(e)[:200]}")

    if model is None:
        results["error"] = "Could not load model"
        save()
        return

    model.eval()
    results["model_class"] = type(model).__name__

    # Dump config
    log("Dumping config...")
    if hasattr(model, 'cfg') and hasattr(model.cfg, 'encoder'):
        enc = model.cfg.encoder
        for attr in ["d_model", "n_heads", "n_layers", "ff_expansion_factor",
                     "subsampling_factor", "subsampling", "conv_kernel_size",
                     "att_context_size", "att_context_style", "att_chunk_context_size"]:
            if hasattr(enc, attr):
                val = getattr(enc, attr)
                results.setdefault("encoder_params", {})[attr] = str(val)
                log(f"  encoder.{attr} = {val}")

    # State dict summary
    sd = model.state_dict()
    results["n_keys"] = len(sd)
    log(f"  {len(sd)} state dict keys")

    # Inference on JFK
    log("NeMo inference on JFK...")
    jfk = str(cdir / "samples" / "jfk.wav")
    try:
        transcripts = model.transcribe([jfk])
        t = transcripts[0] if isinstance(transcripts[0], str) else str(transcripts[0])
        results["nemo_transcript"] = t
        log(f"  NeMo transcript: {t}")
    except Exception as e:
        results["inference_error"] = str(e)
        log(f"  Inference error: {e}")

    # Try converter
    log("Running convert-parakeet-to-gguf.py...")
    converter = str(cdir / "models" / "convert-parakeet-to-gguf.py")
    out_f16 = str(WORK / "parakeet-unified-en-0.6b-f16.gguf")
    r = subprocess.run([sys.executable, converter,
                       "--nemo", "nvidia/parakeet-unified-en-0.6b",
                       "--output", out_f16],
                      capture_output=True, text=True, timeout=600,
                      env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    results["converter_rc"] = r.returncode
    results["converter_stderr"] = r.stderr[-1000:]
    log(f"  Converter rc={r.returncode}")
    if r.returncode != 0:
        log(f"  stderr: {r.stderr[-500:]}")
    else:
        sz = os.path.getsize(out_f16) / (1024*1024)
        results["gguf_size_mb"] = round(sz, 1)
        log(f"  GGUF: {sz:.1f} MB")

        # Test with CrispASR
        log("Testing GGUF with CrispASR...")
        CLI = str(bdir / "bin" / "crispasr")
        r2 = subprocess.run([CLI, "--backend", "parakeet", "-m", out_f16, "-f", jfk, "-np"],
                           capture_output=True, text=True, timeout=300)
        lines = [l.strip() for l in r2.stdout.strip().split('\n') if l.strip()]
        transcript = lines[-1] if lines else ""
        results["crispasr_rc"] = r2.returncode
        results["crispasr_transcript"] = transcript
        log(f"  CrispASR rc={r2.returncode}: {transcript[:80]}")

        # Quantize Q4_K
        if r2.returncode == 0:
            log("Quantizing Q4_K...")
            QUANT = str(bdir / "bin" / "crispasr-quantize")
            out_q4k = str(WORK / "parakeet-unified-en-0.6b-q4_k.gguf")
            subprocess.run([QUANT, out_f16, out_q4k, "q4_k"],
                          capture_output=True, timeout=300)
            if os.path.exists(out_q4k):
                results["q4k_size_mb"] = round(os.path.getsize(out_q4k)/(1024*1024), 1)
                log(f"  Q4_K: {results['q4k_size_mb']} MB")

                # Upload to HF
                log("Uploading to HF...")
                try:
                    from huggingface_hub import HfApi
                    api = HfApi(token=os.environ.get("HF_TOKEN"))
                    for fname in [out_f16, out_q4k]:
                        api.upload_file(
                            path_or_fileobj=fname,
                            path_in_repo=os.path.basename(fname),
                            repo_id="cstr/parakeet-unified-en-0.6b-GGUF",
                            commit_message=f"Add {os.path.basename(fname)}",
                            repo_type="model",
                        )
                        log(f"  Uploaded {os.path.basename(fname)}")
                except Exception as e:
                    results["upload_error"] = str(e)
                    log(f"  Upload error: {e}")

    save()
    log("\nDONE")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        results["_crash"] = str(e)
        results["_tb"] = traceback.format_exc()
        save()
        log(f"CRASH: {e}")
        traceback.print_exc()

# Cleanup: remove large files so kernels_output only has results.json + progress.txt
import shutil
for p in [Path("/tmp/CrispASR"), WORK / "models", WORK / ".hf"]:
    shutil.rmtree(str(p), ignore_errors=True)
for f in WORK.glob("*.gguf"):
    f.unlink(missing_ok=True)
