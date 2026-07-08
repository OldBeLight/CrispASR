#!/usr/bin/env python3
"""CrispASR canary-qwen backend validation on Kaggle (GPU, 13 GB RAM).

Builds from the feature branch, converts nvidia/canary-qwen-2.5b to GGUF,
quantizes to Q4_K, and runs transcription on jfk.wav to verify the full
pipeline end-to-end.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except (AttributeError, ValueError):
    pass

WORK = Path("/kaggle/working")
REPO = WORK / "CrispASR"
BUILD = WORK / "build"
PROGRESS = WORK / "progress.jsonl"
T0 = time.time()


def _step(name, **extra):
    """Local step logger that works even before kh is imported."""
    rec = {"ts": round(time.time() - T0, 2), "step": name, **extra}
    print(f"[step {rec['ts']:>7.1f}s] {name}" + (f"  {extra}" if extra else ""),
          flush=True)
    try:
        with PROGRESS.open("a") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:
        pass


def sh(cmd, cwd=None):
    print(f"$ {cmd}", flush=True)
    subprocess.check_call(cmd, shell=True, cwd=str(cwd) if cwd else None)


_step("start")

try:
    # ── Bootstrap: clone CrispASR from feature branch ─────────────────
    CRISPASR_URL = "https://github.com/CrispStrobe/CrispASR.git"
    BRANCH = "worktree-feat-canary-qwen"

    _step("clone")
    if not REPO.exists():
        try:
            subprocess.check_call(
                ["git", "clone", "--recursive", "--branch", BRANCH,
                 CRISPASR_URL, str(REPO)])
            sys.path.insert(0, str(REPO / "tools" / "kaggle"))
        except Exception as e:
            _step("clone_failed", error=str(e))
            raise
    else:
        sh(f"git fetch origin && git checkout {BRANCH}", cwd=REPO)
        sh("git submodule update --init --recursive", cwd=REPO)

    # Import harness — fall back to bundled copy
    if str(REPO / "tools" / "kaggle") not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
    import kaggle_harness as kh
    kh.init_progress()

    kh.step("harness_loaded")

    # ── HF auth via harness (3-tier: secret → dataset → env) ─────────
    kh.step("hf_auth")
    hf_token = kh.kaggle_secret("HF_TOKEN")
    if not hf_token:
        hf_token = kh.kaggle_token_from_dataset("hf_token.txt")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
        print(f"HF token loaded ({len(hf_token)} chars)", flush=True)
    else:
        print("WARNING: no HF token — model download may fail", flush=True)

    # ── Install build toolchain via harness ───────────────────────────
    kh.step("install_toolchain")
    # Also install libopenblas-dev which the harness doesn't include
    subprocess.run("apt-get update -qq && apt-get install -y --no-install-recommends "
                   "libopenblas-dev 2>/dev/null || true", shell=True)
    kh.install_build_toolchain()

    # ── pip deps (small only — never pip install torch) ───────────────
    kh.step("pip_deps")
    subprocess.run("pip install -q safetensors gguf huggingface_hub",
                   shell=True, capture_output=True)

    # ── nvidia-smi for diagnostics ────────────────────────────────────
    subprocess.run("nvidia-smi", shell=True)

    # ── Build flags via harness ───────────────────────────────────────
    kh.step("build_configure")
    build_flags = kh.cache_and_link_flags()
    try:
        arch = kh.detect_cuda_arch()
        build_flags += kh.cuda_build_flags(arch)
        print(f"CUDA arch: {arch}", flush=True)
    except Exception:
        print("No CUDA detected, building CPU-only", flush=True)

    sh(f"cmake -S {REPO} -B {BUILD} -G Ninja "
       f"-DCMAKE_BUILD_TYPE=Release "
       f"-DCRISPASR_BUILD_TESTS=OFF "
       f"-DCRISPASR_BUILD_EXAMPLES=ON "
       f"-DCRISPASR_BUILD_SERVER=OFF "
       + " ".join(build_flags))

    # ── Build with heartbeat ──────────────────────────────────────────
    kh.step("build_compile")
    n_jobs = kh.safe_build_jobs(gpu=True)
    with kh.build_heartbeat("cmake.build"):
        sh(f"stdbuf -oL -eL cmake --build {BUILD} "
           f"--target crispasr-cli crispasr-quantize -j{n_jobs}")

    CRISPASR_BIN = BUILD / "bin" / "crispasr"
    QUANTIZE_BIN = BUILD / "bin" / "crispasr-quantize"
    assert CRISPASR_BIN.exists(), f"crispasr not found at {CRISPASR_BIN}"
    kh.step("build_done")

    # ── ccache stats ──────────────────────────────────────────────────
    subprocess.run("ccache -s 2>/dev/null | grep -E 'hit|miss|size'",
                   shell=True)

    # ── Download model from HF ───────────────────────────────────────
    kh.step("download_model")
    MODEL_DIR = WORK / "canary-qwen-hf"
    GGUF_F16 = WORK / "canary-qwen-2.5b-f16.gguf"
    GGUF_Q4K = WORK / "canary-qwen-2.5b-q4_k.gguf"

    from huggingface_hub import snapshot_download, hf_hub_download
    snapshot_download("nvidia/canary-qwen-2.5b",
                      local_dir=str(MODEL_DIR),
                      token=os.environ.get("HF_TOKEN"))
    kh.step("model_downloaded", free_gb=kh.free_gb())

    # The canary-qwen model doesn't ship a tokenizer.json — it uses the
    # Qwen3 tokenizer. Download it into the model dir so the converter
    # finds it.
    kh.step("download_tokenizer")
    for tok_file in ("tokenizer.json", "tokenizer_config.json"):
        if not (MODEL_DIR / tok_file).exists():
            hf_hub_download("Qwen/Qwen3-1.7B", tok_file,
                            local_dir=str(MODEL_DIR))
    kh.step("tokenizer_ready")

    # ── Convert to GGUF ──────────────────────────────────────────────
    kh.step("convert_to_gguf")
    os.environ["TMPDIR"] = str(WORK / "tmp")
    (WORK / "tmp").mkdir(exist_ok=True)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    sh(f"python {REPO}/models/convert-canary-qwen-to-gguf.py "
       f"--input {MODEL_DIR} --output {GGUF_F16}")
    kh.step("convert_done",
            size_gb=round(GGUF_F16.stat().st_size / 1e9, 2),
            free_gb=kh.free_gb())

    # ── Quantize to Q4_K ─────────────────────────────────────────────
    kh.step("quantize")
    sh(f"{QUANTIZE_BIN} {GGUF_F16} {GGUF_Q4K} q4_k")
    kh.step("quantize_done",
            size_gb=round(GGUF_Q4K.stat().st_size / 1e9, 2))

    # Free disk space
    GGUF_F16.unlink(missing_ok=True)
    shutil.rmtree(str(MODEL_DIR), ignore_errors=True)
    kh.step("cleanup_done", free_gb=kh.free_gb())

    # ── Transcribe jfk.wav ───────────────────────────────────────────
    kh.step("transcribe_jfk")
    JFK_WAV = REPO / "samples" / "jfk.wav"
    assert JFK_WAV.exists(), f"jfk.wav not found at {JFK_WAV}"

    result = subprocess.run(
        [str(CRISPASR_BIN), "--backend", "canary-qwen",
         "-m", str(GGUF_Q4K),
         "-f", str(JFK_WAV),
         "-t", "4", "-l", "en",
         "--no-prints"],
        capture_output=True, text=True, timeout=600,
        env={**os.environ,
             "CANARY_QWEN_BENCH": "1",
             "CRISPASR_CANARY_QWEN_DEBUG": "1"})

    print("=== STDOUT ===", flush=True)
    print(result.stdout, flush=True)
    print("=== STDERR ===", flush=True)
    print(result.stderr, flush=True)
    print(f"=== EXIT CODE: {result.returncode} ===", flush=True)
    kh.step("transcribe_done", exit_code=result.returncode)

    # ── Write results ────────────────────────────────────────────────
    with open(WORK / "result.txt", "w") as f:
        f.write(f"exit_code={result.returncode}\n")
        f.write(f"stdout={result.stdout}\n")
        f.write(f"stderr={result.stderr[-2000:]}\n")

    # ── Validate transcript ──────────────────────────────────────────
    JFK_EXPECTED = "and so my fellow americans ask not what your country can do for you ask what you can do for your country"
    transcript = result.stdout.strip().lower()
    transcript_clean = re.sub(r'\[.*?\]', '', transcript).strip()
    transcript_clean = re.sub(r'\d{2}:\d{2}:\d{2}.*?-->', '', transcript_clean).strip()
    transcript_clean = re.sub(r'\s+', ' ', transcript_clean).strip()
    transcript_clean = re.sub(r'[^\w\s]', '', transcript_clean).strip()

    kh.step("validation",
            transcript=transcript_clean[:200],
            expected=JFK_EXPECTED[:80])

    if JFK_EXPECTED in transcript_clean:
        print("\n✓ PASS — transcript matches expected JFK reference", flush=True)
        kh.step("PASS")
    elif result.returncode == 0 and len(transcript_clean) > 10:
        print(f"\n~ PARTIAL — got output but doesn't match exactly:", flush=True)
        print(f"  got:      {transcript_clean[:200]}", flush=True)
        print(f"  expected: {JFK_EXPECTED}", flush=True)
        kh.step("PARTIAL", got=transcript_clean[:200])
    else:
        print(f"\n✗ FAIL — exit {result.returncode}, len {len(transcript_clean)}",
              flush=True)
        kh.step("FAIL", exit_code=result.returncode)

except Exception as exc:
    tb = traceback.format_exc()
    print(f"\n=== EXCEPTION ===\n{tb}", flush=True)
    _step("EXCEPTION", error=str(exc), traceback=tb[-1000:])
    # Write traceback to a file so kernels_output can retrieve it
    with open(WORK / "error.txt", "w") as f:
        f.write(tb)

# ── Save ccache for future runs ──────────────────────────────────────
_step("save_ccache")
try:
    os.chdir(str(WORK))
    subprocess.run("tar cf ccache.tar .ccache/", shell=True, check=True)
except Exception:
    pass

_step("done", total_s=round(time.time() - T0, 1))
print(f"\nTotal runtime: {time.time() - T0:.0f}s", flush=True)
