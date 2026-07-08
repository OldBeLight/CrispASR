#!/usr/bin/env python3
"""CrispASR canary-qwen backend validation on Kaggle (GPU, 13 GB RAM).

Builds from the feature branch, converts nvidia/canary-qwen-2.5b to GGUF,
quantizes to Q4_K, and runs transcription on jfk.wav to verify the full
pipeline end-to-end.
"""

import json
import os
import shutil
import subprocess
import sys
import time
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

# ── Bootstrap: clone CrispASR from feature branch and import harness ──
CRISPASR_URL = "https://github.com/CrispStrobe/CrispASR.git"
BRANCH = "worktree-feat-canary-qwen"

if not REPO.exists():
    try:
        subprocess.check_call(
            ["git", "clone", "--recursive", "--branch", BRANCH,
             CRISPASR_URL, str(REPO)])
        sys.path.insert(0, str(REPO / "tools" / "kaggle"))
    except Exception:
        pass  # fall through to bundled copy

if str(REPO / "tools" / "kaggle") not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import kaggle_harness as kh
kh.init_progress()

kh.step("start")

# ── HF auth via harness (3-tier: secret → dataset → env) ─────────────
hf_token = kh.kaggle_secret("HF_TOKEN")
if not hf_token:
    hf_token = kh.kaggle_token_from_dataset("hf_token.txt")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
    print(f"HF token loaded ({len(hf_token)} chars)", flush=True)
else:
    print("WARNING: no HF token found — model download may fail for gated models", flush=True)

# ── Install build toolchain via harness ───────────────────────────────
kh.step("install_toolchain")
kh.install_build_toolchain()

# ── pip deps (small only — never pip install torch, see gotcha #11) ───
kh.step("pip_deps")
subprocess.run("pip install -q safetensors gguf huggingface_hub",
               shell=True, capture_output=True)

# ── Build flags via harness ───────────────────────────────────────────
kh.step("build_configure")

build_flags = kh.cache_and_link_flags()
try:
    arch = kh.detect_cuda_arch()
    build_flags += kh.cuda_build_flags(arch)
    print(f"CUDA arch: {arch}", flush=True)
except Exception:
    print("No CUDA detected, building CPU-only", flush=True)

subprocess.check_call(
    f"cmake -S {REPO} -B {BUILD} -G Ninja "
    f"-DCMAKE_BUILD_TYPE=Release "
    f"-DCRISPASR_BUILD_TESTS=OFF "
    f"-DCRISPASR_BUILD_EXAMPLES=ON "
    f"-DCRISPASR_BUILD_SERVER=OFF "
    + " ".join(build_flags),
    shell=True)

# ── Build with heartbeat via harness ──────────────────────────────────
kh.step("build_compile")
n_jobs = kh.safe_build_jobs(gpu=True)

with kh.build_heartbeat("cmake.build"):
    subprocess.check_call(
        f"stdbuf -oL -eL cmake --build {BUILD} "
        f"--target crispasr-cli crispasr-quantize "
        f"-j{n_jobs}",
        shell=True)

CRISPASR_BIN = BUILD / "bin" / "crispasr"
QUANTIZE_BIN = BUILD / "bin" / "crispasr-quantize"
assert CRISPASR_BIN.exists(), f"crispasr binary not found at {CRISPASR_BIN}"
kh.step("build_done")

# ── Download model from HF ───────────────────────────────────────────
kh.step("download_model")
MODEL_DIR = WORK / "canary-qwen-hf"
GGUF_F16 = WORK / "canary-qwen-2.5b-f16.gguf"
GGUF_Q4K = WORK / "canary-qwen-2.5b-q4_k.gguf"

if not GGUF_Q4K.exists():
    from huggingface_hub import snapshot_download
    snapshot_download("nvidia/canary-qwen-2.5b",
                      local_dir=str(MODEL_DIR),
                      token=os.environ.get("HF_TOKEN"))
    kh.step("model_downloaded")

    # ── Convert to GGUF ──────────────────────────────────────────────
    kh.step("convert_to_gguf")
    os.environ["TMPDIR"] = str(WORK / "tmp")
    (WORK / "tmp").mkdir(exist_ok=True)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    subprocess.check_call([
        sys.executable,
        str(REPO / "models" / "convert-canary-qwen-to-gguf.py"),
        "--input", str(MODEL_DIR),
        "--output", str(GGUF_F16),
    ])
    kh.step("convert_done", size_gb=round(GGUF_F16.stat().st_size / 1e9, 2))

    # ── Quantize to Q4_K ─────────────────────────────────────────────
    kh.step("quantize")
    subprocess.check_call([
        str(QUANTIZE_BIN), str(GGUF_F16), str(GGUF_Q4K), "q4_k"])
    kh.step("quantize_done", size_gb=round(GGUF_Q4K.stat().st_size / 1e9, 2))

    # Free disk space — remove F16 GGUF + HF snapshot
    GGUF_F16.unlink(missing_ok=True)
    shutil.rmtree(str(MODEL_DIR), ignore_errors=True)

kh.step("disk_free_gb", free=kh.free_gb())

# ── Transcribe jfk.wav ───────────────────────────────────────────────
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

# ── Write results to progress file for retrieval ─────────────────────
with open(WORK / "result.txt", "w") as f:
    f.write(f"exit_code={result.returncode}\n")
    f.write(f"stdout={result.stdout}\n")
    f.write(f"stderr={result.stderr[-2000:]}\n")

# ── Validate transcript ──────────────────────────────────────────────
import re
JFK_EXPECTED = "and so my fellow americans ask not what your country can do for you ask what you can do for your country"
transcript = result.stdout.strip().lower()
transcript_clean = re.sub(r'\[.*?\]', '', transcript).strip()
transcript_clean = re.sub(r'\d{2}:\d{2}:\d{2}.*?-->', '', transcript_clean).strip()
transcript_clean = re.sub(r'\s+', ' ', transcript_clean).strip()
transcript_clean = re.sub(r'[^\w\s]', '', transcript_clean).strip()

kh.step("validation", transcript=transcript_clean[:200], expected=JFK_EXPECTED[:80])

if JFK_EXPECTED in transcript_clean:
    print("\n✓ PASS — transcript matches expected JFK reference", flush=True)
    kh.step("PASS")
elif result.returncode == 0 and len(transcript_clean) > 10:
    print(f"\n~ PARTIAL — got output but doesn't match exactly:", flush=True)
    print(f"  got:      {transcript_clean[:200]}", flush=True)
    print(f"  expected: {JFK_EXPECTED}", flush=True)
    kh.step("PARTIAL", got=transcript_clean[:200])
else:
    print(f"\n✗ FAIL — exit code {result.returncode}, output len {len(transcript_clean)}", flush=True)
    kh.step("FAIL", exit_code=result.returncode)

# ── Save ccache for future runs ──────────────────────────────────────
kh.step("save_ccache")
try:
    os.chdir(str(WORK))
    subprocess.run("tar cf ccache.tar .ccache/", shell=True, check=True)
except Exception:
    pass

kh.step("done", total_s=round(time.time() - kh._T0, 1))
print(f"\nTotal runtime: {time.time() - kh._T0:.0f}s", flush=True)
