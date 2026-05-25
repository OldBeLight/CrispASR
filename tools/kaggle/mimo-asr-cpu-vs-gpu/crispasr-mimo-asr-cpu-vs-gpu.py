"""
CrispASR — mimo-asr CPU vs GPU regression bisect (PLAN #115)

Question: on current main, mimo-asr produces zero output on JFK (11 s) on
M1 Metal and segfaults on a 5 min clip. The smoking-gun commit by
inspection is `89111260` ("perf #72: load weights to GPU when use_gpu=true")
which flipped `core_gguf::load_weights(..., ctx->backend_cpu, ...)` to
`..., ctx->backend, ...`. The same commit message foresees the regression:
"If a platform regresses, add a CRISPASR_FORCE_CPU_WEIGHTS=1 escape hatch
— none seen yet." Now we have one.

We cannot safely repro on the local M1 box (4.5 GB CPU mimo Q4_K + already
loaded benchmark sweep risks OOM). Kaggle T4 has 16 GB RAM + a CUDA GPU,
so we can answer:

    A) Does HEAD mimo-asr produce a JFK transcript at all on CUDA (--gpu)?
       — reproduces the Metal failure if CUDA also breaks
    B) Does HEAD mimo-asr produce a JFK transcript with --no-gpu (CPU only)?
       — isolates the regression to the GPU residency code path

Reference (HISTORY §56): the JFK transcript should be verbatim
"And so, my fellow Americans, ask not what your country can do for you.
Ask what you can do for your country."

Run pattern shape mirrors tools/kaggle-issue81-cuda-ab.py and
tools/kaggle/crispasr-s3gen-cuda-bisect (script kernel, no .ipynb noise).
"""

import multiprocessing
import os
import re
import subprocess
import sys
import time
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "CrispASR"
BUILD = WORK / "build"
MODELS = WORK / "models"
SAMPLE = WORK / "jfk.wav"

# Branch-parametrized like s3gen-cuda-bisect — change via Kaggle env var.
CRISPASR_REF = os.environ.get("CRISPASR_REF", "main")

# Expected JFK transcript from HISTORY §56 (the last-known-good ground
# truth that matches the Python MimoAudio.asr_sft reference).
EXPECTED_JFK = "ask not what your country can do for you"


def run(cmd, env=None, check=True, capture=False, timeout=600):
    merged = {**os.environ, **(env or {})}
    r = subprocess.run(cmd, shell=isinstance(cmd, str), env=merged,
                       capture_output=capture, text=True, timeout=timeout)
    if check and r.returncode != 0:
        out = (r.stdout or "") + (r.stderr or "")
        raise RuntimeError(f"Command failed (rc={r.returncode}): {cmd}\n{out[-3000:]}")
    return r


# ── Step 1: clone CrispASR ───────────────────────────────────────────────

print(f"=== Step 1: clone CrispASR @ {CRISPASR_REF} ===")
if not REPO.exists():
    run(f"git clone --depth 1 --branch {CRISPASR_REF} "
        f"https://github.com/CrispStrobe/CrispASR.git {REPO}")
else:
    run(f"git -C {REPO} pull --ff-only", check=False)
sha = run(f"git -C {REPO} rev-parse HEAD", capture=True).stdout.strip()
print(f"HEAD = {sha}")

# ── Step 2: detect CUDA, build with CUDA when available ──────────────────

print("\n=== Step 2: build CrispASR ===")
sm_detect = run("nvidia-smi --query-gpu=compute_cap --format=csv,noheader",
                capture=True, check=False)
has_cuda = sm_detect.returncode == 0 and bool(sm_detect.stdout.strip())
cuda_arch = sm_detect.stdout.strip().split("\n")[0].replace(".", "") if has_cuda else None
print(f"CUDA: {'yes (sm_' + cuda_arch + ')' if has_cuda else 'no — CPU-only build'}")

# Link unversioned libcuda.so if needed (same workaround as
# crispasr-s3gen-cuda-bisect — Kaggle ships libcuda.so.1 only).
if has_cuda:
    r_find = run("find /usr /lib /usr/local /opt -maxdepth 8 -name 'libcuda.so*' 2>/dev/null",
                 capture=True, check=False)
    libcuda_files = [l.strip() for l in r_find.stdout.strip().splitlines() if l.strip()]
    cuda_driver_lib = None
    for name in ["libcuda.so", "libcuda.so.1"]:
        for f in libcuda_files:
            if f.endswith("/" + name):
                cuda_driver_lib = f
                break
        if cuda_driver_lib:
            break
    if cuda_driver_lib:
        run(f"ln -sf {cuda_driver_lib} /usr/lib/x86_64-linux-gnu/libcuda.so", check=False)
        run(f"ln -sf {cuda_driver_lib} /usr/local/lib/libcuda.so", check=False)
        run("ldconfig", check=False)

BUILD.mkdir(exist_ok=True)
cmake_cmd = [
    "cmake", str(REPO), f"-B{BUILD}", "-GNinja",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DBUILD_SHARED_LIBS=ON",
    "-DCRISPASR_BUILD_TESTS=OFF",
]
if has_cuda:
    cmake_cmd += ["-DGGML_CUDA=ON", f"-DCMAKE_CUDA_ARCHITECTURES={cuda_arch}"]
run(cmake_cmd)
njobs = max(4, multiprocessing.cpu_count())
print(f"Building crispasr-cli with -j{njobs}")
run(["cmake", "--build", str(BUILD), "--target", "crispasr-cli", "--", f"-j{njobs}"],
    timeout=3600)
CRISPASR = BUILD / "bin" / "crispasr"
print(f"Built: {CRISPASR}")

# ── Step 3: download mimo-asr + tokenizer GGUFs ──────────────────────────

print("\n=== Step 3: download mimo-asr-q4_k.gguf + mimo-tokenizer-q4_k.gguf ===")
MODELS.mkdir(exist_ok=True)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
run("pip install -q huggingface_hub hf_transfer")
run(
    f"""python -c "
from huggingface_hub import hf_hub_download
for f in ['mimo-asr-q4_k.gguf']:
    p = hf_hub_download('cstr/mimo-asr-GGUF', f, local_dir='{MODELS}', local_dir_use_symlinks=False)
    print(p)
for f in ['mimo-tokenizer-q4_k.gguf']:
    p = hf_hub_download('cstr/mimo-tokenizer-GGUF', f, local_dir='{MODELS}', local_dir_use_symlinks=False)
    print(p)
" """,
    timeout=900,
)
mimo_path = MODELS / "mimo-asr-q4_k.gguf"
tok_path = MODELS / "mimo-tokenizer-q4_k.gguf"
print(f"mimo:  {mimo_path} ({mimo_path.stat().st_size // (1 << 20)} MiB)")
print(f"token: {tok_path}  ({tok_path.stat().st_size // (1 << 20)} MiB)")

# JFK sample from repo
run(f"cp {REPO}/samples/jfk.wav {SAMPLE}", check=False)
assert SAMPLE.is_file(), "jfk.wav missing"

# ── Step 4: helper to run a single config ────────────────────────────────

def run_mimo(label: str, extra_args: list[str]) -> tuple[str, bool]:
    """Run mimo-asr on jfk.wav. Returns (transcript, ok)."""
    out_stem = WORK / f"mimo-jfk-{label}"
    # Clean any prior outputs so a stale file doesn't mask an empty run.
    for ext in [".txt", ".srt"]:
        f = out_stem.with_suffix(ext)
        if f.exists():
            f.unlink()

    cmd = [
        str(CRISPASR),
        "-m", str(mimo_path),
        "--backend", "mimo-asr",
        "-f", str(SAMPLE),
        "-of", str(out_stem),
        "-otxt",
        "-np",
    ] + extra_args

    print(f"\n{'─' * 60}")
    print(f"CONFIG: {label}")
    print(f"CMD:    {' '.join(cmd)}")
    t0 = time.time()
    r = run(cmd, env={"MIMO_ASR_BENCH": "1"}, check=False, capture=True, timeout=600)
    dt = time.time() - t0
    log = (r.stdout or "") + (r.stderr or "")
    # Pull the bench line if present — confirms prefill+decode actually ran.
    bench = re.search(r"mimo_asr_bench:.*", log)
    print(f"exit={r.returncode}  wall={dt:.1f}s  bench={'yes — ' + bench.group(0) if bench else 'NOT EMITTED (prefill/decode did not complete)'}")
    txt_path = out_stem.with_suffix(".txt")
    if txt_path.exists() and txt_path.stat().st_size > 0:
        text = txt_path.read_text().strip()
        ok = EXPECTED_JFK in text.lower()
        marker = "✓ matches reference" if ok else "✗ output present but does not match JFK reference"
        print(f"text: {text[:200]}\n      {marker}")
        return text, ok
    print("text: <no output file produced>")
    # Surface tail of stderr for diagnosis
    print("--- last 30 lines of stderr ---")
    print("\n".join(log.splitlines()[-30:]))
    return "", False


# ── Step 5: A/B ──────────────────────────────────────────────────────────

print("\n=== Step 5: A/B runs ===")
results = {}

# A. HEAD default (GPU when available)
results["A-default-gpu" if has_cuda else "A-default-cpu"] = run_mimo(
    "default", []
)

# B. HEAD with --no-gpu (forced CPU path even when CUDA available)
if has_cuda:
    results["B-no-gpu"] = run_mimo("no-gpu", ["--no-gpu"])

# ── Summary ──────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print(f"SUMMARY — mimo-asr JFK on HEAD ({sha[:8]})")
print("=" * 70)
for label, (text, ok) in results.items():
    status = "PASS" if ok else ("EMPTY" if not text else "WRONG")
    print(f"  {label:25s} {status:6s} {text[:80]}")
print()
print("Interpretation:")
if not has_cuda:
    print("  CPU-only run — if EMPTY/WRONG, regression also affects pure CPU path")
    print("  and is not limited to the GPU residency change in commit 89111260.")
else:
    a_ok = results.get("A-default-gpu", ("", False))[1]
    b_ok = results.get("B-no-gpu", ("", False))[1]
    if a_ok and b_ok:
        print("  Both GPU and CPU produce the JFK transcript on CUDA — bug may")
        print("  be Metal-specific. Repro on M1 still needed for full picture.")
    elif b_ok and not a_ok:
        print("  CPU works, GPU does not — confirms PLAN #72 (89111260) GPU")
        print("  residency change broke the prefill path. Fix: either revert")
        print("  the one-line backend_cpu→backend swap, or fix the GPU tensor")
        print("  routing in mimo_asr_build_prefill_graph.")
    elif not b_ok and not a_ok:
        print("  Both broken — there are more regressions stacked on top of #72.")
        print("  Bisect further from HISTORY §56 baseline.")
    else:
        print("  GPU works, CPU does not — unexpected; investigate.")
