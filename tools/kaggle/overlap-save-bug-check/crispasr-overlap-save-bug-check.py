"""
CrispASR — overlap-save A/B harness on Kaggle CPU for the inconclusive backends

PLAN #114 audit follow-up. The local-M1 run of `tools/check-overlap-save-bug.sh`
on 2026-05-25 came back inconclusive for granite-4.1 and omniasr-llm because
the per-run wallclock budget (~20 min on M1) wasn't enough for those two LLM-AR
backends to finish a 5-min clip. Kaggle CPU has 4 vCPUs + 30 GB RAM + a 9-hour
wall budget — fine for two backends × two runs (default overlap vs nooverlap)
× ~90 s audio.

Scope:
  - Build crispasr CPU-only from current main.
  - Download granite-4.1 + omniasr-llm GGUFs.
  - Synthesize a ~90 s audio by concatenating samples/jfk.wav eight times.
  - Run each backend twice (default --chunk-overlap 3.0 vs --chunk-overlap 0).
  - Compare SRT outputs: last timestamp, segment count, char count.
  - If nooverlap produces materially more output → SUSPECTED-BUG, add backend
    to kBlocked in examples/cli/crispasr_chunk_context_gate.h.

Patterns lifted from tools/kaggle/mimo-asr-cpu-vs-gpu/ (HISTORY 2026-05-26).
"""

import contextlib
import json
import multiprocessing
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import wave
from datetime import datetime, timezone
from pathlib import Path

# ── Working dirs ────────────────────────────────────────────────────────
ROOT = Path("/kaggle/working")
REPO = ROOT / "CrispASR"
BUILD = ROOT / "build"
MODELS = ROOT / "models"
AUDIO_DIR = ROOT / "audio"
OUT_DIR = ROOT / "results"

for d in (BUILD, MODELS, AUDIO_DIR, OUT_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ── Unbuffered I/O + progress.jsonl checkpointer ────────────────────────
PROGRESS = ROOT / "progress.jsonl"
HF_PROGRESS_REPO = os.environ.get("HF_PROGRESS_REPO", "cstr/crispasr-kaggle-progress")
HF_PROGRESS_FILE = os.environ.get("HF_PROGRESS_FILE", "overlap-save-bug-check.jsonl")

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

_progress_lock = threading.Lock()
_progress_last_push = 0.0


def _push_progress_to_hf(force: bool = False) -> None:
    global _progress_last_push
    if not os.environ.get("HF_TOKEN"):
        return
    now = time.time()
    if not force and now - _progress_last_push < 60:
        return
    _progress_last_push = now
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        api.upload_file(
            path_or_fileobj=str(PROGRESS),
            path_in_repo=HF_PROGRESS_FILE,
            repo_id=HF_PROGRESS_REPO,
            repo_type="dataset",
            commit_message=f"progress checkpoint {datetime.now(timezone.utc).isoformat()}",
        )
    except Exception as e:
        print(f"  [progress mirror failed: {e}]", flush=True)


def step(name: str, **extra) -> None:
    rec = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "step": name,
        **extra,
    }
    line = json.dumps(rec, default=str)
    with _progress_lock:
        with open(PROGRESS, "a") as f:
            f.write(line + "\n")
    print(f"[step] {name}  {json.dumps(extra, default=str) if extra else ''}", flush=True)
    _push_progress_to_hf()


# ── Build-step heartbeat + Popen line streamer ──────────────────────────


def sh_with_progress(cmd: str, cwd: Path | None = None) -> None:
    step("sh.begin", cmd=cmd)
    t0 = time.time()
    proc = subprocess.Popen(
        cmd,
        shell=True,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    last_progress_line = ""
    while True:
        line = proc.stdout.readline()
        if not line and proc.poll() is not None:
            break
        if line:
            line = line.rstrip()
            # Throttle ninja [X/N] noise
            m = re.match(r"^\[(\d+)/(\d+)\]", line)
            if m:
                last_progress_line = line
            else:
                if last_progress_line:
                    print(f"  {last_progress_line}", flush=True)
                    last_progress_line = ""
                print(f"  {line}", flush=True)
    if last_progress_line:
        print(f"  {last_progress_line}", flush=True)
    rc = proc.wait()
    dt = time.time() - t0
    step("sh.done", cmd=cmd, rc=rc, wall_s=round(dt, 1))
    if rc != 0:
        raise SystemExit(f"command failed (rc={rc}): {cmd}")


@contextlib.contextmanager
def build_heartbeat(label: str, interval_s: float = 30.0):
    stop = threading.Event()

    def beat() -> None:
        i = 0
        while not stop.wait(interval_s):
            i += 1
            extra = {"tick": i}
            try:
                used_mb = int(open("/proc/self/status").read().split("VmRSS:")[1].split()[0]) // 1024
                extra["rss_mib"] = used_mb
            except Exception:
                pass
            step(f"{label}.heartbeat", **extra)

    th = threading.Thread(target=beat, daemon=True)
    th.start()
    try:
        yield
    finally:
        stop.set()
        th.join(timeout=2)


# ── HF auth: Kaggle Dataset first, Secrets API fallback ─────────────────


def _read_hf_token() -> str | None:
    p = Path("/kaggle/input/crispasr-hf-token/hf_token.txt")
    if p.exists():
        tok = p.read_text().strip()
        if tok:
            return tok
    try:
        from kaggle_secrets import UserSecretsClient

        for attempt in range(3):
            try:
                return UserSecretsClient().get_secret("HF_TOKEN")
            except Exception:
                time.sleep(5 * (attempt + 1))
    except Exception:
        pass
    return None


_tok = _read_hf_token()
if _tok:
    os.environ["HF_TOKEN"] = _tok
    print("HF auth: token loaded (progress mirror enabled)", flush=True)
else:
    print("HF auth: anonymous (progress mirror disabled; local JSONL only)", flush=True)

step("script.start")

# ── Step 1: clone ───────────────────────────────────────────────────────

CRISPASR_REF = os.environ.get("CRISPASR_REF", "main")
step("clone.begin", ref=CRISPASR_REF)
if not REPO.exists():
    sh_with_progress(
        f"git clone --depth 1 --branch {CRISPASR_REF} "
        f"https://github.com/CrispStrobe/CrispASR.git {REPO}"
    )
else:
    sh_with_progress(f"git -C {REPO} pull --ff-only")
sha = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
step("clone.done", sha=sha)

# ── Step 2: CPU-only build ──────────────────────────────────────────────

step("build.begin")
BUILD.mkdir(exist_ok=True)
cmake_cmd = (
    f"cmake {REPO} -B{BUILD} -GNinja "
    "-DCMAKE_BUILD_TYPE=Release "
    "-DBUILD_SHARED_LIBS=ON "
    "-DCRISPASR_BUILD_TESTS=OFF"
)
njobs = max(4, multiprocessing.cpu_count())
with build_heartbeat("cmake-configure"):
    sh_with_progress(cmake_cmd)
step("build.configured")
with build_heartbeat("cmake-build"):
    sh_with_progress(
        f"stdbuf -oL -eL cmake --build {BUILD} --target crispasr-cli -- -j{njobs}"
    )
CRISPASR = BUILD / "bin" / "crispasr"
assert CRISPASR.is_file(), f"crispasr binary missing at {CRISPASR}"
step("build.done", binary=str(CRISPASR))

# ── Step 3: download GGUFs (granite-4.1 + omniasr-llm) ──────────────────

step("download.begin")
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
sh_with_progress("pip install -q huggingface_hub hf_transfer")
from huggingface_hub import hf_hub_download

# The two inconclusive backends from the M1 harness run on 2026-05-25.
# Add more here to widen the audit later.
BACKEND_GGUFS = [
    ("granite-4.1", "cstr/granite-speech-4.1-2b-GGUF", "granite-speech-4.1-2b-q4_k.gguf"),
    ("omniasr-llm", "cstr/omniasr-llm-300m-v2-GGUF", "omniasr-llm-300m-v2-q4_k.gguf"),
]
backend_model = {}
for backend, repo_id, fname in BACKEND_GGUFS:
    step(f"download.{backend}.begin", repo=repo_id, file=fname)
    p = hf_hub_download(repo_id=repo_id, filename=fname, local_dir=str(MODELS), local_dir_use_symlinks=False)
    backend_model[backend] = Path(p)
    step(f"download.{backend}.done", path=p, size_mib=Path(p).stat().st_size // (1 << 20))
step("download.done")

# ── Step 4: build a ~88 s audio by concatenating samples/jfk.wav 8x ─────

step("audio.begin")
SRC_WAV = REPO / "samples" / "jfk.wav"
LONG_WAV = AUDIO_DIR / "jfk_x8.wav"
assert SRC_WAV.is_file(), f"missing jfk.wav at {SRC_WAV}"
with wave.open(str(SRC_WAV), "rb") as src:
    params = src.getparams()
    data = src.readframes(src.getnframes())
n_copies = 8
with wave.open(str(LONG_WAV), "wb") as dst:
    dst.setparams(params)
    for _ in range(n_copies):
        dst.writeframes(data)
src_secs = params.nframes / params.framerate
dst_secs = src_secs * n_copies
step("audio.done", file=str(LONG_WAV), secs=round(dst_secs, 1))

# ── Step 5: A/B sweep ───────────────────────────────────────────────────


def parse_srt(srt_path: Path) -> dict:
    """Mirror parse_srt() in tools/check-overlap-save-bug.sh."""
    if not srt_path.is_file() or srt_path.stat().st_size == 0:
        return {"last_s": 0, "segs": 0, "chars": 0}
    text = srt_path.read_text(errors="replace")
    times = re.findall(r"\d{2}:\d{2}:\d{2}[,.]\d{3} --> (\d{2}:\d{2}:\d{2})[,.]\d{3}", text)
    if not times:
        return {"last_s": 0, "segs": 0, "chars": len(text)}
    last_s = max(
        int(h) * 3600 + int(m) * 60 + int(s) for h, m, s in (t.split(":") for t in times)
    )
    segs = text.count(" --> ")
    chars = sum(
        len(line) for line in text.splitlines() if line and not line[0].isdigit() and " --> " not in line
    )
    return {"last_s": last_s, "segs": segs, "chars": chars}


def run_one(backend: str, model: Path, label: str, extra: list[str]) -> dict:
    out_stem = OUT_DIR / f"{backend}-{label}"
    for ext in (".txt", ".srt", ".json"):
        f = out_stem.with_suffix(ext)
        if f.exists():
            f.unlink()
    cmd = [
        str(CRISPASR),
        "-m", str(model),
        "--backend", backend,
        "-f", str(LONG_WAV),
        "-of", str(out_stem),
        "-osrt",
        "-np",
    ] + extra
    step(f"run.{backend}.{label}.begin", cmd=" ".join(cmd))
    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        rc = r.returncode
        log = (r.stdout or "") + (r.stderr or "")
    except subprocess.TimeoutExpired as e:
        rc = -1
        log = f"TIMEOUT after {round(time.time() - t0, 1)}s\n" + (e.stdout or "") + (e.stderr or "")
    dt = time.time() - t0
    srt_stats = parse_srt(out_stem.with_suffix(".srt"))
    step(f"run.{backend}.{label}.done", rc=rc, wall_s=round(dt, 1), **srt_stats)
    if rc != 0:
        print(f"--- last 30 lines of log for {backend}/{label} ---", flush=True)
        for line in log.splitlines()[-30:]:
            print(line, flush=True)
    return {"rc": rc, "wall_s": round(dt, 1), **srt_stats}


verdicts = {}
for backend, _, _ in BACKEND_GGUFS:
    model = backend_model[backend]
    step(f"sweep.{backend}.begin")
    default = run_one(backend, model, "default", ["--chunk-overlap", "3.0"])
    nooverlap = run_one(backend, model, "nooverlap", ["--chunk-overlap", "0"])
    # Verdict (mirror tools/check-overlap-save-bug.sh logic): if nooverlap
    # produces materially more output (later last_s or larger chars), the
    # backend has the bug. Threshold: 25 % more chars or >= 10 s later.
    chars_d, chars_n = default["chars"], nooverlap["chars"]
    last_d, last_n = default["last_s"], nooverlap["last_s"]
    chars_delta = chars_n - chars_d
    last_delta = last_n - last_d
    rel_chars = (chars_delta / chars_d) if chars_d > 0 else 0.0
    if rel_chars > 0.25 or last_delta >= 10:
        verdict = "SUSPECTED-BUG"
    elif default["rc"] != 0 and nooverlap["rc"] == 0:
        verdict = "DEFAULT-FAILS"
    elif default["rc"] != 0 and nooverlap["rc"] != 0:
        verdict = "BOTH-FAIL"
    else:
        verdict = "OK"
    verdicts[backend] = {
        "verdict": verdict,
        "default": default,
        "nooverlap": nooverlap,
        "rel_chars": round(rel_chars, 2),
        "last_delta_s": last_delta,
    }
    step(
        f"sweep.{backend}.done",
        verdict=verdict,
        rel_chars=round(rel_chars, 2),
        last_delta_s=last_delta,
    )

# ── Summary ─────────────────────────────────────────────────────────────

step("summary", verdicts=verdicts, sha=sha)
print("\n" + "=" * 76)
print(f"SUMMARY — overlap-save A/B on HEAD ({sha[:8]}) — Kaggle CPU")
print("=" * 76)
print(f"  audio: {LONG_WAV.name} ({round(dst_secs, 1)} s)")
for backend, v in verdicts.items():
    print()
    print(f"  {backend}: {v['verdict']}")
    print(f"    default  rc={v['default']['rc']:>3}  wall={v['default']['wall_s']:>6}s  "
          f"chars={v['default']['chars']:>6}  last_s={v['default']['last_s']:>4}")
    print(f"    nooverlap rc={v['nooverlap']['rc']:>3}  wall={v['nooverlap']['wall_s']:>6}s  "
          f"chars={v['nooverlap']['chars']:>6}  last_s={v['nooverlap']['last_s']:>4}")
    print(f"    rel_chars_delta={v['rel_chars']:+.2f}  last_delta_s={v['last_delta_s']:+d}")

print()
print("Interpretation:")
for backend, v in verdicts.items():
    if v["verdict"] == "SUSPECTED-BUG":
        print(f"  {backend}: add to kBlocked in examples/cli/crispasr_chunk_context_gate.h")
    elif v["verdict"] == "OK":
        print(f"  {backend}: opt-out NOT needed; leave default chunking on")
    elif v["verdict"] == "BOTH-FAIL":
        print(f"  {backend}: both arms failed — backend is broken at this length, separate bug")
    elif v["verdict"] == "DEFAULT-FAILS":
        print(f"  {backend}: default fails but nooverlap works — add to kBlocked + investigate")

_push_progress_to_hf(force=True)
step("script.end")
