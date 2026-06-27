# %% [markdown]
# # TADA language voice reference GGUFs
#
# Generates tada-ref-{ar,ch,de,es,fr,it,ja,pl,pt}.gguf from FLEURS CC-BY-4.0
# clips using the HumeAI/tada-codec language-specific aligners.
#
# NO HF upload from here — Kaggle Secrets flake (HTTPError 400).
# Output lands in /kaggle/working/lang-refs/; fetch locally then upload:
#
#   kaggle kernels output chr1str/tada-language-voice-reference-ggufs \
#       -p /Volumes/backups/code/tada-lang-refs-stash/kaggle-out/
#   python tools/upload_tada_lang_refs.py \
#       --dir /Volumes/backups/code/tada-lang-refs-stash/kaggle-out/lang-refs/

# %% [code]
import os, sys, subprocess, time, json
from datetime import datetime, timezone
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except (AttributeError, ValueError):
    pass

_PROGRESS = Path("/kaggle/working/progress.jsonl")
_T0 = time.time()

def _step(name, **kw):
    rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "elapsed_s": round(time.time() - _T0, 2), "step": name, **kw}
    _PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    with _PROGRESS.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"[boot {rec['elapsed_s']:>6.1f}s] {name}", flush=True)

_step("bootstrap.start")

WORK = Path("/kaggle/working")
REPO = WORK / "CrispASR"
OUT  = WORK / "lang-refs"
OUT.mkdir(exist_ok=True)

# %% [code]  clone
if not REPO.exists():
    _step("git-clone.begin")
    subprocess.check_call([
        "git", "clone", "--depth", "1",
        "https://github.com/CrispStrobe/CrispASR.git", str(REPO),
    ])
    _step("git-clone.done")
else:
    _step("git-pull.begin")
    subprocess.check_call(["git", "-C", str(REPO), "pull", "--ff-only"])
    _step("git-pull.done")

# %% [code]  install deps
_step("pip-install.begin")
# Install base deps first so we can read the torch version before
# installing torchaudio (must match torch exactly to avoid ABI errors).
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "gguf", "datasets", "soundfile", "scipy",
])
import importlib.util as _ilu
_torch_ver = None
if _ilu.find_spec("torch"):
    import torch as _t; _torch_ver = _t.__version__.split("+")[0]
_torchaudio_pin = f"torchaudio=={_torch_ver}" if _torch_ver else "torchaudio"
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    _torchaudio_pin,
    "git+https://github.com/HumeAI/tada.git",
])
_step("pip-install.done")

# %% [code]  generate
_step("gen.begin")
result = subprocess.run(
    [sys.executable, str(REPO / "tools" / "gen_tada_lang_refs.py"),
     "--output-dir", str(OUT),
     "--skip-existing"],
    check=False,
)
_step("gen.done", returncode=result.returncode)

# %% [code]  report
files = sorted(OUT.glob("*.gguf"))
print(f"\n=== generated {len(files)} lang ref(s) ===")
for p in files:
    print(f"  {p.name}  ({p.stat().st_size // 1024} KB)")

print("\n=== fetch + upload locally (NOT on Kaggle) ===")
print("  kaggle kernels output chr1str/tada-language-voice-reference-ggufs \\")
print("      -p /Volumes/backups/code/tada-lang-refs-stash/kaggle-out/")
print("  python tools/upload_tada_lang_refs.py \\")
print("      --dir /Volumes/backups/code/tada-lang-refs-stash/kaggle-out/lang-refs/")
