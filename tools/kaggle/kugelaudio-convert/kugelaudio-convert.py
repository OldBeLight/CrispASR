#!/usr/bin/env python3
"""Kaggle kernel: convert KugelAudio-0-Open to GGUF.

CPU-only (30 GB RAM is enough for the 18.7 GB model).
Reference dump (Phase 3) requires GPU — skipped on CPU kernels,
run separately when GPU quota is available.

Push: python -m kaggle kernels push -p tools/kaggle/kugelaudio-convert
"""

import os, sys, time
from pathlib import Path

REPO = Path("/kaggle/working/CrispASR")
BUILD = Path("/kaggle/working/build")
OUTPUT = Path("/kaggle/working/output")

# ── Kaggle harness ──────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(str(REPO), "tools", "kaggle"))
import kaggle_harness as kh

kh.init_progress()

# ── Phase 0: Clone repo + install deps ──────────────────────────────────────
kh.report_status("cloning repo")
if not REPO.exists():
    kh.sh_with_progress("git clone --depth 1 https://github.com/CrispStrobe/CrispASR /kaggle/working/CrispASR")

kh.report_status("installing deps")
kh.sh_with_progress("pip install -q safetensors gguf transformers huggingface_hub torch")

# Try installing kugelaudio-open (may not be on PyPI)
try:
    kh.sh_with_progress("pip install -q kugelaudio-open 2>/dev/null || true")
except Exception:
    pass

# ── Phase 1: Download model ────────────────────────────────────────────────
kh.report_status("downloading model")
MODEL_ID = "kugelaudio/kugelaudio-0-open"
token = kh.resolve_hf_token()

from huggingface_hub import snapshot_download
model_dir = snapshot_download(
    MODEL_ID,
    cache_dir="/kaggle/working/hf-cache",
    token=token,
)
print(f"model downloaded to: {model_dir}")

# ── Phase 2: GGUF conversion ───────────────────────────────────────────────
kh.report_status("converting to GGUF")
OUTPUT.mkdir(parents=True, exist_ok=True)
gguf_path = OUTPUT / "kugelaudio-0-open-f16.gguf"

sys.path.insert(0, str(REPO / "models"))
# Run converter directly
converter = str(REPO / "models" / "convert-kugelaudio-to-gguf.py")
kh.sh_with_progress(
    f"python {converter} "
    f"--input {model_dir} "
    f"--output {gguf_path} "
    f"--no-encoders "
    f"--type f16"
)

print(f"GGUF written: {gguf_path} ({gguf_path.stat().st_size / 1e9:.2f} GB)")

# ── Phase 3: Tensor map dump (CPU-safe, no inference) ─────────────────────
kh.report_status("dumping tensor map")
ref_dir = OUTPUT / "reference"
ref_script = str(REPO / "tools" / "reference_backends" / "kugelaudio.py")
try:
    kh.sh_with_progress(
        f"python {ref_script} "
        f"--model {model_dir} "
        f"--output-dir {ref_dir} "
        f"--dump-tensor-map"
    )
    print(f"tensor map dumped to {ref_dir}")
except Exception as e:
    print(f"tensor map dump failed: {e}")

# ── Phase 4: Upload artifacts ───────────────────────────────────────────────
kh.report_status("listing outputs")
for f in sorted(OUTPUT.rglob("*")):
    if f.is_file():
        print(f"  {f.relative_to(OUTPUT)}: {f.stat().st_size / 1e6:.1f} MB")

kh.report_status("done")
print("All done. Download GGUF from Kaggle output tab.")
