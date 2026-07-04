#!/usr/bin/env python3
"""
Kaggle kernel: Convert Aratako/Irodori-TTS-500M-v3 → GGUF + quantize to Q4_K.
Also dumps a Python reference for crispasr-diff parity validation.

Outputs (in /kaggle/working):
  - irodori-tts-500m-v3-f16.gguf     (~1 GB)
  - irodori-tts-500m-v3-q4_k.gguf    (~250 MB)
  - irodori-tts-ref.gguf              (reference activations for diff)

Run on chr1s4 account with GPU enabled (for faster HF downloads + builds).
"""

import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

WORK = Path("/kaggle/working")
os.chdir(str(WORK))

# ── Bootstrap harness ────────────────────────────────────────────────

CRISPASR_URL = "https://github.com/CrispStrobe/CrispASR.git"
_CRISPASR_DIR = WORK / "CrispASR"
if not _CRISPASR_DIR.exists():
    try:
        subprocess.check_call(
            ["git", "clone", "--depth", "1", CRISPASR_URL, str(_CRISPASR_DIR)]
        )
        sys.path.insert(0, str(_CRISPASR_DIR / "tools" / "kaggle"))
    except Exception:
        pass

if str(_CRISPASR_DIR / "tools" / "kaggle") not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import kaggle_harness as kh

kh.init_progress()

# ── Install deps ─────────────────────────────────────────────────────

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q",
     "gguf", "safetensors", "transformers", "sentencepiece", "huggingface_hub"],
    check=True,
)

# ── Authenticate HF ─────────────────────────────────────────────────

hf_token = kh.kaggle_secret("hf_token", "HF_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

# ── Download model ───────────────────────────────────────────────────

print("=" * 60)
print("STEP 1: Download Irodori-TTS-500M-v3 checkpoint")
print("=" * 60)

from huggingface_hub import hf_hub_download, list_repo_files

REPO_ID = "Aratako/Irodori-TTS-500M-v3"

# Find the safetensors file
files = list_repo_files(REPO_ID)
st_files = [f for f in files if f.endswith(".safetensors")]
print(f"  Repo files: {files}")
print(f"  Safetensors: {st_files}")

if not st_files:
    print("ERROR: No .safetensors found in repo")
    sys.exit(1)

ckpt_path = hf_hub_download(repo_id=REPO_ID, filename=st_files[0])
print(f"  Downloaded: {ckpt_path}")
print(f"  Size: {os.path.getsize(ckpt_path) / 1024 / 1024:.1f} MB")

# ── Convert to GGUF (F16) ───────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 2: Convert to GGUF (F16)")
print("=" * 60)

converter_path = _CRISPASR_DIR / "models" / "convert-irodori-tts-to-gguf.py"
output_f16 = WORK / "irodori-tts-500m-v3-f16.gguf"

cmd = [
    sys.executable, str(converter_path),
    "--checkpoint", ckpt_path,
    "--output", str(output_f16),
    "--tokenizer-repo", "sbintuitions/sarashina2.2-0.5b",
]
print(f"  Running: {' '.join(cmd)}")
t0 = time.time()
subprocess.run(cmd, check=True)
elapsed = time.time() - t0
print(f"  Done in {elapsed:.1f}s")
print(f"  F16 GGUF: {output_f16.stat().st_size / 1024 / 1024:.1f} MB")

# ── Build CrispASR for quantization ─────────────────────────────────

print("\n" + "=" * 60)
print("STEP 3: Build crispasr-quantize")
print("=" * 60)

kh.install_build_toolchain()
build_dir = _CRISPASR_DIR / "build"

cmake_flags = [
    "-DCMAKE_BUILD_TYPE=Release",
    "-G", "Ninja",
]
# Add ccache if available
cmake_flags += kh.cache_and_link_flags()

subprocess.run(
    ["cmake", "-B", str(build_dir), "-S", str(_CRISPASR_DIR)] + cmake_flags,
    check=True, cwd=str(_CRISPASR_DIR),
)

jobs = kh.safe_build_jobs(gpu=True)
with kh.build_heartbeat():
    subprocess.run(
        ["cmake", "--build", str(build_dir), "--target", "crispasr-quantize",
         f"-j{jobs}"],
        check=True, cwd=str(_CRISPASR_DIR),
    )

quantize_bin = build_dir / "bin" / "crispasr-quantize"
if not quantize_bin.exists():
    # Try alternative path
    quantize_bin = build_dir / "examples" / "crispasr-quantize" / "crispasr-quantize"
if not quantize_bin.exists():
    # Search for it
    result = subprocess.run(
        ["find", str(build_dir), "-name", "crispasr-quantize", "-type", "f"],
        capture_output=True, text=True,
    )
    candidates = result.stdout.strip().split("\n")
    if candidates and candidates[0]:
        quantize_bin = Path(candidates[0])
    else:
        print("ERROR: crispasr-quantize not found")
        sys.exit(1)

print(f"  quantize binary: {quantize_bin}")

# ── Quantize to Q4_K ────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 4: Quantize to Q4_K")
print("=" * 60)

output_q4k = WORK / "irodori-tts-500m-v3-q4_k.gguf"

t0 = time.time()
subprocess.run(
    [str(quantize_bin), str(output_f16), str(output_q4k), "q4_k"],
    check=True,
)
elapsed = time.time() - t0
print(f"  Done in {elapsed:.1f}s")
print(f"  Q4_K GGUF: {output_q4k.stat().st_size / 1024 / 1024:.1f} MB")

# ── Python reference dump (for crispasr-diff) ───────────────────────

print("\n" + "=" * 60)
print("STEP 5: Reference dump (text encoder + speaker encoder first layers)")
print("=" * 60)

# We'll do a lightweight reference dump — encode a short text and capture
# intermediate activations at the text encoder and first DiT block.
# Full reference dump can be added later.

try:
    import torch
    import numpy as np
    from safetensors import safe_open

    # Load model config
    with safe_open(ckpt_path, framework="pt", device="cpu") as f:
        metadata = f.metadata() or {}
        config_json = metadata.get("config_json")
        if config_json:
            config = json.loads(config_json)
            print(f"  Model config: latent_dim={config.get('latent_dim')}, "
                  f"model_dim={config.get('model_dim')}, "
                  f"num_layers={config.get('num_layers')}")

    # Load tokenizer and encode a test string
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("sbintuitions/sarashina2.2-0.5b", use_fast=True)
    test_text = "こんにちは、世界。"
    token_ids = tok.encode(test_text, add_special_tokens=False)
    token_ids = [tok.bos_token_id] + token_ids
    print(f"  Test text: '{test_text}' → {len(token_ids)} tokens")
    print(f"  Token IDs (first 10): {token_ids[:10]}")

    # Save reference info
    ref_info = {
        "test_text": test_text,
        "token_ids": token_ids,
        "vocab_size": len(tok),
        "bos_token_id": tok.bos_token_id,
        "pad_token_id": tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
    }
    ref_path = WORK / "irodori-tts-ref-info.json"
    ref_path.write_text(json.dumps(ref_info, ensure_ascii=False, indent=2))
    print(f"  Saved reference info: {ref_path}")

    # For a full reference dump, we'd load the model and run forward passes.
    # That requires ~2GB RAM for the model alone. On Kaggle P100/T4 (13GB RAM),
    # this is feasible. We'll implement the full dump as a follow-up.
    print("  NOTE: Full activation dump will be added in follow-up kernel push")

except Exception as e:
    print(f"  Reference dump skipped: {e}")

# ── Upload to HuggingFace ────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 6: Upload GGUF files to HuggingFace")
print("=" * 60)

try:
    from huggingface_hub import HfApi

    api = HfApi()
    repo_id = "cstr/irodori-tts-GGUF"

    # Create repo if it doesn't exist
    try:
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        print(f"  Repo: {repo_id}")
    except Exception as e:
        print(f"  Repo creation: {e}")

    # Upload F16
    print(f"  Uploading F16 ({output_f16.stat().st_size / 1024 / 1024:.0f} MB)...")
    api.upload_file(
        path_or_fileobj=str(output_f16),
        path_in_repo="irodori-tts-500m-v3-f16.gguf",
        repo_id=repo_id,
        repo_type="model",
    )

    # Upload Q4_K
    print(f"  Uploading Q4_K ({output_q4k.stat().st_size / 1024 / 1024:.0f} MB)...")
    api.upload_file(
        path_or_fileobj=str(output_q4k),
        path_in_repo="irodori-tts-500m-v3-q4_k.gguf",
        repo_id=repo_id,
        repo_type="model",
    )
    print("  Upload complete!")

except Exception as e:
    print(f"  HF upload failed (non-fatal): {e}")
    print("  GGUF files are still in /kaggle/working for manual download")

# ── Summary ──────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
print(f"  F16:  {output_f16}  ({output_f16.stat().st_size / 1024 / 1024:.1f} MB)")
print(f"  Q4_K: {output_q4k}  ({output_q4k.stat().st_size / 1024 / 1024:.1f} MB)")
print(f"\nQ4_K is small enough to run on the VPS (8 GB RAM).")
