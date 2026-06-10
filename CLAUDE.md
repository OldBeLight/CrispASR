# CrispASR — Claude Code project instructions

## Build

```bash
CCACHE_DIR=/mnt/volume1/.ccache cmake -G Ninja -B build \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

## Test

```bash
# Unit tests (no models needed, ~5s):
ctest --test-dir build -L unit --timeout 30

# Integration / live tests (need models):
export CRISPASR_MODELS_DIR=/mnt/storage/gguf-models
source tests/env-live-tests.sh
ctest --test-dir build --rerun-failed --output-on-failure --timeout 300
```

Key env vars: `CRISPASR_MODELS_DIR`, `CRISPASR_MODEL_WHISPER`, `PARAFORMER_MODEL`.
See `tests/env-live-tests.sh` for the full list.

## Lint

```bash
./tools/format.sh --fix <changed .cpp/.h files>
```

Must use clang-format v18 — `tools/format.sh` enforces this.

## Commit workflow

- No PRs — merge locally and push to main directly
- Always run `tools/format.sh --fix` on changed C/C++ files before committing
- Work in git worktrees for non-trivial changes
