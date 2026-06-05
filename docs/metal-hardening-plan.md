# PufferLib Metal Hardening Plan

## Goal

Turn the macOS Metal backend from a prototype that can run selected smoke tests into a backend that is reliable, maintainable, testable, and suitable for upstream review.

This work lives on a personal fork branch first. Upstream `origin` remains `PufferAI/PufferLib`; pushes should go to a user-owned remote and later open a PR against upstream.

## Quality Bar

- Prefer small, reviewable commits with one clear purpose each.
- Keep changes modular: scripts, tests, docs, and runtime fixes should be separated when possible.
- Do not hide unsupported modes. Fail early with clear errors.
- Avoid broad rewrites unless a local abstraction removes real duplication or fixes a concrete correctness issue.
- Comments should explain non-obvious Metal/CUDA-compat behavior only.
- Every behavior claim needs a command, test, or code path that proves it.

## Current Baseline

- Local branch: `codex/puffer-metal-hardening`.
- Upstream remote: `https://github.com/PufferAI/PufferLib.git`.
- Known untracked local artifact: root-level `breakout` Mach-O executable.
- Verified locally before hardening:
  - `PATH="$PWD/.venv/bin:$PATH" ./build.sh breakout`
  - `./.venv/bin/python -c "import pufferlib._C as C; print(C.env_name, C.gpu, C.precision_bytes)"`
  - brief `train breakout` smoke run for 8192 steps.
- Python is not available as `python` in the default shell. The repo-local `.venv/bin` must be on `PATH` for `build.sh`.
- `pytest` is not installed in the current virtual environment.

## Workstreams

### 1. Repository Hygiene

- Keep generated artifacts out of commits:
  - `build/`
  - `pufferlib/_C*.so`
  - `checkpoints/`
  - `logs/`
  - root-level standalone env binaries such as `breakout`.
- Add ignore rules only for reproducible build outputs.
- Keep upstream remote intact and add a personal fork remote for pushing.

### 2. Build Reliability

Required checks:

- Metal extension build for representative envs:
  - `breakout`
  - `pong`
  - `snake`
  - `cartpole`
- Build modes:
  - default Metal extension
  - `--float`
  - `--debug`
  - `--cpu`
- Darwin path must not break Linux/CUDA build script behavior.
- Unsupported Darwin modes must exit with useful messages.

### 3. Runtime Smoke Coverage

For each supported smoke env:

- Import `_C`.
- Check `_C.env_name`.
- Create `PuffeRL`.
- Run at least one rollout.
- Run at least one train step.
- Log metrics.
- Save and load weights when supported.
- Close without crash.
- Remove generated checkpoint/log artifacts after tests.

### 4. Correctness Coverage

Add or enable targeted checks for:

- config parsing and validation
- constant ring reservation and overflow behavior
- buffer wrapping and pointer lookup
- dtype and shape assumptions
- action masks
- rollout buffer copy/select behavior
- GAE/advantage path
- PPO loss accumulation
- MinGRU forward/backward path
- Muon update path
- fp32 training baseline
- `cpu_inference`
- `overlap`
- `train_fp16`

Feature combinations to verify:

- default fp32
- `cpu_inference=1`
- `overlap=1`
- `train_fp16=1`
- `cpu_inference=1` plus `train_fp16=1`
- `overlap=1` plus `train_fp16=1`

### 5. Stability Coverage

Run staged durations:

- tiny: 8k to 16k steps
- short: 100k steps
- medium: 1M steps when local runtime is acceptable

Track:

- crash or abort
- NaN losses/logits/weights
- increasing process memory
- command buffer sync frequency
- constant ring pressure
- multiple buffers
- multiple threads
- repeated create/close cycles in one process

### 6. Performance Coverage

Record baseline SPS and timing breakdown for:

- CPU backend
- Metal default
- Metal `cpu_inference`
- Metal `overlap`
- Metal `train_fp16`

Use the existing dashboard fields first:

- rollout
- eval GPU
- eval env
- train
- train sub-phases
- sync counts and sync time when available

Optimize only after correctness and stability failures are handled.

### 7. Runtime Hardening Targets

Audit and fix, in this order:

1. `src/metal_platform.h` and `src/metal_platform.mm`
   - stream lifecycle
   - command buffer lifecycle
   - constant ring handling
   - buffer lookup
   - sync semantics
   - Metal error reporting
2. `src/metal_bindings.mm`
   - Python API parity
   - config parsing
   - unsupported mode errors
   - environment mismatch errors
3. `src/metal_pufferlib.mm`
   - `PuffeRL` ownership
   - allocator lifetime
   - rollout/training synchronization
   - overlap weight copy correctness
   - CPU inference path
   - fp16 boundary buffers
4. `src/metal_kernels.mm` and `src/metal_shader_src.h`
   - dispatch bounds
   - shape assumptions
   - fp16/fp32 conversion
   - action mask behavior
   - reduction correctness
   - fallback GEMM path

## Test Harness Deliverables

Add scripts that can run without requiring pytest:

- `scripts/metal_smoke.py`
  - Python smoke runner for import/build metadata/runtime train checks.
- `scripts/test_metal_smoke.sh`
  - Builds an env and runs `metal_smoke.py`.
- `scripts/test_metal_matrix.sh`
  - Runs a small matrix of envs and modes.
- `scripts/test_metal_unit.sh`
  - Compiles and runs ObjC++/C++ unit checks such as `tests/test_metal_const_ring.mm`.

Scripts must:

- use repo root consistently
- prefer `.venv/bin/python` when present
- clean up their own checkpoints/logs where practical
- fail non-zero on real errors
- print commands and important metadata

## Initial Commit Plan

1. `docs: add Metal hardening plan`
2. `test: add Metal smoke harness`
3. `test: add Metal unit test runner`
4. `test: add Metal matrix runner`
5. `build: ignore local standalone env binaries`
6. First runtime fix commit, based on test failures

## Push Plan

1. Confirm GitHub identity with `gh auth status` and `gh api user`.
2. Create or reuse a personal fork of `PufferAI/PufferLib`.
3. Add personal remote. Current local remote name: `fork`.
4. Push `codex/puffer-metal-hardening` to the personal remote.
5. Open a draft PR against `PufferAI/PufferLib` only after the first useful test/fix set is committed.

## Done Criteria

This backend is not considered hardened until there is evidence for all of:

- representative env build matrix passes
- runtime smoke matrix passes
- targeted unit checks pass
- unsupported modes fail clearly
- generated artifacts stay out of commits
- docs explain setup, commands, support status, and troubleshooting
- commits are reviewable and separated by concern
- branch is pushed to the personal fork
- draft PR can be opened with exact verification commands and known limitations
