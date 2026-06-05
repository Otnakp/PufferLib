#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ENV_NAME="${1:-breakout}"

PYTHON="${PYTHON:-}"
if [ -z "$PYTHON" ]; then
  if [ -x "$ROOT/.venv/bin/python" ]; then
    PYTHON="$ROOT/.venv/bin/python"
  else
    PYTHON="python3"
  fi
fi

export PATH="$(dirname "$PYTHON"):$PATH"

check_import() {
  local expected_env="$1"
  EXPECT_ENV="$expected_env" "$PYTHON" - <<'PY'
import os
from pufferlib import _C

expected = os.environ["EXPECT_ENV"]
actual = getattr(_C, "env_name", None)
if actual != expected:
    raise SystemExit(f"compiled env mismatch: got {actual!r}, expected {expected!r}")

print(
    "test_metal_build_modes:",
    f"env={actual}",
    f"gpu={getattr(_C, 'gpu', None)}",
    f"precision_bytes={getattr(_C, 'precision_bytes', None)}",
)
PY
}

run_build() {
  local label="$1"
  shift

  echo "test_metal_build_modes: build env=$ENV_NAME mode=$label"
  ./build.sh "$ENV_NAME" "$@"
  check_import "$ENV_NAME"
}

run_build cpu --cpu
run_build debug --debug
run_build float --float
run_build default

echo "test_metal_build_modes: ok"
