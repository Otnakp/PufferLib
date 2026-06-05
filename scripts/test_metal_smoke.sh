#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ENV_NAME="${1:-breakout}"
shift || true

PYTHON="${PYTHON:-}"
if [ -z "$PYTHON" ]; then
  if [ -x "$ROOT/.venv/bin/python" ]; then
    PYTHON="$ROOT/.venv/bin/python"
  else
    PYTHON="python3"
  fi
fi

export PATH="$(dirname "$PYTHON"):$PATH"

echo "test_metal_smoke: build env=$ENV_NAME python=$PYTHON"
./build.sh "$ENV_NAME"

echo "test_metal_smoke: run env=$ENV_NAME"
"$PYTHON" scripts/metal_smoke.py --env "$ENV_NAME" "$@"
