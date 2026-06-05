#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CXX="${CXX:-clang++}"
OUT_DIR="$ROOT/build/tests"
OUT="$OUT_DIR/test_metal_const_ring"

mkdir -p "$OUT_DIR"

echo "test_metal_unit: compile tests/test_metal_const_ring.mm"
"$CXX" \
  -std=c++17 \
  -ObjC++ \
  -fobjc-arc \
  -I. \
  -Isrc \
  tests/test_metal_const_ring.mm \
  -framework Metal \
  -framework Foundation \
  -framework Accelerate \
  -o "$OUT"

echo "test_metal_unit: run $OUT"
"$OUT"

echo "test_metal_unit: ok"
