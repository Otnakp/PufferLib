#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

FULL=0
ENVS=()

while [ "$#" -gt 0 ]; do
  case "$1" in
    --full)
      FULL=1
      ;;
    -h|--help)
      echo "Usage: scripts/test_metal_matrix.sh [--full] [env ...]"
      exit 0
      ;;
    *)
      ENVS+=("$1")
      ;;
  esac
  shift
done

if [ "${#ENVS[@]}" -eq 0 ]; then
  ENVS=(breakout)
fi

FEATURES=(default)
if [ "$FULL" -eq 1 ]; then
  FEATURES=(default cpu_inference overlap train_fp16)
fi

run_feature() {
  local env_name="$1"
  local feature="$2"
  local args=(
    --steps 8192
    --total-agents 128
    --num-buffers 2
    --num-threads 4
    --horizon 8
    --minibatch-size 1024
    --iterations 2
  )

  case "$feature" in
    default)
      ;;
    cpu_inference)
      args+=(--cpu-inference)
      ;;
    overlap)
      args+=(--overlap)
      ;;
    train_fp16)
      args+=(--train-fp16)
      ;;
    *)
      echo "test_metal_matrix: unknown feature $feature" >&2
      return 2
      ;;
  esac

  echo "test_metal_matrix: env=$env_name feature=$feature"
  scripts/test_metal_smoke.sh "$env_name" "${args[@]}"
}

for env_name in "${ENVS[@]}"; do
  for feature in "${FEATURES[@]}"; do
    run_feature "$env_name" "$feature"
  done
done

echo "test_metal_matrix: ok"
