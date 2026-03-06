#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

shopt -s nullglob

cleanup_patterns=(
  ".pytest_cache"
  ".mypy_cache"
  ".ruff_cache"
  ".coverage"
  ".coverage.*"
  ".codex_runs"
  "ai-sdk-tool-parser-*.tgz"
  "dist"
  "*.tsbuildinfo"
  "experiments/*/__pycache__"
  "experiments/*/.pytest_cache"
  "experiments/*/runtime"
  "experiments/*/runtime-*"
)

removed_any=0

for pattern in "${cleanup_patterns[@]}"; do
  matches=( $pattern )
  if [ ${#matches[@]} -eq 0 ]; then
    continue
  fi

  rm -rf -- "${matches[@]}"
  removed_any=1
  printf 'removed %s\n' "${matches[@]}"
done

if [ "$removed_any" -eq 0 ]; then
  echo "nothing to clean"
fi
