#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APPLY=0
KEEP_RUNTIME_NAME=""

usage() {
  cat <<'EOF'
Usage: release_guard.sh [--apply] [--keep-runtime <name>]

Options:
  --apply                 Delete detected generated artifacts.
  --keep-runtime <name>   Keep one runtime directory (for example: runtime-live-benchmark-final).
  --help                  Show this help message.

Behavior:
  1) Lists generated experiment artifacts that should not be uploaded.
  2) Deletes them only when --apply is set.
  3) Scans for potential key leakage strings and exits non-zero if found.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --apply)
      APPLY=1
      shift
      ;;
    --keep-runtime)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --keep-runtime" >&2
        exit 1
      fi
      KEEP_RUNTIME_NAME="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

is_kept_runtime() {
  local path="$1"
  if [[ -z "${KEEP_RUNTIME_NAME}" ]]; then
    return 1
  fi
  local base
  base="$(basename "$path")"
  [[ "$base" == "$KEEP_RUNTIME_NAME" ]]
}

collect_cleanup_targets() {
  local -a targets=()
  local path
  for path in \
    "$SCRIPT_DIR/runtime" \
    "$SCRIPT_DIR"/runtime-* \
    "$SCRIPT_DIR/__pycache__" \
    "$SCRIPT_DIR/.pytest_cache"
  do
    [[ -e "$path" ]] || continue
    if is_kept_runtime "$path"; then
      continue
    fi
    targets+=("$path")
  done
  if [[ ${#targets[@]} -eq 0 ]]; then
    return 0
  fi
  printf '%s\n' "${targets[@]}"
}

echo "== Release Guard =="
echo "Root: $SCRIPT_DIR"
if [[ -n "$KEEP_RUNTIME_NAME" ]]; then
  echo "Keeping runtime directory: $KEEP_RUNTIME_NAME"
fi

declare -a CLEANUP_TARGETS=()
while IFS= read -r line; do
  [[ -n "$line" ]] || continue
  CLEANUP_TARGETS+=("$line")
done < <(collect_cleanup_targets)
if [[ ${#CLEANUP_TARGETS[@]} -eq 0 ]]; then
  echo "No generated artifacts found."
else
  echo "Generated artifacts:"
  printf '  - %s\n' "${CLEANUP_TARGETS[@]}"
  if [[ $APPLY -eq 1 ]]; then
    rm -rf "${CLEANUP_TARGETS[@]}"
    echo "Deleted generated artifacts."
  else
    echo "Dry-run mode. Re-run with --apply to delete."
  fi
fi

if ! command -v rg >/dev/null 2>&1; then
  echo "rg not found; skipping key leakage scan."
  exit 0
fi

echo
echo "Scanning for key-like strings..."
set +e
SCAN_OUTPUT="$(rg -n --hidden --no-messages \
  '(Bearer[[:space:]]+xai-[A-Za-z0-9]{20,}|(GROK_API_KEY|OPENAI_API_KEY|XAI_API_KEY)=xai-[A-Za-z0-9]{20,}|xai-[A-Za-z0-9]{40,})' \
  "$SCRIPT_DIR")"
SCAN_STATUS=$?
set -e

if [[ $SCAN_STATUS -eq 0 ]]; then
  echo "Potential key leakage found:"
  echo "$SCAN_OUTPUT"
  exit 2
fi

if [[ $SCAN_STATUS -eq 1 ]]; then
  echo "No key-like strings found."
  exit 0
fi

echo "Key scan failed with status $SCAN_STATUS" >&2
exit "$SCAN_STATUS"
