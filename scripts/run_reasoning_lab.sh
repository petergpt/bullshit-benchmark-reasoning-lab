#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_FIRST=1
BUILD_ONLY=0
SERVER_ARGS=()

while (($#)); do
  case "$1" in
    --skip-build)
      BUILD_FIRST=0
      ;;
    --build-only)
      BUILD_ONLY=1
      ;;
    --help|-h)
      cat <<'EOF'
Usage: ./scripts/run_reasoning_lab.sh [--skip-build] [--build-only] [-- server-args...]

Options:
  --skip-build  Start the annotation server without rebuilding derived data.
  --build-only  Rebuild derived data and exit without starting the server.
  --            Pass remaining arguments through to reasoning_annotation_server.py.
EOF
      exit 0
      ;;
    --)
      shift
      SERVER_ARGS+=("$@")
      break
      ;;
    *)
      SERVER_ARGS+=("$1")
      ;;
  esac
  shift
done

if ((BUILD_FIRST)); then
  "${ROOT_DIR}/scripts/build_reasoning_lab.sh"
fi

if ((BUILD_ONLY)); then
  exit 0
fi

if ((${#SERVER_ARGS[@]})); then
  exec python3 "${ROOT_DIR}/scripts/reasoning_annotation_server.py" "${SERVER_ARGS[@]}"
fi

exec python3 "${ROOT_DIR}/scripts/reasoning_annotation_server.py"
