#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -f "${ROOT_DIR}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/.env"
  set +a
fi

python3 "${ROOT_DIR}/scripts/build_gpt54_reasoning_atlas.py"
python3 "${ROOT_DIR}/scripts/export_reasoning_label_examples.py"
python3 "${ROOT_DIR}/claude-sonnet-4.6-gemini-single-judge/scripts/build_sonnet46_reasoning_bundle.py"
exec python3 "${ROOT_DIR}/scripts/reasoning_annotation_server.py" "$@"
