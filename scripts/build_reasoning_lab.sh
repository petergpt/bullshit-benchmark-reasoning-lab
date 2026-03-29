#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -f "${ROOT_DIR}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/.env"
  set +a
fi

python3 "${ROOT_DIR}/scripts/export_reasoning_label_examples.py"
python3 "${ROOT_DIR}/scripts/sonnet46/build_reasoning_bundle.py"
