#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "OPENROUTER_API_KEY is required." >&2
  exit 1
fi

run_dataset() {
  local dataset_key="$1"
  local config_path="$2"

  local timestamp
  timestamp="$(date -u +%Y%m%d_%H%M%S)"
  local run_id="sonnet46_reasoning_${dataset_key}_${timestamp}"
  local run_output_dir="runs/${dataset_key}"
  local run_dir="${run_output_dir}/${run_id}"
  local responses_file="${run_dir}/responses.jsonl"

  python3 scripts/openrouter_benchmark.py collect \
    --config "${config_path}" \
    --output-dir "${run_output_dir}" \
    --run-id "${run_id}"

  local grade_slug="judge_google_gemini-3.1-pro-preview"
  local grade_id="${run_id}__${grade_slug}"
  python3 scripts/openrouter_benchmark.py grade \
    --config "${config_path}" \
    --responses-file "${responses_file}" \
    --output-dir "${run_dir}" \
    --grade-id "${grade_id}" \
    --judge-model "google/gemini-3.1-pro-preview"

  local grade_dir="${run_dir}/grades/${grade_id}"
  local aggregate_dir
  aggregate_dir="$(
    python3 scripts/build_single_judge_aggregate.py \
      --responses-file "${responses_file}" \
      --grade-dir "${grade_dir}" \
      --output-dir "${run_dir}" \
      --aggregate-id "${grade_id}__aggregate"
  )"

  python3 scripts/publish_reasoning_dataset.py \
    --special-root "${ROOT_DIR}" \
    --run-dir "${run_dir}" \
    --grade-dir "${grade_dir}" \
    --aggregate-dir "${aggregate_dir}" \
    --output-dir "${ROOT_DIR}/data/viewer-input/${dataset_key}" \
    --dataset-key "${dataset_key}"
}

run_dataset "v1" "configs/config.v1.json"
run_dataset "v2" "configs/config.v2.json"
python3 scripts/build_sonnet46_reasoning_bundle.py
