# Claude Sonnet 4.6 + Gemini Single-Judge Reasoning Capture

This folder is an isolated one-off pipeline for collecting Claude Sonnet 4.6
reasoning traces without affecting the main benchmark datasets under `data/`.

Design constraints:

- No edits to the core benchmark pipeline in the repo root.
- The benchmark runner is copied locally for isolation.
- Outputs are written only under this folder.
- The resulting artifacts mirror the main benchmark shape closely enough for
  later viewer ingestion, but they are intentionally not published into the
  main `data/latest` or `data/v2/latest` trees.

What this flow does:

1. Collect full v1 and v2 runs for `anthropic/claude-sonnet-4.6`
2. Capture both `reasoning=none` and `reasoning=high`
3. Grade with a single judge: `google/gemini-3.1-pro-preview`
4. Build a single-judge `aggregate.jsonl`
5. Publish viewer-only dataset snapshots under `data/viewer-input/v1` and `data/viewer-input/v2`
6. Build a normalized browser bundle for later viewer work

Key outputs:

- `data/viewer-input/v1/`
- `data/viewer-input/v2/`
- `data/claude-sonnet-4.6-gemini-single-judge.data.js`

Repro entrypoint:

- `scripts/run_special_capture.sh`
