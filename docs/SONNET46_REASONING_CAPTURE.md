# Sonnet 4.6 Reasoning Capture

This repo keeps the Sonnet 4.6 special-run data in the same overall structure
as the rest of the lab:

- shared viewer-ready data under `data/sonnet46/`
- maintenance scripts under `scripts/sonnet46/`

## Shared Data

Checked-in Sonnet 4.6 data lives under:

- `data/sonnet46/viewer-input/v1/`
- `data/sonnet46/viewer-input/v2/`
- `data/sonnet46/reasoning.data.js`

The repo keeps only the published viewer inputs needed by the lab:

- `responses.jsonl`
- `aggregate.jsonl`
- `manifest.json`

## Local-Only Data

Raw recapture runs are intentionally local-only and ignored:

- `data/sonnet46/runs/`

Those runs are useful for recapture, debugging, and backfill work, but they are
not required to reproduce the current shared lab UI.

## Maintenance Scripts

The Sonnet 4.6 maintenance workflow lives under `scripts/sonnet46/`:

- `build_reasoning_bundle.py`
- `run_special_capture.sh`
- `backfill_missing_traces.py`
- `openrouter_benchmark.py`
- `publish_reasoning_dataset.py`
- `build_single_judge_aggregate.py`

The capture configs and question sets used by that workflow live alongside the
scripts:

- `scripts/sonnet46/configs/`
- `scripts/sonnet46/questions/`

## Runtime Notes

- `OPENROUTER_API_KEY` is required for `scripts/sonnet46/run_special_capture.sh`
- `OPENROUTER_REFERER` and `OPENROUTER_APP_NAME` are optional headers
- `OPENAI_API_KEY`, `OPENAI_PROJECT`, and `OPENAI_ORGANIZATION` are only needed
  if you change the provider mix in the recapture workflow

## Main Commands

Rebuild the checked-in Sonnet 4.6 browser bundle:

```bash
python3 scripts/sonnet46/build_reasoning_bundle.py
```

Run the full Sonnet 4.6 recapture flow:

```bash
./scripts/sonnet46/run_special_capture.sh
```
