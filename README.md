# Reasoning Lab

Reasoning Lab is a standalone BullshitBench sub-project for inspecting and
labeling model reasoning traces.

This repo is meant to be privately shareable on its own. It includes the UI,
the saved annotation store, the current published reasoning datasets, and the
scripts needed to rebuild the browser bundles and run the local annotation
server without requiring the parent BullshitBench checkout.

## What is in this repo

- `viewer/`
  Static browser UIs:
  - `reasoning-annotation-studio.html`
  - `gpt54-reasoning-atlas.html`
- `annotations/`
  Checked-in annotation store for the current lab state
  (`annotations/reasoning_lab.json`).
- `data/`
  Generated browser bundles, derived JSONL exports, and the checked-in Sonnet
  4.6 reasoning snapshots used by the viewers.
- `source-data/`
  Bundled GPT-5.4 source snapshot used by the lab by default. This is a
  checked-in local snapshot under `source-data/gpt54/`, not a nested upstream
  checkout.
- `scripts/`
  Local server, builders, AI-labeling utilities, sync helpers, and Sonnet 4.6
  recapture utilities under `scripts/sonnet46/`.
- `docs/`
  Notes on the annotation model, store schema, and workflows.

## What is shared vs local-only

Shared in Git:

- the viewers
- the saved labels and review state
- the bundled GPT-5.4 and Sonnet 4.6 reasoning snapshots
- the generated browser bundles needed to open the current lab state
- the AI-labeling and evaluation scripts
- the Sonnet 4.6 recapture/build scripts and their config/question inputs

Kept local and ignored:

- `.env`
- annotation lock files
- `data/sonnet46/runs/`
- ad hoc AI-labeling eval output under `data/ai_label_runs/` and
  `data/ai_label_evals/`

## Runtime Requirements

- Python `>=3.11`
- macOS or Linux for the direct server flow
- no required third-party Python packages for the base app flow

The local server uses `fcntl`, so the main annotation flow is currently
Unix-style rather than Windows-native.

## Quick Start

1. Clone the repo.
2. Optionally copy `.env.example` to `.env` and fill in any API keys you need.
3. Start the lab:

```bash
cd /Users/peter/bullshit-benchmark/reasoning-lab && ./scripts/run_reasoning_lab.sh
```

4. Open:

- `http://127.0.0.1:8878/viewer/reasoning-annotation-studio.html`
- `http://127.0.0.1:8878/viewer/gpt54-reasoning-atlas.html`

`run_reasoning_lab.sh` rebuilds the checked-in browser data files first, then
starts the annotation server.

## Environment Variables

Optional local env file:

- `.env`
- template: `.env.example`

Supported variables:

- `OPENROUTER_API_KEY`
  Required for AI-assisted labeling and the special Sonnet capture flow.
- `OPENROUTER_REFERER`
  Optional OpenRouter header.
- `OPENROUTER_APP_NAME`
  Optional OpenRouter app name header.
- `OPENAI_API_KEY`
- `OPENAI_PROJECT`
- `OPENAI_ORGANIZATION`
  Only needed if you change the provider mix used by the special-capture flow.
- `REASONING_LAB_BENCHMARK_ROOT`
  Optional override. If set, the GPT-5.4 builders read from that external
  BullshitBench checkout instead of the bundled `source-data/gpt54/` snapshot.

## Useful Commands

Rebuild the GPT-5.4 atlas bundle:

```bash
python3 scripts/build_gpt54_reasoning_atlas.py
```

Refresh the derived label-example export:

```bash
python3 scripts/export_reasoning_label_examples.py
```

Rebuild the Sonnet 4.6 viewer bundle:

```bash
python3 scripts/sonnet46/build_reasoning_bundle.py
```

Sync the bundled GPT-5.4 source snapshot from a local BullshitBench checkout:

```bash
python3 scripts/sync_gpt54_source_data.py --help
```

Inspect the AI-labeling workflow:

```bash
python3 scripts/ai_label_reasoning_trace.py --help
```

Inspect the AI-labeling evaluation workflow:

```bash
python3 scripts/evaluate_ai_labeling_variants.py --help
```

Inspect the Sonnet 4.6 recapture flow:

```bash
python3 scripts/sonnet46/openrouter_benchmark.py --help
```

## Data Model Notes

The lab is self-contained by default and uses these bundled inputs:

- `source-data/gpt54/latest/*`
- `source-data/gpt54/v2/latest/*`
- `data/sonnet46/viewer-input/v1/*`
- `data/sonnet46/viewer-input/v2/*`

Checked-in generated outputs:

- `data/gpt54-reasoning-atlas.data.js`
- `data/reasoning_label_examples.jsonl`
- `data/sonnet46/reasoning.data.js`

For the annotation-store schema and review-session model, see:

- `docs/ANNOTATION_MODEL.md`
- `docs/SONNET46_REASONING_CAPTURE.md`

## Main Entry Points

- `scripts/run_reasoning_lab.sh`
- `scripts/reasoning_annotation_server.py`
- `scripts/build_gpt54_reasoning_atlas.py`
- `scripts/export_reasoning_label_examples.py`
- `scripts/ai_label_reasoning_trace.py`
- `scripts/evaluate_ai_labeling_variants.py`
- `scripts/sync_gpt54_source_data.py`
- `scripts/sonnet46/build_reasoning_bundle.py`
- `scripts/sonnet46/run_special_capture.sh`
