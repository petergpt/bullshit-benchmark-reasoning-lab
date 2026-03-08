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
  Checked-in annotation store for the current lab state.
- `data/`
  Generated browser bundles and derived JSONL exports used by the viewers.
- `source-data/`
  Bundled GPT-5.4 source snapshot used by the lab by default.
- `claude-sonnet-4.6-gemini-single-judge/data/viewer-input/`
  Published Sonnet 4.6 trace snapshots used by the lab.
- `scripts/`
  Local server, builders, AI-labeling utilities, and sync helpers.
- `docs/`
  Notes on the annotation model, store schema, and workflows.

## What is shared vs local-only

Shared in Git:

- the viewers
- the saved labels and review state
- the bundled GPT-5.4 and Sonnet 4.6 reasoning snapshots
- the generated browser bundles needed to open the current lab state
- the AI-labeling and evaluation scripts

Kept local and ignored:

- `.env`
- annotation lock files
- `claude-sonnet-4.6-gemini-single-judge/runs/`
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
  BullshitBench checkout instead of the bundled `source-data/` snapshot.

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
python3 claude-sonnet-4.6-gemini-single-judge/scripts/build_sonnet46_reasoning_bundle.py
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

## Data Model Notes

The lab is self-contained by default and uses these bundled inputs:

- `source-data/bullshit-benchmark/data/latest/*`
- `source-data/bullshit-benchmark/data/v2/latest/*`
- `claude-sonnet-4.6-gemini-single-judge/data/viewer-input/v1/*`
- `claude-sonnet-4.6-gemini-single-judge/data/viewer-input/v2/*`

Checked-in generated outputs:

- `data/gpt54-reasoning-atlas.data.js`
- `data/gpt54_reasoning_label_examples.jsonl`
- `claude-sonnet-4.6-gemini-single-judge/data/claude-sonnet-4.6-gemini-single-judge.data.js`

For the annotation-store schema and review-session model, see:

- `docs/REASONING_ANNOTATION_STUDIO.md`

## Main Entry Points

- `scripts/run_reasoning_lab.sh`
- `scripts/reasoning_annotation_server.py`
- `scripts/build_gpt54_reasoning_atlas.py`
- `scripts/export_reasoning_label_examples.py`
- `scripts/ai_label_reasoning_trace.py`
- `scripts/evaluate_ai_labeling_variants.py`
- `scripts/sync_gpt54_source_data.py`
