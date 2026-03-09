# Reasoning Lab

## Background

- [BullshitBench](https://petergpt.github.io/bullshit-benchmark/viewer/index.v2.html)
  is the main benchmark project for testing how models respond to nonsense
  prompts.
- One of the benchmark findings was that GPT models were not performing very
  well.
- To understand why, it was not enough to look only at final scores. We needed
  to inspect the underlying reasoning traces.

## What This Repo Is

Reasoning Lab is the separate workspace for viewing, comparing, and annotating
those reasoning traces.

It is meant for close qualitative review: reading full traces, marking where a
model notices the bogus premise or goes along with it, comparing GPT-5.4 with
Sonnet 4.6, and building a reviewed label set for later AI-assisted labeling.

## Key Features

- Browser-based trace viewer and annotation UI.
- Span-level labels plus overall document notes.
- Side-by-side support for GPT-5.4 and Sonnet 4.6 trace sets.
- Shared JSON annotation store with review history and AI-review sessions.
- AI-labeling and evaluation scripts built on top of the human-reviewed labels.
- Checked-in datasets and browser bundles so the lab can run independently of
  the main BullshitBench repo.

## How To Use It

1. Start the local server.
2. Open `viewer/reasoning-annotation-studio.html` to review traces and save
   labels.
3. Open `viewer/gpt54-reasoning-atlas.html` to browse the GPT-5.4 dataset in a
   read-only way.
4. Save annotations into `annotations/reasoning_lab.json`.
5. Optionally run the AI-labeling and evaluation scripts to test or improve the
   labeling workflow.

## Repo Layout

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

For repeated local restarts, you can skip the rebuild step:

```bash
./scripts/run_reasoning_lab.sh --skip-build
```

If you only want to refresh the derived browser files, use:

```bash
./scripts/build_reasoning_lab.sh
```

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

- `scripts/build_reasoning_lab.sh`
- `scripts/run_reasoning_lab.sh`
- `scripts/reasoning_annotation_server.py`
- `scripts/build_gpt54_reasoning_atlas.py`
- `scripts/export_reasoning_label_examples.py`
- `scripts/ai_label_reasoning_trace.py`
- `scripts/evaluate_ai_labeling_variants.py`
- `scripts/sync_gpt54_source_data.py`
- `scripts/sonnet46/build_reasoning_bundle.py`
- `scripts/sonnet46/run_special_capture.sh`
