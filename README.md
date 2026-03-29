# Reasoning Lab

Reasoning Lab is a local review studio for inspecting how models handled
BullshitBench prompts.

Instead of only looking at final benchmark scores, this repo lets you open the
full saved traces, compare models side by side, and mark the important moments
in the reasoning.

## What This Repo Contains

This repo already comes with:

- saved reasoning traces for `GPT-5.4 xHigh`
- saved reasoning traces for `Sonnet 4.6`
- both benchmark sets:
  - `Benchmark v2` with 100 prompts
  - `Benchmark v1` with 55 prompts
- a checked-in annotation store with reviewed work already in it
- the browser-ready data files needed to run the lab locally

In the current checked-in state, the repo includes:

- `928` saved span annotations
- `200` overall trace notes

## What You Can Do In The Studio

Open the studio and you can:

- browse all benchmark cases
- compare GPT-5.4 and Sonnet 4.6 on the same prompt
- see which cases were clear pushback, partial challenge, or accepted nonsense
- read the full answer and reasoning trace for each model
- add or edit span-level labels
- add an overall note for each trace
- review existing annotation progress
- export the current annotation state

The main app to share is:

- `viewer/reasoning-annotation-studio.html`

That is the one URL I would send to someone else.

If you want a zero-install link, the same studio can also be published as a
read-only GitHub Pages snapshot. The local server is still the editing mode.

## Quick Start

1. Clone the repo.
2. Start the local server:

```bash
cd /path/to/reasoning-lab && ./scripts/run_reasoning_lab.sh
```

3. Open:

- `http://127.0.0.1:8878/viewer/reasoning-annotation-studio.html`

`run_reasoning_lab.sh` rebuilds the checked-in browser data first, then starts
the local server.

If you are just restarting the app and do not need a rebuild:

```bash
./scripts/run_reasoning_lab.sh --skip-build
```

If you only want to refresh the derived browser data:

```bash
./scripts/build_reasoning_lab.sh
```

## What Someone Should Expect When They Open It

The studio has four main views:

- `Report`
  - high-level summary of how each model behaved across the benchmark
- `Explore`
  - sortable table of cases with grades, review status, and quick open actions
- `Review`
  - queue for looking through the current labelled set
- `Labelling`
  - detailed case view for reading traces and editing annotations

The default and intended shared surface is the studio itself, not a separate
dashboard or upstream repo.

## What Is Already Checked In

Shared in Git:

- the studio UI
- the current annotation store
- the bundled GPT-5.4 source snapshot used by this repo
- the bundled Sonnet 4.6 viewer data
- the generated browser data files
- the AI-labelling and evaluation scripts

Local-only and ignored:

- `.env`
- annotation lock files
- `data/sonnet46/runs/`
- `data/ai_label_runs/`
- `data/ai_label_evals/`
- `tmp/` QA screenshots and review artifacts

## Runtime Requirements

- Python `>=3.11`
- macOS or Linux for the built-in local server flow

The base app flow does not require any third-party Python packages.

## Optional API Keys

You only need API keys if you want to run the optional AI-assisted workflows.

Optional local env file:

- `.env`
- template: `.env.example`

Most people using the studio do not need to touch this.

Useful variables:

- `OPENROUTER_API_KEY`
  - needed for AI-assisted labelling and the special Sonnet capture flow
- `REASONING_LAB_BENCHMARK_ROOT`
  - optional override if you want to rebuild GPT-5.4 inputs from another local
    BullshitBench checkout instead of the bundled snapshot

## Repo Structure

- `viewer/`
  - browser UI files
- `annotations/`
  - saved annotation store
- `data/`
  - generated browser bundles and derived exports
- `source-data/`
  - bundled GPT-5.4 source snapshot used by this repo
- `scripts/`
  - local server, builders, and optional AI workflow scripts
- `docs/`
  - schema and workflow notes

## Optional Maintenance Commands

Rebuild the derived local data used by the studio:

```bash
./scripts/build_reasoning_lab.sh
```

Refresh just the derived label-example export:

```bash
python3 scripts/export_reasoning_label_examples.py
```

Build the read-only GitHub Pages snapshot:

```bash
python3 scripts/build_pages_site.py
```

Rebuild just the Sonnet 4.6 browser bundle:

```bash
python3 scripts/sonnet46/build_reasoning_bundle.py
```

Inspect the optional AI-labelling workflow:

```bash
python3 scripts/ai_label_reasoning_trace.py --help
```

Inspect the evaluation workflow:

```bash
python3 scripts/evaluate_ai_labeling_variants.py --help
```

## Notes

For the annotation schema and workflow details, see:

- `docs/ANNOTATION_MODEL.md`
- `docs/SONNET46_REASONING_CAPTURE.md`
