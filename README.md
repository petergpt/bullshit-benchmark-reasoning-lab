# Reasoning Lab

This repo isolates the reasoning-trace tooling from the wider BullshitBench
project while staying runnable on its own.

It contains:

- `viewer/`: the reasoning atlas and annotation UI
- `data/`: the generated browser bundle and derived JSONL exports
- `source-data/`: bundled GPT-5.4 benchmark source snapshot used by the lab
- `scripts/`: the local server, data builders, and export tools
- `annotations/`: the saved labeling store
- `assets/`: local branding assets used by the reasoning viewers
- `docs/`: notes about the annotation model and storage format

The lab keeps its own UI, scripts, notes, saved labels, and the exact dataset
snapshots needed to reproduce the current reasoning views.

Bundled inputs:

- `source-data/bullshit-benchmark/data/latest/*`
- `source-data/bullshit-benchmark/data/v2/latest/*`
- `claude-sonnet-4.6-gemini-single-judge/data/viewer-input/v1/*`
- `claude-sonnet-4.6-gemini-single-judge/data/viewer-input/v2/*`

Checked-in generated outputs:

- `data/gpt54-reasoning-atlas.data.js`
- `data/gpt54_reasoning_label_examples.jsonl`
- `claude-sonnet-4.6-gemini-single-judge/data/claude-sonnet-4.6-gemini-single-judge.data.js`

The only remaining benchmark path dependency is optional:

- `REASONING_LAB_BENCHMARK_ROOT`
  If set, the lab will read GPT-5.4 source inputs from that external
  BullshitBench checkout instead of the bundled `source-data/` snapshot.

Runtime notes:

- Python `>=3.11`
- macOS/Linux supported directly
- Optional local env file: `.env` (template in `.env.example`)
- Base annotation app runs without API keys
- AI-assisted labeling requires `OPENROUTER_API_KEY`
- The special Sonnet capture scripts require `OPENROUTER_API_KEY`
- OpenAI env vars are only needed if you change the special-capture provider mix

Primary entrypoints:

- `scripts/run_reasoning_lab.sh`
- `scripts/reasoning_annotation_server.py`
- `scripts/sync_gpt54_source_data.py`
- `scripts/build_gpt54_reasoning_atlas.py`
- `scripts/export_reasoning_label_examples.py`
- `viewer/reasoning-annotation-studio.html`
- `viewer/gpt54-reasoning-atlas.html`
