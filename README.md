# Reasoning Lab

This folder isolates the GPT-5.4 reasoning-trace tooling from the rest of the
BullshitBench project.

It contains:

- `viewer/`: the reasoning atlas and annotation UI
- `data/`: the generated browser bundle and derived JSONL exports
- `scripts/`: the local server, data builders, and export tools
- `annotations/`: the saved labeling store
- `assets/`: local branding assets used by the reasoning viewers
- `docs/`: notes about the annotation model and storage format

The lab keeps its own UI, generated bundle, scripts, notes, and saved labels.
It only reads the benchmark source data from the parent repo:

- `../data/latest/*`
- `../data/v2/latest/*`

Primary entrypoints:

- `scripts/reasoning_annotation_server.py`
- `scripts/build_gpt54_reasoning_atlas.py`
- `scripts/export_reasoning_label_examples.py`
- `viewer/reasoning-annotation-studio.html`
- `viewer/gpt54-reasoning-atlas.html`
