# GPT-5.4 Reasoning Annotation Studio

This studio is a local annotation tool for BullshitBench reasoning traces.

Its purpose is not just to read traces, but to create reusable supervision about
how GPT-5.4 handles nonsense prompts:

- where it clearly challenges the bogus premise
- where it questions whether the user is joking or testing it
- where it quietly goes along with the nonsense
- where it provides explanatory context
- where it actively attempts to solve
- where it seems confused or loses the thread

## Current Stack

- Frontend: `viewer/reasoning-annotation-studio.html`
- Local server: `scripts/reasoning_annotation_server.py`
- Shared dataset builder: `scripts/reasoning_atlas_data.py`
- Annotation store: `annotations/gpt54_reasoning_lab.json`
- Generated atlas bundle: `data/gpt54-reasoning-atlas.data.js`
- Derived model examples export: `data/gpt54_reasoning_label_examples.jsonl`
- JSONL exporter: `scripts/export_reasoning_label_examples.py`
- Branding asset: `assets/bsbench.png`

The lab lives under `reasoning-lab/`, but it reads the published benchmark
datasets from the main repo as immutable source inputs:

- `../data/latest/*`
- `../data/v2/latest/*`

The annotation store is intentionally a JSON sidecar:

- human-readable
- easy to diff
- easy to export/import/share
- easy for future AI-assisted labeling scripts to consume

If the annotation workload grows into multi-user review queues or batch jobs, the
same schema can be moved into SQLite later without changing the browser model.

## Annotation Model

Each saved annotation contains:

- `document_id`: stable reasoning document target such as `v2:fin_af_01:xhigh:reasoning`
- `dataset_key`, `question_id`, `variant_key`, `surface`
- `start`, `end`: character offsets into the canonical reasoning document
- `span_length`
- `quote`: exact selected text
- `normalized_quote`
- `quote_sha1`
- `prefix`, `suffix`: local context for recovery and later review
- `label_id`: taxonomy category
- `confidence`: `clear` or `borderline`
- `label_snapshot`: category id, name, guidance, and colors frozen at annotation time
- `comment`: freeform interpretation
- `author`
- `status`
- `provenance`: who/what labeled it, from which interface, and using what selection mode
- `created_at_utc`, `updated_at_utc`
- `source_text_hash`: hash of the underlying reasoning document

The store can also carry one document-level note per reasoning trace:

- `document_id`
- `summary`: overall note for the trace
- `label_id`: optional overall label from the same taxonomy
- `confidence`: optional `clear` or `borderline` when an overall label is set
- `label_snapshot`
- `author`
- `created_at_utc`, `updated_at_utc`
- `source_text_hash`
- `provenance`

This gives you three useful properties:

1. The labels survive page reloads and exports.
2. The data is machine-readable for future AI labeling.
3. Every annotation remains tied to a stable source span, not a fragile DOM node.
4. Category meaning survives future taxonomy edits because each annotation carries a label snapshot.

## Current Taxonomy

The seeded categories are deliberately broad and editable:

- `going_along_with_it`: treating the bogus premise as valid and reasoning within it
- `challenges_the_premise`: pushing back on the premise as wrong, incoherent, or invented
- `questions_intent`: questioning whether the user is joking, baiting, or testing rather than asking sincerely
- `provides_context`: adding background or framing without clearly solving or rejecting
- `attempts_to_solve`: actively working toward an answer, method, or procedure
- `confused`: misreading the question or losing the thread

These live inside the store itself, so if you rename, recolor, or add categories,
that taxonomy travels with the exported annotations.

## Selection Rules

- Selections are made against the normalized `xhigh` reasoning document.
- Spans are non-overlapping within a document.

The non-overlap rule is intentional. It keeps the first-pass labels clean enough
to act as primary supervision later.

## Future AI Assist

The intended workflow is:

1. You annotate a few cases yourself.
2. The store captures both your span choices and your category definitions.
3. A later assist pass can read those examples and propose labels on unlabeled
   traces using your implicit style.
4. Human-approved labels remain the ground truth; AI suggestions should be added
   as suggestions, not silent overwrites.

The store now also carries annotation defaults and provenance fields so later AI
passes can distinguish human gold labels from future machine suggestions.

## Model-Facing Export

The canonical store stays normalized for integrity and editing. For prompting or
training-style workflows, use the derived JSONL export instead:

- `data/gpt54_reasoning_label_examples.jsonl`

Each row is denormalized and includes the surrounding case context:

- question
- nonsense rationale
- final answer
- full reasoning trace
- selected span plus local context
- label and confidence
- overall trace label and note when available

This keeps the saved store compact and durable while making it easy to feed
later model-assisted labeling workflows.

## Files to Know

- `viewer/reasoning-annotation-studio.html`
- `viewer/gpt54-reasoning-atlas.html`
- `data/gpt54-reasoning-atlas.data.js`
- `data/gpt54_reasoning_label_examples.jsonl`
- `scripts/reasoning_annotation_server.py`
- `scripts/reasoning_atlas_data.py`
- `scripts/export_reasoning_label_examples.py`
- `annotations/gpt54_reasoning_lab.json`
