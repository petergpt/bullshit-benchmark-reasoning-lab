# Reasoning Lab Annotation Model

This studio is the shared annotation layer for Reasoning Lab across the GPT-5.4
and Sonnet 4.6 trace sets.

For external sharing, treat `viewer/reasoning-annotation-studio.html` as the
canonical product surface.

Its purpose is not just to read traces, but to create reusable supervision about
how models handle nonsense prompts:

- where it clearly challenges the bogus premise
- where it questions whether the user is joking or testing it
- where it quietly goes along with the nonsense
- where it provides explanatory context
- where it actively attempts to solve
- where it seems confused or loses the thread

## Current Stack

- Frontend: `viewer/reasoning-annotation-studio.html`
- Local server: `scripts/reasoning_annotation_server.py`
- Shared multi-model dataset builder: `scripts/reasoning_lab_data.py`
- Shared path resolver: `scripts/reasoning_lab_paths.py`
- Annotation store: `annotations/reasoning_lab.json`
- Derived model examples export: `data/reasoning_label_examples.jsonl`
- JSONL exporter: `scripts/export_reasoning_label_examples.py`
- Branding asset: `assets/bsbench.png`

The lab is now self-contained by default. It reads immutable bundled source
inputs from this repo:

- `source-data/gpt54/latest/*`
- `source-data/gpt54/v2/latest/*`
- `data/sonnet46/viewer-input/v1/*`
- `data/sonnet46/viewer-input/v2/*`

Optional override:

- `REASONING_LAB_BENCHMARK_ROOT`
  If set, the GPT-5.4 source builders read from that external BullshitBench
  checkout instead of the bundled `source-data/gpt54/`.

## Runtime Notes

- Python `>=3.11`
- The local annotation server currently uses `fcntl`, so the direct server flow
  is for macOS/Linux
- Optional local env file: `.env` (template in `.env.example`)
- The base app works without API keys
- AI-assisted labeling requires `OPENROUTER_API_KEY`
- The isolated Sonnet 4.6 recapture flow also requires `OPENROUTER_API_KEY`
- Optional provider envs for the special-capture scripts:
  `OPENROUTER_REFERER`, `OPENROUTER_APP_NAME`, `OPENAI_API_KEY`,
  `OPENAI_PROJECT` / `OPENAI_PROJECT_ID`, and `OPENAI_ORGANIZATION` /
  `OPENAI_ORG` / `OPENAI_ORG_ID`

The annotation store is intentionally a JSON sidecar:

- human-readable
- easy to diff
- easy to export/import/share
- easy for future AI-assisted labeling scripts to consume

If the annotation workload grows into multi-user review queues or batch jobs, the
same schema can be moved into SQLite later without changing the browser model.

## Annotation Model

Each saved annotation contains:

- `document_id`: stable reasoning document target such as `gpt54_high:v2:fin_af_01:xhigh:reasoning` or `sonnet46_high:v2:sw_cds_01:high:reasoning`
- `dataset_key`, `question_id`, `variant_key`, `surface`
- `start`, `end`: character offsets into the canonical reasoning document
- `span_length`
- `quote`: exact selected text
- `normalized_quote`
- `quote_sha1`
- `prefix`, `suffix`: local context for recovery and later review
- `label_id`: taxonomy category
- `label_snapshot`: category id, name, guidance, and colors frozen at annotation time
- `comment`: freeform interpretation
- `author`
- `status`
- `ai_labelled`: whether the span was created by the AI labelling workflow
- `human_reviewed`: whether a human has explicitly reviewed or created this span
- `reviewed_by`, `reviewed_at_utc`
- `review_session_id`: review-session linkage when the span came from or was edited during an AI review pass
- `review_origin`: either `ai_suggestion` or `human_addition` when linked to a review session
- `origin_suggestion_id`: stable id for the original AI suggestion this span came from
- `provenance`: who/what labeled it, from which interface, and using what selection mode
- `created_at_utc`, `updated_at_utc`
- `source_text_hash`: hash of the underlying reasoning document

The store can also carry one document-level note per reasoning trace:

- `document_id`
- `summary`: overall note for the trace
- `label_id`: optional overall label from the same taxonomy
- `label_snapshot`
- `author`
- `ai_labelled`
- `human_reviewed`
- `reviewed_by`, `reviewed_at_utc`
- `review_session_id`
- `review_origin`
- `origin_suggestion_id`
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

- Selections are made against the source's primary reasoning document.
- For GPT-5.4 xHigh that primary document is `xhigh`; for Sonnet 4.6 it is `high`.
- Spans are non-overlapping within a document.

The non-overlap rule is intentional. It keeps the first-pass labels clean enough
to act as primary supervision later.

## Future AI Assist

The intended workflow is:

1. You annotate a few cases yourself.
2. The store captures both your span choices and your category definitions.
3. A later assist pass can read those examples and propose labels on not labelled
   traces using your implicit style.
4. Human-approved labels remain the ground truth; AI suggestions should be added
   as suggestions, not silent overwrites.

The store now also carries annotation defaults, provenance fields, immutable AI
review sessions, and append-only review events so later analysis can separate:

- the original AI suggestion
- what the human kept as-is
- what the human edited
- what the human deleted
- what the human added during review

The current integrated flow is one trace per request:

1. The UI sends `POST /api/ai-label` for a single `document_id`.
2. The server calls OpenRouter with the current default config.
3. The response is normalized into the same store schema as manual labels.
4. AI-created items are marked `ai_labelled: true` and `human_reviewed: false`.
5. Once a human reviews or edits those labels, the same records stay in place but
   flip to `human_reviewed: true`, and human-added replacement/split spans stay
   linked to the active review session as `review_origin: human_addition`.

There are now two server-side AI labelling modes:

- `replace`: for traces with no human-reviewed labels yet; existing labels on that
  trace are replaced by the new AI suggestion set
- `complete_existing`: for traces that already have human-reviewed labels; the
  server passes those accepted human labels into the prompt, keeps them fixed,
  removes only unreviewed AI suggestions on that trace, and saves only the
  complementary AI additions

To make later prompt-tuning and calibration work possible without guessing,
every AI run also creates an immutable `review_session` snapshot in the store:

- `review_sessions`: one entry per AI suggestion pass, including full suggested
  note/spans plus model/prompt/config provenance
- `review_events`: append-only human actions against those sessions, with
  `added`, `accepted`, `edited`, and `deleted` events carrying before/after
  snapshots and optional freeform `feedback` text

This means you can later reconstruct the exact delta between:

- what the model proposed
- what remained after human review
- which corrections were systematic enough to feed back into prompts/settings

The default production config is versioned in provenance so later experiments can
coexist without ambiguity:

- `ai_label_workflow`
- `ai_label_workflow_version`
- `label_mode`
- `ai_label_config_id`
- `provider`, `model_id`, `reasoning_effort`, `temperature`
- `prompt_profile`, `prompt_version`
- `response_id`, `response_created`

This means an “AI labelled” suggestion can later be reviewed and amended in the
same UI without losing the original model/prompt lineage.

## Prompt-Tuning Notes

- Overall notes should describe the model's general approach to the reasoning
  trace, not the question-specific content. The goal is to make those notes easy
  to scan across domains, so they should emphasize the broad response pattern
  (for example: mostly pushback, soft reframing, sustained solving, going along
  with the frame) rather than listing the concrete domain details from one
  prompt.
- Use the simplest, clearest language possible in overall notes. Avoid jargon,
  dense phrasing, and overly specific detail.
- Prefer direct summaries of the response style over factual retelling. Phrases
  like `goes along with the nonsense` or `mostly rejects the premise` are better
  than walking through the details of one trace.
- Be careful with `attempts_to_solve`. It is closer to a negative label for when
  the model starts genuinely working the bogus frame or drifts into nonsense
  operationalization. Do not use it just because the model notices the premise
  is wrong but still tries to be helpful in a legitimate way. Helpful reframing
  or sensible redirection can still be the right response and should not be
  over-labeled as `attempts_to_solve`. Judge it in the context of the wider
  reasoning trace, not from one sentence in isolation.
- For the overall label, if the trace does contain some solving moves but the
  broader approach is mainly rejecting the premise, prefer the rejecting label
  over `attempts_to_solve`.
- If the overall trace is mostly `going_along_with_it` but it has brief moments
  of real doubt or explicit recognition that the premise may not make sense,
  those spans are still worth picking out as `challenges_the_premise`. Those
  exceptions are interesting and should be surfaced.
- Review-derived trend so far: the AI often gets the broad overall label roughly
  right, but span-level corrections have mostly been to add more
  `challenges_the_premise` or `going_along_with_it` moments and to trim
  over-eager `attempts_to_solve` / `provides_context` spans.

## Model-Facing Export

The canonical store stays normalized for integrity and editing. For prompting or
training-style workflows, use the derived JSONL export instead:

- `data/reasoning_label_examples.jsonl`

Each row is denormalized and includes the surrounding case context:

- question
- nonsense rationale
- final answer
- full reasoning trace
- selected span plus local context
- label
- overall trace label and note when available
- review-session linkage (`review_session_id`, `review_origin`,
  `origin_suggestion_id`)
- review-session status and event count

This keeps the saved store compact and durable while making it easy to feed
later model-assisted labeling workflows.

## Files to Know

- `viewer/reasoning-annotation-studio.html`
- `data/reasoning_label_examples.jsonl`
- `source-data/gpt54/`
- `scripts/reasoning_annotation_server.py`
- `scripts/run_reasoning_lab.sh`
- `scripts/sync_gpt54_source_data.py`
- `scripts/reasoning_lab_paths.py`
- `scripts/reasoning_lab_data.py`
- `scripts/export_reasoning_label_examples.py`
- `annotations/reasoning_lab.json`
