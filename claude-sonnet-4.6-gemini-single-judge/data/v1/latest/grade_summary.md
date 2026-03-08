# BullshitBench Results

- Grade ID: `sonnet46_reasoning_v1_20260307_205051__judge_google_gemini-3.1-pro-preview`
- Timestamp (UTC): `2026-03-07T20:58:07.490402+00:00`
- Source responses: `/Users/peter/bullshit-benchmark/reasoning-lab/claude-sonnet-4.6-gemini-single-judge/runs/v1/sonnet46_reasoning_v1_20260307_205051/responses.jsonl`
- Judge model: `google/gemini-3.1-pro-preview`
- Records: `110`
- Scored: `110`
- Errors: `0`

| Rank | Model | Avg Score | Detected (2) | Fooled (0) | 0/1/2/3 | Errors |
|---|---|---:|---:|---:|---|---:|
| 1 | `anthropic/claude-sonnet-4.6@reasoning=none` | 1.9091 | 0.9455 | 0.0364 | 2/1/52/0 | 0 |
| 2 | `anthropic/claude-sonnet-4.6@reasoning=high` | 1.8364 | 0.8909 | 0.0545 | 3/3/49/0 | 0 |

## Per-Technique Average Score

### `anthropic/claude-sonnet-4.6@reasoning=none`
| Technique | Avg Score |
|---|---:|
| `authoritative_framing_of_nothing` | 2.0000 |
| `causal_chimera` | 2.0000 |
| `cross_domain_concept_stitching` | 1.8571 |
| `false_granularity` | 2.0000 |
| `inverted_nonexistent_dependency` | 2.0000 |
| `misapplied_mechanism` | 2.0000 |
| `plausible_nonexistent_framework` | 1.3333 |
| `reified_metaphor` | 2.0000 |
| `temporal_category_error` | 1.6000 |
| `wrong_unit_of_analysis` | 2.0000 |

### `anthropic/claude-sonnet-4.6@reasoning=high`
| Technique | Avg Score |
|---|---:|
| `authoritative_framing_of_nothing` | 2.0000 |
| `causal_chimera` | 2.0000 |
| `cross_domain_concept_stitching` | 1.7143 |
| `false_granularity` | 2.0000 |
| `inverted_nonexistent_dependency` | 2.0000 |
| `misapplied_mechanism` | 2.0000 |
| `plausible_nonexistent_framework` | 1.3333 |
| `reified_metaphor` | 1.8000 |
| `temporal_category_error` | 1.2000 |
| `wrong_unit_of_analysis` | 2.0000 |

## Run Stability

### `anthropic/claude-sonnet-4.6@reasoning=none`
- run 1: 1.9091
- run avg stddev: n/a

### `anthropic/claude-sonnet-4.6@reasoning=high`
- run 1: 1.8364
- run avg stddev: n/a

