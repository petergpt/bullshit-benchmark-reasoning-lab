# BullshitBench Results

- Grade ID: `sonnet46_reasoning_v2_20260307_210007__judge_google_gemini-3.1-pro-preview`
- Timestamp (UTC): `2026-03-07T21:19:01.976772+00:00`
- Source responses: `/Users/peter/bullshit-benchmark/reasoning-lab/claude-sonnet-4.6-gemini-single-judge/runs/v2/sonnet46_reasoning_v2_20260307_210007/responses.jsonl`
- Judge model: `google/gemini-3.1-pro-preview`
- Records: `200`
- Scored: `200`
- Errors: `0`

| Rank | Model | Avg Score | Detected (2) | Fooled (0) | 0/1/2/3 | Errors |
|---|---|---:|---:|---:|---|---:|
| 1 | `anthropic/claude-sonnet-4.6@reasoning=high` | 1.8600 | 0.9200 | 0.0600 | 6/2/92/0 | 0 |
| 2 | `anthropic/claude-sonnet-4.6@reasoning=none` | 1.8500 | 0.9100 | 0.0600 | 6/3/91/0 | 0 |

## Per-Technique Average Score

### `anthropic/claude-sonnet-4.6@reasoning=high`
| Technique | Avg Score |
|---|---:|
| `authoritative_framing` | 2.0000 |
| `confident_extrapolation` | 2.0000 |
| `cross_domain_stitching` | 1.6000 |
| `fabricated_authority` | 2.0000 |
| `false_granularity` | 2.0000 |
| `misapplied_mechanism` | 1.8462 |
| `nested_nonsense` | 1.8571 |
| `plausible_nonexistent_framework` | 1.8750 |
| `reified_metaphor` | 2.0000 |
| `specificity_trap` | 1.7500 |
| `sunk_cost_framing` | 1.7143 |
| `temporal_category_error` | 1.5000 |
| `wrong_unit_of_analysis` | 2.0000 |

### `anthropic/claude-sonnet-4.6@reasoning=none`
| Technique | Avg Score |
|---|---:|
| `authoritative_framing` | 2.0000 |
| `confident_extrapolation` | 2.0000 |
| `cross_domain_stitching` | 2.0000 |
| `fabricated_authority` | 2.0000 |
| `false_granularity` | 2.0000 |
| `misapplied_mechanism` | 1.8462 |
| `nested_nonsense` | 1.2857 |
| `plausible_nonexistent_framework` | 1.8750 |
| `reified_metaphor` | 2.0000 |
| `specificity_trap` | 1.7500 |
| `sunk_cost_framing` | 1.7143 |
| `temporal_category_error` | 1.6667 |
| `wrong_unit_of_analysis` | 2.0000 |

## Run Stability

### `anthropic/claude-sonnet-4.6@reasoning=high`
- run 1: 1.8600
- run avg stddev: n/a

### `anthropic/claude-sonnet-4.6@reasoning=none`
- run 1: 1.8500
- run avg stddev: n/a

