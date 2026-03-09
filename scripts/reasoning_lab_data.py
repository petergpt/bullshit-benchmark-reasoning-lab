#!/usr/bin/env python3
"""Shared multi-model data builder for the BullshitBench reasoning lab."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from reasoning_lab_paths import LAB_ROOT, benchmark_dataset_dir

SOURCES = {
    "gpt54_high": {
        "key": "gpt54_high",
        "label": "GPT-5.4 xHigh",
        "description": "GPT-5.4 reasoning traces from the main BullshitBench runs.",
        "model_id": "openai/gpt-5.4",
        "primary_variant_key": "xhigh",
        "baseline_variant_key": "none",
        "variant_models": {
            "none": "openai/gpt-5.4@reasoning=none",
            "xhigh": "openai/gpt-5.4@reasoning=xhigh",
        },
        "datasets": {
            "v2": {
                "label": "Benchmark v2",
                "base_dir": benchmark_dataset_dir("v2"),
                "description": "100-question refreshed benchmark",
                "expected_question_count": 100,
            },
            "v1": {
                "label": "Benchmark v1",
                "base_dir": benchmark_dataset_dir("v1"),
                "description": "55-question original benchmark",
                "expected_question_count": 55,
            },
        },
    },
    "sonnet46_high": {
        "key": "sonnet46_high",
        "label": "Sonnet 4.6",
        "description": "Claude Sonnet 4.6 reasoning traces from the isolated special run.",
        "model_id": "anthropic/claude-sonnet-4.6",
        "judge_model": "google/gemini-3.1-pro-preview",
        "primary_variant_key": "high",
        "baseline_variant_key": "none",
        "variant_models": {
            "none": "anthropic/claude-sonnet-4.6@reasoning=none",
            "high": "anthropic/claude-sonnet-4.6@reasoning=high",
        },
        "datasets": {
            "v2": {
                "label": "Benchmark v2",
                "base_dir": LAB_ROOT
                / "data"
                / "sonnet46"
                / "viewer-input"
                / "v2",
                "description": "100-question isolated Sonnet 4.6 reasoning capture",
                "expected_question_count": 100,
            },
            "v1": {
                "label": "Benchmark v1",
                "base_dir": LAB_ROOT
                / "data"
                / "sonnet46"
                / "viewer-input"
                / "v1",
                "description": "55-question isolated Sonnet 4.6 reasoning capture",
                "expected_question_count": 55,
            },
        },
    },
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object JSON in {path}")
    return payload


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected object JSONL row in {path}")
            rows.append(payload)
    return rows


def text_or_empty(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def first_message_payload(response_raw: Any) -> dict[str, Any]:
    if not isinstance(response_raw, dict):
        return {}
    choices = response_raw.get("choices")
    if not isinstance(choices, list) or not choices:
        return {}
    first = choices[0]
    if not isinstance(first, dict):
        return {}
    message = first.get("message")
    return message if isinstance(message, dict) else {}


def collect_reasoning_payload(response_raw: Any) -> dict[str, Any]:
    message = first_message_payload(response_raw)
    summary = text_or_empty(message.get("reasoning")).strip()

    details: list[str] = []
    seen: set[str] = set()

    def push(value: Any) -> None:
        candidate = text_or_empty(value).strip()
        if candidate and candidate not in seen:
            seen.add(candidate)
            details.append(candidate)

    push(summary)
    raw_details = message.get("reasoning_details")
    if isinstance(raw_details, list):
        for item in raw_details:
            if not isinstance(item, dict):
                continue
            push(item.get("summary"))
            push(item.get("text"))

    return {
        "summary": summary,
        "details": details,
    }


def compact_judges(aggregate_row: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(aggregate_row, dict):
        return []
    judges: list[dict[str, Any]] = []
    for index in range(1, 4):
        model = text_or_empty(aggregate_row.get(f"judge_{index}_model")).strip()
        score = aggregate_row.get(f"judge_{index}_score")
        justification = text_or_empty(
            aggregate_row.get(f"judge_{index}_justification")
        ).strip()
        error = text_or_empty(aggregate_row.get(f"judge_{index}_error")).strip()
        status = text_or_empty(aggregate_row.get(f"judge_{index}_status")).strip()
        if not any([model, justification, error, status, score is not None]):
            continue
        judges.append(
            {
                "model": model,
                "score": score,
                "justification": justification,
                "error": error,
                "status": status,
            }
        )
    return judges


def split_reasoning_sections(text: str) -> list[dict[str, str]]:
    source = str(text or "").replace("\r", "").strip()
    if not source:
        return []

    heading_pattern = re.compile(r"\*\*([^*\n]{2,120})\*\*\s*")
    matches = list(heading_pattern.finditer(source))
    if not matches:
        return [{"title": "Summary", "body": source}]

    sections: list[dict[str, str]] = []
    for index, match in enumerate(matches):
        title = text_or_empty(match.group(1)).strip() or "Reasoning note"
        body_start = match.end()
        body_end = matches[index + 1].start() if index + 1 < len(matches) else len(source)
        body = source[body_start:body_end].strip()
        if title or body:
            sections.append({"title": title, "body": body})

    return sections or [{"title": "Summary", "body": source}]


def dedupe_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text)
    return output


def build_reasoning_document(
    source_key: str,
    dataset_key: str,
    question_id: str,
    variant_name: str,
    reasoning_summary: str,
    reasoning_details: list[str],
    *,
    include_legacy_id: bool = False,
) -> dict[str, Any]:
    blocks = dedupe_strings(list(reasoning_details) + [reasoning_summary])
    sections: list[dict[str, Any]] = []
    parts: list[str] = []
    cursor = 0

    for block in blocks:
        for section in split_reasoning_sections(block):
            title = text_or_empty(section.get("title")).strip() or "Reasoning note"
            body = text_or_empty(section.get("body")).strip()
            if not title and not body:
                continue
            section_text = f"{title}\n{body}".rstrip()
            title_start = cursor
            title_end = title_start + len(title)
            body_start = title_end + 1
            body_end = body_start + len(body)
            end = title_start + len(section_text)
            section_id = (
                f"{source_key}:{dataset_key}:{question_id}:{variant_name}:reasoning:"
                f"{len(sections) + 1}"
            )
            parts.append(section_text)
            sections.append(
                {
                    "id": section_id,
                    "title": title,
                    "body": body,
                    "start": title_start,
                    "end": end,
                    "title_start": title_start,
                    "title_end": title_end,
                    "body_start": body_start,
                    "body_end": body_end,
                }
            )
            cursor = end
            parts.append("\n\n")
            cursor += 2

    text = "".join(parts).rstrip() or "No reasoning summary was saved."
    document_id = f"{source_key}:{dataset_key}:{question_id}:{variant_name}:reasoning"
    legacy_ids = (
        [f"{dataset_key}:{question_id}:{variant_name}:reasoning"]
        if include_legacy_id
        else []
    )
    return {
        "document_id": document_id,
        "legacy_document_ids": legacy_ids,
        "kind": "reasoning",
        "variant": variant_name,
        "text": text,
        "sections": sections,
    }


def build_variant(
    source_key: str,
    dataset_key: str,
    question_id: str,
    variant_name: str,
    response_row: dict[str, Any] | None,
    aggregate_row: dict[str, Any] | None,
    *,
    primary_variant_key: str,
    include_legacy_id: bool = False,
) -> dict[str, Any]:
    response_row = response_row or {}
    aggregate_row = aggregate_row or {}
    reasoning = collect_reasoning_payload(response_row.get("response_raw"))
    response_text = text_or_empty(
        response_row.get("response_text") or aggregate_row.get("response_text")
    )

    return {
        "sample_id": text_or_empty(
            response_row.get("sample_id") or aggregate_row.get("sample_id")
        ),
        "model": text_or_empty(response_row.get("model") or aggregate_row.get("model")),
        "model_reasoning_level": text_or_empty(
            response_row.get("model_reasoning_level")
            or aggregate_row.get("model_reasoning_level")
        ),
        "response_reasoning_effort": response_row.get("response_reasoning_effort"),
        "response_text": response_text,
        "response_char_count": len(response_text),
        "response_latency_ms": response_row.get("response_latency_ms"),
        "response_cost_usd": response_row.get("response_cost_usd"),
        "response_prompt_tokens": response_row.get("response_prompt_tokens"),
        "response_completion_tokens": response_row.get("response_completion_tokens"),
        "response_reasoning_tokens": response_row.get("response_reasoning_tokens"),
        "response_total_tokens": response_row.get("response_total_tokens"),
        "status": text_or_empty(aggregate_row.get("status") or response_row.get("status")),
        "error": text_or_empty(aggregate_row.get("error") or response_row.get("error")),
        "consensus_score": aggregate_row.get("consensus_score"),
        "consensus_method": text_or_empty(aggregate_row.get("consensus_method")),
        "judge_valid_scores": aggregate_row.get("judge_valid_scores") or [],
        "judges": compact_judges(aggregate_row),
        "reasoning_summary": reasoning["summary"],
        "reasoning_details": reasoning["details"],
        "reasoning_document": build_reasoning_document(
            source_key=source_key,
            dataset_key=dataset_key,
            question_id=question_id,
            variant_name=variant_name,
            reasoning_summary=reasoning["summary"],
            reasoning_details=reasoning["details"],
            include_legacy_id=include_legacy_id,
        )
        if variant_name == primary_variant_key
        else None,
    }


def build_dataset(source_key: str, dataset_key: str, source_config: dict[str, Any], dataset_config: dict[str, Any]) -> dict[str, Any]:
    base_dir = Path(dataset_config["base_dir"])
    responses = load_jsonl(base_dir / "responses.jsonl")
    aggregates = load_jsonl(base_dir / "aggregate.jsonl")
    manifest = load_json(base_dir / "manifest.json")
    variant_models = source_config["variant_models"]
    primary_variant_key = text_or_empty(source_config.get("primary_variant_key"))
    baseline_variant_key = text_or_empty(source_config.get("baseline_variant_key"))
    include_legacy_id = source_key == "gpt54_high"

    response_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    aggregate_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    question_ids: set[str] = set()

    for row in responses:
        model = text_or_empty(row.get("model")).strip()
        for variant_name, model_label in variant_models.items():
            if model != model_label:
                continue
            question_id = text_or_empty(row.get("question_id")).strip()
            key = (question_id, variant_name)
            if key in response_lookup:
                raise ValueError(f"Duplicate response row for {source_key}:{dataset_key}:{question_id}:{variant_name}")
            response_lookup[(question_id, variant_name)] = row
            question_ids.add(question_id)

    for row in aggregates:
        model = text_or_empty(row.get("model")).strip()
        for variant_name, model_label in variant_models.items():
            if model != model_label:
                continue
            question_id = text_or_empty(row.get("question_id")).strip()
            key = (question_id, variant_name)
            if key in aggregate_lookup:
                raise ValueError(f"Duplicate aggregate row for {source_key}:{dataset_key}:{question_id}:{variant_name}")
            aggregate_lookup[(question_id, variant_name)] = row
            question_ids.add(question_id)

    expected_question_count = int(dataset_config.get("expected_question_count") or 0)
    if expected_question_count and len(question_ids) != expected_question_count:
        raise ValueError(
            f"Expected {expected_question_count} questions for {source_key}:{dataset_key}, found {len(question_ids)}."
        )

    cases: list[dict[str, Any]] = []
    for question_id in sorted(question_ids):
        base_row = {}
        for variant_name in (primary_variant_key, baseline_variant_key, *variant_models.keys()):
            base_row = (
                response_lookup.get((question_id, variant_name))
                or aggregate_lookup.get((question_id, variant_name))
                or base_row
            )
            if base_row:
                break

        case = {
            "question_id": question_id,
            "question": text_or_empty(base_row.get("question")),
            "nonsensical_element": text_or_empty(base_row.get("nonsensical_element")),
            "domain": text_or_empty(base_row.get("domain")),
            "technique": text_or_empty(base_row.get("technique")),
            "is_control": bool(base_row.get("is_control", False)),
            "variants": {},
        }

        for variant_name in variant_models:
            response_row = response_lookup.get((question_id, variant_name))
            aggregate_row = aggregate_lookup.get((question_id, variant_name))
            if response_row is None and aggregate_row is None:
                continue
            case["variants"][variant_name] = build_variant(
                source_key=source_key,
                dataset_key=dataset_key,
                question_id=question_id,
                variant_name=variant_name,
                response_row=response_row,
                aggregate_row=aggregate_row,
                primary_variant_key=primary_variant_key,
                include_legacy_id=include_legacy_id,
            )

        cases.append(case)

    primary_cases = [case for case in cases if primary_variant_key in case["variants"]]
    improved = 0
    worsened = 0
    unchanged = 0
    if baseline_variant_key:
        for case in primary_cases:
            primary_score = case["variants"][primary_variant_key].get("consensus_score")
            baseline_score = case["variants"].get(baseline_variant_key, {}).get("consensus_score")
            if not isinstance(primary_score, (int, float)) or not isinstance(
                baseline_score, (int, float)
            ):
                continue
            if primary_score > baseline_score:
                improved += 1
            elif primary_score < baseline_score:
                worsened += 1
            else:
                unchanged += 1

    total_trace_tokens = sum(
        int(case["variants"][primary_variant_key].get("response_reasoning_tokens") or 0)
        for case in primary_cases
    )

    return {
        "key": dataset_key,
        "label": dataset_config["label"],
        "description": dataset_config["description"],
        "generated_at_utc": manifest.get("generated_at_utc"),
        "case_count": len(cases),
        "primary_case_count": len(primary_cases),
        "improved_count": improved,
        "unchanged_count": unchanged,
        "worsened_count": worsened,
        "trace_token_total": total_trace_tokens,
        "cases": cases,
    }


def build_payload() -> dict[str, Any]:
    sources: dict[str, Any] = {}
    generated_values: list[str] = []

    for source_key, source_config in SOURCES.items():
        datasets: dict[str, Any] = {}
        for dataset_key, dataset_config in source_config.get("datasets", {}).items():
            dataset = build_dataset(source_key, dataset_key, source_config, dataset_config)
            datasets[dataset_key] = dataset
            generated_values.append(text_or_empty(dataset.get("generated_at_utc")))

        sources[source_key] = {
            "key": source_key,
            "label": source_config["label"],
            "description": source_config["description"],
            "model_id": source_config.get("model_id"),
            "judge_model": source_config.get("judge_model"),
            "primary_variant_key": source_config.get("primary_variant_key"),
            "baseline_variant_key": source_config.get("baseline_variant_key"),
            "datasets": datasets,
        }

    return {
        "generated_at_utc": max(generated_values) if generated_values else "",
        "default_source_key": "gpt54_high",
        "source_order": list(SOURCES.keys()),
        "sources": sources,
    }
