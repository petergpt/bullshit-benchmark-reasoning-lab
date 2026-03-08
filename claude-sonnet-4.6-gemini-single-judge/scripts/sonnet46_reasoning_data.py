#!/usr/bin/env python3
"""Build a local browser bundle from isolated Sonnet 4.6 reasoning runs."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "data" / "claude-sonnet-4.6-gemini-single-judge.data.js"

DATASETS = {
    "v2": {
        "label": "Benchmark v2",
        "base_dir": ROOT / "data" / "viewer-input" / "v2",
        "description": "100-question Sonnet 4.6 reasoning capture with Gemini single-judge grading",
    },
    "v1": {
        "label": "Benchmark v1",
        "base_dir": ROOT / "data" / "viewer-input" / "v1",
        "description": "55-question Sonnet 4.6 reasoning capture with Gemini single-judge grading",
    },
}

VARIANT_MODELS = {
    "none": "anthropic/claude-sonnet-4.6@reasoning=none",
    "high": "anthropic/claude-sonnet-4.6@reasoning=high",
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
            push(item.get("text"))
            push(item.get("summary"))

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
    dataset_key: str,
    question_id: str,
    variant_name: str,
    reasoning_summary: str,
    reasoning_details: list[str],
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
                f"{dataset_key}:{question_id}:{variant_name}:reasoning:"
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
    return {
        "document_id": f"{dataset_key}:{question_id}:{variant_name}:reasoning",
        "kind": "reasoning",
        "variant": variant_name,
        "text": text,
        "sections": sections,
    }


def build_variant(
    dataset_key: str,
    question_id: str,
    variant_name: str,
    response_row: dict[str, Any] | None,
    aggregate_row: dict[str, Any] | None,
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
            dataset_key=dataset_key,
            question_id=question_id,
            variant_name=variant_name,
            reasoning_summary=reasoning["summary"],
            reasoning_details=reasoning["details"],
        )
        if variant_name == "high"
        else None,
    }


def build_dataset(dataset_key: str, config: dict[str, Any]) -> dict[str, Any]:
    base_dir = Path(config["base_dir"])
    responses = load_jsonl(base_dir / "responses.jsonl")
    aggregates = load_jsonl(base_dir / "aggregate.jsonl")
    manifest = load_json(base_dir / "manifest.json")

    response_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    aggregate_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    question_ids: set[str] = set()

    for row in responses:
        model = text_or_empty(row.get("model")).strip()
        for variant_name, model_label in VARIANT_MODELS.items():
            if model == model_label:
                question_id = text_or_empty(row.get("question_id")).strip()
                response_lookup[(question_id, variant_name)] = row
                question_ids.add(question_id)

    for row in aggregates:
        model = text_or_empty(row.get("model")).strip()
        for variant_name, model_label in VARIANT_MODELS.items():
            if model == model_label:
                question_id = text_or_empty(row.get("question_id")).strip()
                aggregate_lookup[(question_id, variant_name)] = row
                question_ids.add(question_id)

    cases: list[dict[str, Any]] = []
    for question_id in sorted(question_ids):
        base_row = (
            response_lookup.get((question_id, "high"))
            or response_lookup.get((question_id, "none"))
            or aggregate_lookup.get((question_id, "high"))
            or aggregate_lookup.get((question_id, "none"))
            or {}
        )
        case = {
            "question_id": question_id,
            "question": text_or_empty(base_row.get("question")),
            "nonsensical_element": text_or_empty(base_row.get("nonsensical_element")),
            "domain": text_or_empty(base_row.get("domain")),
            "technique": text_or_empty(base_row.get("technique")),
            "is_control": bool(base_row.get("is_control", False)),
            "variants": {},
        }
        for variant_name in ("none", "high"):
            response_row = response_lookup.get((question_id, variant_name))
            aggregate_row = aggregate_lookup.get((question_id, variant_name))
            if response_row is None and aggregate_row is None:
                continue
            case["variants"][variant_name] = build_variant(
                dataset_key,
                question_id,
                variant_name,
                response_row,
                aggregate_row,
            )
        cases.append(case)

    high_cases = [case for case in cases if "high" in case["variants"]]
    improved = 0
    worsened = 0
    unchanged = 0
    for case in high_cases:
        high_score = case["variants"]["high"].get("consensus_score")
        none_score = case["variants"].get("none", {}).get("consensus_score")
        if not isinstance(high_score, (int, float)) or not isinstance(none_score, (int, float)):
            continue
        if high_score > none_score:
            improved += 1
        elif high_score < none_score:
            worsened += 1
        else:
            unchanged += 1

    trace_token_total = sum(
        int(case["variants"]["high"].get("response_reasoning_tokens") or 0)
        for case in high_cases
    )

    return {
        "key": dataset_key,
        "label": config["label"],
        "description": config["description"],
        "generated_at_utc": manifest.get("generated_at_utc"),
        "case_count": len(cases),
        "high_case_count": len(high_cases),
        "improved_count": improved,
        "unchanged_count": unchanged,
        "worsened_count": worsened,
        "trace_token_total": trace_token_total,
        "cases": cases,
    }


def build_payload() -> dict[str, Any]:
    datasets = {key: build_dataset(key, config) for key, config in DATASETS.items()}
    return {
        "generated_at_utc": max(
            text_or_empty(dataset.get("generated_at_utc")) for dataset in datasets.values()
        ),
        "model": "anthropic/claude-sonnet-4.6",
        "judge_model": "google/gemini-3.1-pro-preview",
        "datasets": datasets,
    }


def write_browser_bundle(payload: dict[str, Any], output_path: Path = OUTPUT_PATH) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    output_path.write_text(
        "window.CLAUDE_SONNET_46_REASONING_DATA=" + encoded + ";\n",
        encoding="utf-8",
    )
