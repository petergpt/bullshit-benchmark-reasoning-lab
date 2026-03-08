#!/usr/bin/env python3
"""Export denormalized reasoning-label examples for later model supervision."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from reasoning_lab_data import LAB_ROOT as ROOT, build_payload, text_or_empty


DEFAULT_STORE_PATH = ROOT / "annotations" / "gpt54_reasoning_lab.json"
DEFAULT_OUTPUT_PATH = ROOT / "data" / "gpt54_reasoning_label_examples.jsonl"

SCORE_LABELS = {
    "clear": "Clear pushback",
    "partial": "Partial challenge",
    "accepted": "Accepted nonsense",
    "missing": "Missing",
}


def verdict_key(score: Any) -> str:
    if isinstance(score, (int, float)):
        if score >= 1.5:
            return "clear"
        if score >= 0.5:
            return "partial"
        if score >= 0:
            return "accepted"
    return "missing"


def load_store(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_case_index(payload: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    cases: dict[str, dict[str, Any]] = {}
    documents: dict[str, dict[str, Any]] = {}
    for source in (payload.get("sources") or {}).values():
        source_key = text_or_empty(source.get("key"))
        primary_variant_key = text_or_empty(source.get("primary_variant_key"))
        for dataset in (source.get("datasets") or {}).values():
            for case in dataset.get("cases") or []:
                case_key = (
                    f"{source_key}:"
                    f"{text_or_empty(dataset.get('key'))}:"
                    f"{text_or_empty(case.get('question_id'))}"
                )
                cases[case_key] = case
                variant = (case.get("variants") or {}).get(primary_variant_key) or {}
                document = variant.get("reasoning_document")
                if isinstance(document, dict):
                    document_id = text_or_empty(document.get("document_id"))
                    if document_id:
                        bundle = {
                            "document": document,
                            "case": case,
                            "dataset": dataset,
                            "source": source,
                            "variant": variant,
                        }
                        documents[document_id] = bundle
                        for legacy_id in document.get("legacy_document_ids") or []:
                            legacy_text = text_or_empty(legacy_id).strip()
                            if legacy_text:
                                documents[legacy_text] = bundle
    return cases, documents


def compact_case_context(bundle: dict[str, Any]) -> dict[str, Any]:
    case = bundle["case"]
    dataset = bundle["dataset"]
    source = bundle["source"]
    variant = bundle["variant"]
    verdict = verdict_key(variant.get("consensus_score"))
    return {
        "source_key": text_or_empty(source.get("key")),
        "source_label": text_or_empty(source.get("label")),
        "model_id": text_or_empty(source.get("model_id")),
        "dataset_key": text_or_empty(dataset.get("key")),
        "dataset_label": text_or_empty(dataset.get("label")),
        "question_id": text_or_empty(case.get("question_id")),
        "domain": text_or_empty(case.get("domain")),
        "technique": text_or_empty(case.get("technique")),
        "question": text_or_empty(case.get("question")),
        "nonsense_rationale": text_or_empty(case.get("nonsensical_element")),
        "benchmark_grade_key": verdict,
        "benchmark_grade": SCORE_LABELS[verdict],
        "benchmark_consensus_score": variant.get("consensus_score"),
        "final_answer": text_or_empty(variant.get("response_text")),
    }


def export_examples(store: dict[str, Any], payload: dict[str, Any]) -> list[dict[str, Any]]:
    _cases, documents = build_case_index(payload)
    notes_by_document = {
        text_or_empty(note.get("document_id")): note
        for note in store.get("document_notes") or []
    }
    category_lookup = {
        text_or_empty(category.get("id")): category
        for category in store.get("categories") or []
    }

    examples: list[dict[str, Any]] = []

    for annotation in sorted(
        store.get("annotations") or [],
        key=lambda item: (
            text_or_empty(item.get("source_key")),
            text_or_empty(item.get("dataset_key")),
            text_or_empty(item.get("question_id")),
            text_or_empty(item.get("document_id")),
            int(item.get("start") or 0),
        ),
    ):
        document_id = text_or_empty(annotation.get("document_id"))
        bundle = documents.get(document_id)
        if not bundle:
            continue
        document = bundle["document"]
        note = notes_by_document.get(document_id) or {}
        label_id = text_or_empty(annotation.get("label_id"))
        label_snapshot = annotation.get("label_snapshot") or category_lookup.get(label_id) or {}
        note_label_id = text_or_empty(note.get("label_id"))
        note_label_snapshot = note.get("label_snapshot") or category_lookup.get(note_label_id) or {}
        case_context = compact_case_context(bundle)
        reasoning_trace = text_or_empty(document.get("text"))

        examples.append(
            {
                "example_type": "span_annotation",
                "id": text_or_empty(annotation.get("id")),
                **case_context,
                "variant_key": text_or_empty(annotation.get("variant_key")) or text_or_empty(bundle["source"].get("primary_variant_key")),
                "document_id": document_id,
                "reasoning_trace": reasoning_trace,
                "reasoning_trace_sha1": hashlib.sha1(reasoning_trace.encode("utf-8")).hexdigest(),
                "selected_span": {
                    "start": annotation.get("start"),
                    "end": annotation.get("end"),
                    "quote": text_or_empty(annotation.get("quote")),
                    "normalized_quote": text_or_empty(annotation.get("normalized_quote")),
                    "prefix": text_or_empty(annotation.get("prefix")),
                    "suffix": text_or_empty(annotation.get("suffix")),
                },
                "label": {
                    "id": label_id,
                    "name": text_or_empty(label_snapshot.get("name")) or label_id,
                    "description": text_or_empty(label_snapshot.get("description")),
                    "guidance": text_or_empty(label_snapshot.get("guidance")),
                },
                "confidence": text_or_empty(annotation.get("confidence")) or "clear",
                "why_label": text_or_empty(annotation.get("comment")),
                "overall_trace_label": {
                    "id": note_label_id,
                    "name": text_or_empty(note_label_snapshot.get("name")) if note_label_id else "",
                    "confidence": text_or_empty(note.get("confidence")),
                } if (note_label_id or note.get("confidence")) else None,
                "overall_trace_note": text_or_empty(note.get("summary")),
                "author": text_or_empty(annotation.get("author")),
                "status": text_or_empty(annotation.get("status")),
                "provenance": annotation.get("provenance") or {},
                "created_at_utc": text_or_empty(annotation.get("created_at_utc")),
                "updated_at_utc": text_or_empty(annotation.get("updated_at_utc")),
                "source_text_hash": text_or_empty(annotation.get("source_text_hash")),
            }
        )

    for note in sorted(
        store.get("document_notes") or [],
        key=lambda item: (
            text_or_empty(item.get("source_key")),
            text_or_empty(item.get("dataset_key")),
            text_or_empty(item.get("question_id")),
            text_or_empty(item.get("document_id")),
        ),
    ):
        document_id = text_or_empty(note.get("document_id"))
        bundle = documents.get(document_id)
        if not bundle:
            continue
        document = bundle["document"]
        case_context = compact_case_context(bundle)
        label_id = text_or_empty(note.get("label_id"))
        label_snapshot = note.get("label_snapshot") or category_lookup.get(label_id) or {}
        reasoning_trace = text_or_empty(document.get("text"))
        examples.append(
            {
                "example_type": "overall_trace_annotation",
                "id": f"{document_id}:overall",
                **case_context,
                "variant_key": text_or_empty(note.get("variant_key")) or text_or_empty(bundle["source"].get("primary_variant_key")),
                "document_id": document_id,
                "reasoning_trace": reasoning_trace,
                "reasoning_trace_sha1": hashlib.sha1(reasoning_trace.encode("utf-8")).hexdigest(),
                "label": {
                    "id": label_id,
                    "name": text_or_empty(label_snapshot.get("name")) or label_id,
                    "description": text_or_empty(label_snapshot.get("description")),
                    "guidance": text_or_empty(label_snapshot.get("guidance")),
                } if label_id else None,
                "confidence": text_or_empty(note.get("confidence")),
                "overall_trace_note": text_or_empty(note.get("summary")),
                "author": text_or_empty(note.get("author")),
                "provenance": note.get("provenance") or {},
                "created_at_utc": text_or_empty(note.get("created_at_utc")),
                "updated_at_utc": text_or_empty(note.get("updated_at_utc")),
                "source_text_hash": text_or_empty(note.get("source_text_hash")),
            }
        )

    return examples


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", default=str(DEFAULT_STORE_PATH))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_payload()
    store = load_store(Path(args.store).resolve())
    rows = export_examples(store, payload)
    output_path = Path(args.output).resolve()
    write_jsonl(rows, output_path)
    print(f"Wrote {output_path.relative_to(ROOT)} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
