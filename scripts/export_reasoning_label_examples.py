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


def normalize_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = text_or_empty(value).strip().lower()
    if not text:
        return default
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def record_is_ai_labelled(record: dict[str, Any]) -> bool:
    if "ai_labelled" in record:
        return normalize_bool(record.get("ai_labelled"))
    provenance = record.get("provenance") or {}
    return (
        text_or_empty(provenance.get("labeler_type")).strip() == "model"
        or text_or_empty(provenance.get("source")).strip() == "ai_labelling"
    )


def record_is_human_reviewed(record: dict[str, Any]) -> bool:
    if "human_reviewed" in record:
        return normalize_bool(record.get("human_reviewed"))
    provenance = record.get("provenance") or {}
    return (
        text_or_empty(provenance.get("labeler_type")).strip() == "human"
        and not record_is_ai_labelled(record)
    )


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
                if case_key in cases:
                    raise ValueError(f"Duplicate case key in atlas payload: {case_key}")
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
                        if document_id in documents and documents[document_id] != bundle:
                            raise ValueError(f"Duplicate document_id in atlas payload: {document_id}")
                        documents[document_id] = bundle
                        for legacy_id in document.get("legacy_document_ids") or []:
                            legacy_text = text_or_empty(legacy_id).strip()
                            if legacy_text:
                                if legacy_text in documents and documents[legacy_text] != bundle:
                                    raise ValueError(
                                        f"Duplicate legacy document_id in atlas payload: {legacy_text}"
                                    )
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
    notes_by_document: dict[str, dict[str, Any]] = {}
    for note in store.get("document_notes") or []:
        document_id = text_or_empty(note.get("document_id"))
        if document_id in notes_by_document:
            raise ValueError(f"Duplicate document note in store export: {document_id}")
        notes_by_document[document_id] = note
    category_lookup: dict[str, dict[str, Any]] = {}
    for category in store.get("categories") or []:
        category_id = text_or_empty(category.get("id"))
        if category_id in category_lookup:
            raise ValueError(f"Duplicate category id in store export: {category_id}")
        category_lookup[category_id] = category
    review_sessions_by_id: dict[str, dict[str, Any]] = {}
    for session in store.get("review_sessions") or []:
        session_id = text_or_empty(session.get("id"))
        if session_id in review_sessions_by_id:
            raise ValueError(f"Duplicate review session id in store export: {session_id}")
        review_sessions_by_id[session_id] = session
    review_event_counts_by_session: dict[str, int] = {}
    review_feedback_by_record: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for event in store.get("review_events") or []:
        session_id = text_or_empty(event.get("session_id"))
        if not session_id:
            continue
        review_event_counts_by_session[session_id] = review_event_counts_by_session.get(session_id, 0) + 1
        feedback = text_or_empty(event.get("feedback")).strip()
        if not feedback:
            continue
        record_id = text_or_empty(event.get("record_id"))
        review_feedback_by_record.setdefault((session_id, record_id), []).append(
            {
                "record_kind": text_or_empty(event.get("record_kind")),
                "action": text_or_empty(event.get("action")),
                "feedback": feedback,
                "created_at_utc": text_or_empty(event.get("created_at_utc")),
            }
        )

    examples: list[dict[str, Any]] = []
    unresolved_document_ids: set[str] = set()

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
            unresolved_document_ids.add(document_id)
            continue
        document = bundle["document"]
        note = notes_by_document.get(document_id) or {}
        label_id = text_or_empty(annotation.get("label_id"))
        label_snapshot = annotation.get("label_snapshot") or category_lookup.get(label_id) or {}
        note_label_id = text_or_empty(note.get("label_id"))
        note_label_snapshot = note.get("label_snapshot") or category_lookup.get(note_label_id) or {}
        case_context = compact_case_context(bundle)
        reasoning_trace = text_or_empty(document.get("text"))
        review_session_id = text_or_empty(annotation.get("review_session_id"))
        review_session = review_sessions_by_id.get(review_session_id) or {}

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
                "why_label": text_or_empty(annotation.get("comment")),
                "overall_trace_label": {
                    "id": note_label_id,
                    "name": text_or_empty(note_label_snapshot.get("name")) if note_label_id else "",
                } if note_label_id else None,
                "overall_trace_note": text_or_empty(note.get("summary")),
                "author": text_or_empty(annotation.get("author")),
                "status": text_or_empty(annotation.get("status")),
                "ai_labelled": record_is_ai_labelled(annotation),
                "human_reviewed": record_is_human_reviewed(annotation),
                "reviewed_by": text_or_empty(annotation.get("reviewed_by")),
                "reviewed_at_utc": text_or_empty(annotation.get("reviewed_at_utc")),
                "review_session_id": review_session_id,
                "review_origin": text_or_empty(annotation.get("review_origin")),
                "origin_suggestion_id": text_or_empty(annotation.get("origin_suggestion_id")),
                "review_session_status": text_or_empty(review_session.get("status")),
                "review_event_count": int(review_event_counts_by_session.get(review_session_id) or 0),
                "review_feedback": review_feedback_by_record.get(
                    (review_session_id, text_or_empty(annotation.get("id")))
                ) or [],
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
            unresolved_document_ids.add(document_id)
            continue
        document = bundle["document"]
        case_context = compact_case_context(bundle)
        label_id = text_or_empty(note.get("label_id"))
        label_snapshot = note.get("label_snapshot") or category_lookup.get(label_id) or {}
        reasoning_trace = text_or_empty(document.get("text"))
        review_session_id = text_or_empty(note.get("review_session_id"))
        review_session = review_sessions_by_id.get(review_session_id) or {}
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
                "overall_trace_note": text_or_empty(note.get("summary")),
                "author": text_or_empty(note.get("author")),
                "ai_labelled": record_is_ai_labelled(note),
                "human_reviewed": record_is_human_reviewed(note),
                "reviewed_by": text_or_empty(note.get("reviewed_by")),
                "reviewed_at_utc": text_or_empty(note.get("reviewed_at_utc")),
                "review_session_id": review_session_id,
                "review_origin": text_or_empty(note.get("review_origin")),
                "origin_suggestion_id": text_or_empty(note.get("origin_suggestion_id")),
                "review_session_status": text_or_empty(review_session.get("status")),
                "review_event_count": int(review_event_counts_by_session.get(review_session_id) or 0),
                "review_feedback": review_feedback_by_record.get(
                    (review_session_id, f"{document_id}:overall")
                ) or [],
                "provenance": note.get("provenance") or {},
                "created_at_utc": text_or_empty(note.get("created_at_utc")),
                "updated_at_utc": text_or_empty(note.get("updated_at_utc")),
                "source_text_hash": text_or_empty(note.get("source_text_hash")),
            }
        )

    if unresolved_document_ids:
        missing = ", ".join(sorted(unresolved_document_ids))
        raise ValueError(f"Unresolved document ids during export: {missing}")

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
