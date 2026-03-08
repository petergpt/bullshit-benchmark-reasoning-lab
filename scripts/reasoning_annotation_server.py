#!/usr/bin/env python3
"""Local annotation server for BullshitBench reasoning traces."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import threading
import urllib.parse
import uuid
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from reasoning_lab_data import LAB_ROOT as ROOT, build_payload, text_or_empty


DEFAULT_STORE_PATH = ROOT / "annotations" / "gpt54_reasoning_lab.json"
STORE_SCHEMA_VERSION = 2
VALID_CONFIDENCE = {"clear", "borderline"}

DEFAULT_CATEGORIES = [
    {
        "id": "going_along_with_it",
        "name": "Going Along With It",
        "description": "Treats the bogus premise as valid and reasons within it.",
        "color": "#d1495b",
        "text_color": "#ffffff",
        "guidance": "Use when the trace accepts the nonsense frame and proceeds as if it were real.",
        "order": 10,
        "active": True,
    },
    {
        "id": "challenges_the_premise",
        "name": "Challenges The Premise",
        "description": "Pushes back on the underlying premise as wrong, incoherent, or invented.",
        "color": "#1f8f5c",
        "text_color": "#ffffff",
        "guidance": "Use when the trace says the premise itself does not hold or should not be accepted.",
        "order": 20,
        "active": True,
    },
    {
        "id": "questions_intent",
        "name": "Joke or Test?",
        "description": "Questions whether the user is joking, testing, or baiting rather than asking sincerely.",
        "color": "#7a55d1",
        "text_color": "#ffffff",
        "guidance": "Use when the trace pauses to ask whether the prompt is a joke, bait, or some kind of test.",
        "order": 25,
        "active": True,
    },
    {
        "id": "provides_context",
        "name": "Provides Context",
        "description": "Adds background, framing, or explanatory context without clearly solving or rejecting.",
        "color": "#2878c7",
        "text_color": "#ffffff",
        "guidance": "Use when the trace is orienting, explaining, or giving setup rather than taking a strong stance.",
        "order": 30,
        "active": True,
    },
    {
        "id": "attempts_to_solve",
        "name": "Attempts to Solve",
        "description": "Starts working toward an answer, method, formula, or procedure.",
        "color": "#d28a13",
        "text_color": "#1e1402",
        "guidance": "Use when the trace actively tries to compute, derive, estimate, or operationalize a response.",
        "order": 40,
        "active": True,
    },
    {
        "id": "confused",
        "name": "Confused",
        "description": "Misreads the question, loses the thread, or responds in a way that shows confusion.",
        "color": "#7c8b86",
        "text_color": "#ffffff",
        "guidance": "Use when the trace appears to misunderstand what is being asked or drifts into incoherence.",
        "order": 50,
        "active": True,
    },
]

DEFAULT_GUIDANCE = {
    "purpose": (
        "Label reasoning spans that show how BullshitBench models handle "
        "nonsense prompts, especially whether it goes along with them, "
        "challenges them, questions the user's intent, provides context, "
        "attempts to solve them, or becomes confused."
    ),
    "annotation_unit": (
        "Prefer short spans that express one distinct reasoning move. Use comments "
        "for interpretation, not for repeating the selected text."
    ),
    "overlap_policy": (
        "Spans in the same document are non-overlapping so each piece of text has "
        "at most one primary label."
    ),
    "future_ai_use": (
        "These labels are designed to become a few-shot supervision set for later "
        "AI-assisted annotation and retrieval."
    ),
}

DEFAULT_ANNOTATION_DEFAULTS = {
    "status": "confirmed",
    "labeler_type": "human",
    "interface": "reasoning-annotation-studio",
    "selection_mode": "freeform",
    "confidence": "clear",
}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def make_store_skeleton() -> dict[str, Any]:
    now = utc_now()
    return {
        "schema_version": STORE_SCHEMA_VERSION,
        "store_name": "BullshitBench Reasoning Lab",
        "store_format": "reasoning-annotation-store",
        "created_at_utc": now,
        "updated_at_utc": now,
        "guidance": dict(DEFAULT_GUIDANCE),
        "annotation_defaults": dict(DEFAULT_ANNOTATION_DEFAULTS),
        "categories": DEFAULT_CATEGORIES,
        "document_notes": [],
        "annotations": [],
    }


class AtlasIndex:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.documents: dict[str, dict[str, Any]] = {}
        self.aliases: dict[str, str] = {}
        self._index_documents()

    def _index_documents(self) -> None:
        for source_key, source in (self.payload.get("sources") or {}).items():
            source_label = text_or_empty(source.get("label"))
            for dataset_key, dataset in (source.get("datasets") or {}).items():
                for case in dataset.get("cases") or []:
                    question_id = text_or_empty(case.get("question_id"))
                    variants = case.get("variants") or {}
                    for variant_key, variant in variants.items():
                        document = variant.get("reasoning_document")
                        if not isinstance(document, dict):
                            continue
                        document_id = text_or_empty(document.get("document_id"))
                        if not document_id:
                            continue
                        text = text_or_empty(document.get("text"))
                        record = {
                            "document_id": document_id,
                            "source_key": source_key,
                            "source_label": source_label,
                            "dataset_key": dataset_key,
                            "question_id": question_id,
                            "variant_key": variant_key,
                            "kind": text_or_empty(document.get("kind")),
                            "text": text,
                            "text_hash": hashlib.sha1(text.encode("utf-8")).hexdigest(),
                        }
                        self.documents[document_id] = record
                        for legacy_id in document.get("legacy_document_ids") or []:
                            alias = text_or_empty(legacy_id).strip()
                            if alias:
                                self.aliases[alias] = document_id

    def get_document(self, document_id: str) -> dict[str, Any] | None:
        canonical_id = self.aliases.get(document_id, document_id)
        return self.documents.get(canonical_id)


class AnnotationStore:
    def __init__(self, path: Path, atlas_index: AtlasIndex) -> None:
        self.path = path
        self.atlas_index = atlas_index
        self.lock = threading.Lock()
        self._ensure_store()

    def _ensure_store(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write(make_store_skeleton())
            return
        existing = self._read()
        normalized = self._normalize_store(existing)
        if normalized != existing:
            self._write(normalized)

    def _read(self) -> dict[str, Any]:
        with self.path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _write(self, payload: dict[str, Any]) -> None:
        payload["updated_at_utc"] = utc_now()
        temp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        temp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temp_path.replace(self.path)

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return self._normalize_store(self._read())

    def add_category(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            store = self._read()
            category = self._normalize_category(
                payload,
                fallback_order=(len(store["categories"]) + 1) * 10,
            )
            category_id = text_or_empty(category.get("id"))
            if any(text_or_empty(item.get("id")) == category_id for item in store["categories"]):
                raise ValueError(f"Category already exists: {category_id}")
            store["categories"].append(category)
            store["categories"] = sorted(
                store["categories"],
                key=lambda item: (int(item.get("order") or 999), text_or_empty(item.get("name"))),
            )
            self._write(store)
            return category

    def update_category(self, category_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            store = self._read()
            for category in store["categories"]:
                if text_or_empty(category.get("id")) != category_id:
                    continue
                for field in ("name", "description", "color", "text_color", "guidance"):
                    if field in payload:
                        category[field] = text_or_empty(payload.get(field)).strip()
                if "order" in payload:
                    category["order"] = int(payload.get("order") or 999)
                if "active" in payload:
                    category["active"] = bool(payload.get("active"))
                normalized = self._normalize_category(category, fallback_order=int(category.get("order") or 999))
                category.update(normalized)
                store["categories"] = sorted(
                    store["categories"],
                    key=lambda item: (int(item.get("order") or 999), text_or_empty(item.get("name"))),
                )
                self._write(store)
                return category
            raise KeyError(f"Unknown category: {category_id}")

    def add_annotation(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            store = self._read()
            annotation = self._normalize_annotation_payload(
                payload=payload,
                existing_annotations=store["annotations"],
                replace_id=None,
                category_lookup={
                    text_or_empty(category.get("id")): category
                    for category in store["categories"]
                },
                valid_category_ids={
                    text_or_empty(category.get("id")) for category in store["categories"]
                },
            )
            store["annotations"].append(annotation)
            self._write(store)
            return annotation

    def update_annotation(self, annotation_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            store = self._read()
            for index, current in enumerate(store["annotations"]):
                if text_or_empty(current.get("id")) != annotation_id:
                    continue
                merged = dict(current)
                for field in (
                    "label_id",
                    "comment",
                    "confidence",
                    "status",
                    "author",
                    "start",
                    "end",
                    "quote",
                    "document_id",
                    "selection_mode",
                    "interface",
                    "labeler_type",
                ):
                    if field in payload:
                        merged[field] = payload[field]
                annotation = self._normalize_annotation_payload(
                    payload=merged,
                    existing_annotations=store["annotations"],
                    replace_id=annotation_id,
                    created_at_utc=text_or_empty(current.get("created_at_utc")) or utc_now(),
                    category_lookup={
                        text_or_empty(category.get("id")): category
                        for category in store["categories"]
                    },
                    valid_category_ids={
                        text_or_empty(category.get("id")) for category in store["categories"]
                    },
                )
                store["annotations"][index] = annotation
                self._write(store)
                return annotation
            raise KeyError(f"Unknown annotation: {annotation_id}")

    def upsert_document_note(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        with self.lock:
            store = self._read()
            notes = store.get("document_notes") or []
            normalized = self._normalize_document_note_payload(
                payload=payload,
                created_at_utc=next(
                    (
                        text_or_empty(item.get("created_at_utc")) or utc_now()
                        for item in notes
                        if text_or_empty(item.get("document_id"))
                        == text_or_empty(payload.get("document_id")).strip()
                    ),
                    utc_now(),
                ),
                category_lookup={
                    text_or_empty(category.get("id")): category
                    for category in store["categories"]
                },
                valid_category_ids={
                    text_or_empty(category.get("id")) for category in store["categories"]
                },
            )
            document_id = text_or_empty(payload.get("document_id")).strip()
            store["document_notes"] = [
                item for item in notes if text_or_empty(item.get("document_id")) != document_id
            ]
            if normalized is not None:
                store["document_notes"].append(normalized)
                store["document_notes"] = sorted(
                    store["document_notes"],
                    key=lambda item: (
                        text_or_empty(item.get("source_key")),
                        text_or_empty(item.get("dataset_key")),
                        text_or_empty(item.get("question_id")),
                        text_or_empty(item.get("document_id")),
                    ),
                )
            self._write(store)
            return normalized

    def delete_annotation(self, annotation_id: str) -> None:
        with self.lock:
            store = self._read()
            before = len(store["annotations"])
            store["annotations"] = [
                item
                for item in store["annotations"]
                if text_or_empty(item.get("id")) != annotation_id
            ]
            if len(store["annotations"]) == before:
                raise KeyError(f"Unknown annotation: {annotation_id}")
            self._write(store)

    def import_store(self, incoming: dict[str, Any], mode: str) -> dict[str, Any]:
        if mode not in {"merge", "replace"}:
            raise ValueError("Import mode must be merge or replace.")
        with self.lock:
            if mode == "replace":
                replacement = make_store_skeleton()
                replacement["categories"] = []
                replacement["document_notes"] = []
                replacement["annotations"] = []
            else:
                replacement = self._read()

            incoming_categories = incoming.get("categories") or []
            existing_categories = {
                text_or_empty(item.get("id")): item for item in replacement["categories"]
            }
            for category in incoming_categories:
                normalized_category = self._normalize_category(
                    category,
                    fallback_order=(len(existing_categories) + 1) * 10,
                )
                category_id = text_or_empty(normalized_category.get("id"))
                if not category_id:
                    continue
                existing_categories[category_id] = normalized_category
            replacement["categories"] = sorted(
                existing_categories.values(),
                key=lambda item: (int(item.get("order") or 999), text_or_empty(item.get("name"))),
            )
            valid_category_ids = set(existing_categories.keys())
            category_lookup = dict(existing_categories)

            existing_notes = replacement.get("document_notes") or []
            notes_by_document_id = {
                text_or_empty(item.get("document_id")): item for item in existing_notes
            }
            for note in incoming.get("document_notes") or []:
                candidate = self._normalize_document_note_payload(
                    payload=note,
                    created_at_utc=text_or_empty(note.get("created_at_utc")) or utc_now(),
                    category_lookup=category_lookup,
                    valid_category_ids=valid_category_ids,
                )
                if candidate is None:
                    continue
                notes_by_document_id[candidate["document_id"]] = candidate
            replacement["document_notes"] = sorted(
                notes_by_document_id.values(),
                key=lambda item: (
                    text_or_empty(item.get("source_key")),
                    text_or_empty(item.get("dataset_key")),
                    text_or_empty(item.get("question_id")),
                    text_or_empty(item.get("document_id")),
                ),
            )

            existing_annotations = replacement.get("annotations") or []
            existing_by_id = {
                text_or_empty(item.get("id")): item for item in existing_annotations
            }
            for annotation in incoming.get("annotations") or []:
                candidate = self._normalize_annotation_payload(
                    payload=annotation,
                    existing_annotations=list(existing_by_id.values()),
                    replace_id=text_or_empty(annotation.get("id")) or None,
                    created_at_utc=text_or_empty(annotation.get("created_at_utc")) or utc_now(),
                    preserve_id=text_or_empty(annotation.get("id")) or None,
                    category_lookup=category_lookup,
                    valid_category_ids=valid_category_ids,
                )
                existing_by_id[candidate["id"]] = candidate
            replacement["annotations"] = sorted(
                existing_by_id.values(),
                key=lambda item: (
                    text_or_empty(item.get("source_key")),
                    text_or_empty(item.get("dataset_key")),
                    text_or_empty(item.get("question_id")),
                    text_or_empty(item.get("document_id")),
                    int(item.get("start") or 0),
                ),
            )

            self._write(replacement)
            return replacement

    def _normalize_store(self, payload: dict[str, Any]) -> dict[str, Any]:
        base = make_store_skeleton()
        store = dict(payload or {})
        normalized: dict[str, Any] = {
            "schema_version": STORE_SCHEMA_VERSION,
            "store_name": text_or_empty(store.get("store_name")).strip() or base["store_name"],
            "store_format": text_or_empty(store.get("store_format")).strip() or base["store_format"],
            "created_at_utc": text_or_empty(store.get("created_at_utc")).strip() or base["created_at_utc"],
            "updated_at_utc": text_or_empty(store.get("updated_at_utc")).strip() or base["updated_at_utc"],
            "guidance": dict(DEFAULT_GUIDANCE),
            "annotation_defaults": dict(DEFAULT_ANNOTATION_DEFAULTS),
        }
        raw_guidance = store.get("guidance") if isinstance(store.get("guidance"), dict) else {}
        for key in DEFAULT_GUIDANCE:
            value = text_or_empty(raw_guidance.get(key)).strip()
            if value:
                normalized["guidance"][key] = value
        raw_defaults = (
            store.get("annotation_defaults")
            if isinstance(store.get("annotation_defaults"), dict)
            else {}
        )
        for key in DEFAULT_ANNOTATION_DEFAULTS:
            value = text_or_empty(raw_defaults.get(key)).strip()
            if value:
                normalized["annotation_defaults"][key] = value

        raw_categories = store.get("categories") or DEFAULT_CATEGORIES
        categories: list[dict[str, Any]] = []
        seen_category_ids: set[str] = set()
        for index, category in enumerate(raw_categories):
            normalized_category = self._normalize_category(category, fallback_order=(index + 1) * 10)
            category_id = text_or_empty(normalized_category.get("id"))
            if not category_id or category_id in seen_category_ids:
                continue
            seen_category_ids.add(category_id)
            categories.append(normalized_category)
        if not categories:
            categories = [
                self._normalize_category(category, fallback_order=(index + 1) * 10)
                for index, category in enumerate(DEFAULT_CATEGORIES)
            ]
        categories = sorted(
            categories,
            key=lambda item: (int(item.get("order") or 999), text_or_empty(item.get("name"))),
        )
        normalized["categories"] = categories
        category_lookup = {text_or_empty(item.get("id")): item for item in categories}
        valid_category_ids = set(category_lookup.keys())

        document_notes: list[dict[str, Any]] = []
        for note in store.get("document_notes") or []:
            normalized_note = self._normalize_document_note_payload(
                payload=note,
                created_at_utc=text_or_empty(note.get("created_at_utc")) or utc_now(),
                category_lookup=category_lookup,
                valid_category_ids=valid_category_ids,
            )
            if normalized_note is None:
                continue
            document_notes = [
                item
                for item in document_notes
                if text_or_empty(item.get("document_id")) != normalized_note["document_id"]
            ]
            document_notes.append(normalized_note)
        normalized["document_notes"] = sorted(
            document_notes,
            key=lambda item: (
                text_or_empty(item.get("source_key")),
                text_or_empty(item.get("dataset_key")),
                text_or_empty(item.get("question_id")),
                text_or_empty(item.get("document_id")),
            ),
        )

        annotations: list[dict[str, Any]] = []
        for annotation in store.get("annotations") or []:
            normalized_annotation = self._normalize_annotation_payload(
                payload=annotation,
                existing_annotations=annotations,
                replace_id=text_or_empty(annotation.get("id")) or None,
                created_at_utc=text_or_empty(annotation.get("created_at_utc")) or utc_now(),
                preserve_id=text_or_empty(annotation.get("id")) or None,
                category_lookup=category_lookup,
                valid_category_ids=valid_category_ids,
            )
            annotations.append(normalized_annotation)
        normalized["annotations"] = annotations
        return normalized

    def _normalize_category(self, payload: dict[str, Any], fallback_order: int = 999) -> dict[str, Any]:
        category_id = self._slugify_category_id(payload.get("id") or payload.get("name"))
        if not category_id:
            raise ValueError("Category id/name is required.")
        return {
            "id": category_id,
            "name": text_or_empty(payload.get("name")).strip() or category_id,
            "description": text_or_empty(payload.get("description")).strip(),
            "color": text_or_empty(payload.get("color")).strip() or "#4b9b8c",
            "text_color": text_or_empty(payload.get("text_color")).strip() or "#ffffff",
            "guidance": text_or_empty(payload.get("guidance")).strip(),
            "order": int(payload.get("order") or fallback_order),
            "active": bool(payload.get("active", True)),
        }

    def _label_snapshot(
        self,
        label_id: str,
        category_lookup: dict[str, dict[str, Any]] | None,
    ) -> dict[str, Any]:
        category = (category_lookup or {}).get(label_id) or {}
        return {
            "id": label_id,
            "name": text_or_empty(category.get("name")).strip() or label_id,
            "description": text_or_empty(category.get("description")).strip(),
            "guidance": text_or_empty(category.get("guidance")).strip(),
            "color": text_or_empty(category.get("color")).strip() or "#4b9b8c",
            "text_color": text_or_empty(category.get("text_color")).strip() or "#ffffff",
        }

    def _normalize_confidence(self, value: Any, *, allow_blank: bool = False) -> str:
        confidence = text_or_empty(value).strip().lower()
        if not confidence:
            return "" if allow_blank else DEFAULT_ANNOTATION_DEFAULTS["confidence"]
        if confidence not in VALID_CONFIDENCE:
            raise ValueError(
                f"Unknown confidence: {confidence}. Expected one of {sorted(VALID_CONFIDENCE)}."
            )
        return confidence

    def _normalize_document_note_payload(
        self,
        payload: dict[str, Any],
        created_at_utc: str | None = None,
        category_lookup: dict[str, dict[str, Any]] | None = None,
        valid_category_ids: set[str] | None = None,
    ) -> dict[str, Any] | None:
        document_id = text_or_empty(payload.get("document_id")).strip()
        document = self.atlas_index.get_document(document_id)
        if not document:
            raise ValueError(f"Unknown document_id: {document_id}")
        document_id = text_or_empty(document.get("document_id")).strip()

        summary = text_or_empty(payload.get("summary")).strip()
        label_id = self._slugify_category_id(payload.get("label_id"))
        if valid_category_ids is None:
            valid_category_ids = {
                text_or_empty(category.get("id"))
                for category in self._read().get("categories", [])
            }
        if label_id and label_id not in valid_category_ids:
            raise ValueError(f"Unknown category: {label_id}")
        if not summary and not label_id:
            return None
        confidence = self._normalize_confidence(
            payload.get("confidence"),
            allow_blank=not bool(label_id),
        )
        if not label_id:
            confidence = ""

        incoming_source_text_hash = text_or_empty(payload.get("source_text_hash")).strip()
        if incoming_source_text_hash and incoming_source_text_hash != document["text_hash"]:
            raise ValueError(
                "Document note source_text_hash does not match the current reasoning document."
            )

        interface = (
            text_or_empty(payload.get("interface")).strip()
            or text_or_empty((payload.get("provenance") or {}).get("interface")).strip()
            or DEFAULT_ANNOTATION_DEFAULTS["interface"]
        )
        labeler_type = (
            text_or_empty(payload.get("labeler_type")).strip()
            or text_or_empty((payload.get("provenance") or {}).get("labeler_type")).strip()
            or DEFAULT_ANNOTATION_DEFAULTS["labeler_type"]
        )

        return {
            "document_id": document["document_id"],
            "source_key": document["source_key"],
            "source_label": document["source_label"],
            "dataset_key": document["dataset_key"],
            "question_id": document["question_id"],
            "variant_key": document["variant_key"],
            "surface": document["kind"],
            "summary": summary,
            "normalized_summary": " ".join(summary.split()),
            "summary_sha1": hashlib.sha1(summary.encode("utf-8")).hexdigest() if summary else "",
            "label_id": label_id or "",
            "label_snapshot": self._label_snapshot(label_id, category_lookup) if label_id else None,
            "confidence": confidence,
            "author": text_or_empty(payload.get("author")).strip(),
            "created_at_utc": created_at_utc or utc_now(),
            "updated_at_utc": utc_now(),
            "source_text_hash": document["text_hash"],
            "provenance": {
                "labeler_type": labeler_type,
                "interface": interface,
            },
        }

    def _slugify_category_id(self, value: Any) -> str:
        text = text_or_empty(value).strip().lower()
        output = []
        last_separator = False
        for char in text:
            if char.isalnum() or char == "_":
                output.append(char)
                last_separator = False
            elif not last_separator:
                output.append("_")
                last_separator = True
        return "".join(output).strip("_")

    def _normalize_annotation_payload(
        self,
        payload: dict[str, Any],
        existing_annotations: list[dict[str, Any]],
        replace_id: str | None,
        created_at_utc: str | None = None,
        preserve_id: str | None = None,
        category_lookup: dict[str, dict[str, Any]] | None = None,
        valid_category_ids: set[str] | None = None,
    ) -> dict[str, Any]:
        document_id = text_or_empty(payload.get("document_id")).strip()
        document = self.atlas_index.get_document(document_id)
        if not document:
            raise ValueError(f"Unknown document_id: {document_id}")
        document_id = text_or_empty(document.get("document_id")).strip()

        start = int(payload.get("start"))
        end = int(payload.get("end"))
        if start < 0 or end <= start:
            raise ValueError("Annotation start/end offsets are invalid.")
        document_text = document["text"]
        if end > len(document_text):
            raise ValueError("Annotation end offset exceeds document length.")

        quote = document_text[start:end]
        if text_or_empty(payload.get("quote")).strip():
            if text_or_empty(payload.get("quote")) != quote:
                raise ValueError("Submitted quote does not match document text.")
        incoming_source_text_hash = text_or_empty(payload.get("source_text_hash")).strip()
        if incoming_source_text_hash and incoming_source_text_hash != document["text_hash"]:
            raise ValueError(
                "Annotation source_text_hash does not match the current reasoning document."
            )

        label_id = self._slugify_category_id(payload.get("label_id"))
        if not label_id:
            raise ValueError("label_id is required.")
        if valid_category_ids is None:
            valid_category_ids = {
                text_or_empty(category.get("id"))
                for category in self._read().get("categories", [])
            }
        if label_id not in valid_category_ids:
            raise ValueError(f"Unknown category: {label_id}")
        confidence = self._normalize_confidence(payload.get("confidence"))

        for current in existing_annotations:
            if replace_id and text_or_empty(current.get("id")) == replace_id:
                continue
            if text_or_empty(current.get("document_id")) != document_id:
                continue
            current_start = int(current.get("start") or 0)
            current_end = int(current.get("end") or 0)
            overlaps = start < current_end and end > current_start
            if overlaps:
                raise ValueError(
                    "Overlapping annotations are not allowed within the same document."
                )

        prefix = document_text[max(0, start - 48) : start]
        suffix = document_text[end : min(len(document_text), end + 48)]
        selection_mode = (
            text_or_empty(payload.get("selection_mode")).strip()
            or text_or_empty((payload.get("provenance") or {}).get("selection_mode")).strip()
            or DEFAULT_ANNOTATION_DEFAULTS["selection_mode"]
        )
        interface = (
            text_or_empty(payload.get("interface")).strip()
            or text_or_empty((payload.get("provenance") or {}).get("interface")).strip()
            or DEFAULT_ANNOTATION_DEFAULTS["interface"]
        )
        labeler_type = (
            text_or_empty(payload.get("labeler_type")).strip()
            or text_or_empty((payload.get("provenance") or {}).get("labeler_type")).strip()
            or DEFAULT_ANNOTATION_DEFAULTS["labeler_type"]
        )
        normalized_quote = " ".join(quote.split())

        return {
            "id": preserve_id or replace_id or str(uuid.uuid4()),
            "document_id": document["document_id"],
            "source_key": document["source_key"],
            "source_label": document["source_label"],
            "dataset_key": document["dataset_key"],
            "question_id": document["question_id"],
            "variant_key": document["variant_key"],
            "surface": document["kind"],
            "start": start,
            "end": end,
            "span_length": end - start,
            "quote": quote,
            "normalized_quote": normalized_quote,
            "quote_sha1": hashlib.sha1(quote.encode("utf-8")).hexdigest(),
            "prefix": prefix,
            "suffix": suffix,
            "label_id": label_id,
            "label_snapshot": self._label_snapshot(label_id, category_lookup),
            "confidence": confidence,
            "comment": text_or_empty(payload.get("comment")).strip(),
            "author": text_or_empty(payload.get("author")).strip(),
            "status": text_or_empty(payload.get("status")).strip() or "confirmed",
            "created_at_utc": created_at_utc or utc_now(),
            "updated_at_utc": utc_now(),
            "source_text_hash": document["text_hash"],
            "provenance": {
                "labeler_type": labeler_type,
                "interface": interface,
                "selection_mode": selection_mode,
            },
        }


class AnnotationRequestHandler(SimpleHTTPRequestHandler):
    server_version = "ReasoningAnnotationServer/1.0"

    def __init__(self, *args: Any, directory: str | None = None, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(ROOT), **kwargs)

    @property
    def app(self) -> "AnnotationHTTPServer":
        return self.server  # type: ignore[return-value]

    def log_message(self, format: str, *args: Any) -> None:
        return super().log_message(format, *args)

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        try:
            if parsed.path == "/":
                self.send_response(HTTPStatus.FOUND)
                self.send_header("Location", "/viewer/reasoning-annotation-studio.html")
                self.end_headers()
                return
            if parsed.path == "/api/bootstrap":
                self._send_json(
                    {
                        "app": {
                            "name": "BullshitBench Reasoning Lab",
                            "store_path": str(self.app.store.path.relative_to(ROOT)),
                            "generated_at_utc": self.app.payload.get("generated_at_utc"),
                        },
                        "atlas": self.app.payload,
                        "store": self.app.store.snapshot(),
                    }
                )
                return
            if parsed.path == "/api/export":
                payload = self.app.store.snapshot()
                encoded = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
                body = encoded.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header(
                    "Content-Disposition",
                    'attachment; filename="reasoning_lab.export.json"',
                )
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
        except Exception as exc:  # noqa: BLE001
            self._send_error_payload(exc)
            return
        return super().do_GET()

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        try:
            payload = self._read_json_body()
            if parsed.path == "/api/categories":
                category = self.app.store.add_category(payload)
                self._send_json({"ok": True, "category": category}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/annotations":
                annotation = self.app.store.add_annotation(payload)
                self._send_json(
                    {"ok": True, "annotation": annotation},
                    status=HTTPStatus.CREATED,
                )
                return
            if parsed.path == "/api/document-notes":
                document_note = self.app.store.upsert_document_note(payload)
                self._send_json(
                    {"ok": True, "document_note": document_note},
                    status=HTTPStatus.CREATED,
                )
                return
            if parsed.path == "/api/import":
                query = urllib.parse.parse_qs(parsed.query)
                mode = text_or_empty(query.get("mode", ["merge"])[0]).strip() or "merge"
                store = self.app.store.import_store(payload, mode=mode)
                self._send_json({"ok": True, "store": store})
                return
        except Exception as exc:  # noqa: BLE001
            self._send_error_payload(exc)
            return
        self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

    def do_PATCH(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        try:
            payload = self._read_json_body()
            if parsed.path.startswith("/api/categories/"):
                category_id = parsed.path.rsplit("/", 1)[-1]
                category = self.app.store.update_category(category_id, payload)
                self._send_json({"ok": True, "category": category})
                return
            if parsed.path.startswith("/api/annotations/"):
                annotation_id = parsed.path.rsplit("/", 1)[-1]
                annotation = self.app.store.update_annotation(annotation_id, payload)
                self._send_json({"ok": True, "annotation": annotation})
                return
        except Exception as exc:  # noqa: BLE001
            self._send_error_payload(exc)
            return
        self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

    def do_DELETE(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        try:
            if parsed.path.startswith("/api/annotations/"):
                annotation_id = parsed.path.rsplit("/", 1)[-1]
                self.app.store.delete_annotation(annotation_id)
                self._send_json({"ok": True})
                return
        except Exception as exc:  # noqa: BLE001
            self._send_error_payload(exc)
            return
        self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

    def _read_json_body(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(content_length) if content_length else b"{}"
        if not raw.strip():
            return {}
        return json.loads(raw.decode("utf-8"))

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error_payload(self, exc: Exception) -> None:
        if isinstance(exc, KeyError):
            status = HTTPStatus.NOT_FOUND
        else:
            status = HTTPStatus.BAD_REQUEST
        self._send_json(
            {
                "ok": False,
                "error": str(exc),
                "error_type": exc.__class__.__name__,
            },
            status=status,
        )


class AnnotationHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        request_handler_class: type[AnnotationRequestHandler],
        payload: dict[str, Any],
        store: AnnotationStore,
    ) -> None:
        super().__init__(server_address, request_handler_class)
        self.payload = payload
        self.store = store


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8878)
    parser.add_argument(
        "--annotation-store",
        default=str(DEFAULT_STORE_PATH),
        help="Path to the JSON annotation store.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_payload()
    atlas_index = AtlasIndex(payload)
    store = AnnotationStore(Path(args.annotation_store).resolve(), atlas_index)
    try:
        server = AnnotationHTTPServer(
            (args.host, args.port),
            AnnotationRequestHandler,
            payload=payload,
            store=store,
        )
    except OSError as exc:
        if exc.errno in {48, 98}:
            raise SystemExit(
                f"Port {args.port} is already in use on {args.host}.\n"
                f"If the reasoning lab is already running, open "
                f"http://{args.host}:{args.port}/viewer/reasoning-annotation-studio.html\n"
                f"Otherwise stop the process on that port or rerun with --port <other-port>."
            ) from None
        raise
    print(
        "Serving BullshitBench Reasoning Lab on "
        f"http://{args.host}:{args.port}/viewer/reasoning-annotation-studio.html"
    )
    print(f"Annotation store: {store.path}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
