#!/usr/bin/env python3
"""Local annotation server for BullshitBench reasoning traces."""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import os
import threading
import time
import urllib.parse
import uuid
from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from ai_label_reasoning_trace import (
    AI_LABEL_MODE_COMPLETE_EXISTING,
    AI_LABEL_MODE_REPLACE,
    AI_LABEL_WORKFLOW_VERSION,
    DEFAULT_AI_LABEL_CONFIG_ID,
    list_ai_label_configs,
    run_ai_labeling,
)
from reasoning_lab_data import LAB_ROOT as ROOT, build_payload, text_or_empty


DEFAULT_STORE_PATH = ROOT / "annotations" / "reasoning_lab.json"
STORE_SCHEMA_VERSION = 7
AI_LABEL_RUN_STATUSES = {
    "queued",
    "running",
    "completed",
    "completed_with_errors",
    "cancelled",
}
AI_LABEL_RUN_ITEM_STATUSES = {
    "queued",
    "running",
    "applied",
    "failed",
    "cancelled",
}
AI_LABEL_RUN_TERMINAL_STATUSES = {
    "completed",
    "completed_with_errors",
    "cancelled",
}
AI_LABEL_RUN_ITEM_TERMINAL_STATUSES = {
    "applied",
    "failed",
    "cancelled",
}
DEFAULT_AI_LABEL_RUN_MAX_ATTEMPTS = 2
DEFAULT_AI_LABEL_RUN_CONCURRENCY = 10
REVIEW_SESSION_TYPE_AI_SUGGESTION = "ai_suggestion_review"
REVIEW_SESSION_STATUSES = {
    "open",
    "in_review",
    "resolved",
    "dismissed",
    "superseded",
}
REVIEW_ORIGINS = {
    "",
    "ai_suggestion",
    "human_addition",
}
REVIEW_EVENT_ACTIONS = {
    "added",
    "accepted",
    "edited",
    "deleted",
}
ANNOTATION_REVIEW_FIELDS = (
    "label_id",
    "comment",
    "start",
    "end",
    "quote",
    "selection_mode",
    "author",
    "human_reviewed",
    "reviewed_by",
    "reviewed_at_utc",
)
DOCUMENT_NOTE_REVIEW_FIELDS = (
    "label_id",
    "summary",
    "author",
    "human_reviewed",
    "reviewed_by",
    "reviewed_at_utc",
)

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
        "description": "Clearly pushes back on the underlying premise as wrong, incoherent, invented, or not answerable as asked.",
        "color": "#1f8f5c",
        "text_color": "#ffffff",
        "guidance": "Use when the trace clearly says the premise itself does not hold, should not be accepted, or directly rejects the question as posed.",
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
        "description": "Adds explanatory framing or soft pushback that helps explain why the question may be off, without flatly rejecting it.",
        "color": "#2878c7",
        "text_color": "#ffffff",
        "guidance": "Use when the trace is explaining why the question may be incorrect, mismatched, or non-standard, but stops short of a strong outright rejection.",
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
}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


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
    labeler_type = text_or_empty(provenance.get("labeler_type")).strip()
    source = text_or_empty(provenance.get("source")).strip()
    return labeler_type == "model" or source == "ai_labelling"


def record_is_human_reviewed(record: dict[str, Any]) -> bool:
    if "human_reviewed" in record:
        return normalize_bool(record.get("human_reviewed"))
    provenance = record.get("provenance") or {}
    return text_or_empty(provenance.get("labeler_type")).strip() == "human"


def document_note_sort_key(item: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        text_or_empty(item.get("source_key")),
        text_or_empty(item.get("dataset_key")),
        text_or_empty(item.get("question_id")),
        text_or_empty(item.get("document_id")),
    )


def annotation_sort_key(item: dict[str, Any]) -> tuple[str, str, str, str, int]:
    return (
        text_or_empty(item.get("source_key")),
        text_or_empty(item.get("dataset_key")),
        text_or_empty(item.get("question_id")),
        text_or_empty(item.get("document_id")),
        int(item.get("start") or 0),
    )


def review_session_sort_key(item: dict[str, Any]) -> tuple[str, str, str]:
    return (
        text_or_empty(item.get("document_id")),
        text_or_empty(item.get("created_at_utc")),
        text_or_empty(item.get("id")),
    )


def review_event_sort_key(item: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        text_or_empty(item.get("created_at_utc")),
        text_or_empty(item.get("document_id")),
        text_or_empty(item.get("session_id")),
        text_or_empty(item.get("record_kind")),
        text_or_empty(item.get("id")),
    )


def ai_label_run_sort_key(item: dict[str, Any]) -> tuple[str, str]:
    return (
        text_or_empty(item.get("created_at_utc")),
        text_or_empty(item.get("id")),
    )


def ai_label_run_item_sort_key(item: dict[str, Any]) -> tuple[str, int, str]:
    return (
        text_or_empty(item.get("run_id")),
        int(item.get("queue_index") or 0),
        text_or_empty(item.get("id")),
    )


def json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))


def parse_utc_timestamp(value: Any) -> dt.datetime | None:
    text = text_or_empty(value).strip()
    if not text:
        return None
    normalized = text.replace("Z", "+00:00")
    try:
        return dt.datetime.fromisoformat(normalized)
    except ValueError:
        return None


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


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
        "review_sessions": [],
        "review_events": [],
        "ai_label_runs": [],
        "ai_label_run_items": [],
    }


class ReasoningDocumentIndex:
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
    def __init__(self, path: Path, document_index: ReasoningDocumentIndex) -> None:
        self.path = path
        self.lock_path = path.with_suffix(path.suffix + ".lock")
        self.document_index = document_index
        self.lock = threading.Lock()
        self._ensure_store()

    @contextlib.contextmanager
    def _locked_store(self):
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock, self.lock_path.open("a+", encoding="utf-8") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    def _ensure_store(self) -> None:
        with self._locked_store():
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
        with self._locked_store():
            return self._normalize_store(self._read())

    def raw_snapshot(self) -> dict[str, Any]:
        with self._locked_store():
            return json_copy(self._read())

    def _document_metadata(self, document_id: str) -> dict[str, Any]:
        document = self.document_index.get_document(document_id)
        if not document:
            raise ValueError(f"Unknown document_id: {document_id}")
        return {
            "document_id": text_or_empty(document.get("document_id")).strip(),
            "source_key": text_or_empty(document.get("source_key")).strip(),
            "source_label": text_or_empty(document.get("source_label")).strip(),
            "dataset_key": text_or_empty(document.get("dataset_key")).strip(),
            "question_id": text_or_empty(document.get("question_id")).strip(),
            "variant_key": text_or_empty(document.get("variant_key")).strip(),
            "surface": text_or_empty(document.get("kind")).strip(),
        }

    def _review_record_id(self, kind: str, record: dict[str, Any] | None) -> str:
        if not isinstance(record, dict):
            return ""
        if kind == "annotation":
            return text_or_empty(record.get("id")).strip()
        if kind == "document_note":
            document_id = text_or_empty(record.get("document_id")).strip()
            return f"{document_id}:overall" if document_id else ""
        return ""

    def _default_origin_suggestion_id(
        self,
        kind: str,
        record: dict[str, Any],
        session_id: str,
    ) -> str:
        if kind == "annotation":
            base = "|".join(
                [
                    session_id,
                    text_or_empty(record.get("document_id")).strip(),
                    str(int(record.get("start") or 0)),
                    str(int(record.get("end") or 0)),
                    text_or_empty(record.get("quote_sha1")).strip()
                    or hashlib.sha1(text_or_empty(record.get("quote")).encode("utf-8")).hexdigest(),
                    text_or_empty(record.get("label_id")).strip(),
                ]
            )
            prefix = "ann"
        else:
            base = "|".join(
                [
                    session_id,
                    text_or_empty(record.get("document_id")).strip(),
                    text_or_empty(record.get("summary_sha1")).strip()
                    or hashlib.sha1(text_or_empty(record.get("summary")).encode("utf-8")).hexdigest(),
                    text_or_empty(record.get("label_id")).strip(),
                ]
            )
            prefix = "note"
        return f"{prefix}_{hashlib.sha1(base.encode('utf-8')).hexdigest()[:16]}"

    def _normalize_review_lineage_fields(
        self,
        payload: dict[str, Any],
        *,
        ai_labelled: bool,
    ) -> dict[str, str]:
        review_session_id = text_or_empty(payload.get("review_session_id")).strip()
        review_origin = text_or_empty(payload.get("review_origin")).strip().lower()
        origin_suggestion_id = text_or_empty(payload.get("origin_suggestion_id")).strip()
        if review_origin not in REVIEW_ORIGINS:
            if ai_labelled and (review_session_id or origin_suggestion_id):
                review_origin = "ai_suggestion"
            elif review_session_id and not ai_labelled:
                review_origin = "human_addition"
            else:
                review_origin = ""
        if review_origin != "ai_suggestion":
            origin_suggestion_id = ""
        return {
            "review_session_id": review_session_id,
            "review_origin": review_origin,
            "origin_suggestion_id": origin_suggestion_id,
        }

    def _non_review_field_changes(
        self,
        kind: str,
        before: dict[str, Any] | None,
        after: dict[str, Any] | None,
    ) -> list[str]:
        return [
            field
            for field in self._diff_review_fields(kind, before, after)
            if field not in {"human_reviewed", "reviewed_by", "reviewed_at_utc"}
        ]

    def _review_actor(self, record: dict[str, Any] | None, fallback: str = "") -> str:
        if not isinstance(record, dict):
            return fallback.strip()
        return (
            text_or_empty(record.get("reviewed_by")).strip()
            or text_or_empty(record.get("author")).strip()
            or fallback.strip()
        )

    def _promote_human_review(
        self,
        record: dict[str, Any],
        *,
        actor: str = "",
        reviewed_at_utc: str = "",
    ) -> dict[str, Any]:
        promoted = dict(record)
        promoted["human_reviewed"] = True
        promoted["reviewed_by"] = text_or_empty(promoted.get("reviewed_by")).strip() or actor.strip()
        promoted["reviewed_at_utc"] = (
            text_or_empty(promoted.get("reviewed_at_utc")).strip()
            or reviewed_at_utc.strip()
            or utc_now()
        )
        return promoted

    def _auto_promote_ai_review(
        self,
        *,
        kind: str,
        before: dict[str, Any] | None,
        after: dict[str, Any],
        payload: dict[str, Any],
        actor: str = "",
    ) -> dict[str, Any]:
        if not before or not record_is_ai_labelled(before):
            return after
        if "human_reviewed" in payload:
            return after
        if record_is_human_reviewed(after):
            return after
        if not self._non_review_field_changes(kind, before, after):
            return after
        return self._promote_human_review(after, actor=actor)

    def _meaningful_review_state(self, kind: str, record: dict[str, Any]) -> dict[str, Any]:
        fields = ANNOTATION_REVIEW_FIELDS if kind == "annotation" else DOCUMENT_NOTE_REVIEW_FIELDS
        state: dict[str, Any] = {}
        provenance = record.get("provenance") if isinstance(record.get("provenance"), dict) else {}
        for field in fields:
            if field == "selection_mode":
                state[field] = text_or_empty(
                    provenance.get("selection_mode") or record.get("selection_mode")
                ).strip()
            elif field in {"start", "end"}:
                state[field] = int(record.get(field) or 0)
            elif field == "human_reviewed":
                state[field] = record_is_human_reviewed(record)
            else:
                state[field] = text_or_empty(record.get(field)).strip()
        return state

    def _diff_review_fields(
        self,
        kind: str,
        before: dict[str, Any] | None,
        after: dict[str, Any] | None,
    ) -> list[str]:
        before_state = self._meaningful_review_state(kind, before or {})
        after_state = self._meaningful_review_state(kind, after or {})
        field_order = ANNOTATION_REVIEW_FIELDS if kind == "annotation" else DOCUMENT_NOTE_REVIEW_FIELDS
        return [
            field
            for field in field_order
            if before_state.get(field) != after_state.get(field)
        ]

    def _detect_review_action(
        self,
        kind: str,
        before: dict[str, Any] | None,
        after: dict[str, Any] | None,
    ) -> str:
        if before is None and after is not None:
            return "added"
        if before is not None and after is None:
            return "deleted"
        if before is None or after is None:
            return ""
        changed_fields = self._diff_review_fields(kind, before, after)
        if not changed_fields:
            return ""
        non_review_fields = [
            field
            for field in changed_fields
            if field not in {"human_reviewed", "reviewed_by", "reviewed_at_utc"}
        ]
        if (
            text_or_empty(before.get("review_origin")).strip() == "ai_suggestion"
            and not record_is_human_reviewed(before)
            and record_is_human_reviewed(after)
            and not non_review_fields
        ):
            return "accepted"
        return "edited"

    def _find_review_session(
        self,
        store: dict[str, Any],
        session_id: str,
    ) -> dict[str, Any] | None:
        for session in store.get("review_sessions") or []:
            if text_or_empty(session.get("id")).strip() == session_id:
                return session
        return None

    def _active_review_session_for_document(
        self,
        store: dict[str, Any],
        document_id: str,
    ) -> dict[str, Any] | None:
        matching = [
            session
            for session in store.get("review_sessions") or []
            if text_or_empty(session.get("document_id")).strip() == document_id
            and text_or_empty(session.get("status")).strip() in {"open", "in_review"}
        ]
        if not matching:
            return None
        return sorted(matching, key=review_session_sort_key)[-1]

    def _review_events_for_record(
        self,
        store: dict[str, Any],
        *,
        session_id: str,
        record_kind: str,
        record_id: str,
    ) -> list[dict[str, Any]]:
        if not session_id or not record_id:
            return []
        return [
            event
            for event in store.get("review_events") or []
            if text_or_empty(event.get("session_id")).strip() == session_id
            and text_or_empty(event.get("record_kind")).strip() == record_kind
            and text_or_empty(event.get("record_id")).strip() == record_id
        ]

    def _record_is_inferred_backfill_addition(
        self,
        store: dict[str, Any],
        *,
        record_kind: str,
        record: dict[str, Any],
    ) -> bool:
        if text_or_empty(record.get("review_origin")).strip() != "human_addition":
            return False
        session_id = text_or_empty(record.get("review_session_id")).strip()
        record_id = self._review_record_id(record_kind, record)
        events = self._review_events_for_record(
            store,
            session_id=session_id,
            record_kind=record_kind,
            record_id=record_id,
        )
        if not events:
            return False
        actions = {
            text_or_empty(event.get("action")).strip()
            for event in events
            if text_or_empty(event.get("action")).strip()
        }
        if not actions or not actions.issubset({"added"}):
            return False
        return all(normalize_bool(event.get("inferred"), default=False) for event in events)

    def _append_review_event(
        self,
        store: dict[str, Any],
        *,
        session_id: str,
        record_kind: str,
        before: dict[str, Any] | None,
        after: dict[str, Any] | None,
        actor: str = "",
        feedback: str = "",
    ) -> dict[str, Any] | None:
        if not session_id or record_kind not in {"annotation", "document_note"}:
            return None
        action = self._detect_review_action(record_kind, before, after)
        if action not in REVIEW_EVENT_ACTIONS:
            return None
        session = self._find_review_session(store, session_id)
        record = after or before or {}
        document_id = text_or_empty(record.get("document_id")).strip() or (
            text_or_empty(session.get("document_id")).strip() if session else ""
        )
        if not document_id:
            return None
        metadata = self._document_metadata(document_id)
        origin_source = after or before or {}
        event = {
            "id": str(uuid.uuid4()),
            **metadata,
            "session_id": session_id,
            "record_kind": record_kind,
            "record_id": self._review_record_id(record_kind, record),
            "action": action,
            "review_origin": text_or_empty(origin_source.get("review_origin")).strip(),
            "origin_suggestion_id": text_or_empty(origin_source.get("origin_suggestion_id")).strip(),
            "actor": actor.strip(),
            "created_at_utc": utc_now(),
            "changed_fields": self._diff_review_fields(record_kind, before, after),
            "before": json_copy(before) if before is not None else None,
            "after": json_copy(after) if after is not None else None,
            "feedback": feedback.strip(),
            "inferred": False,
            "inference_note": "",
        }
        store.setdefault("review_events", []).append(event)
        store["review_events"] = sorted(store["review_events"], key=review_event_sort_key)
        self._refresh_review_session_status(store, session_id)
        return event

    def _append_inferred_review_event(
        self,
        store: dict[str, Any],
        *,
        session_id: str,
        record_kind: str,
        before: dict[str, Any] | None,
        after: dict[str, Any] | None,
        actor: str = "",
        created_at_utc: str = "",
        inference_note: str = "",
    ) -> dict[str, Any] | None:
        if not session_id or record_kind not in {"annotation", "document_note"}:
            return None
        action = self._detect_review_action(record_kind, before, after)
        if action not in REVIEW_EVENT_ACTIONS:
            return None
        session = self._find_review_session(store, session_id)
        record = after or before or {}
        document_id = text_or_empty(record.get("document_id")).strip() or (
            text_or_empty(session.get("document_id")).strip() if session else ""
        )
        if not document_id:
            return None
        metadata = self._document_metadata(document_id)
        origin_source = after or before or {}
        event_payload = {
            "id": str(uuid.uuid4()),
            **metadata,
            "session_id": session_id,
            "record_kind": record_kind,
            "record_id": self._review_record_id(record_kind, record),
            "action": action,
            "review_origin": text_or_empty(origin_source.get("review_origin")).strip(),
            "origin_suggestion_id": text_or_empty(origin_source.get("origin_suggestion_id")).strip(),
            "actor": actor.strip(),
            "created_at_utc": created_at_utc or utc_now(),
            "changed_fields": self._diff_review_fields(record_kind, before, after),
            "before": json_copy(before) if before is not None else None,
            "after": json_copy(after) if after is not None else None,
            "feedback": "",
            "inferred": True,
            "inference_note": inference_note.strip(),
        }
        normalized_event = self._normalize_review_event_payload(event_payload)
        if normalized_event is None:
            return None
        store.setdefault("review_events", []).append(normalized_event)
        store["review_events"] = sorted(store["review_events"], key=review_event_sort_key)
        self._refresh_review_session_status(store, session_id)
        return normalized_event

    def _refresh_review_session_status(self, store: dict[str, Any], session_id: str) -> None:
        session = self._find_review_session(store, session_id)
        if not session:
            return
        if text_or_empty(session.get("status")).strip() == "superseded":
            return
        live_annotations = [
            item
            for item in store.get("annotations") or []
            if text_or_empty(item.get("review_session_id")).strip() == session_id
        ]
        live_notes = [
            item
            for item in store.get("document_notes") or []
            if text_or_empty(item.get("review_session_id")).strip() == session_id
        ]
        live_items = [*live_notes, *live_annotations]
        pending_ai_items = [
            item
            for item in live_items
            if text_or_empty(item.get("review_origin")).strip() == "ai_suggestion"
            and record_is_ai_labelled(item)
            and not record_is_human_reviewed(item)
        ]
        has_events = any(
            text_or_empty(event.get("session_id")).strip() == session_id
            for event in store.get("review_events") or []
        )
        has_review_activity = has_events or any(record_is_human_reviewed(item) for item in live_items)
        if pending_ai_items:
            next_status = "in_review" if has_review_activity else "open"
        elif live_items:
            next_status = "resolved"
        elif has_review_activity:
            next_status = "dismissed"
        else:
            next_status = "resolved"
        session["status"] = next_status
        session["updated_at_utc"] = utc_now()
        session["closed_at_utc"] = (
            text_or_empty(session.get("closed_at_utc")).strip() or utc_now()
            if next_status in {"resolved", "dismissed", "superseded"}
            else ""
        )

    def _supersede_active_review_sessions(
        self,
        store: dict[str, Any],
        *,
        document_id: str,
        superseded_by_session_id: str,
    ) -> None:
        now = utc_now()
        for session in store.get("review_sessions") or []:
            if text_or_empty(session.get("document_id")).strip() != document_id:
                continue
            if text_or_empty(session.get("status")).strip() not in {"open", "in_review"}:
                continue
            session["status"] = "superseded"
            session["superseded_by_session_id"] = superseded_by_session_id
            session["updated_at_utc"] = now
            session["closed_at_utc"] = now

    def _backfill_review_sessions(self, store: dict[str, Any]) -> None:
        existing_sessions = store.setdefault("review_sessions", [])
        existing_session_ids = {
            text_or_empty(item.get("id")).strip()
            for item in existing_sessions
            if text_or_empty(item.get("id")).strip()
        }
        existing_sessions_by_document: dict[str, list[dict[str, Any]]] = {}
        for session in existing_sessions:
            document_id = text_or_empty(session.get("document_id")).strip()
            if not document_id:
                continue
            existing_sessions_by_document.setdefault(document_id, []).append(session)
        grouped_items: dict[str, list[dict[str, Any]]] = {}

        for collection_key in ("document_notes", "annotations"):
            for item in store.get(collection_key) or []:
                if not record_is_ai_labelled(item):
                    continue
                document_id = text_or_empty(item.get("document_id")).strip()
                session_id = text_or_empty(item.get("review_session_id")).strip()
                if not session_id:
                    existing_for_document = sorted(
                        existing_sessions_by_document.get(document_id) or [],
                        key=lambda session: (
                            -len(session.get("suggested_annotations") or []),
                            text_or_empty(session.get("created_at_utc")).strip(),
                            text_or_empty(session.get("id")).strip(),
                        ),
                    )
                    if existing_for_document:
                        session_id = text_or_empty(existing_for_document[0].get("id")).strip()
                    else:
                        base = "|".join(
                            [
                                document_id,
                                text_or_empty(item.get("created_at_utc")).strip(),
                                text_or_empty(item.get("updated_at_utc")).strip(),
                            ]
                        )
                        session_id = f"backfill_{hashlib.sha1(base.encode('utf-8')).hexdigest()[:16]}"
                    item["review_session_id"] = session_id
                if text_or_empty(item.get("review_origin")).strip() != "ai_suggestion":
                    item["review_origin"] = "ai_suggestion"
                if not text_or_empty(item.get("origin_suggestion_id")).strip():
                    kind = "annotation" if collection_key == "annotations" else "document_note"
                    item["origin_suggestion_id"] = self._default_origin_suggestion_id(
                        kind,
                        item,
                        session_id,
                    )
                grouped_items.setdefault(session_id, []).append(item)

        category_lookup = {
            text_or_empty(category.get("id")): category
            for category in store.get("categories") or []
        }
        valid_category_ids = set(category_lookup.keys())

        for session_id, items in grouped_items.items():
            if session_id in existing_session_ids:
                continue
            by_kind = {
                "document_note": next(
                    (item for item in items if not text_or_empty(item.get("id")).strip()),
                    None,
                ),
                "annotations": sorted(
                    [
                        item
                        for item in items
                        if text_or_empty(item.get("id")).strip()
                    ],
                    key=annotation_sort_key,
                ),
            }
            first_item = items[0]
            suggestion_provenance = first_item.get("provenance") if isinstance(first_item.get("provenance"), dict) else {}
            created_at_utc = min(
                [
                    text_or_empty(item.get("created_at_utc")).strip() or utc_now()
                    for item in items
                ]
            )
            updated_at_utc = max(
                [
                    text_or_empty(item.get("updated_at_utc")).strip() or created_at_utc
                    for item in items
                ]
            )
            session = self._normalize_review_session_payload(
                payload={
                    "id": session_id,
                    "document_id": text_or_empty(first_item.get("document_id")).strip(),
                    "session_type": REVIEW_SESSION_TYPE_AI_SUGGESTION,
                    "status": "resolved" if all(record_is_human_reviewed(item) for item in items) else "open",
                    "created_at_utc": created_at_utc,
                    "updated_at_utc": updated_at_utc,
                    "closed_at_utc": (
                        max(
                            [
                                text_or_empty(item.get("reviewed_at_utc")).strip()
                                for item in items
                                if text_or_empty(item.get("reviewed_at_utc")).strip()
                            ],
                            default="",
                        )
                    ),
                    "created_by": text_or_empty(first_item.get("author")).strip(),
                    "suggestion_provenance": suggestion_provenance,
                    "suggested_document_note": by_kind["document_note"],
                    "suggested_annotations": by_kind["annotations"],
                },
                category_lookup=category_lookup,
                valid_category_ids=valid_category_ids,
                preserve_id=session_id,
            )
            existing_sessions.append(session)
            existing_session_ids.add(session_id)

        store["review_sessions"] = sorted(existing_sessions, key=review_session_sort_key)
        for session in store["review_sessions"]:
            self._refresh_review_session_status(store, text_or_empty(session.get("id")).strip())

    def _consolidate_review_sessions(self, store: dict[str, Any]) -> None:
        sessions_by_document: dict[str, list[dict[str, Any]]] = {}
        for session in store.get("review_sessions") or []:
            document_id = text_or_empty(session.get("document_id")).strip()
            if not document_id:
                continue
            sessions_by_document.setdefault(document_id, []).append(session)

        sessions_to_remove: set[str] = set()
        for document_id, sessions in sessions_by_document.items():
            if len(sessions) < 2:
                continue
            ordered = sorted(
                sessions,
                key=lambda session: (
                    -len(session.get("suggested_annotations") or []),
                    -int(bool(session.get("suggested_document_note"))),
                    text_or_empty(session.get("created_at_utc")).strip(),
                    text_or_empty(session.get("id")).strip(),
                ),
            )
            keeper = ordered[0]
            keeper_id = text_or_empty(keeper.get("id")).strip()
            keeper_note = keeper.get("suggested_document_note") if isinstance(keeper.get("suggested_document_note"), dict) else None
            for duplicate in ordered[1:]:
                duplicate_id = text_or_empty(duplicate.get("id")).strip()
                if not duplicate_id:
                    continue
                duplicate_note = duplicate.get("suggested_document_note") if isinstance(duplicate.get("suggested_document_note"), dict) else None
                same_note = (
                    bool(keeper_note)
                    and bool(duplicate_note)
                    and text_or_empty(keeper_note.get("summary")).strip() == text_or_empty(duplicate_note.get("summary")).strip()
                    and text_or_empty(keeper_note.get("label_id")).strip() == text_or_empty(duplicate_note.get("label_id")).strip()
                )
                if len(duplicate.get("suggested_annotations") or []) > 0 and not same_note:
                    continue
                for collection_key in ("document_notes", "annotations"):
                    for item in store.get(collection_key) or []:
                        if text_or_empty(item.get("review_session_id")).strip() == duplicate_id:
                            item["review_session_id"] = keeper_id
                for event in store.get("review_events") or []:
                    if text_or_empty(event.get("session_id")).strip() == duplicate_id:
                        event["session_id"] = keeper_id
                sessions_to_remove.add(duplicate_id)

        if sessions_to_remove:
            store["review_sessions"] = [
                session
                for session in (store.get("review_sessions") or [])
                if text_or_empty(session.get("id")).strip() not in sessions_to_remove
            ]
            deduped_events: dict[tuple[str, str, str, str], dict[str, Any]] = {}
            for event in sorted(store.get("review_events") or [], key=review_event_sort_key):
                signature = (
                    text_or_empty(event.get("session_id")).strip(),
                    text_or_empty(event.get("record_kind")).strip(),
                    text_or_empty(event.get("action")).strip(),
                    text_or_empty(event.get("record_id")).strip(),
                )
                if signature in deduped_events:
                    continue
                deduped_events[signature] = event
            store["review_events"] = sorted(deduped_events.values(), key=review_event_sort_key)

    def _backfill_review_events(self, store: dict[str, Any]) -> None:
        existing_event_signatures = {
            (
                text_or_empty(event.get("session_id")).strip(),
                text_or_empty(event.get("record_kind")).strip(),
                text_or_empty(event.get("action")).strip(),
                text_or_empty(event.get("record_id")).strip(),
            )
            for event in (store.get("review_events") or [])
            if text_or_empty(event.get("session_id")).strip()
        }
        document_notes = store.get("document_notes") or []
        annotations = store.get("annotations") or []
        review_sessions = sorted(store.get("review_sessions") or [], key=review_session_sort_key)
        sessions_by_document: dict[str, list[dict[str, Any]]] = {}
        for session in review_sessions:
            document_id = text_or_empty(session.get("document_id")).strip()
            session_id = text_or_empty(session.get("id")).strip()
            if not document_id or not session_id:
                continue
            sessions_by_document.setdefault(document_id, []).append(session)

        for item in [*document_notes, *annotations]:
            if not isinstance(item, dict):
                continue
            if text_or_empty(item.get("review_session_id")).strip():
                continue
            if record_is_ai_labelled(item):
                continue
            document_id = text_or_empty(item.get("document_id")).strip()
            if not document_id:
                continue
            item_created = (
                parse_utc_timestamp(item.get("created_at_utc"))
                or parse_utc_timestamp(item.get("updated_at_utc"))
            )
            eligible_sessions = [
                session
                for session in sessions_by_document.get(document_id, [])
                if not (
                    (session_created := parse_utc_timestamp(session.get("created_at_utc")))
                    and item_created
                    and item_created < session_created
                )
            ]
            if not eligible_sessions:
                continue
            session_id = text_or_empty(eligible_sessions[-1].get("id")).strip()
            if not session_id:
                continue
            item["review_session_id"] = session_id
            item["review_origin"] = "human_addition"
            item["origin_suggestion_id"] = ""

        for session in review_sessions:
            session_id = text_or_empty(session.get("id")).strip()
            document_id = text_or_empty(session.get("document_id")).strip()
            if not session_id or not document_id:
                continue

            suggested_note = session.get("suggested_document_note") if isinstance(session.get("suggested_document_note"), dict) else None
            suggested_annotations = [
                item
                for item in (session.get("suggested_annotations") or [])
                if isinstance(item, dict)
            ]
            live_note = next(
                (
                    item
                    for item in document_notes
                    if text_or_empty(item.get("document_id")).strip() == document_id
                ),
                None,
            )
            live_annotations = [
                item
                for item in annotations
                if text_or_empty(item.get("document_id")).strip() == document_id
            ]

            live_note = next(
                (
                    item
                    for item in document_notes
                    if text_or_empty(item.get("document_id")).strip() == document_id
                ),
                None,
            )
            live_annotations = [
                item
                for item in annotations
                if text_or_empty(item.get("document_id")).strip() == document_id
            ]

            if suggested_note is not None:
                if live_note is None:
                    event = self._append_inferred_review_event(
                        store,
                        session_id=session_id,
                        record_kind="document_note",
                        before=suggested_note,
                        after=None,
                        created_at_utc=text_or_empty(session.get("updated_at_utc")).strip() or utc_now(),
                        inference_note="Reconstructed deleted AI overall label because the suggested note no longer exists in live labels.",
                    )
                    if event is not None:
                        existing_event_signatures.add(
                            (session_id, event["record_kind"], event["action"], event["record_id"])
                        )
                else:
                    candidate_action = self._detect_review_action("document_note", suggested_note, live_note)
                    candidate_record_id = self._review_record_id("document_note", live_note)
                    if (session_id, "document_note", candidate_action, candidate_record_id) in existing_event_signatures:
                        pass
                    else:
                        event = self._append_inferred_review_event(
                            store,
                            session_id=session_id,
                            record_kind="document_note",
                            before=suggested_note,
                            after=live_note,
                            actor=text_or_empty(live_note.get("reviewed_by")).strip()
                            or text_or_empty(live_note.get("author")).strip(),
                            created_at_utc=text_or_empty(live_note.get("reviewed_at_utc")).strip()
                            or text_or_empty(live_note.get("updated_at_utc")).strip()
                            or utc_now(),
                            inference_note=(
                                "Reconstructed review of the AI overall label by comparing the saved suggestion snapshot with the current live note."
                            ),
                        )
                        if event is not None:
                            existing_event_signatures.add(
                                (session_id, event["record_kind"], event["action"], event["record_id"])
                            )

            live_by_origin = {
                text_or_empty(item.get("origin_suggestion_id")).strip(): item
                for item in live_annotations
                if text_or_empty(item.get("origin_suggestion_id")).strip()
            }
            for suggested in suggested_annotations:
                origin_suggestion_id = text_or_empty(suggested.get("origin_suggestion_id")).strip()
                live_match = live_by_origin.get(origin_suggestion_id)
                if live_match is None:
                    candidate_record_id = self._review_record_id("annotation", suggested)
                    if (session_id, "annotation", "deleted", candidate_record_id) not in existing_event_signatures:
                        event = self._append_inferred_review_event(
                            store,
                            session_id=session_id,
                            record_kind="annotation",
                            before=suggested,
                            after=None,
                            created_at_utc=text_or_empty(session.get("updated_at_utc")).strip() or utc_now(),
                            inference_note="Reconstructed deleted AI span because the suggested span no longer exists in the live labels.",
                        )
                        if event is not None:
                            existing_event_signatures.add(
                                (session_id, event["record_kind"], event["action"], event["record_id"])
                            )
                    continue
                candidate_action = self._detect_review_action("annotation", suggested, live_match)
                candidate_record_id = self._review_record_id("annotation", live_match)
                if (session_id, "annotation", candidate_action, candidate_record_id) in existing_event_signatures:
                    continue
                event = self._append_inferred_review_event(
                    store,
                    session_id=session_id,
                    record_kind="annotation",
                    before=suggested,
                    after=live_match,
                    actor=text_or_empty(live_match.get("reviewed_by")).strip()
                    or text_or_empty(live_match.get("author")).strip(),
                    created_at_utc=text_or_empty(live_match.get("reviewed_at_utc")).strip()
                    or text_or_empty(live_match.get("updated_at_utc")).strip()
                    or utc_now(),
                    inference_note="Reconstructed review of the AI span by comparing the saved suggestion snapshot with the current live span.",
                )
                if event is not None:
                    existing_event_signatures.add(
                        (session_id, event["record_kind"], event["action"], event["record_id"])
                    )

            for item in live_annotations:
                if text_or_empty(item.get("review_session_id")).strip() != session_id:
                    continue
                if text_or_empty(item.get("review_origin")).strip() != "human_addition":
                    continue
                candidate_record_id = self._review_record_id("annotation", item)
                if (session_id, "annotation", "added", candidate_record_id) in existing_event_signatures:
                    continue
                event = self._append_inferred_review_event(
                    store,
                    session_id=session_id,
                    record_kind="annotation",
                    before=None,
                    after=item,
                    actor=text_or_empty(item.get("reviewed_by")).strip()
                    or text_or_empty(item.get("author")).strip(),
                    created_at_utc=text_or_empty(item.get("created_at_utc")).strip()
                    or text_or_empty(item.get("updated_at_utc")).strip()
                    or utc_now(),
                    inference_note="Reconstructed human-added span during AI review because the live span was created after the AI suggestion and has no matching suggested origin.",
                )
                if event is not None:
                    existing_event_signatures.add(
                        (session_id, event["record_kind"], event["action"], event["record_id"])
                    )

            if (
                live_note is not None
                and text_or_empty(live_note.get("review_session_id")).strip() == session_id
                and text_or_empty(live_note.get("review_origin")).strip() == "human_addition"
            ):
                candidate_record_id = self._review_record_id("document_note", live_note)
                if (session_id, "document_note", "added", candidate_record_id) in existing_event_signatures:
                    continue
                event = self._append_inferred_review_event(
                    store,
                    session_id=session_id,
                    record_kind="document_note",
                    before=None,
                    after=live_note,
                    actor=text_or_empty(live_note.get("reviewed_by")).strip()
                    or text_or_empty(live_note.get("author")).strip(),
                    created_at_utc=text_or_empty(live_note.get("created_at_utc")).strip()
                    or text_or_empty(live_note.get("updated_at_utc")).strip()
                    or utc_now(),
                    inference_note="Reconstructed human-added overall label during AI review because the live note was created after the AI suggestion and has no matching suggested origin.",
                )
                if event is not None:
                    existing_event_signatures.add(
                        (session_id, event["record_kind"], event["action"], event["record_id"])
                    )
            self._refresh_review_session_status(store, session_id)

    def _normalize_review_session_payload(
        self,
        payload: dict[str, Any],
        *,
        category_lookup: dict[str, dict[str, Any]],
        valid_category_ids: set[str],
        preserve_id: str | None = None,
    ) -> dict[str, Any]:
        document_id = text_or_empty(payload.get("document_id")).strip()
        metadata = self._document_metadata(document_id)
        session_id = text_or_empty(payload.get("id")).strip() or preserve_id or str(uuid.uuid4())
        status = text_or_empty(payload.get("status")).strip()
        if status not in REVIEW_SESSION_STATUSES:
            status = "open"
        suggestion_provenance = payload.get("suggestion_provenance")
        normalized_suggestion_provenance: dict[str, Any] = {}
        if isinstance(suggestion_provenance, dict):
            for key, value in suggestion_provenance.items():
                normalized_key = text_or_empty(key).strip()
                if not normalized_key or value is None:
                    continue
                if isinstance(value, (str, int, float, bool, list, dict)):
                    normalized_suggestion_provenance[normalized_key] = value
                else:
                    normalized_value = text_or_empty(value).strip()
                    if normalized_value:
                        normalized_suggestion_provenance[normalized_key] = normalized_value

        suggested_note_payload = payload.get("suggested_document_note")
        normalized_suggested_note = None
        if isinstance(suggested_note_payload, dict):
            normalized_suggested_note = self._normalize_document_note_payload(
                payload={
                    **suggested_note_payload,
                    "document_id": metadata["document_id"],
                    "ai_labelled": True,
                    "human_reviewed": False,
                    "reviewed_by": "",
                    "reviewed_at_utc": "",
                    "review_session_id": session_id,
                    "review_origin": "ai_suggestion",
                },
                created_at_utc=text_or_empty(suggested_note_payload.get("created_at_utc")).strip()
                or text_or_empty(payload.get("created_at_utc")).strip()
                or utc_now(),
                category_lookup=category_lookup,
                valid_category_ids=valid_category_ids,
            )

        normalized_suggested_annotations: list[dict[str, Any]] = []
        for annotation in payload.get("suggested_annotations") or []:
            if not isinstance(annotation, dict):
                continue
            normalized_annotation = self._normalize_annotation_payload(
                payload={
                    **annotation,
                    "document_id": metadata["document_id"],
                    "ai_labelled": True,
                    "human_reviewed": False,
                    "reviewed_by": "",
                    "reviewed_at_utc": "",
                    "review_session_id": session_id,
                    "review_origin": "ai_suggestion",
                },
                existing_annotations=normalized_suggested_annotations,
                replace_id=text_or_empty(annotation.get("id")).strip() or None,
                created_at_utc=text_or_empty(annotation.get("created_at_utc")).strip()
                or text_or_empty(payload.get("created_at_utc")).strip()
                or utc_now(),
                preserve_id=text_or_empty(annotation.get("id")).strip() or None,
                category_lookup=category_lookup,
                valid_category_ids=valid_category_ids,
            )
            normalized_suggested_annotations.append(normalized_annotation)

        resolution_warnings = [
            text_or_empty(item).strip()
            for item in (payload.get("resolution_warnings") or [])
            if text_or_empty(item).strip()
        ]
        created_at_utc = text_or_empty(payload.get("created_at_utc")).strip() or utc_now()
        updated_at_utc = text_or_empty(payload.get("updated_at_utc")).strip() or created_at_utc
        closed_at_utc = text_or_empty(payload.get("closed_at_utc")).strip()
        if status in {"open", "in_review"}:
            closed_at_utc = ""
        ai_label_run_id = text_or_empty(payload.get("ai_label_run_id")).strip() or text_or_empty(
            normalized_suggestion_provenance.get("ai_label_run_id")
        ).strip()
        ai_label_run_item_id = text_or_empty(payload.get("ai_label_run_item_id")).strip() or text_or_empty(
            normalized_suggestion_provenance.get("ai_label_run_item_id")
        ).strip()
        return {
            "id": session_id,
            **metadata,
            "session_type": text_or_empty(payload.get("session_type")).strip()
            or REVIEW_SESSION_TYPE_AI_SUGGESTION,
            "status": status,
            "created_at_utc": created_at_utc,
            "updated_at_utc": updated_at_utc,
            "closed_at_utc": closed_at_utc,
            "created_by": text_or_empty(payload.get("created_by")).strip(),
            "superseded_by_session_id": text_or_empty(payload.get("superseded_by_session_id")).strip(),
            "ai_label_run_id": ai_label_run_id,
            "ai_label_run_item_id": ai_label_run_item_id,
            "suggestion_provenance": normalized_suggestion_provenance,
            "resolution_warnings": resolution_warnings,
            "suggested_document_note": normalized_suggested_note,
            "suggested_annotations": sorted(
                normalized_suggested_annotations,
                key=annotation_sort_key,
            ),
        }

    def _normalize_review_event_payload(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        session_id = text_or_empty(payload.get("session_id")).strip()
        record_kind = text_or_empty(payload.get("record_kind")).strip()
        action = text_or_empty(payload.get("action")).strip()
        if not session_id or record_kind not in {"annotation", "document_note"}:
            return None
        if action not in REVIEW_EVENT_ACTIONS:
            return None
        before = payload.get("before") if isinstance(payload.get("before"), dict) else None
        after = payload.get("after") if isinstance(payload.get("after"), dict) else None
        source_record = after or before or {}
        document_id = (
            text_or_empty(source_record.get("document_id")).strip()
            or text_or_empty(payload.get("document_id")).strip()
        )
        if not document_id:
            return None
        metadata = self._document_metadata(document_id)
        changed_fields = [
            text_or_empty(field).strip()
            for field in (payload.get("changed_fields") or [])
            if text_or_empty(field).strip()
        ] or self._diff_review_fields(record_kind, before, after)
        return {
            "id": text_or_empty(payload.get("id")).strip() or str(uuid.uuid4()),
            **metadata,
            "session_id": session_id,
            "record_kind": record_kind,
            "record_id": text_or_empty(payload.get("record_id")).strip()
            or self._review_record_id(record_kind, source_record),
            "action": action,
            "review_origin": text_or_empty(payload.get("review_origin")).strip()
            or text_or_empty(source_record.get("review_origin")).strip(),
            "origin_suggestion_id": text_or_empty(payload.get("origin_suggestion_id")).strip()
            or text_or_empty(source_record.get("origin_suggestion_id")).strip(),
            "actor": text_or_empty(payload.get("actor")).strip(),
            "created_at_utc": text_or_empty(payload.get("created_at_utc")).strip() or utc_now(),
            "changed_fields": changed_fields,
            "before": json_copy(before) if before is not None else None,
            "after": json_copy(after) if after is not None else None,
            "feedback": text_or_empty(payload.get("feedback")).strip(),
            "inferred": normalize_bool(payload.get("inferred"), default=False),
            "inference_note": text_or_empty(payload.get("inference_note")).strip(),
        }

    def _normalize_ai_label_run_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        status = text_or_empty(payload.get("status")).strip()
        if status not in AI_LABEL_RUN_STATUSES:
            status = "queued"
        document_ids = [
            text_or_empty(document_id).strip()
            for document_id in (payload.get("document_ids") or [])
            if text_or_empty(document_id).strip()
        ]
        unique_document_ids: list[str] = []
        seen_document_ids: set[str] = set()
        for document_id in document_ids:
            try:
                canonical_document_id = self._document_metadata(document_id)["document_id"]
            except ValueError:
                canonical_document_id = document_id
            if canonical_document_id in seen_document_ids:
                continue
            seen_document_ids.add(canonical_document_id)
            unique_document_ids.append(canonical_document_id)
        skipped_document_ids = [
            text_or_empty(document_id).strip()
            for document_id in (payload.get("skipped_document_ids") or [])
            if text_or_empty(document_id).strip()
        ]
        config = payload.get("config") if isinstance(payload.get("config"), dict) else {}
        return {
            "id": text_or_empty(payload.get("id")).strip() or str(uuid.uuid4()),
            "status": status,
            "created_at_utc": text_or_empty(payload.get("created_at_utc")).strip() or utc_now(),
            "updated_at_utc": text_or_empty(payload.get("updated_at_utc")).strip()
            or text_or_empty(payload.get("created_at_utc")).strip()
            or utc_now(),
            "started_at_utc": text_or_empty(payload.get("started_at_utc")).strip(),
            "finished_at_utc": text_or_empty(payload.get("finished_at_utc")).strip(),
            "created_by": text_or_empty(payload.get("created_by")).strip(),
            "config_id": text_or_empty(payload.get("config_id")).strip(),
            "config": json_copy(config),
            "mode": text_or_empty(payload.get("mode")).strip() or "auto",
            "document_ids": unique_document_ids,
            "requested_document_count": int(
                payload.get("requested_document_count") or len(unique_document_ids)
            ),
            "skipped_document_ids": skipped_document_ids,
            "total_count": int(payload.get("total_count") or len(unique_document_ids)),
            "queued_count": int(payload.get("queued_count") or 0),
            "running_count": int(payload.get("running_count") or 0),
            "applied_count": int(payload.get("applied_count") or 0),
            "failed_count": int(payload.get("failed_count") or 0),
            "cancelled_count": int(payload.get("cancelled_count") or 0),
            "last_error": text_or_empty(payload.get("last_error")).strip(),
        }

    def _normalize_ai_label_run_item_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        document_id = text_or_empty(payload.get("document_id")).strip()
        metadata = {}
        if document_id:
            try:
                metadata = self._document_metadata(document_id)
            except ValueError:
                metadata = {
                    "document_id": document_id,
                    "source_key": text_or_empty(payload.get("source_key")).strip(),
                    "source_label": text_or_empty(payload.get("source_label")).strip(),
                    "dataset_key": text_or_empty(payload.get("dataset_key")).strip(),
                    "question_id": text_or_empty(payload.get("question_id")).strip(),
                    "variant_key": text_or_empty(payload.get("variant_key")).strip(),
                    "surface": text_or_empty(payload.get("surface")).strip() or "reasoning",
                }
        status = text_or_empty(payload.get("status")).strip()
        if status not in AI_LABEL_RUN_ITEM_STATUSES:
            status = "queued"
        resolution_warnings = [
            text_or_empty(item).strip()
            for item in (payload.get("resolution_warnings") or [])
            if text_or_empty(item).strip()
        ]
        error = text_or_empty(payload.get("error")).strip()
        return {
            "id": text_or_empty(payload.get("id")).strip() or str(uuid.uuid4()),
            "run_id": text_or_empty(payload.get("run_id")).strip(),
            **metadata,
            "queue_index": int(payload.get("queue_index") or 0),
            "status": status,
            "mode_requested": text_or_empty(payload.get("mode_requested")).strip() or "auto",
            "label_mode": text_or_empty(payload.get("label_mode")).strip(),
            "config_id": text_or_empty(payload.get("config_id")).strip(),
            "attempt_count": int(payload.get("attempt_count") or 0),
            "max_attempts": int(
                payload.get("max_attempts") or DEFAULT_AI_LABEL_RUN_MAX_ATTEMPTS
            ),
            "started_at_utc": text_or_empty(payload.get("started_at_utc")).strip(),
            "finished_at_utc": text_or_empty(payload.get("finished_at_utc")).strip(),
            "review_session_id": text_or_empty(payload.get("review_session_id")).strip(),
            "response_id": text_or_empty(payload.get("response_id")).strip(),
            "resolution_warnings": resolution_warnings,
            "error": error,
            "last_error_at_utc": text_or_empty(payload.get("last_error_at_utc")).strip(),
        }

    def _find_ai_label_run(
        self,
        store: dict[str, Any],
        run_id: str,
    ) -> dict[str, Any] | None:
        for run in store.get("ai_label_runs") or []:
            if text_or_empty(run.get("id")).strip() == run_id:
                return run
        return None

    def _find_ai_label_run_item(
        self,
        store: dict[str, Any],
        item_id: str,
    ) -> dict[str, Any] | None:
        for item in store.get("ai_label_run_items") or []:
            if text_or_empty(item.get("id")).strip() == item_id:
                return item
        return None

    def _refresh_ai_label_run_status(
        self,
        store: dict[str, Any],
        run_id: str,
    ) -> None:
        run = self._find_ai_label_run(store, run_id)
        if not run:
            return
        items = [
            item
            for item in (store.get("ai_label_run_items") or [])
            if text_or_empty(item.get("run_id")).strip() == run_id
        ]
        queued_count = sum(
            1 for item in items if text_or_empty(item.get("status")).strip() == "queued"
        )
        running_count = sum(
            1 for item in items if text_or_empty(item.get("status")).strip() == "running"
        )
        applied_count = sum(
            1 for item in items if text_or_empty(item.get("status")).strip() == "applied"
        )
        failed_count = sum(
            1 for item in items if text_or_empty(item.get("status")).strip() == "failed"
        )
        cancelled_count = sum(
            1 for item in items if text_or_empty(item.get("status")).strip() == "cancelled"
        )
        run["total_count"] = len(items)
        run["queued_count"] = queued_count
        run["running_count"] = running_count
        run["applied_count"] = applied_count
        run["failed_count"] = failed_count
        run["cancelled_count"] = cancelled_count
        if running_count or queued_count:
            run["status"] = "running" if run.get("started_at_utc") else "queued"
        elif failed_count:
            run["status"] = "completed_with_errors"
        elif cancelled_count and not applied_count:
            run["status"] = "cancelled"
        else:
            run["status"] = "completed"
        if (applied_count or failed_count or cancelled_count) and not text_or_empty(
            run.get("started_at_utc")
        ).strip():
            first_started = min(
                (
                    text_or_empty(item.get("started_at_utc")).strip()
                    for item in items
                    if text_or_empty(item.get("started_at_utc")).strip()
                ),
                default="",
            )
            run["started_at_utc"] = first_started
        if run["status"] in AI_LABEL_RUN_TERMINAL_STATUSES:
            run["finished_at_utc"] = text_or_empty(run.get("finished_at_utc")).strip() or utc_now()
        else:
            run["finished_at_utc"] = ""
        if failed_count:
            latest_failed = max(
                (
                    item
                    for item in items
                    if text_or_empty(item.get("status")).strip() == "failed"
                ),
                key=lambda item: text_or_empty(item.get("last_error_at_utc")).strip()
                or text_or_empty(item.get("finished_at_utc")).strip()
                or "",
                default=None,
            )
            run["last_error"] = text_or_empty((latest_failed or {}).get("error")).strip()
        elif run["status"] != "completed_with_errors":
            run["last_error"] = ""
        run["updated_at_utc"] = utc_now()

    def _refresh_ai_label_runs(self, store: dict[str, Any]) -> None:
        for run in store.get("ai_label_runs") or []:
            run_id = text_or_empty(run.get("id")).strip()
            if run_id:
                self._refresh_ai_label_run_status(store, run_id)

    def add_category(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._locked_store():
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
        with self._locked_store():
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
        with self._locked_store():
            store = self._read()
            document_id = text_or_empty(payload.get("document_id")).strip()
            metadata = self._document_metadata(document_id)
            active_session = self._active_review_session_for_document(
                store,
                metadata["document_id"],
            )
            enriched_payload = dict(payload)
            if active_session and not text_or_empty(enriched_payload.get("review_session_id")).strip():
                enriched_payload["review_session_id"] = text_or_empty(active_session.get("id")).strip()
                if not text_or_empty(enriched_payload.get("review_origin")).strip():
                    enriched_payload["review_origin"] = "human_addition"
            annotation = self._normalize_annotation_payload(
                payload=enriched_payload,
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
            store["annotations"] = sorted(store["annotations"], key=annotation_sort_key)
            session_id = text_or_empty(annotation.get("review_session_id")).strip()
            if session_id:
                self._append_review_event(
                    store,
                    session_id=session_id,
                    record_kind="annotation",
                    before=None,
                    after=annotation,
                    actor=text_or_empty(annotation.get("author")).strip(),
                )
            self._write(store)
            return annotation

    def update_annotation(self, annotation_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        with self._locked_store():
            store = self._read()
            for index, current in enumerate(store["annotations"]):
                if text_or_empty(current.get("id")) != annotation_id:
                    continue
                merged = dict(current)
                for field in (
                    "label_id",
                    "comment",
                    "status",
                    "author",
                    "start",
                    "end",
                    "quote",
                    "document_id",
                    "selection_mode",
                    "interface",
                    "labeler_type",
                    "provenance",
                    "ai_labelled",
                    "human_reviewed",
                    "reviewed_by",
                    "reviewed_at_utc",
                    "source_text_hash",
                    "review_session_id",
                    "review_origin",
                    "origin_suggestion_id",
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
                annotation = self._auto_promote_ai_review(
                    kind="annotation",
                    before=current,
                    after=annotation,
                    payload=payload,
                    actor=(
                        text_or_empty(payload.get("reviewed_by")).strip()
                        or text_or_empty(payload.get("author")).strip()
                        or self._review_actor(current)
                    ),
                )
                store["annotations"][index] = annotation
                session_id = text_or_empty(annotation.get("review_session_id")).strip() or text_or_empty(current.get("review_session_id")).strip()
                if session_id:
                    self._append_review_event(
                        store,
                        session_id=session_id,
                        record_kind="annotation",
                        before=current,
                        after=annotation,
                        actor=text_or_empty(annotation.get("reviewed_by")).strip()
                        or text_or_empty(annotation.get("author")).strip(),
                    )
                self._write(store)
                return annotation
            raise KeyError(f"Unknown annotation: {annotation_id}")

    def upsert_document_note(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        with self._locked_store():
            store = self._read()
            notes = store.get("document_notes") or []
            document_id = text_or_empty(payload.get("document_id")).strip()
            document = self.document_index.get_document(document_id)
            if not document:
                raise ValueError(f"Unknown document_id: {document_id}")
            document_id = text_or_empty(document.get("document_id")).strip()
            existing_note = next(
                (
                    item
                    for item in notes
                    if text_or_empty(item.get("document_id")) == document_id
                ),
                None,
            )
            merged = dict(existing_note or {})
            active_session = None
            if document_id:
                active_session = self._active_review_session_for_document(
                    store,
                    self._document_metadata(document_id)["document_id"],
                )
            for field in (
                "document_id",
                "summary",
                "label_id",
                "author",
                "interface",
                "labeler_type",
                "provenance",
                "ai_labelled",
                "human_reviewed",
                "reviewed_by",
                "reviewed_at_utc",
                "source_text_hash",
                "review_session_id",
                "review_origin",
                "origin_suggestion_id",
            ):
                if field in payload:
                    merged[field] = payload[field]
            if (
                active_session
                and not existing_note
                and not text_or_empty(merged.get("review_session_id")).strip()
            ):
                merged["review_session_id"] = text_or_empty(active_session.get("id")).strip()
                if not text_or_empty(merged.get("review_origin")).strip():
                    merged["review_origin"] = "human_addition"
            normalized = self._normalize_document_note_payload(
                payload=merged,
                created_at_utc=text_or_empty((existing_note or {}).get("created_at_utc")) or utc_now(),
                category_lookup={
                    text_or_empty(category.get("id")): category
                    for category in store["categories"]
                },
                valid_category_ids={
                    text_or_empty(category.get("id")) for category in store["categories"]
                },
            )
            if normalized is not None:
                normalized = self._auto_promote_ai_review(
                    kind="document_note",
                    before=existing_note,
                    after=normalized,
                    payload=payload,
                    actor=(
                        text_or_empty(payload.get("reviewed_by")).strip()
                        or text_or_empty(payload.get("author")).strip()
                        or self._review_actor(existing_note)
                    ),
                )
            store["document_notes"] = [
                item for item in notes if text_or_empty(item.get("document_id")) != document_id
            ]
            if normalized is not None:
                store["document_notes"].append(normalized)
                store["document_notes"] = sorted(store["document_notes"], key=document_note_sort_key)
            session_id = text_or_empty((normalized or existing_note or {}).get("review_session_id")).strip()
            if session_id:
                self._append_review_event(
                    store,
                    session_id=session_id,
                    record_kind="document_note",
                    before=existing_note,
                    after=normalized,
                    actor=text_or_empty((normalized or existing_note or {}).get("reviewed_by")).strip()
                    or text_or_empty((normalized or existing_note or {}).get("author")).strip(),
                )
            self._write(store)
            return normalized

    def delete_annotation(self, annotation_id: str) -> None:
        with self._locked_store():
            store = self._read()
            current = next(
                (
                    item
                    for item in store["annotations"]
                    if text_or_empty(item.get("id")) == annotation_id
                ),
                None,
            )
            before = len(store["annotations"])
            store["annotations"] = [
                item
                for item in store["annotations"]
                if text_or_empty(item.get("id")) != annotation_id
            ]
            if len(store["annotations"]) == before:
                raise KeyError(f"Unknown annotation: {annotation_id}")
            session_id = text_or_empty((current or {}).get("review_session_id")).strip()
            if session_id and current:
                self._append_review_event(
                    store,
                    session_id=session_id,
                    record_kind="annotation",
                    before=current,
                    after=None,
                    actor=text_or_empty(current.get("reviewed_by")).strip()
                    or text_or_empty(current.get("author")).strip(),
                )
            self._write(store)

    def _normalize_selection_payload(
        self,
        payload: dict[str, Any],
    ) -> tuple[dict[str, Any], str, int, int, str]:
        document_id = text_or_empty(payload.get("document_id")).strip()
        document = self.document_index.get_document(document_id)
        if not document:
            raise ValueError(f"Unknown document_id: {document_id}")
        canonical_document_id = text_or_empty(document.get("document_id")).strip()
        start = int(payload.get("start"))
        end = int(payload.get("end"))
        if start < 0 or end <= start:
            raise ValueError("Annotation start/end offsets are invalid.")
        document_text = text_or_empty(document.get("text"))
        if end > len(document_text):
            raise ValueError("Annotation end offset exceeds document length.")
        quote = document_text[start:end]
        submitted_quote = text_or_empty(payload.get("quote")).strip()
        if submitted_quote and submitted_quote != quote:
            raise ValueError("Submitted quote does not match document text.")
        incoming_source_text_hash = text_or_empty(payload.get("source_text_hash")).strip()
        if incoming_source_text_hash and incoming_source_text_hash != document["text_hash"]:
            raise ValueError(
                "Annotation source_text_hash does not match the current reasoning document."
            )
        return document, canonical_document_id, start, end, quote

    def _annotations_overlapping_range(
        self,
        store: dict[str, Any],
        *,
        document_id: str,
        start: int,
        end: int,
    ) -> list[dict[str, Any]]:
        return sorted(
            [
                item
                for item in store.get("annotations") or []
                if text_or_empty(item.get("document_id")).strip() == document_id
                and start < int(item.get("end") or 0)
                and end > int(item.get("start") or 0)
            ],
            key=annotation_sort_key,
        )

    def _annotation_segments_after_clear(
        self,
        annotation: dict[str, Any],
        *,
        start: int,
        end: int,
    ) -> list[tuple[int, int]]:
        annotation_start = int(annotation.get("start") or 0)
        annotation_end = int(annotation.get("end") or 0)
        segments: list[tuple[int, int]] = []
        if start > annotation_start:
            segments.append((annotation_start, min(start, annotation_end)))
        if end < annotation_end:
            segments.append((max(end, annotation_start), annotation_end))
        return [(left, right) for left, right in segments if right > left]

    def _annotation_payload_for_segment(
        self,
        annotation: dict[str, Any],
        *,
        start: int,
        end: int,
        review_actor: str = "",
        preserve_existing_lineage: bool,
    ) -> dict[str, Any]:
        document = self.document_index.get_document(text_or_empty(annotation.get("document_id")).strip())
        if not document:
            raise ValueError(
                f"Unknown document_id: {text_or_empty(annotation.get('document_id')).strip()}"
            )
        provenance = (
            dict(annotation.get("provenance"))
            if isinstance(annotation.get("provenance"), dict)
            else {}
        )
        if not preserve_existing_lineage:
            provenance["labeler_type"] = "human"
        payload = {
            "document_id": text_or_empty(annotation.get("document_id")).strip(),
            "start": start,
            "end": end,
            "quote": text_or_empty(document.get("text"))[start:end],
            "label_id": text_or_empty(annotation.get("label_id")).strip(),
            "comment": text_or_empty(annotation.get("comment")).strip(),
            "author": text_or_empty(annotation.get("author")).strip() or review_actor.strip(),
            "status": text_or_empty(annotation.get("status")).strip() or "confirmed",
            "selection_mode": text_or_empty(
                (annotation.get("provenance") or {}).get("selection_mode")
                or annotation.get("selection_mode")
            ).strip()
            or "freeform",
            "interface": text_or_empty(
                (annotation.get("provenance") or {}).get("interface")
            ).strip()
            or "reasoning-annotation-studio",
            "labeler_type": text_or_empty(provenance.get("labeler_type")).strip() or "human",
            "provenance": provenance,
            "ai_labelled": (
                record_is_ai_labelled(annotation) if preserve_existing_lineage else False
            ),
            "human_reviewed": (
                True
                if record_is_ai_labelled(annotation)
                else record_is_human_reviewed(annotation) or not preserve_existing_lineage
            ),
            "reviewed_by": (
                text_or_empty(annotation.get("reviewed_by")).strip()
                or review_actor.strip()
                or text_or_empty(annotation.get("author")).strip()
            ),
            "reviewed_at_utc": (
                text_or_empty(annotation.get("reviewed_at_utc")).strip()
                or utc_now()
            ),
            "source_text_hash": text_or_empty(annotation.get("source_text_hash")).strip()
            or text_or_empty(document.get("text_hash")).strip(),
        }
        if preserve_existing_lineage:
            payload["review_session_id"] = text_or_empty(annotation.get("review_session_id")).strip()
            payload["review_origin"] = text_or_empty(annotation.get("review_origin")).strip()
            payload["origin_suggestion_id"] = text_or_empty(annotation.get("origin_suggestion_id")).strip()
        else:
            session_id = text_or_empty(annotation.get("review_session_id")).strip()
            payload["review_session_id"] = session_id
            payload["review_origin"] = "human_addition" if session_id else ""
            payload["origin_suggestion_id"] = ""
        return payload

    def delete_annotations_batch(self, annotation_ids: list[str]) -> dict[str, Any]:
        with self._locked_store():
            store = self._read()
            unique_ids = [
                annotation_id
                for annotation_id in dict.fromkeys(
                    text_or_empty(item).strip() for item in annotation_ids or []
                )
                if annotation_id
            ]
            if not unique_ids:
                return {"deleted_annotation_ids": [], "document_annotations": {}}
            current_by_id = {
                text_or_empty(item.get("id")).strip(): item
                for item in store.get("annotations") or []
            }
            missing = [annotation_id for annotation_id in unique_ids if annotation_id not in current_by_id]
            if missing:
                raise KeyError(f"Unknown annotation: {missing[0]}")
            deleted_annotations = [current_by_id[annotation_id] for annotation_id in unique_ids]
            store["annotations"] = [
                item
                for item in store.get("annotations") or []
                if text_or_empty(item.get("id")).strip() not in set(unique_ids)
            ]
            for current in deleted_annotations:
                session_id = text_or_empty(current.get("review_session_id")).strip()
                if not session_id:
                    continue
                self._append_review_event(
                    store,
                    session_id=session_id,
                    record_kind="annotation",
                    before=current,
                    after=None,
                    actor=self._review_actor(current),
                )
            self._write(store)
            document_annotations: dict[str, list[dict[str, Any]]] = {}
            changed_document_ids = {
                text_or_empty(item.get("document_id")).strip() for item in deleted_annotations
            }
            for document_id in changed_document_ids:
                document_annotations[document_id] = [
                    item
                    for item in store.get("annotations") or []
                    if text_or_empty(item.get("document_id")).strip() == document_id
                ]
            return {
                "deleted_annotation_ids": unique_ids,
                "document_annotations": document_annotations,
            }

    def _rewrite_annotation_selection(
        self,
        payload: dict[str, Any],
        *,
        clear_only: bool,
    ) -> dict[str, Any]:
        document, canonical_document_id, start, end, quote = self._normalize_selection_payload(payload)
        store = self._read()
        overlaps = self._annotations_overlapping_range(
            store,
            document_id=canonical_document_id,
            start=start,
            end=end,
        )
        if not overlaps and clear_only:
            return {
                "document_id": canonical_document_id,
                "annotation": None,
                "document_annotations": [
                    item
                    for item in store.get("annotations") or []
                    if text_or_empty(item.get("document_id")).strip() == canonical_document_id
                ],
                "deleted_annotation_ids": [],
            }

        review_actor = (
            text_or_empty(payload.get("reviewed_by")).strip()
            or text_or_empty(payload.get("author")).strip()
        )
        active_session = self._active_review_session_for_document(store, canonical_document_id)
        active_session_id = text_or_empty((active_session or {}).get("id")).strip()
        working_annotations = list(store.get("annotations") or [])
        deleted_annotation_ids: list[str] = []
        created_annotations: list[dict[str, Any]] = []
        created_annotation: dict[str, Any] | None = None

        def sync_working_annotations() -> None:
            store["annotations"] = sorted(working_annotations, key=annotation_sort_key)

        def replace_working(annotation_id: str, replacement: dict[str, Any] | None) -> None:
            nonlocal working_annotations
            next_annotations: list[dict[str, Any]] = []
            replaced = False
            for item in working_annotations:
                if text_or_empty(item.get("id")).strip() != annotation_id:
                    next_annotations.append(item)
                    continue
                replaced = True
                if replacement is not None:
                    next_annotations.append(replacement)
            if not replaced:
                raise KeyError(f"Unknown annotation: {annotation_id}")
            working_annotations = next_annotations

        for overlap in overlaps:
            remaining_segments = self._annotation_segments_after_clear(
                overlap,
                start=start,
                end=end,
            )
            if not remaining_segments:
                replace_working(text_or_empty(overlap.get("id")).strip(), None)
                sync_working_annotations()
                deleted_annotation_ids.append(text_or_empty(overlap.get("id")).strip())
                session_id = text_or_empty(overlap.get("review_session_id")).strip()
                if session_id:
                    self._append_review_event(
                        store,
                        session_id=session_id,
                        record_kind="annotation",
                        before=overlap,
                        after=None,
                        actor=self._review_actor(overlap, review_actor),
                    )
                continue

            first_start, first_end = remaining_segments[0]
            updated_payload = self._annotation_payload_for_segment(
                overlap,
                start=first_start,
                end=first_end,
                review_actor=review_actor,
                preserve_existing_lineage=True,
            )
            updated_annotation = self._normalize_annotation_payload(
                payload=updated_payload,
                existing_annotations=working_annotations,
                replace_id=text_or_empty(overlap.get("id")).strip(),
                created_at_utc=text_or_empty(overlap.get("created_at_utc")).strip() or utc_now(),
                category_lookup={
                    text_or_empty(category.get("id")): category
                    for category in store["categories"]
                },
                valid_category_ids={
                    text_or_empty(category.get("id")) for category in store["categories"]
                },
            )
            if record_is_ai_labelled(overlap):
                updated_annotation = self._promote_human_review(
                    updated_annotation,
                    actor=self._review_actor(overlap, review_actor),
                )
            replace_working(text_or_empty(overlap.get("id")).strip(), updated_annotation)
            sync_working_annotations()
            session_id = text_or_empty(updated_annotation.get("review_session_id")).strip() or text_or_empty(overlap.get("review_session_id")).strip()
            if session_id:
                self._append_review_event(
                    store,
                    session_id=session_id,
                    record_kind="annotation",
                    before=overlap,
                    after=updated_annotation,
                    actor=self._review_actor(updated_annotation, review_actor),
                )

            for extra_start, extra_end in remaining_segments[1:]:
                extra_payload = self._annotation_payload_for_segment(
                    overlap,
                    start=extra_start,
                    end=extra_end,
                    review_actor=review_actor,
                    preserve_existing_lineage=False,
                )
                extra_annotation = self._normalize_annotation_payload(
                    payload=extra_payload,
                    existing_annotations=working_annotations,
                    replace_id=None,
                    category_lookup={
                        text_or_empty(category.get("id")): category
                        for category in store["categories"]
                    },
                    valid_category_ids={
                        text_or_empty(category.get("id")) for category in store["categories"]
                    },
                )
                working_annotations.append(extra_annotation)
                sync_working_annotations()
                created_annotations.append(extra_annotation)
                session_id = text_or_empty(extra_annotation.get("review_session_id")).strip()
                if session_id:
                    self._append_review_event(
                        store,
                        session_id=session_id,
                        record_kind="annotation",
                        before=None,
                        after=extra_annotation,
                        actor=self._review_actor(extra_annotation, review_actor),
                    )

        if not clear_only:
            selection_payload = {
                "document_id": canonical_document_id,
                "start": start,
                "end": end,
                "quote": quote,
                "label_id": text_or_empty(payload.get("label_id")).strip(),
                "comment": text_or_empty(payload.get("comment")).strip(),
                "author": text_or_empty(payload.get("author")).strip() or review_actor,
                "status": text_or_empty(payload.get("status")).strip() or "confirmed",
                "selection_mode": text_or_empty(payload.get("selection_mode")).strip() or "freeform",
                "interface": text_or_empty(payload.get("interface")).strip() or "reasoning-annotation-studio",
                "labeler_type": "human",
                "provenance": {
                    **(
                        dict(payload.get("provenance"))
                        if isinstance(payload.get("provenance"), dict)
                        else {}
                    ),
                    "labeler_type": "human",
                },
                "ai_labelled": False,
                "human_reviewed": True,
                "reviewed_by": review_actor or text_or_empty(payload.get("author")).strip(),
                "reviewed_at_utc": text_or_empty(payload.get("reviewed_at_utc")).strip() or utc_now(),
                "source_text_hash": text_or_empty(payload.get("source_text_hash")).strip()
                or text_or_empty(document.get("text_hash")).strip(),
                "review_session_id": active_session_id,
                "review_origin": "human_addition" if active_session_id else "",
                "origin_suggestion_id": "",
            }
            created_annotation = self._normalize_annotation_payload(
                payload=selection_payload,
                existing_annotations=working_annotations,
                replace_id=None,
                category_lookup={
                    text_or_empty(category.get("id")): category
                    for category in store["categories"]
                },
                valid_category_ids={
                    text_or_empty(category.get("id")) for category in store["categories"]
                },
            )
            working_annotations.append(created_annotation)
            sync_working_annotations()
            created_annotations.append(created_annotation)
            session_id = text_or_empty(created_annotation.get("review_session_id")).strip()
            if session_id:
                self._append_review_event(
                    store,
                    session_id=session_id,
                    record_kind="annotation",
                    before=None,
                    after=created_annotation,
                    actor=self._review_actor(created_annotation, review_actor),
                )

        sync_working_annotations()
        self._write(store)
        return {
            "document_id": canonical_document_id,
            "annotation": created_annotation,
            "created_annotations": created_annotations,
            "deleted_annotation_ids": deleted_annotation_ids,
            "document_annotations": [
                item
                for item in store.get("annotations") or []
                if text_or_empty(item.get("document_id")).strip() == canonical_document_id
            ],
        }

    def apply_annotation_selection(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._locked_store():
            return self._rewrite_annotation_selection(payload, clear_only=False)

    def clear_annotation_selection(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._locked_store():
            return self._rewrite_annotation_selection(payload, clear_only=True)

    def _assert_document_ai_label_writeable(
        self,
        store: dict[str, Any],
        canonical_document_id: str,
    ) -> None:
        existing_annotations = [
            item
            for item in store.get("annotations") or []
            if text_or_empty(item.get("document_id")) == canonical_document_id
        ]
        existing_note = next(
            (
                item
                for item in store.get("document_notes") or []
                if text_or_empty(item.get("document_id")) == canonical_document_id
            ),
            None,
        )
        if any(record_is_human_reviewed(item) for item in existing_annotations):
            raise ValueError(
                "Refusing to AI-label a reasoning trace that already has human-reviewed highlights."
            )
        if any(not record_is_ai_labelled(item) for item in existing_annotations):
            raise ValueError(
                "Refusing to AI-label a reasoning trace that already has non-AI highlights."
            )
        if existing_note and record_is_human_reviewed(existing_note):
            raise ValueError(
                "Refusing to AI-label a reasoning trace that already has a human-reviewed overall label."
            )
        if existing_note and not record_is_ai_labelled(existing_note):
            raise ValueError(
                "Refusing to AI-label a reasoning trace that already has a non-AI overall label."
            )

    def _write_ai_suggestion_session(
        self,
        *,
        store: dict[str, Any],
        canonical_document_id: str,
        session_id: str,
        document_note: dict[str, Any] | None,
        annotations: list[dict[str, Any]],
        suggestion_provenance: dict[str, Any] | None,
        created_by: str,
        resolution_warnings: list[str] | None,
    ) -> dict[str, Any]:
        category_lookup = {
            text_or_empty(category.get("id")): category
            for category in store["categories"]
        }
        valid_category_ids = {
            text_or_empty(category.get("id")) for category in store["categories"]
        }

        normalized_note = None
        run_provenance = {
            "ai_label_run_id": text_or_empty((suggestion_provenance or {}).get("ai_label_run_id")).strip(),
            "ai_label_run_item_id": text_or_empty((suggestion_provenance or {}).get("ai_label_run_item_id")).strip(),
        }
        if document_note is not None:
            note_provenance = (
                json_copy(document_note.get("provenance"))
                if isinstance(document_note.get("provenance"), dict)
                else {}
            )
            note_provenance.update({key: value for key, value in run_provenance.items() if value})
            normalized_note = self._normalize_document_note_payload(
                payload={
                    **dict(document_note),
                    "provenance": note_provenance,
                    "review_session_id": session_id,
                    "review_origin": "ai_suggestion",
                },
                created_at_utc=utc_now(),
                category_lookup=category_lookup,
                valid_category_ids=valid_category_ids,
            )
            if normalized_note is not None:
                store["document_notes"].append(normalized_note)

        normalized_annotations: list[dict[str, Any]] = []
        for annotation in annotations:
            annotation_provenance = (
                json_copy(annotation.get("provenance"))
                if isinstance(annotation.get("provenance"), dict)
                else {}
            )
            annotation_provenance.update({key: value for key, value in run_provenance.items() if value})
            normalized_annotation = self._normalize_annotation_payload(
                payload={
                    **dict(annotation),
                    "provenance": annotation_provenance,
                    "review_session_id": session_id,
                    "review_origin": "ai_suggestion",
                },
                existing_annotations=store["annotations"] + normalized_annotations,
                replace_id=None,
                category_lookup=category_lookup,
                valid_category_ids=valid_category_ids,
            )
            normalized_annotations.append(normalized_annotation)
        store["annotations"].extend(normalized_annotations)
        has_suggestions = bool(normalized_note) or bool(normalized_annotations)
        created_at = utc_now()
        review_session = self._normalize_review_session_payload(
            payload={
                "id": session_id,
                "document_id": canonical_document_id,
                "session_type": REVIEW_SESSION_TYPE_AI_SUGGESTION,
                "status": "open" if has_suggestions else "resolved",
                "created_at_utc": created_at,
                "updated_at_utc": created_at,
                "closed_at_utc": "" if has_suggestions else created_at,
                "created_by": created_by,
                "ai_label_run_id": run_provenance["ai_label_run_id"],
                "ai_label_run_item_id": run_provenance["ai_label_run_item_id"],
                "suggestion_provenance": suggestion_provenance or {},
                "resolution_warnings": resolution_warnings or [],
                "suggested_document_note": normalized_note,
                "suggested_annotations": normalized_annotations,
            },
            category_lookup=category_lookup,
            valid_category_ids=valid_category_ids,
            preserve_id=session_id,
        )
        store.setdefault("review_sessions", []).append(review_session)
        store["review_sessions"] = sorted(store["review_sessions"], key=review_session_sort_key)
        store["document_notes"] = sorted(store["document_notes"], key=document_note_sort_key)
        store["annotations"] = sorted(store["annotations"], key=annotation_sort_key)
        self._write(store)
        return {
            "document_note": normalized_note,
            "annotations": normalized_annotations,
            "review_session": review_session,
        }

    def replace_document_ai_labels(
        self,
        *,
        document_id: str,
        document_note: dict[str, Any] | None,
        annotations: list[dict[str, Any]],
        suggestion_provenance: dict[str, Any] | None = None,
        created_by: str = "",
        resolution_warnings: list[str] | None = None,
    ) -> dict[str, Any]:
        with self._locked_store():
            store = self._read()
            document = self.document_index.get_document(document_id)
            if not document:
                raise ValueError(f"Unknown document_id: {document_id}")
            canonical_document_id = text_or_empty(document.get("document_id")).strip()
            self._assert_document_ai_label_writeable(store, canonical_document_id)
            session_id = str(uuid.uuid4())
            self._supersede_active_review_sessions(
                store,
                document_id=canonical_document_id,
                superseded_by_session_id=session_id,
            )

            store["annotations"] = [
                item
                for item in store.get("annotations") or []
                if text_or_empty(item.get("document_id")) != canonical_document_id
            ]
            store["document_notes"] = [
                item
                for item in store.get("document_notes") or []
                if text_or_empty(item.get("document_id")) != canonical_document_id
            ]
            return self._write_ai_suggestion_session(
                store=store,
                canonical_document_id=canonical_document_id,
                session_id=session_id,
                document_note=document_note,
                annotations=annotations,
                suggestion_provenance=suggestion_provenance,
                created_by=created_by,
                resolution_warnings=resolution_warnings,
            )

    def complete_document_ai_labels(
        self,
        *,
        document_id: str,
        document_note: dict[str, Any] | None,
        annotations: list[dict[str, Any]],
        suggestion_provenance: dict[str, Any] | None = None,
        created_by: str = "",
        resolution_warnings: list[str] | None = None,
    ) -> dict[str, Any]:
        with self._locked_store():
            store = self._read()
            document = self.document_index.get_document(document_id)
            if not document:
                raise ValueError(f"Unknown document_id: {document_id}")
            canonical_document_id = text_or_empty(document.get("document_id")).strip()
            session_id = str(uuid.uuid4())
            self._supersede_active_review_sessions(
                store,
                document_id=canonical_document_id,
                superseded_by_session_id=session_id,
            )

            existing_human_note = next(
                (
                    item
                    for item in store.get("document_notes") or []
                    if text_or_empty(item.get("document_id")).strip() == canonical_document_id
                    and record_is_human_reviewed(item)
                ),
                None,
            )
            if existing_human_note is not None:
                document_note = None

            store["annotations"] = [
                item
                for item in store.get("annotations") or []
                if not (
                    text_or_empty(item.get("document_id")).strip() == canonical_document_id
                    and record_is_ai_labelled(item)
                    and not record_is_human_reviewed(item)
                )
            ]
            store["document_notes"] = [
                item
                for item in store.get("document_notes") or []
                if not (
                    text_or_empty(item.get("document_id")).strip() == canonical_document_id
                    and record_is_ai_labelled(item)
                    and not record_is_human_reviewed(item)
                )
            ]

            return self._write_ai_suggestion_session(
                store=store,
                canonical_document_id=canonical_document_id,
                session_id=session_id,
                document_note=document_note,
                annotations=annotations,
                suggestion_provenance=suggestion_provenance,
                created_by=created_by,
                resolution_warnings=resolution_warnings,
            )

    def clear_document_ai_labels(self, document_id: str, *, feedback: str = "") -> dict[str, Any]:
        with self._locked_store():
            store = self._read()
            document = self.document_index.get_document(document_id)
            if not document:
                raise ValueError(f"Unknown document_id: {document_id}")
            canonical_document_id = text_or_empty(document.get("document_id")).strip()

            existing_annotations = store.get("annotations") or []
            removed_annotations = [
                item
                for item in existing_annotations
                if text_or_empty(item.get("document_id")) == canonical_document_id
                and record_is_ai_labelled(item)
                and not record_is_human_reviewed(item)
            ]
            store["annotations"] = [
                item
                for item in existing_annotations
                if not (
                    text_or_empty(item.get("document_id")) == canonical_document_id
                    and record_is_ai_labelled(item)
                    and not record_is_human_reviewed(item)
                )
            ]

            existing_notes = store.get("document_notes") or []
            removed_note = next(
                (
                    item
                    for item in existing_notes
                    if text_or_empty(item.get("document_id")) == canonical_document_id
                    and record_is_ai_labelled(item)
                    and not record_is_human_reviewed(item)
                ),
                None,
            )
            store["document_notes"] = [
                item
                for item in existing_notes
                if not (
                    text_or_empty(item.get("document_id")) == canonical_document_id
                    and record_is_ai_labelled(item)
                    and not record_is_human_reviewed(item)
                )
            ]
            remaining_annotations = [
                item
                for item in store.get("annotations") or []
                if text_or_empty(item.get("document_id")) == canonical_document_id
            ]
            remaining_note = next(
                (
                    item
                    for item in store.get("document_notes") or []
                    if text_or_empty(item.get("document_id")) == canonical_document_id
                ),
                None,
            )

            store["document_notes"] = sorted(store["document_notes"], key=document_note_sort_key)
            store["annotations"] = sorted(store["annotations"], key=annotation_sort_key)
            for item in removed_annotations:
                session_id = text_or_empty(item.get("review_session_id")).strip()
                if session_id:
                    self._append_review_event(
                        store,
                        session_id=session_id,
                        record_kind="annotation",
                        before=item,
                        after=None,
                        feedback=feedback,
                    )
            if removed_note:
                session_id = text_or_empty(removed_note.get("review_session_id")).strip()
                if session_id:
                    self._append_review_event(
                        store,
                        session_id=session_id,
                        record_kind="document_note",
                        before=removed_note,
                        after=None,
                        feedback=feedback,
                    )
            self._write(store)
            return {
                "document_id": canonical_document_id,
                "removed_note": removed_note,
                "removed_annotations": removed_annotations,
                "removed_annotation_count": len(removed_annotations),
                "removed_note_count": 1 if removed_note else 0,
                "removed_inferred_annotation_count": 0,
                "removed_inferred_note_count": 0,
                "remaining_annotation_count": len(remaining_annotations),
                "remaining_note_count": 1 if remaining_note else 0,
                "remaining_human_annotation_count": sum(
                    1
                    for item in remaining_annotations
                    if record_is_human_reviewed(item)
                ),
                "remaining_ai_annotation_count": sum(
                    1
                    for item in remaining_annotations
                    if record_is_ai_labelled(item)
                ),
                "remaining_note_is_ai": bool(remaining_note and record_is_ai_labelled(remaining_note)),
                "remaining_note_is_human_reviewed": bool(
                    remaining_note and record_is_human_reviewed(remaining_note)
                ),
                "feedback": feedback.strip(),
            }

    def assert_document_ai_label_writeable(self, document_id: str) -> str:
        with self._locked_store():
            store = self._read()
            document = self.document_index.get_document(document_id)
            if not document:
                raise ValueError(f"Unknown document_id: {document_id}")
            canonical_document_id = text_or_empty(document.get("document_id")).strip()
            self._assert_document_ai_label_writeable(store, canonical_document_id)
            return canonical_document_id

    def suggest_ai_label_mode(self, document_id: str) -> str:
        with self._locked_store():
            store = self._read()
            document = self.document_index.get_document(document_id)
            if not document:
                raise ValueError(f"Unknown document_id: {document_id}")
            canonical_document_id = text_or_empty(document.get("document_id")).strip()
            related_annotations = [
                item
                for item in (store.get("annotations") or [])
                if text_or_empty(item.get("document_id")).strip() == canonical_document_id
            ]
            related_note = next(
                (
                    item
                    for item in (store.get("document_notes") or [])
                    if text_or_empty(item.get("document_id")).strip() == canonical_document_id
                ),
                None,
            )
            related_items = [*related_annotations, *([related_note] if related_note else [])]
            has_locked_human_state = any(
                record_is_human_reviewed(item) or not record_is_ai_labelled(item)
                for item in related_items
            )
            return (
                AI_LABEL_MODE_COMPLETE_EXISTING
                if has_locked_human_state
                else AI_LABEL_MODE_REPLACE
            )

    def list_ai_label_runs(self, limit: int = 8) -> list[dict[str, Any]]:
        with self._locked_store():
            store = self._read()
            runs = sorted(
                (store.get("ai_label_runs") or []),
                key=ai_label_run_sort_key,
                reverse=True,
            )
            return json_copy(runs[: max(0, int(limit))])

    def create_ai_label_run(
        self,
        *,
        document_ids: list[str],
        config_id: str,
        created_by: str = "reasoning-annotation-studio",
    ) -> dict[str, Any]:
        with self._locked_store():
            store = self._read()
            requested_document_ids: list[str] = []
            seen_document_ids: set[str] = set()
            for document_id in document_ids:
                metadata = self._document_metadata(document_id)
                canonical_document_id = text_or_empty(metadata.get("document_id")).strip()
                if not canonical_document_id or canonical_document_id in seen_document_ids:
                    continue
                seen_document_ids.add(canonical_document_id)
                requested_document_ids.append(canonical_document_id)
            if not requested_document_ids:
                raise ValueError("At least one reasoning trace is required.")

            active_document_ids = {
                text_or_empty(item.get("document_id")).strip()
                for item in (store.get("ai_label_run_items") or [])
                if text_or_empty(item.get("status")).strip() in {"queued", "running"}
            }
            eligible_document_ids = [
                document_id
                for document_id in requested_document_ids
                if document_id not in active_document_ids
            ]
            skipped_document_ids = [
                document_id
                for document_id in requested_document_ids
                if document_id in active_document_ids
            ]

            run_id = str(uuid.uuid4())
            run_config = next(
                (
                    config
                    for config in list_ai_label_configs()
                    if text_or_empty(config.get("id")).strip() == config_id
                ),
                None,
            )
            if run_config is None:
                raise ValueError(f"Unknown AI label config: {config_id}")
            run = self._normalize_ai_label_run_payload(
                {
                    "id": run_id,
                    "status": "queued" if eligible_document_ids else "completed",
                    "created_at_utc": utc_now(),
                    "updated_at_utc": utc_now(),
                    "created_by": created_by,
                    "config_id": config_id,
                    "config": run_config,
                    "mode": "auto",
                    "document_ids": eligible_document_ids,
                    "requested_document_count": len(requested_document_ids),
                    "skipped_document_ids": skipped_document_ids,
                    "total_count": len(eligible_document_ids),
                    "queued_count": len(eligible_document_ids),
                }
            )
            items: list[dict[str, Any]] = []
            for index, document_id in enumerate(eligible_document_ids, start=1):
                items.append(
                    self._normalize_ai_label_run_item_payload(
                        {
                            "id": str(uuid.uuid4()),
                            "run_id": run_id,
                            "document_id": document_id,
                            "queue_index": index,
                            "status": "queued",
                            "mode_requested": "auto",
                            "config_id": config_id,
                            "attempt_count": 0,
                            "max_attempts": DEFAULT_AI_LABEL_RUN_MAX_ATTEMPTS,
                        }
                    )
                )
            store.setdefault("ai_label_runs", []).append(run)
            store.setdefault("ai_label_run_items", []).extend(items)
            store["ai_label_runs"] = sorted(store["ai_label_runs"], key=ai_label_run_sort_key)
            store["ai_label_run_items"] = sorted(
                store["ai_label_run_items"],
                key=ai_label_run_item_sort_key,
            )
            self._refresh_ai_label_run_status(store, run_id)
            self._write(store)
            return {
                "run": run,
                "items": items,
                "skipped_document_ids": skipped_document_ids,
            }

    def mark_ai_label_run_item_running(
        self,
        *,
        run_id: str,
        item_id: str,
        attempt_count: int,
        label_mode: str,
    ) -> dict[str, Any]:
        with self._locked_store():
            store = self._read()
            run = self._find_ai_label_run(store, run_id)
            item = self._find_ai_label_run_item(store, item_id)
            if not run or not item:
                raise ValueError("AI label run item could not be found.")
            now = utc_now()
            item["status"] = "running"
            item["attempt_count"] = max(int(item.get("attempt_count") or 0), int(attempt_count))
            item["label_mode"] = label_mode
            item["started_at_utc"] = text_or_empty(item.get("started_at_utc")).strip() or now
            item["finished_at_utc"] = ""
            item["error"] = ""
            item["last_error_at_utc"] = ""
            if not text_or_empty(run.get("started_at_utc")).strip():
                run["started_at_utc"] = now
            self._refresh_ai_label_run_status(store, run_id)
            self._write(store)
            return json_copy(item)

    def mark_ai_label_run_item_applied(
        self,
        *,
        run_id: str,
        item_id: str,
        label_mode: str,
        review_session_id: str,
        response_id: str,
        resolution_warnings: list[str] | None = None,
    ) -> dict[str, Any]:
        with self._locked_store():
            store = self._read()
            item = self._find_ai_label_run_item(store, item_id)
            if not item:
                raise ValueError("AI label run item could not be found.")
            item["status"] = "applied"
            item["label_mode"] = label_mode
            item["review_session_id"] = review_session_id
            item["response_id"] = response_id
            item["resolution_warnings"] = [
                text_or_empty(entry).strip()
                for entry in (resolution_warnings or [])
                if text_or_empty(entry).strip()
            ]
            item["finished_at_utc"] = utc_now()
            item["error"] = ""
            item["last_error_at_utc"] = ""
            self._refresh_ai_label_run_status(store, run_id)
            self._write(store)
            return json_copy(item)

    def mark_ai_label_run_item_failed(
        self,
        *,
        run_id: str,
        item_id: str,
        label_mode: str,
        error: str,
    ) -> dict[str, Any]:
        with self._locked_store():
            store = self._read()
            item = self._find_ai_label_run_item(store, item_id)
            if not item:
                raise ValueError("AI label run item could not be found.")
            item["status"] = "failed"
            item["label_mode"] = label_mode
            item["error"] = text_or_empty(error).strip()
            item["last_error_at_utc"] = utc_now()
            item["finished_at_utc"] = text_or_empty(item.get("finished_at_utc")).strip() or utc_now()
            self._refresh_ai_label_run_status(store, run_id)
            self._write(store)
            return json_copy(item)

    def mark_incomplete_ai_label_runs_interrupted(self) -> None:
        with self._locked_store():
            store = self._read()
            changed = False
            interruption_note = "Interrupted because the local annotation server restarted."
            for item in store.get("ai_label_run_items") or []:
                if text_or_empty(item.get("status")).strip() not in {"queued", "running"}:
                    continue
                item["status"] = "failed"
                item["error"] = interruption_note
                item["last_error_at_utc"] = utc_now()
                item["finished_at_utc"] = text_or_empty(item.get("finished_at_utc")).strip() or utc_now()
                changed = True
            if not changed:
                return
            self._refresh_ai_label_runs(store)
            self._write(store)

    def import_store(self, incoming: dict[str, Any], mode: str) -> dict[str, Any]:
        if mode not in {"merge", "replace"}:
            raise ValueError("Import mode must be merge or replace.")
        with self._locked_store():
            if mode == "replace":
                replacement = make_store_skeleton()
                replacement["categories"] = []
                replacement["document_notes"] = []
                replacement["annotations"] = []
                replacement["review_sessions"] = []
                replacement["review_events"] = []
                replacement["ai_label_runs"] = []
                replacement["ai_label_run_items"] = []
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
                existing_note = notes_by_document_id.get(candidate["document_id"])
                if (
                    mode == "merge"
                    and existing_note is not None
                    and canonical_json(existing_note) != canonical_json(candidate)
                ):
                    raise ValueError(
                        f"Import conflict for document note {candidate['document_id']}."
                    )
                notes_by_document_id[candidate["document_id"]] = candidate
            replacement["document_notes"] = sorted(
                notes_by_document_id.values(),
                key=document_note_sort_key,
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
                existing_annotation = existing_by_id.get(candidate["id"])
                if (
                    mode == "merge"
                    and existing_annotation is not None
                    and canonical_json(existing_annotation) != canonical_json(candidate)
                ):
                    raise ValueError(f"Import conflict for annotation {candidate['id']}.")
                existing_by_id[candidate["id"]] = candidate
            replacement["annotations"] = sorted(
                existing_by_id.values(),
                key=annotation_sort_key,
            )
            existing_sessions = {
                text_or_empty(item.get("id")).strip(): item
                for item in (replacement.get("review_sessions") or [])
                if text_or_empty(item.get("id")).strip()
            }
            for session in incoming.get("review_sessions") or []:
                if not isinstance(session, dict):
                    continue
                candidate = self._normalize_review_session_payload(
                    payload=session,
                    category_lookup=category_lookup,
                    valid_category_ids=valid_category_ids,
                    preserve_id=text_or_empty(session.get("id")).strip() or None,
                )
                existing_session = existing_sessions.get(candidate["id"])
                if (
                    mode == "merge"
                    and existing_session is not None
                    and canonical_json(existing_session) != canonical_json(candidate)
                ):
                    raise ValueError(f"Import conflict for review session {candidate['id']}.")
                existing_sessions[candidate["id"]] = candidate
            replacement["review_sessions"] = sorted(
                existing_sessions.values(),
                key=review_session_sort_key,
            )

            existing_events = {
                text_or_empty(item.get("id")).strip(): item
                for item in (replacement.get("review_events") or [])
                if text_or_empty(item.get("id")).strip()
            }
            for event in incoming.get("review_events") or []:
                if not isinstance(event, dict):
                    continue
                candidate = self._normalize_review_event_payload(event)
                if candidate is None:
                    raise ValueError("Import review_event is missing a resolvable document_id.")
                existing_event = existing_events.get(candidate["id"])
                if (
                    mode == "merge"
                    and existing_event is not None
                    and canonical_json(existing_event) != canonical_json(candidate)
                ):
                    raise ValueError(f"Import conflict for review event {candidate['id']}.")
                existing_events[candidate["id"]] = candidate
            replacement["review_events"] = sorted(
                existing_events.values(),
                key=review_event_sort_key,
            )
            existing_runs = {
                text_or_empty(item.get("id")).strip(): item
                for item in (replacement.get("ai_label_runs") or [])
                if text_or_empty(item.get("id")).strip()
            }
            for run in incoming.get("ai_label_runs") or []:
                if not isinstance(run, dict):
                    continue
                candidate = self._normalize_ai_label_run_payload(run)
                existing_run = existing_runs.get(candidate["id"])
                if (
                    mode == "merge"
                    and existing_run is not None
                    and canonical_json(existing_run) != canonical_json(candidate)
                ):
                    raise ValueError(f"Import conflict for ai_label_run {candidate['id']}.")
                existing_runs[candidate["id"]] = candidate
            replacement["ai_label_runs"] = sorted(
                existing_runs.values(),
                key=ai_label_run_sort_key,
            )

            existing_run_items = {
                text_or_empty(item.get("id")).strip(): item
                for item in (replacement.get("ai_label_run_items") or [])
                if text_or_empty(item.get("id")).strip()
            }
            for run_item in incoming.get("ai_label_run_items") or []:
                if not isinstance(run_item, dict):
                    continue
                candidate = self._normalize_ai_label_run_item_payload(run_item)
                existing_run_item = existing_run_items.get(candidate["id"])
                if (
                    mode == "merge"
                    and existing_run_item is not None
                    and canonical_json(existing_run_item) != canonical_json(candidate)
                ):
                    raise ValueError(
                        f"Import conflict for ai_label_run_item {candidate['id']}."
                    )
                existing_run_items[candidate["id"]] = candidate
            replacement["ai_label_run_items"] = sorted(
                existing_run_items.values(),
                key=ai_label_run_item_sort_key,
            )
            self._backfill_review_sessions(replacement)
            self._backfill_review_events(replacement)
            self._refresh_ai_label_runs(replacement)

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
            "review_sessions": [],
            "review_events": [],
            "ai_label_runs": [],
            "ai_label_run_items": [],
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
            if not category_id:
                continue
            if category_id in seen_category_ids:
                raise ValueError(f"Duplicate category id in store: {category_id}")
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
        seen_note_document_ids: set[str] = set()
        for note in store.get("document_notes") or []:
            normalized_note = self._normalize_document_note_payload(
                payload=note,
                created_at_utc=text_or_empty(note.get("created_at_utc")) or utc_now(),
                category_lookup=category_lookup,
                valid_category_ids=valid_category_ids,
            )
            if normalized_note is None:
                continue
            if normalized_note["document_id"] in seen_note_document_ids:
                raise ValueError(
                    f"Duplicate document note in store: {normalized_note['document_id']}"
                )
            seen_note_document_ids.add(normalized_note["document_id"])
            document_notes.append(normalized_note)
        normalized["document_notes"] = sorted(
            document_notes,
            key=document_note_sort_key,
        )

        annotations: list[dict[str, Any]] = []
        seen_annotation_ids: set[str] = set()
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
            annotation_id = text_or_empty(normalized_annotation.get("id")).strip()
            if not annotation_id:
                raise ValueError("Annotation id is required.")
            if annotation_id in seen_annotation_ids:
                raise ValueError(f"Duplicate annotation id in store: {annotation_id}")
            seen_annotation_ids.add(annotation_id)
            annotations.append(normalized_annotation)
        normalized["annotations"] = sorted(annotations, key=annotation_sort_key)
        review_sessions: list[dict[str, Any]] = []
        seen_session_ids: set[str] = set()
        for session in store.get("review_sessions") or []:
            if not isinstance(session, dict):
                continue
            normalized_session = self._normalize_review_session_payload(
                payload=session,
                category_lookup=category_lookup,
                valid_category_ids=valid_category_ids,
                preserve_id=text_or_empty(session.get("id")).strip() or None,
            )
            session_id = text_or_empty(normalized_session.get("id")).strip()
            if not session_id:
                continue
            if session_id in seen_session_ids:
                raise ValueError(f"Duplicate review session id in store: {session_id}")
            seen_session_ids.add(session_id)
            review_sessions.append(normalized_session)
        normalized["review_sessions"] = sorted(review_sessions, key=review_session_sort_key)

        review_events: list[dict[str, Any]] = []
        seen_event_ids: set[str] = set()
        for event in store.get("review_events") or []:
            if not isinstance(event, dict):
                continue
            normalized_event = self._normalize_review_event_payload(event)
            if normalized_event is None:
                raise ValueError("Review event is missing a resolvable document_id.")
            event_id = text_or_empty(normalized_event.get("id")).strip()
            if not event_id:
                continue
            if event_id in seen_event_ids:
                raise ValueError(f"Duplicate review event id in store: {event_id}")
            seen_event_ids.add(event_id)
            review_events.append(normalized_event)
        normalized["review_events"] = sorted(review_events, key=review_event_sort_key)
        ai_label_runs: list[dict[str, Any]] = []
        seen_run_ids: set[str] = set()
        for run in store.get("ai_label_runs") or []:
            if not isinstance(run, dict):
                continue
            normalized_run = self._normalize_ai_label_run_payload(run)
            run_id = text_or_empty(normalized_run.get("id")).strip()
            if not run_id:
                continue
            if run_id in seen_run_ids:
                raise ValueError(f"Duplicate ai_label_run id in store: {run_id}")
            seen_run_ids.add(run_id)
            ai_label_runs.append(normalized_run)
        normalized["ai_label_runs"] = sorted(ai_label_runs, key=ai_label_run_sort_key)

        ai_label_run_items: list[dict[str, Any]] = []
        seen_run_item_ids: set[str] = set()
        for run_item in store.get("ai_label_run_items") or []:
            if not isinstance(run_item, dict):
                continue
            normalized_run_item = self._normalize_ai_label_run_item_payload(run_item)
            run_item_id = text_or_empty(normalized_run_item.get("id")).strip()
            if not run_item_id:
                continue
            if run_item_id in seen_run_item_ids:
                raise ValueError(
                    f"Duplicate ai_label_run_item id in store: {run_item_id}"
                )
            seen_run_item_ids.add(run_item_id)
            ai_label_run_items.append(normalized_run_item)
        normalized["ai_label_run_items"] = sorted(
            ai_label_run_items,
            key=ai_label_run_item_sort_key,
        )
        self._backfill_review_sessions(normalized)
        self._backfill_review_events(normalized)
        self._refresh_ai_label_runs(normalized)
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
    def _normalize_provenance(
        self,
        payload: dict[str, Any],
        *,
        include_selection_mode: bool,
    ) -> dict[str, Any]:
        raw_provenance = (
            payload.get("provenance")
            if isinstance(payload.get("provenance"), dict)
            else {}
        )
        interface = (
            text_or_empty(payload.get("interface")).strip()
            or text_or_empty(raw_provenance.get("interface")).strip()
            or DEFAULT_ANNOTATION_DEFAULTS["interface"]
        )
        labeler_type = (
            text_or_empty(payload.get("labeler_type")).strip()
            or text_or_empty(raw_provenance.get("labeler_type")).strip()
            or DEFAULT_ANNOTATION_DEFAULTS["labeler_type"]
        )
        provenance: dict[str, Any] = {
            "labeler_type": labeler_type,
            "interface": interface,
        }
        if include_selection_mode:
            selection_mode = (
                text_or_empty(payload.get("selection_mode")).strip()
                or text_or_empty(raw_provenance.get("selection_mode")).strip()
                or DEFAULT_ANNOTATION_DEFAULTS["selection_mode"]
            )
            provenance["selection_mode"] = selection_mode

        for key, value in raw_provenance.items():
            normalized_key = text_or_empty(key).strip()
            if not normalized_key or normalized_key in provenance:
                continue
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool, list, dict)):
                provenance[normalized_key] = value
                continue
            normalized_value = text_or_empty(value).strip()
            if normalized_value:
                provenance[normalized_key] = normalized_value
        return provenance

    def _normalize_review_fields(
        self,
        payload: dict[str, Any],
        *,
        provenance: dict[str, Any],
        author: str,
        reviewed_at_fallback: str,
    ) -> dict[str, Any]:
        ai_labelled = normalize_bool(
            payload.get("ai_labelled"),
            default=(
                text_or_empty(provenance.get("labeler_type")).strip() == "model"
                or text_or_empty(provenance.get("source")).strip() == "ai_labelling"
            ),
        )
        if "human_reviewed" in payload:
            human_reviewed = normalize_bool(payload.get("human_reviewed"))
        else:
            human_reviewed = not ai_labelled
        reviewed_by = text_or_empty(payload.get("reviewed_by")).strip()
        reviewed_at_utc = text_or_empty(payload.get("reviewed_at_utc")).strip()
        if human_reviewed:
            if not reviewed_by and not ai_labelled:
                reviewed_by = author
            if not reviewed_at_utc:
                reviewed_at_utc = reviewed_at_fallback
        else:
            reviewed_by = ""
            reviewed_at_utc = ""
        return {
            "ai_labelled": ai_labelled,
            "human_reviewed": human_reviewed,
            "reviewed_by": reviewed_by,
            "reviewed_at_utc": reviewed_at_utc,
        }

    def _normalize_document_note_payload(
        self,
        payload: dict[str, Any],
        created_at_utc: str | None = None,
        category_lookup: dict[str, dict[str, Any]] | None = None,
        valid_category_ids: set[str] | None = None,
    ) -> dict[str, Any] | None:
        document_id = text_or_empty(payload.get("document_id")).strip()
        document = self.document_index.get_document(document_id)
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

        incoming_source_text_hash = text_or_empty(payload.get("source_text_hash")).strip()
        if incoming_source_text_hash and incoming_source_text_hash != document["text_hash"]:
            raise ValueError(
                "Document note source_text_hash does not match the current reasoning document."
            )
        author = text_or_empty(payload.get("author")).strip()
        updated_at_utc = utc_now()
        provenance = self._normalize_provenance(
            payload,
            include_selection_mode=False,
        )
        review_fields = self._normalize_review_fields(
            payload,
            provenance=provenance,
            author=author,
            reviewed_at_fallback=text_or_empty(payload.get("reviewed_at_utc")).strip()
            or text_or_empty(payload.get("updated_at_utc")).strip()
            or text_or_empty(created_at_utc).strip()
            or updated_at_utc,
        )
        lineage_fields = self._normalize_review_lineage_fields(
            payload,
            ai_labelled=review_fields["ai_labelled"],
        )
        if lineage_fields["review_origin"] == "ai_suggestion" and not lineage_fields["origin_suggestion_id"]:
            lineage_fields["origin_suggestion_id"] = self._default_origin_suggestion_id(
                "document_note",
                {
                    "document_id": document["document_id"],
                    "summary": summary,
                    "summary_sha1": hashlib.sha1(summary.encode("utf-8")).hexdigest() if summary else "",
                    "label_id": label_id or "",
                },
                lineage_fields["review_session_id"],
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
            "author": author,
            "created_at_utc": created_at_utc or utc_now(),
            "updated_at_utc": updated_at_utc,
            "source_text_hash": document["text_hash"],
            "provenance": provenance,
            **review_fields,
            **lineage_fields,
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
        document = self.document_index.get_document(document_id)
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
        normalized_quote = " ".join(quote.split())
        author = text_or_empty(payload.get("author")).strip()
        updated_at_utc = utc_now()
        provenance = self._normalize_provenance(
            payload,
            include_selection_mode=True,
        )
        review_fields = self._normalize_review_fields(
            payload,
            provenance=provenance,
            author=author,
            reviewed_at_fallback=text_or_empty(payload.get("reviewed_at_utc")).strip()
            or text_or_empty(payload.get("updated_at_utc")).strip()
            or text_or_empty(created_at_utc).strip()
            or updated_at_utc,
        )
        lineage_fields = self._normalize_review_lineage_fields(
            payload,
            ai_labelled=review_fields["ai_labelled"],
        )
        if lineage_fields["review_origin"] == "ai_suggestion" and not lineage_fields["origin_suggestion_id"]:
            lineage_fields["origin_suggestion_id"] = self._default_origin_suggestion_id(
                "annotation",
                {
                    "document_id": document["document_id"],
                    "start": start,
                    "end": end,
                    "quote": quote,
                    "quote_sha1": hashlib.sha1(quote.encode("utf-8")).hexdigest(),
                    "label_id": label_id,
                },
                lineage_fields["review_session_id"],
            )

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
            "comment": text_or_empty(payload.get("comment")).strip(),
            "author": author,
            "status": text_or_empty(payload.get("status")).strip() or "confirmed",
            "created_at_utc": created_at_utc or utc_now(),
            "updated_at_utc": updated_at_utc,
            "source_text_hash": document["text_hash"],
            "provenance": provenance,
            **review_fields,
            **lineage_fields,
        }


class AiLabelRunManager:
    def __init__(
        self,
        *,
        store: AnnotationStore,
        payload: dict[str, Any],
        max_workers: int = DEFAULT_AI_LABEL_RUN_CONCURRENCY,
    ) -> None:
        self.store = store
        self.payload = payload
        self.max_workers = max(1, int(max_workers))
        self.executor = ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="ai-label-run",
        )

    def start_run(
        self,
        *,
        document_ids: list[str],
        config_id: str,
        created_by: str = "reasoning-annotation-studio",
    ) -> dict[str, Any]:
        created = self.store.create_ai_label_run(
            document_ids=document_ids,
            config_id=config_id,
            created_by=created_by,
        )
        run = created.get("run") or {}
        items = list(created.get("items") or [])
        run_id = text_or_empty(run.get("id")).strip()
        for item in items:
            item_id = text_or_empty(item.get("id")).strip()
            document_id = text_or_empty(item.get("document_id")).strip()
            if not run_id or not item_id or not document_id:
                continue
            self.executor.submit(
                self._process_item,
                run_id,
                item_id,
                document_id,
                text_or_empty(run.get("config_id")).strip(),
            )
        return created

    def run_document_sync(
        self,
        *,
        document_id: str,
        config_id: str,
        created_by: str = "reasoning-annotation-studio",
    ) -> dict[str, Any]:
        created = self.store.create_ai_label_run(
            document_ids=[document_id],
            config_id=config_id,
            created_by=created_by,
        )
        run = created.get("run") or {}
        item = next(iter(created.get("items") or []), None)
        if not item:
            raise ValueError("AI labelling is already queued or running for this reasoning trace.")
        run_id = text_or_empty(run.get("id")).strip()
        item_id = text_or_empty(item.get("id")).strip()
        canonical_document_id = text_or_empty(item.get("document_id")).strip()
        if not run_id or not item_id or not canonical_document_id:
            raise ValueError("AI label run could not be initialized.")
        self._process_item(
            run_id,
            item_id,
            canonical_document_id,
            text_or_empty(run.get("config_id")).strip(),
        )
        snapshot = self.store.snapshot()
        refreshed_run = next(
            (
                candidate
                for candidate in (snapshot.get("ai_label_runs") or [])
                if text_or_empty(candidate.get("id")).strip() == run_id
            ),
            None,
        )
        refreshed_item = next(
            (
                candidate
                for candidate in (snapshot.get("ai_label_run_items") or [])
                if text_or_empty(candidate.get("id")).strip() == item_id
            ),
            None,
        )
        if not refreshed_run or not refreshed_item:
            raise ValueError("AI label run state could not be reloaded.")
        if text_or_empty(refreshed_item.get("status")).strip() != "applied":
            raise ValueError(
                text_or_empty(refreshed_item.get("error")).strip()
                or "AI labelling did not complete successfully."
            )
        review_session_id = text_or_empty(refreshed_item.get("review_session_id")).strip()
        review_session = next(
            (
                session
                for session in (snapshot.get("review_sessions") or [])
                if text_or_empty(session.get("id")).strip() == review_session_id
            ),
            None,
        )
        annotations = [
            record
            for record in (snapshot.get("annotations") or [])
            if text_or_empty(record.get("review_session_id")).strip() == review_session_id
        ]
        document_note = next(
            (
                note
                for note in (snapshot.get("document_notes") or [])
                if text_or_empty(note.get("review_session_id")).strip() == review_session_id
            ),
            None,
        )
        return {
            "run": json_copy(refreshed_run),
            "item": json_copy(refreshed_item),
            "review_session": json_copy(review_session) if review_session else None,
            "annotations": json_copy(annotations),
            "document_note": json_copy(document_note) if document_note else None,
        }

    def _retry_delay_seconds(self, attempt_count: int, error_text: str) -> float:
        normalized_error = text_or_empty(error_text).lower()
        if "429" in normalized_error:
            return min(12.0, 2.0 * attempt_count)
        if "408" in normalized_error or "timed out" in normalized_error:
            return min(8.0, 1.5 * attempt_count)
        if "5" in normalized_error and "http" in normalized_error:
            return min(10.0, 2.0 * attempt_count)
        return min(4.0, float(attempt_count))

    def _should_retry(
        self,
        *,
        attempt_count: int,
        max_attempts: int,
        error_text: str,
    ) -> bool:
        if attempt_count >= max_attempts:
            return False
        normalized_error = text_or_empty(error_text).lower()
        if "resolution error:" in normalized_error:
            return attempt_count < max_attempts
        retryable_markers = (
            "http 408",
            "http 409",
            "http 429",
            "http 500",
            "http 502",
            "http 503",
            "http 504",
            "request failed",
            "timed out",
            "temporarily unavailable",
        )
        return any(marker in normalized_error for marker in retryable_markers)

    def _process_item(
        self,
        run_id: str,
        item_id: str,
        document_id: str,
        config_id: str,
    ) -> None:
        max_attempts = DEFAULT_AI_LABEL_RUN_MAX_ATTEMPTS
        attempt_count = 0
        while attempt_count < max_attempts:
            attempt_count += 1
            label_mode = self.store.suggest_ai_label_mode(document_id)
            self.store.mark_ai_label_run_item_running(
                run_id=run_id,
                item_id=item_id,
                attempt_count=attempt_count,
                label_mode=label_mode,
            )
            try:
                store_snapshot = self.store.snapshot()
                job = run_ai_labeling(
                    document_id=document_id,
                    store=store_snapshot,
                    payload=self.payload,
                    config_id=config_id or None,
                    persist_artifact=False,
                    label_mode=label_mode,
                )
                resolution_error = text_or_empty(job.get("resolution_error")).strip()
                if resolution_error:
                    raise RuntimeError(f"Resolution error: {resolution_error}")
                apply_method = (
                    self.store.complete_document_ai_labels
                    if label_mode == AI_LABEL_MODE_COMPLETE_EXISTING
                    else self.store.replace_document_ai_labels
                )
                applied = apply_method(
                    document_id=document_id,
                    document_note=job.get("resolved_document_note"),
                    annotations=list(job.get("resolved_annotations") or []),
                    suggestion_provenance={
                        "workflow": text_or_empty(job.get("workflow")).strip(),
                        "workflow_version": text_or_empty(job.get("workflow_version")).strip(),
                        "label_mode": text_or_empty(job.get("label_mode")).strip() or label_mode,
                        "config_id": text_or_empty(job.get("config_id")).strip(),
                        "config": job.get("config") or {},
                        "prompt_version": text_or_empty(job.get("prompt_version")).strip(),
                        "response_id": text_or_empty(job.get("response_id")).strip(),
                        "response_created": text_or_empty(job.get("response_created")).strip(),
                        "target_existing_human_labels": job.get("target_existing_human_labels") or {},
                        "ai_label_run_id": run_id,
                        "ai_label_run_item_id": item_id,
                    },
                    created_by=text_or_empty((job.get("config") or {}).get("model")).strip(),
                    resolution_warnings=list(job.get("resolution_warnings") or []),
                )
                self.store.mark_ai_label_run_item_applied(
                    run_id=run_id,
                    item_id=item_id,
                    label_mode=label_mode,
                    review_session_id=text_or_empty(
                        (applied.get("review_session") or {}).get("id")
                    ).strip(),
                    response_id=text_or_empty(job.get("response_id")).strip(),
                    resolution_warnings=list(job.get("resolution_warnings") or []),
                )
                return
            except Exception as exc:  # noqa: BLE001
                error_text = str(exc)
                if self._should_retry(
                    attempt_count=attempt_count,
                    max_attempts=max_attempts,
                    error_text=error_text,
                ):
                    time.sleep(self._retry_delay_seconds(attempt_count, error_text))
                    continue
                self.store.mark_ai_label_run_item_failed(
                    run_id=run_id,
                    item_id=item_id,
                    label_mode=label_mode,
                    error=error_text,
                )
                return


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
                try:
                    store_path = str(self.app.store.path.relative_to(ROOT))
                except ValueError:
                    store_path = str(self.app.store.path)
                self._send_json(
                    {
                        "app": {
                            "name": "BullshitBench Reasoning Lab",
                            "store_path": store_path,
                            "generated_at_utc": self.app.payload.get("generated_at_utc"),
                            "ai_labeling": {
                                "enabled": self.app.ai_labeling_enabled,
                                "default_config_id": self.app.ai_labeling_default_config_id,
                                "configs": self.app.ai_labeling_configs,
                                "max_concurrency": self.app.ai_labeling_max_concurrency,
                            },
                        },
                        "lab": self.app.payload,
                        "store": self.app.store.snapshot(),
                    }
                )
                return
            if parsed.path == "/api/export":
                payload = self.app.store.raw_snapshot()
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
            if parsed.path == "/api/annotations/batch-delete":
                result = self.app.store.delete_annotations_batch(
                    list(payload.get("annotation_ids") or [])
                )
                self._send_json({"ok": True, "result": result})
                return
            if parsed.path == "/api/annotations/apply-selection":
                result = self.app.store.apply_annotation_selection(payload)
                self._send_json(
                    {"ok": True, "result": result},
                    status=HTTPStatus.CREATED,
                )
                return
            if parsed.path == "/api/annotations/clear-selection":
                result = self.app.store.clear_annotation_selection(payload)
                self._send_json({"ok": True, "result": result})
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
            if parsed.path == "/api/ai-label":
                if not self.app.ai_labeling_enabled:
                    raise ValueError("AI labelling is not configured on this server.")
                document_id = text_or_empty(payload.get("document_id")).strip()
                if not document_id:
                    raise ValueError("document_id is required.")
                config_id = (
                    text_or_empty(payload.get("config_id")).strip()
                    or self.app.ai_labeling_default_config_id
                )
                applied = self.app.ai_label_run_manager.run_document_sync(
                    document_id=document_id,
                    config_id=config_id,
                    created_by="reasoning-annotation-studio",
                )
                run = applied.get("run") or {}
                item = applied.get("item") or {}
                self._send_json(
                    {
                        "ok": True,
                        "document_note": applied.get("document_note"),
                        "annotations": applied.get("annotations") or [],
                        "review_session": applied.get("review_session"),
                        "ai_label_run": run,
                        "ai_label_run_item": item,
                        "ai_label_job": {
                            "document_id": document_id,
                            "label_mode": text_or_empty(item.get("label_mode")).strip(),
                            "config_id": text_or_empty(run.get("config_id")).strip(),
                            "config": run.get("config") or {},
                            "workflow": "reasoning_trace_ai_labelling",
                            "workflow_version": AI_LABEL_WORKFLOW_VERSION,
                            "resolution_warnings": item.get("resolution_warnings") or [],
                        },
                    },
                    status=HTTPStatus.CREATED,
                )
                return
            if parsed.path == "/api/ai-label-runs":
                if not self.app.ai_labeling_enabled:
                    raise ValueError("AI labelling is not configured on this server.")
                document_ids = [
                    text_or_empty(item).strip()
                    for item in (payload.get("document_ids") or [])
                    if text_or_empty(item).strip()
                ]
                if not document_ids:
                    raise ValueError("document_ids is required.")
                config_id = (
                    text_or_empty(payload.get("config_id")).strip()
                    or self.app.ai_labeling_default_config_id
                )
                created = self.app.ai_label_run_manager.start_run(
                    document_ids=document_ids,
                    config_id=config_id,
                    created_by="reasoning-annotation-studio",
                )
                self._send_json({"ok": True, **created}, status=HTTPStatus.CREATED)
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
            if parsed.path == "/api/ai-label":
                payload = self._read_json_body()
                query = urllib.parse.parse_qs(parsed.query)
                document_id = text_or_empty(query.get("document_id", [""])[0]).strip()
                if not document_id:
                    raise ValueError("document_id is required.")
                result = self.app.store.clear_document_ai_labels(
                    document_id,
                    feedback=text_or_empty(payload.get("feedback")).strip(),
                )
                self._send_json({"ok": True, "result": result})
                return
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
        self.ai_labeling_enabled = bool(os.getenv("OPENROUTER_API_KEY", "").strip())
        self.ai_labeling_default_config_id = DEFAULT_AI_LABEL_CONFIG_ID
        self.ai_labeling_configs = list_ai_label_configs()
        self.ai_labeling_max_concurrency = DEFAULT_AI_LABEL_RUN_CONCURRENCY
        self.store.mark_incomplete_ai_label_runs_interrupted()
        self.ai_label_run_manager = AiLabelRunManager(
            store=self.store,
            payload=self.payload,
            max_workers=self.ai_labeling_max_concurrency,
        )


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
    document_index = ReasoningDocumentIndex(payload)
    store = AnnotationStore(Path(args.annotation_store).resolve(), document_index)
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
