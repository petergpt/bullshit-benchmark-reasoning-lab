#!/usr/bin/env python3
"""Backfill missing Sonnet 4.6 reasoning traces in the isolated viewer dataset."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

import build_single_judge_aggregate as singleagg
import openrouter_benchmark as bench


ROOT = Path(__file__).resolve().parents[2]
EMPTY_PLACEHOLDER = "[Model returned an empty response.]"


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object JSON in {path}")
    return payload


def normalize_prompt_for_recovery(prompt: str) -> str:
    normalized = str(prompt)
    replacements = {
        "What's": "What is",
        "degree-days": "degree days",
        "re-sterilize": "re sterilize",
    }
    for source, target in replacements.items():
        normalized = normalized.replace(source, target)
    return normalized


def trace_payload(message: Any) -> tuple[str, list[dict[str, Any]]]:
    if not isinstance(message, dict):
        return "", []
    reasoning = str(message.get("reasoning") or "").strip()
    details = message.get("reasoning_details")
    if not isinstance(details, list):
        details = []
    return reasoning, [item for item in details if isinstance(item, dict)]


def row_has_trace(row: dict[str, Any]) -> bool:
    raw = row.get("response_raw")
    if not isinstance(raw, dict):
        return False
    choices = raw.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return False
    message = choices[0].get("message")
    reasoning, details = trace_payload(message)
    if reasoning:
        return True
    return any(str(item.get("text") or item.get("summary") or "").strip() for item in details)


def row_needs_backfill(row: dict[str, Any]) -> bool:
    if str(row.get("model", "")).strip() != "anthropic/claude-sonnet-4.6@reasoning=high":
        return False
    if row_has_trace(row):
        return False
    if str(row.get("error", "")).strip():
        return True
    if str(row.get("response_text", "")).strip() == EMPTY_PLACEHOLDER:
        return True
    return True


def build_task(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": row["sample_id"],
        "run_index": row["run_index"],
        "model": row["model"],
        "model_id": row["model_id"],
        "request_model_id": row.get("request_model_id", row["model_id"]),
        "model_org": row["model_org"],
        "model_name": row["model_name"],
        "model_provider": row.get("model_provider", "openrouter"),
        "model_reasoning_level": row["model_reasoning_level"],
        "model_row": row["model_row"],
        "response_reasoning_effort": row.get("response_reasoning_effort"),
        "question": {
            "id": row["question_id"],
            "question": row["question"],
            "nonsensical_element": row["nonsensical_element"],
            "domain": row["domain"],
            "technique": row["technique"],
            "is_control": bool(row.get("is_control", False)),
        },
    }


def recollect_row(base_row: dict[str, Any], clients: dict[str, Any]) -> tuple[dict[str, Any], str]:
    strategies = [
        ("enabled", {"reasoning": {"enabled": True}}, None),
        ("max_tokens_512", {"reasoning": {"max_tokens": 512}}, None),
        ("max_tokens_1024", {"reasoning": {"max_tokens": 1024}}, None),
        (
            "normalized_prompt_anthropic_max_tokens_512",
            {
                "reasoning": {"max_tokens": 512},
                "provider": {
                    "only": ["anthropic"],
                    "allow_fallbacks": False,
                    "require_parameters": True,
                },
            },
            normalize_prompt_for_recovery,
        ),
    ]
    task = build_task(base_row)
    last_row: dict[str, Any] | None = None
    last_label = ""
    for label, override, prompt_transform in strategies:
        task_with_override = copy.deepcopy(task)
        task_with_override["request_overrides"] = override
        original_prompt = str(task_with_override["question"]["question"])
        prompt_override = ""
        if prompt_transform is not None:
            prompt_override = prompt_transform(original_prompt)
            task_with_override["question"]["question"] = prompt_override
        recollected = bench.collect_one(
            task_with_override,
            clients=clients,
            system_prompt="",
            omit_system_prompt=True,
            temperature=None,
            max_tokens=0,
            empty_response_retries=2,
            retries=3,
            pause_seconds=0.0,
            dry_run=False,
            store_request_messages=False,
            store_response_raw=True,
        )
        recollected.setdefault("warnings", [])
        if isinstance(recollected["warnings"], list):
            recollected["warnings"].append(f"trace_backfill_strategy={label}")
            if prompt_override:
                recollected["warnings"].append("trace_backfill_prompt_normalized")
        if prompt_override:
            recollected["trace_backfill_prompt_variant"] = label
            recollected["trace_backfill_prompt_used"] = prompt_override
            recollected["question"] = original_prompt
        last_row = recollected
        last_label = label
        if recollected.get("error"):
            continue
        if str(recollected.get("response_text", "")).strip() == EMPTY_PLACEHOLDER:
            continue
        if row_has_trace(recollected):
            return recollected, label
    if last_row is None:
        raise RuntimeError("Backfill recollection did not run.")
    raise RuntimeError(
        f"Unable to recover reasoning trace for sample_id={base_row['sample_id']} "
        f"after strategies ending with {last_label}."
    )


def make_clients() -> dict[str, Any]:
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required.")
    return {
        "openrouter": bench.OpenRouterClient(
            api_key=api_key,
            timeout_seconds=120,
        )
    }


def grade_row(
    response_row: dict[str, Any],
    *,
    grade_meta: dict[str, Any],
    clients: dict[str, Any],
) -> dict[str, Any]:
    judge_no_hint = bool(grade_meta.get("judge_no_hint"))
    judge_template = (
        bench.DEFAULT_JUDGE_USER_TEMPLATE_NO_HINT
        if judge_no_hint
        else bench.DEFAULT_JUDGE_USER_TEMPLATE
    )
    judge_template_control = "" if judge_no_hint else bench.DEFAULT_JUDGE_USER_TEMPLATE_CONTROL_HINT

    graded = bench.grade_one(
        response_row,
        clients=clients,
        judge_model=str(grade_meta["judge_model"]),
        judge_provider=str(grade_meta.get("judge_provider", "openrouter")),
        judge_system_prompt=str(grade_meta["judge_system_prompt"]),
        judge_user_template=judge_template,
        judge_user_template_control=judge_template_control,
        judge_no_hint=judge_no_hint,
        judge_temperature=grade_meta.get("judge_temperature"),
        judge_reasoning_effort=str(grade_meta.get("judge_reasoning_effort", "off")),
        judge_max_tokens=int(grade_meta.get("judge_max_tokens", 0) or 0),
        judge_output_retries=int(grade_meta.get("judge_output_retries", 2) or 2),
        store_judge_response_raw=bool(grade_meta.get("store_judge_response_raw", True)),
        retries=int(grade_meta.get("retries", 3) or 3),
        pause_seconds=0.0,
        dry_run=False,
    )
    graded.setdefault("judge_warnings", [])
    if isinstance(graded["judge_warnings"], list):
        graded["judge_warnings"].append("trace_backfill_regenerated_grade")
    return graded


def rewrite_collection_outputs(run_dir: Path, rows: list[dict[str, Any]]) -> None:
    bench.write_jsonl(run_dir / "responses.jsonl", rows)
    bench.write_jsonl(run_dir / "responses.partial.jsonl", rows)
    stats_path = run_dir / "collection_stats.json"
    stats = load_json(stats_path)
    stats["generated_at_utc"] = utc_now()
    stats["total_records"] = len(rows)
    stats["error_count"] = sum(1 for row in rows if row.get("error"))
    stats["success_count"] = sum(1 for row in rows if not row.get("error"))
    stats["usage_summary"] = bench.summarize_collect_usage(rows)
    bench.write_json(stats_path, stats)
    bench.write_collect_review_csv(run_dir / "responses_review.csv", rows)


def rewrite_grade_outputs(grade_dir: Path, grade_meta: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    bench.write_jsonl(grade_dir / "grades.jsonl", rows)
    bench.write_jsonl(grade_dir / "grades.partial.jsonl", rows)
    summary = bench.summarize_grades(rows)
    prior_summary_path = grade_dir / "summary.json"
    if prior_summary_path.exists():
        prior_summary = load_json(prior_summary_path)
        summary["elapsed_seconds"] = prior_summary.get("elapsed_seconds")
        summary["resumed"] = prior_summary.get("resumed", True)
        summary["checkpoint_rows_at_start"] = prior_summary.get("checkpoint_rows_at_start", 0)
        summary["new_rows_processed"] = prior_summary.get("new_rows_processed", len(rows))
    bench.write_json(prior_summary_path, summary)
    (grade_dir / "summary.md").write_text(
        bench.render_markdown_summary(grade_meta, summary),
        encoding="utf-8",
    )
    bench.write_grade_review_csv(grade_dir / "review.csv", rows)
    (grade_dir / "review.md").write_text(
        bench.render_grade_review_markdown(rows),
        encoding="utf-8",
    )


def rewrite_aggregate_outputs(
    aggregate_dir: Path,
    *,
    responses: list[dict[str, Any]],
    grades: list[dict[str, Any]],
    grade_meta: dict[str, Any],
) -> None:
    aggregate_meta_path = aggregate_dir / "aggregate_meta.json"
    aggregate_meta = load_json(aggregate_meta_path)
    rows = singleagg.build_rows(
        responses,
        grades,
        judge_model=str(grade_meta["judge_model"]),
        grade_id=str(grade_meta["grade_id"]),
    )
    summary = singleagg.summarize(rows)
    aggregate_meta["timestamp_utc"] = utc_now()
    aggregate_meta["backfilled"] = True
    bench.write_json(aggregate_meta_path, aggregate_meta)
    singleagg.write_jsonl(aggregate_dir / "aggregate.jsonl", rows)
    singleagg.write_json(aggregate_dir / "aggregate_summary.json", summary)
    singleagg.write_json(
        aggregate_dir / "aggregate_events.jsonl.meta.json",
        {
            "note": "Single-judge isolated aggregate; no event stream generated.",
            "backfilled_at_utc": utc_now(),
        },
    )


def publish_and_rebuild(dataset_key: str, manifest: dict[str, Any]) -> None:
    run_dir = ROOT / str(manifest["sources"]["run_dir"])
    grade_dir = ROOT / str(manifest["sources"]["grade_dir"])
    aggregate_dir = ROOT / str(manifest["sources"]["aggregate_dir"])

    publish_cmd = [
        sys.executable,
        "scripts/sonnet46/publish_reasoning_dataset.py",
        "--special-root",
        str(ROOT),
        "--run-dir",
        str(run_dir),
        "--grade-dir",
        str(grade_dir),
        "--aggregate-dir",
        str(aggregate_dir),
        "--output-dir",
        str(ROOT / "data" / "sonnet46" / "viewer-input" / dataset_key),
        "--dataset-key",
        dataset_key,
    ]
    subprocess.run(publish_cmd, cwd=ROOT, check=True)
    subprocess.run(
        [sys.executable, "scripts/sonnet46/build_reasoning_bundle.py"],
        cwd=ROOT,
        check=True,
    )


def main() -> None:
    dataset_key = "v2"
    viewer_manifest_path = (
        ROOT / "data" / "sonnet46" / "viewer-input" / dataset_key / "manifest.json"
    )
    viewer_manifest = load_json(viewer_manifest_path)

    run_dir = ROOT / str(viewer_manifest["sources"]["run_dir"])
    grade_dir = ROOT / str(viewer_manifest["sources"]["grade_dir"])
    aggregate_dir = ROOT / str(viewer_manifest["sources"]["aggregate_dir"])

    responses_path = run_dir / "responses.jsonl"
    grades_path = grade_dir / "grades.jsonl"
    responses = bench.read_jsonl(responses_path)
    grades = bench.read_jsonl(grades_path)
    grade_meta = load_json(grade_dir / "grade_meta.json")

    candidates = [row for row in responses if row_needs_backfill(row)]
    if not candidates:
        print("No missing traces found.")
        return

    backup_dir = run_dir / "backfills"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / f"backfill_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.json"
    backup_payload = {
        "created_at_utc": utc_now(),
        "sample_ids": [row["sample_id"] for row in candidates],
        "responses_before": candidates,
        "grades_before": [
            row for row in grades if str(row.get("sample_id", "")) in {c["sample_id"] for c in candidates}
        ],
    }
    backup_path.write_text(json.dumps(backup_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    clients = make_clients()
    response_index = {str(row.get("sample_id", "")): idx for idx, row in enumerate(responses)}
    grade_index = {str(row.get("sample_id", "")): idx for idx, row in enumerate(grades)}

    for candidate in candidates:
        refreshed_row, strategy = recollect_row(candidate, clients)
        refreshed_row.setdefault("warnings", [])
        if isinstance(refreshed_row["warnings"], list):
            refreshed_row["warnings"].append("trace_backfill_completed")
        responses[response_index[candidate["sample_id"]]] = refreshed_row
        grades[grade_index[candidate["sample_id"]]] = grade_row(
            refreshed_row,
            grade_meta=grade_meta,
            clients=clients,
        )
        print(
            f"Backfilled {candidate['question_id']} via {strategy}: "
            f"trace={'yes' if row_has_trace(refreshed_row) else 'no'}",
            flush=True,
        )

    rewrite_collection_outputs(run_dir, responses)
    rewrite_grade_outputs(grade_dir, grade_meta, grades)
    rewrite_aggregate_outputs(
        aggregate_dir,
        responses=responses,
        grades=grades,
        grade_meta=grade_meta,
    )
    publish_and_rebuild(dataset_key, viewer_manifest)
    print("Backfill complete.", flush=True)


if __name__ == "__main__":
    main()
