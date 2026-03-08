#!/usr/bin/env python3
"""Build a single-judge aggregate.jsonl for isolated reasoning runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def utc_timestamp() -> str:
    from datetime import datetime, UTC

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def is_valid_score(value: Any) -> bool:
    return isinstance(value, int) and value in {0, 1, 2, 3}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_model: dict[str, dict[str, Any]] = {}
    for row in rows:
        model = str(row.get("model", ""))
        stats = by_model.setdefault(
            model,
            {
                "model": model,
                "count": 0,
                "scored_count": 0,
                "nonsense_count": 0,
                "control_count": 0,
                "score_0": 0,
                "score_1": 0,
                "score_2": 0,
                "score_3": 0,
                "avg_score": None,
                "detection_rate_score_2": None,
                "full_engagement_rate_score_0": None,
                "control_correct_rate_score_3": None,
                "error_count": 0,
                "_nonsense_scores": [],
            },
        )
        stats["count"] += 1
        if row.get("status") == "error":
            stats["error_count"] += 1

        score = row.get("consensus_score")
        if not is_valid_score(score):
            continue

        stats["scored_count"] += 1
        if row.get("is_control"):
            stats["control_count"] += 1
            stats["score_3"] += 1 if score == 3 else 0
            continue

        stats["nonsense_count"] += 1
        stats["_nonsense_scores"].append(float(score))
        if score == 0:
            stats["score_0"] += 1
        elif score == 1:
            stats["score_1"] += 1
        elif score == 2:
            stats["score_2"] += 1
        elif score == 3:
            stats["score_3"] += 1

    leaderboard: list[dict[str, Any]] = []
    for stats in by_model.values():
        nonsense_scores = stats.pop("_nonsense_scores")
        if nonsense_scores:
            count = len(nonsense_scores)
            stats["avg_score"] = round(sum(nonsense_scores) / count, 4)
            stats["detection_rate_score_2"] = round(stats["score_2"] / count, 4)
            stats["full_engagement_rate_score_0"] = round(stats["score_0"] / count, 4)
        if stats["control_count"] > 0:
            stats["control_correct_rate_score_3"] = round(
                stats["score_3"] / stats["control_count"], 4
            )
        leaderboard.append(stats)

    leaderboard.sort(
        key=lambda item: (
            item["avg_score"] if isinstance(item["avg_score"], (int, float)) else -1,
            item["detection_rate_score_2"]
            if isinstance(item["detection_rate_score_2"], (int, float))
            else -1,
        ),
        reverse=True,
    )

    return {
        "consensus_method": "single_judge",
        "num_judges": 1,
        "leaderboard": leaderboard,
        "reliability": {
            "pairwise": [],
            "average_pairwise_agreement": None,
            "krippendorff_alpha_ordinal": None,
        },
        "total_records": len(rows),
        "total_error_records": sum(1 for row in rows if row.get("status") == "error"),
        "total_scored_records": sum(1 for row in rows if is_valid_score(row.get("consensus_score"))),
    }


def build_rows(
    responses: list[dict[str, Any]],
    grades: list[dict[str, Any]],
    *,
    judge_model: str,
    grade_id: str,
) -> list[dict[str, Any]]:
    grades_by_sample = {str(row.get("sample_id", "")).strip(): row for row in grades}
    rows: list[dict[str, Any]] = []

    for response in responses:
        sample_id = str(response.get("sample_id", "")).strip()
        if not sample_id:
            continue
        grade = grades_by_sample.get(sample_id)
        row_errors: list[str] = []
        row_identity_mismatch = False

        if grade is None:
            row_errors.append("Missing grade row for sample_id.")
        else:
            for field in (
                "model",
                "model_id",
                "model_org",
                "model_name",
                "model_reasoning_level",
                "model_row",
                "run_index",
                "question_id",
                "response_text",
            ):
                if grade.get(field) != response.get(field):
                    row_identity_mismatch = True
                    row_errors.append(
                        f"Identity mismatch for field={field!r}: "
                        f"response={response.get(field)!r} grade={grade.get(field)!r}"
                    )

        judge_score = grade.get("judge_score") if isinstance(grade, dict) else None
        judge_error = str(grade.get("error", "")).strip() if isinstance(grade, dict) else "Missing grade row."
        if str(response.get("error", "")).strip():
            row_errors.append(str(response.get("error", "")).strip())
        if judge_error:
            row_errors.append(f"{judge_model}: {judge_error}")

        consensus_score = judge_score if is_valid_score(judge_score) and not judge_error else None
        consensus_error = None
        if judge_error:
            consensus_error = "Single judge row has error; consensus unavailable."
        elif not is_valid_score(judge_score):
            consensus_error = "Single judge row missing valid score."
        if consensus_error:
            row_errors.append(consensus_error)

        row = {
            "sample_id": sample_id,
            "model": response.get("model"),
            "model_id": response.get("model_id", response.get("model")),
            "model_org": response.get("model_org", "unknown"),
            "model_name": response.get(
                "model_name", response.get("model_id", response.get("model"))
            ),
            "model_reasoning_level": response.get("model_reasoning_level", "default"),
            "model_row": response.get("model_row", response.get("model")),
            "response_reasoning_effort": response.get("response_reasoning_effort"),
            "run_index": response.get("run_index"),
            "question_id": response.get("question_id"),
            "technique": response.get("technique"),
            "is_control": bool(response.get("is_control", False)),
            "domain": response.get("domain"),
            "question": response.get("question"),
            "nonsensical_element": response.get("nonsensical_element"),
            "response_text": response.get("response_text", ""),
            "row_identity_mismatch": row_identity_mismatch,
            "row_errors": row_errors,
            "judge_1_model": judge_model,
            "judge_1_grade_id": grade_id,
            "judge_1_score": judge_score,
            "judge_1_justification": grade.get("judge_justification", "") if isinstance(grade, dict) else "",
            "judge_1_error": judge_error,
            "judge_1_status": "error" if judge_error else "ok",
            "consensus_score": consensus_score,
            "consensus_method": "single_judge",
            "consensus_error": consensus_error,
            "judge_valid_scores": [judge_score] if is_valid_score(judge_score) and not judge_error else [],
            "status": "error" if row_errors else "ok",
            "error": " | ".join(row_errors),
        }
        rows.append(row)

    rows.sort(
        key=lambda row: (
            str(row.get("model", "")),
            int(row.get("run_index", 0) or 0),
            str(row.get("question_id", "")),
        )
    )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses-file", required=True)
    parser.add_argument("--grade-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--aggregate-id", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    responses_file = Path(args.responses_file).resolve()
    grade_dir = Path(args.grade_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    grade_meta = load_json(grade_dir / "grade_meta.json")
    responses_source = Path(str(grade_meta.get("responses_file", ""))).resolve()
    if responses_source != responses_file:
        raise ValueError(
            "grade_meta.responses_file does not match --responses-file: "
            f"{responses_source} != {responses_file}"
        )

    grades = load_jsonl(grade_dir / "grades.jsonl")
    responses = load_jsonl(responses_file)

    aggregate_id = args.aggregate_id.strip() or f"{grade_meta.get('grade_id', grade_dir.name)}__aggregate"
    aggregate_dir = output_dir / "aggregates" / aggregate_id
    aggregate_dir.mkdir(parents=True, exist_ok=False)

    judge_model = str(grade_meta.get("judge_model", "")).strip()
    grade_id = str(grade_meta.get("grade_id", grade_dir.name)).strip()
    rows = build_rows(responses, grades, judge_model=judge_model, grade_id=grade_id)
    summary = summarize(rows)

    aggregate_meta = {
      "phase": "aggregate",
      "aggregate_id": aggregate_id,
      "timestamp_utc": utc_timestamp(),
      "grade_dirs": [str(grade_dir)],
      "consensus_method": "single_judge",
      "num_judges": 1,
      "judge_models": [judge_model],
      "responses_file": str(responses_file),
      "special_single_judge": True,
    }

    write_json(aggregate_dir / "aggregate_meta.json", aggregate_meta)
    write_jsonl(aggregate_dir / "aggregate.jsonl", rows)
    write_json(aggregate_dir / "aggregate_summary.json", summary)
    write_json(
        aggregate_dir / "aggregate_events.jsonl.meta.json",
        {"note": "Single-judge isolated aggregate; no event stream generated."},
    )
    print(str(aggregate_dir))


if __name__ == "__main__":
    main()
