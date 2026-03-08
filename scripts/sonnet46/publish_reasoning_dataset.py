#!/usr/bin/env python3
"""Publish isolated run artifacts into a stable latest dataset directory."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


COPY_MAP = (
    ("responses.jsonl", "responses.jsonl"),
    ("collection_meta.json", "collection_meta.json"),
    ("collection_stats.json", "collection_stats.json"),
    ("questions_snapshot.json", "questions_snapshot.json"),
)

GRADE_COPY_MAP = (
    ("grade_meta.json", "grade_meta.json"),
    ("grades.jsonl", "grades.jsonl"),
    ("summary.json", "grade_summary.json"),
    ("summary.md", "grade_summary.md"),
    ("review.csv", "grade_review.csv"),
    ("review.md", "grade_review.md"),
)

AGGREGATE_COPY_MAP = (
    ("aggregate_meta.json", "aggregate_meta.json"),
    ("aggregate.jsonl", "aggregate.jsonl"),
    ("aggregate_summary.json", "aggregate_summary.json"),
)


def load_jsonl_count(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def copy_many(source_root: Path, dest_root: Path, mapping: tuple[tuple[str, str], ...]) -> None:
    for source_name, dest_name in mapping:
        source_path = source_root / source_name
        if not source_path.exists():
            continue
        shutil.copy2(source_path, dest_root / dest_name)


def rel_to(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--special-root", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--grade-dir", required=True)
    parser.add_argument("--aggregate-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-key", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    special_root = Path(args.special_root).resolve()
    run_dir = Path(args.run_dir).resolve()
    grade_dir = Path(args.grade_dir).resolve()
    aggregate_dir = Path(args.aggregate_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    copy_many(run_dir, output_dir, COPY_MAP)
    copy_many(grade_dir, output_dir, GRADE_COPY_MAP)
    copy_many(aggregate_dir, output_dir, AGGREGATE_COPY_MAP)

    manifest: dict[str, Any] = {
        "generated_at_utc": json.loads((aggregate_dir / "aggregate_meta.json").read_text(encoding="utf-8")).get("timestamp_utc"),
        "dataset_key": args.dataset_key,
        "special_run": True,
        "sources": {
            "run_dir": rel_to(special_root, run_dir),
            "grade_dir": rel_to(special_root, grade_dir),
            "aggregate_dir": rel_to(special_root, aggregate_dir),
            "responses_file": rel_to(special_root, output_dir / "responses.jsonl"),
            "grades_file": rel_to(special_root, output_dir / "grades.jsonl"),
            "aggregate_rows_file": rel_to(special_root, output_dir / "aggregate.jsonl"),
        },
        "counts": {
            "responses_rows": load_jsonl_count(output_dir / "responses.jsonl"),
            "grades_rows": load_jsonl_count(output_dir / "grades.jsonl"),
            "aggregate_rows": load_jsonl_count(output_dir / "aggregate.jsonl"),
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
