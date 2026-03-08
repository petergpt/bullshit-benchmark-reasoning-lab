#!/usr/bin/env python3
"""Vendor the GPT-5.4 source rows needed by the reasoning lab."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from reasoning_lab_paths import DEFAULT_BENCHMARK_ROOT, LAB_ROOT


MODEL_FILTER = {
    "openai/gpt-5.4@reasoning=none",
    "openai/gpt-5.4@reasoning=xhigh",
}

DATASET_SPECS = {
    "v1": {
        "source_dir": Path("data") / "latest",
        "dest_dir": Path("latest"),
        "expected_question_count": 55,
    },
    "v2": {
        "source_dir": Path("data") / "v2" / "latest",
        "dest_dir": Path("v2") / "latest",
        "expected_question_count": 100,
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
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected object JSONL row in {path}")
            rows.append(payload)
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def filter_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if str(row.get("model", "")).strip() in MODEL_FILTER]


def validate_rows(dataset_key: str, label: str, rows: list[dict[str, Any]], expected_question_count: int) -> None:
    question_ids = {
        str(row.get("question_id", "")).strip()
        for row in rows
        if str(row.get("question_id", "")).strip()
    }
    if len(question_ids) != expected_question_count:
        raise ValueError(
            f"Expected {expected_question_count} unique questions in {dataset_key} {label}, "
            f"found {len(question_ids)}."
        )


def build_manifest(
    *,
    dataset_key: str,
    dest_dir: Path,
    source_manifest: dict[str, Any],
    responses_rows: list[dict[str, Any]],
    aggregate_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "generated_at_utc": source_manifest.get("generated_at_utc"),
        "snapshot_type": "reasoning-lab-gpt54-filtered-source",
        "dataset_key": dataset_key,
        "models": sorted(MODEL_FILTER),
        "sources": {
            "responses_file": str(dest_dir / "responses.jsonl"),
            "aggregate_rows_file": str(dest_dir / "aggregate.jsonl"),
        },
        "counts": {
            "responses_rows": len(responses_rows),
            "aggregate_rows": len(aggregate_rows),
        },
    }


def sync_dataset(dataset_key: str, source_root: Path, dest_root: Path) -> None:
    spec = DATASET_SPECS[dataset_key]
    source_dir = source_root / spec["source_dir"]
    dest_dir = dest_root / spec["dest_dir"]
    responses_rows = filter_rows(load_jsonl(source_dir / "responses.jsonl"))
    aggregate_rows = filter_rows(load_jsonl(source_dir / "aggregate.jsonl"))
    source_manifest = load_json(source_dir / "manifest.json")
    validate_rows(
        dataset_key,
        "responses",
        responses_rows,
        int(spec["expected_question_count"]),
    )
    validate_rows(
        dataset_key,
        "aggregate",
        aggregate_rows,
        int(spec["expected_question_count"]),
    )

    write_jsonl(dest_dir / "responses.jsonl", responses_rows)
    write_jsonl(dest_dir / "aggregate.jsonl", aggregate_rows)
    write_json(
        dest_dir / "manifest.json",
        build_manifest(
            dataset_key=dataset_key,
            dest_dir=spec["dest_dir"],
            source_manifest=source_manifest,
            responses_rows=responses_rows,
            aggregate_rows=aggregate_rows,
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        default=str(LAB_ROOT.parent),
        help="BullshitBench repo root containing data/latest and data/v2/latest.",
    )
    parser.add_argument(
        "--dest-root",
        default=str(DEFAULT_BENCHMARK_ROOT),
        help="Destination root for the bundled GPT-5.4 snapshot tree.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root).expanduser().resolve()
    dest_root = Path(args.dest_root).expanduser().resolve()
    for dataset_key in ("v1", "v2"):
        sync_dataset(dataset_key, source_root, dest_root)
    print(f"Wrote GPT-5.4 source snapshot under {dest_root.relative_to(LAB_ROOT)}")


if __name__ == "__main__":
    main()
