#!/usr/bin/env python3
"""Shared filesystem path helpers for the standalone reasoning lab."""

from __future__ import annotations

import os
from pathlib import Path


LAB_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK_ROOT = LAB_ROOT / "source-data" / "benchmark-snapshot"


def _expand_candidate(path_text: str) -> Path:
    return Path(path_text).expanduser().resolve()


def iter_benchmark_root_candidates() -> list[Path]:
    candidates: list[Path] = []
    for env_name in ("REASONING_LAB_BENCHMARK_ROOT", "BULLSHIT_BENCHMARK_ROOT"):
        value = os.getenv(env_name, "").strip()
        if value:
            candidates.append(_expand_candidate(value))
    candidates.append(DEFAULT_BENCHMARK_ROOT)
    candidates.append(LAB_ROOT.parent)

    unique_candidates: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        unique_candidates.append(candidate)
    return unique_candidates


def is_benchmark_root(path: Path) -> bool:
    return (path / "data" / "latest" / "responses.jsonl").exists() and (
        path / "data" / "v2" / "latest" / "responses.jsonl"
    ).exists()


def resolve_benchmark_root() -> Path:
    candidates = iter_benchmark_root_candidates()
    for candidate in candidates:
        if is_benchmark_root(candidate):
            return candidate
    checked = "\n".join(f"- {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        "Could not locate BullshitBench source data for the reasoning lab.\n"
        "Expected a root containing both data/latest/responses.jsonl and "
        "data/v2/latest/responses.jsonl.\n"
        "Checked:\n"
        f"{checked}\n"
        "Either sync the bundled snapshot into source-data/benchmark-snapshot or "
        "set REASONING_LAB_BENCHMARK_ROOT to a BullshitBench checkout."
    )


def benchmark_dataset_dir(dataset_key: str) -> Path:
    benchmark_root = resolve_benchmark_root()
    if dataset_key == "v1":
        return benchmark_root / "data" / "latest"
    if dataset_key == "v2":
        return benchmark_root / "data" / "v2" / "latest"
    raise ValueError(f"Unsupported dataset key: {dataset_key}")
