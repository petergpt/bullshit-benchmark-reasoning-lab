#!/usr/bin/env python3
"""Run prompt-variant experiments for AI reasoning labeling."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from ai_label_reasoning_trace import PROMPT_PROFILES, record_is_human_reviewed
from export_reasoning_label_examples import load_store
from reasoning_lab_data import LAB_ROOT as ROOT, text_or_empty


STORE_PATH = ROOT / "annotations" / "reasoning_lab.json"
RUNNER_PATH = ROOT / "scripts" / "ai_label_reasoning_trace.py"
DEFAULT_RUN_OUTPUT_DIR = ROOT / "data" / "ai_label_runs"
DEFAULT_EVAL_OUTPUT_DIR = ROOT / "data" / "ai_label_evals"
SOFT_OVERALL_GROUP = {"attempts_to_solve", "going_along_with_it"}
DEFAULT_MEETS_BAR = {
    "min_relaxed_overall_match_rate": 0.8,
    "min_intent_recall": 1.0,
    "min_mean_label_presence_f1": 0.6,
}
VARIANTS: dict[str, dict[str, Any]] = {
    "baseline_t0": {
        "prompt_profile": "baseline",
        "reference_order": "question_id",
        "temperature": 0.0,
        "max_reference_documents": 0,
    },
    "baseline_signal_t0": {
        "prompt_profile": "baseline",
        "reference_order": "signal_first",
        "temperature": 0.0,
        "max_reference_documents": 0,
    },
    "rubric_t0": {
        "prompt_profile": "rubric",
        "reference_order": "question_id",
        "temperature": 0.0,
        "max_reference_documents": 0,
    },
    "rubric_signal_t0": {
        "prompt_profile": "rubric",
        "reference_order": "signal_first",
        "temperature": 0.0,
        "max_reference_documents": 0,
    },
}


def select_labelled_documents(store: dict[str, Any]) -> list[str]:
    document_ids = [
        text_or_empty(item.get("document_id"))
        for item in store.get("document_notes") or []
        if text_or_empty(item.get("label_id"))
        and record_is_human_reviewed(item)
    ]
    return sorted({document_id for document_id in document_ids if document_id})


def run_single_trial(
    *,
    document_id: str,
    variant_name: str,
    variant: dict[str, Any],
    repeat_index: int,
    model: str,
    reasoning_effort: str,
    store_path: Path,
    run_output_dir: Path,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(RUNNER_PATH),
        "--document-id",
        document_id,
        "--model",
        model,
        "--reasoning-effort",
        reasoning_effort,
        "--store",
        str(store_path),
        "--output-dir",
        str(run_output_dir),
        "--prompt-profile",
        text_or_empty(variant.get("prompt_profile")),
        "--reference-order",
        text_or_empty(variant.get("reference_order")),
        "--temperature",
        str(float(variant.get("temperature") or 0.0)),
        "--max-reference-documents",
        str(int(variant.get("max_reference_documents") or 0)),
    ]
    completed = subprocess.run(
        command,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    artifact_rel = ""
    for line in stdout.splitlines():
        if line.startswith("Wrote "):
            artifact_rel = line.removeprefix("Wrote ").strip()
            break
    if not artifact_rel:
        raise RuntimeError(
            f"Could not find artifact path for {variant_name} repeat {repeat_index} {document_id}.\n"
            f"STDOUT:\n{stdout}\nSTDERR:\n{stderr}"
        )
    artifact_path = ROOT / artifact_rel
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    return {
        "variant_name": variant_name,
        "repeat_index": repeat_index,
        "document_id": document_id,
        "artifact_path": str(artifact_path),
        "artifact": artifact,
        "stdout": stdout,
        "stderr": stderr,
        "returncode": completed.returncode,
    }


def label_presence_f1(gold_labels: set[str], predicted_labels: set[str]) -> float:
    if not gold_labels and not predicted_labels:
        return 1.0
    if not gold_labels or not predicted_labels:
        return 0.0
    overlap = len(gold_labels & predicted_labels)
    return (2.0 * overlap) / (len(gold_labels) + len(predicted_labels))


def relaxed_overall_match(gold_label: str, predicted_label: str) -> bool:
    if not gold_label or not predicted_label:
        return False
    if gold_label == predicted_label:
        return True
    return gold_label in SOFT_OVERALL_GROUP and predicted_label in SOFT_OVERALL_GROUP


def summarize_variant(
    *,
    variant_name: str,
    variant: dict[str, Any],
    trial_results: list[dict[str, Any]],
) -> dict[str, Any]:
    strict_match_count = 0
    relaxed_match_count = 0
    resolved_count = 0
    total_label_presence_f1 = 0.0
    total_annotation_budget_violations = 0
    intent_tp = 0
    intent_fp = 0
    intent_fn = 0
    strict_pair_counter: Counter[tuple[str, str]] = Counter()
    docs: list[dict[str, Any]] = []

    for result in trial_results:
        artifact = result["artifact"]
        comparison = artifact.get("comparison_to_human_labels") or {}
        resolution_error = text_or_empty(artifact.get("resolution_error"))
        gold_overall = text_or_empty(comparison.get("gold_overall_label"))
        predicted_overall = text_or_empty(comparison.get("predicted_overall_label"))
        gold_presence = {
            text_or_empty(label_id)
            for label_id in comparison.get("gold_label_presence") or []
            if text_or_empty(label_id)
        }
        predicted_presence = {
            text_or_empty(label_id)
            for label_id in comparison.get("predicted_label_presence") or []
            if text_or_empty(label_id)
        }
        strict_match = bool(gold_overall) and gold_overall == predicted_overall
        relaxed_match_value = relaxed_overall_match(gold_overall, predicted_overall)
        label_presence = label_presence_f1(gold_presence, predicted_presence)
        gold_intent = bool(comparison.get("gold_questions_intent_present"))
        predicted_intent = bool(comparison.get("predicted_questions_intent_present"))
        if not resolution_error:
            resolved_count += 1
        if strict_match:
            strict_match_count += 1
        if relaxed_match_value:
            relaxed_match_count += 1
        if int(comparison.get("predicted_annotation_count") or 0) > 10:
            total_annotation_budget_violations += 1
        total_label_presence_f1 += label_presence
        if gold_intent and predicted_intent:
            intent_tp += 1
        elif gold_intent and not predicted_intent:
            intent_fn += 1
        elif predicted_intent and not gold_intent:
            intent_fp += 1
        strict_pair_counter[(gold_overall, predicted_overall)] += 1
        docs.append(
            {
                "document_id": result["document_id"],
                "repeat_index": result["repeat_index"],
                "artifact_path": result["artifact_path"],
                "prompt_version": text_or_empty(artifact.get("prompt_version")),
                "prompt_profile": text_or_empty(artifact.get("prompt_profile")),
                "resolution_error": resolution_error,
                "gold_overall_label": gold_overall,
                "predicted_overall_label": predicted_overall,
                "strict_overall_match": strict_match,
                "relaxed_overall_match": relaxed_match_value,
                "gold_label_presence": sorted(gold_presence),
                "predicted_label_presence": sorted(predicted_presence),
                "label_presence_f1": label_presence,
                "gold_questions_intent_present": gold_intent,
                "predicted_questions_intent_present": predicted_intent,
                "overall_assessment": (artifact.get("parsed_model_output") or {}).get("overall_assessment"),
                "predicted_summary": text_or_empty(
                    (artifact.get("resolved_document_note") or {}).get("summary")
                ),
            }
        )

    run_count = len(trial_results)
    intent_recall = intent_tp / (intent_tp + intent_fn) if (intent_tp + intent_fn) else 1.0
    intent_precision = intent_tp / (intent_tp + intent_fp) if (intent_tp + intent_fp) else 1.0
    if intent_precision + intent_recall:
        intent_f1 = (2.0 * intent_precision * intent_recall) / (intent_precision + intent_recall)
    else:
        intent_f1 = 0.0
    mean_label_presence_f1 = total_label_presence_f1 / run_count if run_count else 0.0
    strict_overall_match_rate = strict_match_count / run_count if run_count else 0.0
    relaxed_overall_match_rate = relaxed_match_count / run_count if run_count else 0.0
    meets_bar = (
        relaxed_overall_match_rate >= DEFAULT_MEETS_BAR["min_relaxed_overall_match_rate"]
        and intent_recall >= DEFAULT_MEETS_BAR["min_intent_recall"]
        and mean_label_presence_f1 >= DEFAULT_MEETS_BAR["min_mean_label_presence_f1"]
    )

    return {
        "variant_name": variant_name,
        "config": variant,
        "run_count": run_count,
        "resolved_count": resolved_count,
        "strict_overall_match_rate": strict_overall_match_rate,
        "relaxed_overall_match_rate": relaxed_overall_match_rate,
        "mean_label_presence_f1": mean_label_presence_f1,
        "intent_doc_precision": intent_precision,
        "intent_doc_recall": intent_recall,
        "intent_doc_f1": intent_f1,
        "annotation_budget_violations": total_annotation_budget_violations,
        "meets_bar": meets_bar,
        "overall_label_pairs": [
            {"gold": gold, "predicted": predicted, "count": count}
            for (gold, predicted), count in sorted(
                strict_pair_counter.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ],
        "documents": docs,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", default=str(STORE_PATH))
    parser.add_argument("--model", default="anthropic/claude-sonnet-4.6")
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument(
        "--variant",
        dest="variants",
        action="append",
        choices=sorted(VARIANTS),
        help="Variant name to run. Repeat to add multiple. Defaults to all built-in variants.",
    )
    parser.add_argument(
        "--document-id",
        dest="document_ids",
        action="append",
        help="Specific labelled document_id to evaluate. Repeat to add multiple.",
    )
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--run-output-dir", default=str(DEFAULT_RUN_OUTPUT_DIR))
    parser.add_argument("--eval-output-dir", default=str(DEFAULT_EVAL_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    store_path = Path(args.store).resolve()
    run_output_dir = Path(args.run_output_dir).resolve()
    eval_output_dir = Path(args.eval_output_dir).resolve()
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    store = load_store(store_path)
    document_ids = args.document_ids or select_labelled_documents(store)
    if not document_ids:
        raise SystemExit("No labelled documents found to evaluate.")
    variant_names = args.variants or list(VARIANTS)

    summaries = []
    for variant_name in variant_names:
        variant = VARIANTS[variant_name]
        print(f"== {variant_name} ==", flush=True)
        trial_results = []
        for repeat_index in range(1, args.repeat + 1):
            for index, document_id in enumerate(document_ids, start=1):
                print(
                    f"[{variant_name} repeat {repeat_index}/{args.repeat} "
                    f"doc {index}/{len(document_ids)}] {document_id}",
                    flush=True,
                )
                trial_results.append(
                    run_single_trial(
                        document_id=document_id,
                        variant_name=variant_name,
                        variant=variant,
                        repeat_index=repeat_index,
                        model=args.model,
                        reasoning_effort=args.reasoning_effort,
                        store_path=store_path,
                        run_output_dir=run_output_dir,
                    )
                )
        summaries.append(
            summarize_variant(
                variant_name=variant_name,
                variant=variant,
                trial_results=trial_results,
            )
        )

    ranked_summaries = sorted(
        summaries,
        key=lambda item: (
            bool(item.get("meets_bar")),
            float(item.get("intent_doc_recall") or 0.0),
            float(item.get("relaxed_overall_match_rate") or 0.0),
            float(item.get("mean_label_presence_f1") or 0.0),
            float(item.get("strict_overall_match_rate") or 0.0),
        ),
        reverse=True,
    )
    output_payload = {
        "created_at_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "document_ids": document_ids,
        "repeat": args.repeat,
        "meets_bar_thresholds": DEFAULT_MEETS_BAR,
        "variants": ranked_summaries,
    }
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    variant_slug = "-".join(variant_names)
    output_path = eval_output_dir / f"{timestamp}__label-variant-eval__{variant_slug}.json"
    output_path.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    headline = [
        {
            "variant_name": item["variant_name"],
            "strict_overall_match_rate": item["strict_overall_match_rate"],
            "relaxed_overall_match_rate": item["relaxed_overall_match_rate"],
            "mean_label_presence_f1": item["mean_label_presence_f1"],
            "intent_doc_recall": item["intent_doc_recall"],
            "intent_doc_precision": item["intent_doc_precision"],
            "meets_bar": item["meets_bar"],
        }
        for item in ranked_summaries
    ]
    print(f"Wrote {output_path.relative_to(ROOT)}")
    print(json.dumps(headline, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
