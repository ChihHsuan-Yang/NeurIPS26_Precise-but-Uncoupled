"""Helpers for benchmark/runtime and post-analysis API-call accounting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_candidate_correctness_accounting(
    evaluator: Any,
    *,
    component_name: str = "candidate_correctness_evaluator",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "kind": component_name,
        "api_calls": int(getattr(evaluator, "evaluator_calls", 0) or 0),
        "cache_hits": int(getattr(evaluator, "cache_hits", 0) or 0),
        "persistent_cache_hits": int(getattr(evaluator, "persistent_cache_hits", 0) or 0),
        "evaluator_type": str(getattr(evaluator, "evaluator_type", "") or ""),
    }
    if extra:
        payload.update(extra)
    return payload


def build_trace_labeler_accounting(
    labeler: Any,
    *,
    component_name: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "kind": component_name,
        "api_calls": int(getattr(labeler, "llm_request_count", 0) or 0),
        "mode": str(getattr(labeler, "mode", "") or ""),
        "model": str(getattr(labeler, "model", "") or ""),
        "correctness_cache_hits": int(getattr(labeler, "correctness_cache_hits", 0) or 0),
        "feedback_incorporation_cache_hits": int(
            getattr(labeler, "feedback_incorporation_cache_hits", 0) or 0
        ),
        "persistent_feedback_cache_hits": int(
            getattr(labeler, "persistent_feedback_cache_hits", 0) or 0
        ),
    }
    if extra:
        payload.update(extra)
    return payload


def merge_api_accounting(*components: dict[str, Any] | None) -> dict[str, Any]:
    normalized: dict[str, dict[str, Any]] = {}
    total = 0
    for index, component in enumerate(components, start=1):
        if not component:
            continue
        item = dict(component)
        name = str(item.get("kind") or item.get("name") or f"component_{index}")
        item["api_calls"] = int(item.get("api_calls", 0) or 0)
        total += item["api_calls"]
        normalized[name] = item
    return {
        "analysis_api_calls_total": total,
        "components": normalized,
    }


def combine_api_accounting_summaries(*summaries: dict[str, Any] | None) -> dict[str, Any]:
    components: list[dict[str, Any]] = []
    for summary in summaries:
        if not summary:
            continue
        summary_components = summary.get("components", {}) or {}
        if isinstance(summary_components, dict):
            for component in summary_components.values():
                if isinstance(component, dict):
                    components.append(component)
    return merge_api_accounting(*components)


def summarize_benchmark_runtime_accounting(
    metrics_path: Path | None,
) -> dict[str, Any]:
    rows = _load_jsonl(metrics_path)
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    total_model_calls = 0
    runtime_wall_seconds = 0.0

    for row in rows:
        token_row = row.get("total_tokens", {}) or {}
        prompt_tokens += int(token_row.get("prompt_tokens", 0) or 0)
        completion_tokens += int(token_row.get("completion_tokens", 0) or 0)
        total_tokens += int(token_row.get("total_tokens", 0) or 0)
        total_model_calls += int(row.get("total_model_calls", 0) or 0)
        runtime_wall_seconds += float(row.get("wall_time_seconds", 0.0) or 0.0)

    return {
        "kind": "benchmark_runtime_accounting",
        "metrics_path": str(metrics_path) if metrics_path else None,
        "example_count": len(rows),
        "runtime_api_calls_total": total_model_calls,
        "runtime_wall_seconds_total": runtime_wall_seconds,
        "runtime_prompt_tokens_total": prompt_tokens,
        "runtime_completion_tokens_total": completion_tokens,
        "runtime_tokens_total": total_tokens,
    }


def write_api_accounting_sidecar(output_dir: Path, payload: dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "api_accounting.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return path


def read_api_accounting_sidecar(output_dir: Path) -> dict[str, Any]:
    path = output_dir / "api_accounting.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.replace("\x00", "").strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError:
                continue
    return rows
