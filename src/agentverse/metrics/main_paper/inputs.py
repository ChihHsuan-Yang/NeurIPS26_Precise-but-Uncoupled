"""Input loading and deterministic trace parsing for main-paper metrics."""

from __future__ import annotations

import json
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

from agentverse.metrics.candidate_label_cache import default_candidate_label_cache_path
from agentverse.metrics.api_accounting import (
    build_candidate_correctness_accounting,
    merge_api_accounting,
)
from agentverse.metrics.main_paper.evaluator_labeling import (
    CandidateCorrectnessEvaluator,
    infer_evaluator_type,
    normalize_evaluator_type,
)
from agentverse.metrics.main_paper.records import (
    AttemptRecord,
    InnerTransitionRecord,
    MainPaperDataset,
    ProblemRecord,
)
from agentverse.metrics.repair import (
    TraceExample,
    _is_evaluator_turn,
    _parse_evaluator_pass_fail,
    extract_protocol_candidate_sequence,
    parse_trace_file,
)


def load_main_paper_dataset(
    trace_files: list[Path],
    *,
    results_path: str | Path | None = None,
    metrics_path: str | Path | None = None,
    evaluator_type: str = "llm",
    numeric_tolerance: str = "0",
    data_name: str = "omni-math",
    omni_judge_model_path: str | None = None,
    omni_judge_max_new_tokens: int = 300,
    omni_judge_device: str = "auto",
    omni_judge_dtype: str = "auto",
    run_config_path: str | Path | None = None,
    reuse_runtime_final_labels: bool = True,
    cache_candidate_labels: bool = True,
    candidate_label_cache_path: str | Path | None = None,
    include_inner_transitions: bool = True,
) -> MainPaperDataset:
    examples = _load_trace_examples(trace_files)
    base_dir = trace_files[0].parent if trace_files else Path.cwd()
    results_by_idx = _load_results_by_example(
        Path(results_path).expanduser() if results_path else base_dir / "results.jsonl"
    )
    metrics_by_idx = _load_metrics_by_example(
        Path(metrics_path).expanduser()
        if metrics_path
        else _discover_metrics_path(base_dir, trace_files=trace_files)
    )

    selected_evaluator = normalize_evaluator_type(evaluator_type)
    if selected_evaluator == "same_as_benchmark":
        selected_evaluator = infer_evaluator_type(examples)
    resolved_run_config = (
        Path(run_config_path).expanduser()
        if run_config_path
        else _discover_run_config_path(base_dir)
    )
    resolved_cache_path = _resolve_candidate_label_cache_path(
        base_dir=base_dir,
        cache_candidate_labels=cache_candidate_labels,
        candidate_label_cache_path=candidate_label_cache_path,
    )

    attempts = extract_attempt_records(examples)
    reflective_rounds_by_example = extract_reflective_round_counts(examples)
    inner_transitions: list[InnerTransitionRecord] = []
    analysis_accounting = merge_api_accounting()
    if include_inner_transitions:
        candidate_evaluator = CandidateCorrectnessEvaluator(
            evaluator_type=selected_evaluator,
            numeric_tolerance=numeric_tolerance,
            data_name=data_name,
            omni_judge_model_path=omni_judge_model_path,
            omni_judge_max_new_tokens=omni_judge_max_new_tokens,
            omni_judge_device=omni_judge_device,
            omni_judge_dtype=omni_judge_dtype,
            evaluator_agent_config_path=resolved_run_config,
            reuse_runtime_final_labels=reuse_runtime_final_labels,
            cache_candidate_labels=cache_candidate_labels,
            candidate_label_cache_path=resolved_cache_path,
        )
        inner_transitions = extract_inner_transitions(
            examples,
            results_by_idx=results_by_idx,
            evaluator=candidate_evaluator,
        )
        analysis_accounting = merge_api_accounting(
            build_candidate_correctness_accounting(candidate_evaluator)
        )
    problems = build_problem_records(
        examples=examples,
        attempts=attempts,
        inner_transitions=inner_transitions,
        reflective_rounds_by_example=reflective_rounds_by_example,
        results_by_idx=results_by_idx,
        metrics_by_idx=metrics_by_idx,
    )
    return MainPaperDataset(
        examples=examples,
        attempts=attempts,
        problems=problems,
        inner_transitions=inner_transitions,
        system_feedback_episodes=[],
        results_by_idx=results_by_idx,
        metrics_by_idx=metrics_by_idx,
        trace_files=[str(path) for path in trace_files],
        evaluator_type=selected_evaluator,
        analysis_accounting=analysis_accounting,
    )


def extract_attempt_records(examples: list[TraceExample]) -> list[AttemptRecord]:
    attempts: list[AttemptRecord] = []
    for example in examples:
        evaluator_names = {
            name for name in example.agent_map.values() if "evaluator" in name.lower()
        }
        attempt_idx = 0
        for idx, turn in enumerate(example.turns):
            if not _is_evaluator_turn(turn, evaluator_names):
                continue
            passed = _parse_evaluator_pass_fail(turn.content)
            if passed is None:
                continue
            attempt_idx += 1
            attempts.append(
                AttemptRecord(
                    trace_path=example.trace_path,
                    task=example.task,
                    protocol=example.protocol,
                    example_idx=example.example_idx,
                    attempt_idx=attempt_idx,
                    evaluator_turn=turn.turn_idx,
                    correct=passed,
                    hint_used=_attempt_has_hint(example.turns, idx, evaluator_names),
                    feedback=turn.content,
                )
            )
    return attempts


def extract_inner_transitions(
    examples: list[TraceExample],
    *,
    results_by_idx: dict[int, dict[str, Any]],
    evaluator: CandidateCorrectnessEvaluator,
) -> list[InnerTransitionRecord]:
    transitions: list[InnerTransitionRecord] = []
    for example in examples:
        evaluator_names = {
            name for name in example.agent_map.values() if "evaluator" in name.lower()
        }
        eval_positions = [
            idx
            for idx, turn in enumerate(example.turns)
            if _is_evaluator_turn(turn, evaluator_names)
            and _parse_evaluator_pass_fail(turn.content) is not None
        ]
        start_pos = 0
        for attempt_idx, eval_pos in enumerate(eval_positions, start=1):
            interval = example.turns[start_pos:eval_pos]
            candidates = extract_protocol_candidate_sequence(example, interval)
            if not candidates:
                start_pos = eval_pos + 1
                continue

            initial_candidate = candidates[0]
            final_candidate = candidates[-1]
            runtime_final = _parse_evaluator_pass_fail(example.turns[eval_pos].content)
            initial = evaluator.label_candidate(
                problem=example.input_text,
                gold_answer=example.label,
                candidate_answer=initial_candidate,
                final_candidate=final_candidate,
                runtime_final=runtime_final,
            )
            final = evaluator.label_candidate(
                problem=example.input_text,
                gold_answer=example.label,
                candidate_answer=final_candidate,
                final_candidate=final_candidate,
                runtime_final=runtime_final,
            )
            result_row = results_by_idx.get(example.example_idx, {})
            transitions.append(
                InnerTransitionRecord(
                    trace_path=example.trace_path,
                    task=example.task,
                    protocol=example.protocol,
                    example_idx=example.example_idx,
                    attempt_idx=attempt_idx,
                    start_turn=interval[0].turn_idx if interval else 0,
                    end_turn=example.turns[eval_pos].turn_idx,
                    initial_candidate=initial_candidate,
                    final_candidate=final_candidate,
                    candidate_count=len(candidates),
                    initial_correct=initial.value,
                    final_correct=final.value,
                    correctness_source=_join_sources(initial.source, final.source),
                    difficulty_tier=result_row.get("difficulty_tier"),
                    domain=result_row.get("domain"),
                )
            )
            start_pos = eval_pos + 1
    return transitions


def build_problem_records(
    *,
    examples: list[TraceExample],
    attempts: list[AttemptRecord],
    inner_transitions: list[InnerTransitionRecord],
    reflective_rounds_by_example: dict[int, int],
    results_by_idx: dict[int, dict[str, Any]],
    metrics_by_idx: dict[int, dict[str, Any]],
) -> list[ProblemRecord]:
    attempts_by_idx = _group_by_example(attempts)
    records: list[ProblemRecord] = []
    for example in examples:
        idx = example.example_idx
        ex_attempts = attempts_by_idx.get(idx, [])
        metric_row = metrics_by_idx.get(idx, {})
        runtime_attempts = _attempts_from_metric_row(metric_row)
        evaluator_attempts = ex_attempts if ex_attempts else runtime_attempts
        system_tries = _system_tries_from_metric_row(metric_row)
        scoring_attempts = system_tries if system_tries else evaluator_attempts
        first_success = _first_success_attempt(scoring_attempts)
        final_passed = first_success is not None
        first_passed = bool(scoring_attempts and scoring_attempts[0].correct is True)
        correction_loops = len(scoring_attempts)
        first_success_submission = _first_success_attempt(evaluator_attempts)
        hints_used = _hints_before_success(evaluator_attempts, first_success_submission)
        result_row = results_by_idx.get(idx, {})
        token_row = metric_row.get("total_tokens", {}) or {}
        agent_metrics = metric_row.get("agent_metrics", {}) or {}

        records.append(
            ProblemRecord(
                trace_path=example.trace_path,
                task=example.task,
                protocol=example.protocol,
                example_idx=idx,
                final_passed=final_passed,
                first_passed=first_passed,
                first_success_attempt=first_success,
                correction_loops=correction_loops,
                hints_used=hints_used,
                reflective_rounds=int(reflective_rounds_by_example.get(idx, 0)),
                reasoning_turns=_reasoning_turn_count(example),
                evaluator_submission_count=len(evaluator_attempts),
                difficulty_tier=result_row.get("difficulty_tier"),
                domain=result_row.get("domain"),
                source=result_row.get("source"),
                prompt_tokens=int(token_row.get("prompt_tokens", 0) or 0),
                completion_tokens=int(token_row.get("completion_tokens", 0) or 0),
                total_tokens=int(token_row.get("total_tokens", 0) or 0),
                model_calls=_model_calls_from_metric_row(
                    metric_row,
                    agent_metrics,
                    example=example,
                ),
                wall_time_seconds=float(metric_row.get("wall_time_seconds", 0.0) or 0.0),
            )
        )
    return records


def extract_reflective_round_counts(examples: list[TraceExample]) -> dict[int, int]:
    """Count within-attempt candidate revisions without judging correctness.

    This keeps RQ1/RQ2 runs cheap: the count is the sum over attempts of
    `max(0, number_of_distinct_candidates_before_submission - 1)`.
    """

    counts: dict[int, int] = {}
    for example in examples:
        evaluator_names = {
            name for name in example.agent_map.values() if "evaluator" in name.lower()
        }
        eval_positions = [
            idx
            for idx, turn in enumerate(example.turns)
            if _is_evaluator_turn(turn, evaluator_names)
            and _parse_evaluator_pass_fail(turn.content) is not None
        ]
        start_pos = 0
        total = 0
        for eval_pos in eval_positions:
            interval = example.turns[start_pos:eval_pos]
            candidates = extract_protocol_candidate_sequence(example, interval)
            total += max(0, len(candidates) - 1)
            start_pos = eval_pos + 1
        counts[example.example_idx] = total
    return counts


def _load_trace_examples(trace_files: list[Path]) -> list[TraceExample]:
    examples: list[TraceExample] = []
    for trace_file in trace_files:
        examples.extend(parse_trace_file(trace_file))
    return examples


def _discover_metrics_path(
    base_dir: Path,
    *,
    trace_files: list[Path] | None = None,
) -> Optional[Path]:
    matches = sorted(base_dir.glob("*.metrics.jsonl"))
    if not matches:
        return None

    preferred_names = {
        f"{trace_file.name[:-len('.trace.txt')]}.metrics.jsonl"
        for trace_file in (trace_files or [])
        if trace_file.name.endswith(".trace.txt")
    }
    preferred_matches = [path for path in matches if path.name in preferred_names]
    if len(preferred_matches) == 1:
        return preferred_matches[0]
    if preferred_matches:
        matches = preferred_matches

    # Prefer the most complete sidecar when multiple metrics files coexist in one
    # results folder. This avoids accidentally picking an earlier partial artifact.
    def _score(path: Path) -> tuple[int, float, str]:
        return (_count_nonempty_lines(path), path.stat().st_mtime, path.name)

    return max(matches, key=_score)


def _count_nonempty_lines(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def _discover_run_config_path(base_dir: Path) -> Optional[Path]:
    preferred = base_dir / "config.yaml"
    if preferred.exists():
        return preferred
    matches = sorted(base_dir.glob("*.config.yaml"))
    return matches[0] if matches else None


def _resolve_candidate_label_cache_path(
    *,
    base_dir: Path,
    cache_candidate_labels: bool,
    candidate_label_cache_path: str | Path | None,
) -> Path | None:
    if not cache_candidate_labels:
        return None
    if candidate_label_cache_path:
        return Path(candidate_label_cache_path).expanduser().resolve()
    return default_candidate_label_cache_path(base_dir)


def _load_results_by_example(path: Path | None) -> dict[int, dict[str, Any]]:
    rows = _load_jsonl(path)
    out: dict[int, dict[str, Any]] = {}
    for pos, row in enumerate(rows, start=1):
        idx = int(row.get("question_id") or row.get("example_idx") or pos)
        out[idx] = row
    return out


def _load_metrics_by_example(path: Path | None) -> dict[int, dict[str, Any]]:
    rows = _load_jsonl(path)
    out: dict[int, dict[str, Any]] = {}
    for pos, row in enumerate(rows, start=1):
        raw_idx = row.get("example_idx")
        idx = int(raw_idx) + 1 if raw_idx is not None else pos
        out[idx] = row
    return out


def _load_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.replace("\x00", "").strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                warnings.warn(
                    (
                        f"Skipping malformed JSONL row in {path} at line {line_no}: "
                        f"{exc}"
                    ),
                    RuntimeWarning,
                    stacklevel=2,
                )
    return rows


def _attempts_from_metric_row(metric_row: dict[str, Any]) -> list[AttemptRecord]:
    raw_attempts = metric_row.get("evaluation_attempt_results", []) or []
    attempts: list[AttemptRecord] = []
    for passed in raw_attempts:
        if passed not in {True, False}:
            continue
        attempts.append(
            AttemptRecord(
                trace_path="metrics_jsonl",
                task="",
                protocol="",
                example_idx=int(metric_row.get("example_idx", 0) or 0) + 1,
                attempt_idx=len(attempts) + 1,
                evaluator_turn=len(attempts) + 1,
                correct=bool(passed),
                hint_used=False,
                feedback="",
            )
        )
    return attempts


def _system_tries_from_metric_row(metric_row: dict[str, Any]) -> list[AttemptRecord]:
    raw_rounds = metric_row.get("round_metrics", []) or []
    attempts: list[AttemptRecord] = []
    for round_pos, round_row in enumerate(raw_rounds, start=1):
        if not isinstance(round_row, dict):
            continue
        passed = round_row.get("evaluation_passed")
        if passed not in {True, False, None}:
            passed = None
        attempts.append(
            AttemptRecord(
                trace_path="metrics_jsonl_rounds",
                task="",
                protocol="",
                example_idx=int(metric_row.get("example_idx", 0) or 0) + 1,
                attempt_idx=round_pos,
                evaluator_turn=int(round_row.get("round_id", round_pos - 1) or 0),
                correct=passed,
                hint_used=False,
                feedback="",
            )
        )
    return attempts


def _first_success_attempt(attempts: list[AttemptRecord]) -> Optional[int]:
    for attempt in attempts:
        if attempt.correct is True:
            return attempt.attempt_idx
    return None


def _hints_before_success(
    attempts: list[AttemptRecord],
    first_success: Optional[int],
) -> int:
    limit = first_success if first_success is not None else len(attempts) + 1
    return sum(
        1
        for attempt in attempts
        if attempt.attempt_idx < limit and attempt.hint_used
    )


def _model_calls_from_metric_row(
    metric_row: dict[str, Any],
    agent_metrics: dict[str, Any],
    *,
    example: TraceExample | None = None,
) -> int:
    total_model_calls = metric_row.get("total_model_calls")
    metric_count = 0
    if total_model_calls is not None:
        metric_count = int(total_model_calls or 0)
    else:
        metric_count = sum(
            int(agent.get("invocation_count", 0) or 0)
            for agent in agent_metrics.values()
            if isinstance(agent, dict)
        )
    trace_lower_bound = _trace_model_call_lower_bound(example)
    return max(metric_count, trace_lower_bound)


def _trace_model_call_lower_bound(example: TraceExample | None) -> int:
    if example is None:
        return 0
    return sum(
        1
        for turn in example.turns
        if turn.agent_id.lower() != "system"
    )


def _reasoning_turn_count(example: TraceExample) -> int:
    return sum(
        1
        for turn in example.turns
        if "evaluator" not in turn.agent_name.lower()
        and turn.agent_id.lower() not in {"evaluator", "system"}
    )


def _looks_like_hint_feedback(text: str) -> bool:
    raw = str(text or "")
    lower = raw.lower()
    if "evaluation hint:" in lower:
        return bool(raw.split(":", 1)[1].strip()) if ":" in raw else True
    if "Evaluation signal: FAIL" not in raw and "Verifier: FAIL" not in raw:
        return False
    if "(no hint needed" in lower:
        return False

    marker: str | None = None
    if "hint:" in lower:
        marker = "hint:"
    elif "advice:" in lower:
        marker = "advice:"
    if marker is None:
        return False

    start = lower.index(marker) + len(marker)
    payload = raw[start:].strip()
    payload_lower = payload.lower()
    for prefix in (
        "verifier: fail (llm).",
        "verifier: fail.",
        "verifier: fail",
    ):
        if payload_lower.startswith(prefix):
            payload = payload[len(prefix) :].strip()
            payload_lower = payload.lower()
            break
    generic_plain_feedback = {
        "re-check the reasoning and the final reviewer answer.",
        "re-check the reasoning and the final reviewer answer",
        "re-check the reasoning and the final reviewer's boxed answer.",
        "re-check the reasoning and the final reviewer's boxed answer",
        "re-check the reasoning and the final boxed answer.",
        "re-check the reasoning and the final boxed answer",
    }
    if payload_lower in generic_plain_feedback:
        return False
    return bool(payload)


def _attempt_has_hint(
    turns: list[TraceTurn],
    verdict_idx: int,
    evaluator_names: set[str],
) -> bool:
    if verdict_idx < 0 or verdict_idx >= len(turns):
        return False
    if _looks_like_hint_feedback(turns[verdict_idx].content):
        return True

    idx = verdict_idx + 1
    while idx < len(turns):
        turn = turns[idx]
        raw = str(turn.content or "")
        if _parse_evaluator_pass_fail(raw) is not None:
            break
        if _looks_like_hint_feedback(raw):
            return True
        if (
            turn.agent_name in evaluator_names
            or turn.agent_id.lower() in {"evaluator", "system"}
            or "Evaluation hint:" in raw
        ):
            idx += 1
            continue
        break
    return False


def _group_by_example(items):
    grouped: dict[int, list[Any]] = defaultdict(list)
    for item in items:
        grouped[item.example_idx].append(item)
    return dict(grouped)


def _join_sources(*sources: str) -> str:
    return ",".join(sorted({source for source in sources if source}))
