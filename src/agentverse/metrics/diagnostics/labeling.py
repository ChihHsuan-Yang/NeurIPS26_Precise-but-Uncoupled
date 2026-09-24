"""Evaluator-style labels used by supplemental diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from agentverse.evaluation.content_based import ContentJudgeResult
from agentverse.metrics.candidate_label_cache import default_candidate_label_cache_path
from agentverse.metrics.main_paper.evaluator_labeling import (
    CandidateCorrectnessEvaluator,
    infer_evaluator_type,
    normalize_evaluator_type,
)
from agentverse.metrics.repair import (
    TraceExample,
    _extract_any_candidate,
    _find_previous_candidate,
    _is_evaluator_turn,
    _parse_evaluator_pass_fail,
)


class EvaluatorStyleTraceLabeler:
    """Duck-typed correctness labeler for review diagnostics.

    The diagnostic review metrics should not use a separate post-trace LLM judge
    for candidate correctness. This adapter reuses runtime evaluator labels when
    the candidate matches a submitted answer, then falls back to the same
    benchmark-style evaluator used by the main-paper trace analysis.
    """

    def __init__(
        self,
        *,
        examples: list[TraceExample],
        evaluator: CandidateCorrectnessEvaluator,
    ):
        self.evaluator = evaluator
        self._runtime_labels = _build_runtime_label_map(examples)
        self.runtime_hits = 0

    def judge_correctness(
        self,
        *,
        problem: str,
        gold_answer: str,
        candidate_answer: str,
    ) -> ContentJudgeResult:
        candidate = str(candidate_answer or "").strip()
        if not candidate:
            return ContentJudgeResult(None, "missing", "Missing candidate answer.")

        key = _label_key(problem, gold_answer, candidate)
        if key in self._runtime_labels and self._runtime_labels[key] is not None:
            self.runtime_hits += 1
            return ContentJudgeResult(
                bool(self._runtime_labels[key]),
                "runtime_evaluator",
                "Reused saved evaluator PASS/FAIL label for a submitted candidate.",
            )

        label = self.evaluator.label_candidate(
            problem=problem,
            gold_answer=gold_answer,
            candidate_answer=candidate,
        )
        return ContentJudgeResult(label.value, label.source, label.reason)


def build_review_correctness_labeler(
    examples: list[TraceExample],
    *,
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
) -> tuple[EvaluatorStyleTraceLabeler, str]:
    selected = normalize_evaluator_type(evaluator_type)
    if selected == "same_as_benchmark":
        selected = infer_evaluator_type(examples)

    resolved_cache_path = None
    if cache_candidate_labels:
        if candidate_label_cache_path:
            resolved_cache_path = Path(candidate_label_cache_path).expanduser().resolve()
        elif examples:
            resolved_cache_path = default_candidate_label_cache_path(
                Path(examples[0].trace_path).expanduser().resolve().parent
            )

    evaluator = CandidateCorrectnessEvaluator(
        evaluator_type=selected,
        numeric_tolerance=numeric_tolerance,
        data_name=data_name,
        omni_judge_model_path=omni_judge_model_path,
        omni_judge_max_new_tokens=omni_judge_max_new_tokens,
        omni_judge_device=omni_judge_device,
        omni_judge_dtype=omni_judge_dtype,
        evaluator_agent_config_path=run_config_path,
        reuse_runtime_final_labels=reuse_runtime_final_labels,
        cache_candidate_labels=cache_candidate_labels,
        candidate_label_cache_path=resolved_cache_path,
    )
    return EvaluatorStyleTraceLabeler(examples=examples, evaluator=evaluator), selected


def _build_runtime_label_map(
    examples: list[TraceExample],
) -> dict[tuple[str, str, str], Optional[bool]]:
    labels: dict[tuple[str, str, str], Optional[bool]] = {}
    for example in examples:
        evaluator_names = {
            name for name in example.agent_map.values() if "evaluator" in name.lower()
        }
        for turn_pos, turn in enumerate(example.turns):
            if not _is_evaluator_turn(turn, evaluator_names):
                continue
            passed = _parse_evaluator_pass_fail(turn.content)
            if passed is None:
                continue
            candidate = _find_previous_candidate(example.turns, turn_pos)
            candidate = candidate or _extract_any_candidate(turn.content)
            if not candidate:
                continue

            key = _label_key(example.input_text, example.label, candidate)
            existing = labels.get(key)
            if existing is None and key in labels:
                continue
            if existing is not None and existing != passed:
                labels[key] = None
            else:
                labels[key] = bool(passed)
    return labels


def _label_key(problem: str, gold_answer: str, candidate_answer: str) -> tuple[str, str, str]:
    return (
        str(problem or "").strip(),
        str(gold_answer or "").strip(),
        str(candidate_answer or "").strip(),
    )
