from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Tuple

from agentverse.evaluation.common import extract_boxed, extract_final_answer
from agentverse.evaluation.omni_judge import (
    DEFAULT_OMNI_JUDGE_MODEL_ID,
    get_cached_omni_judge_evaluator,
)
from agentverse.evaluation.omni_rule import get_cached_omni_rule_evaluator
from agentverse.message import EvaluatorMessage

from . import evaluator_registry
from .base import BaseEvaluator

if TYPE_CHECKING:
    from agentverse.agents import EvaluatorAgent
    from agentverse.message import ExecutorMessage, SolverMessage

def _boxed_only_candidate(review_text: str) -> str:
    boxed = extract_boxed(review_text or "")
    return f"\\boxed{{{boxed}}}" if boxed else ""


def _to_decimal(s: str) -> Decimal | None:
    if s is None:
        return None
    cleaned = str(s).strip().replace(",", "")
    try:
        return Decimal(cleaned).normalize()
    except (InvalidOperation, ValueError):
        return None


def _numeric_equal(pred_str: str, gt_str: str, tol: str = "0") -> bool:
    pred = _to_decimal(pred_str)
    gt = _to_decimal(gt_str)
    if pred is not None and gt is not None:
        if tol and tol != "0":
            return abs(pred - gt) <= Decimal(tol)
        return pred == gt
    return str(pred_str).strip() == str(gt_str).strip()


def _coerce_score_to_bool(score: Any) -> bool:
    if isinstance(score, bool):
        return score
    if isinstance(score, int):
        return score == 1 or score >= 8
    if isinstance(score, (list, tuple)):
        return bool(score) and all(_coerce_score_to_bool(item) for item in score)
    return bool(score)


def _sanitize_reference_leaks(
    text: str,
    ground_truth: str,
    extra_references: List[str] | None = None,
) -> str:
    raw = str(text or "").strip()
    if not raw:
        return raw

    gt_raw = str(ground_truth or "").strip()
    gt_final = extract_final_answer(gt_raw)
    redaction = "[redacted reference answer]"

    candidates = []
    reference_values = {gt_raw, gt_final}
    for extra in extra_references or []:
        extra_raw = str(extra or "").strip()
        if not extra_raw:
            continue
        reference_values.add(extra_raw)
        reference_values.add(extract_final_answer(extra_raw))

    for value in reference_values:
        clean_value = str(value or "").strip()
        if clean_value:
            candidates.append(clean_value)
            candidates.append(f"\\boxed{{{clean_value}}}")

    sanitized = raw
    for candidate in sorted(set(candidates), key=len, reverse=True):
        sanitized = sanitized.replace(candidate, redaction)
    return sanitized


def _compact_hint_context(text: str, max_chars: int = 4000) -> str:
    raw = str(text or "").strip()
    if len(raw) <= max_chars:
        return raw
    return raw[: max_chars - 27].rstrip() + "\n...[truncated reasoning context]"


def _nonempty_parts(*parts: str) -> List[str]:
    return [str(part or "").strip() for part in parts if str(part or "").strip()]


def _build_hint_history(
    *,
    previous_answer_summary: str,
    previous_reasoning_summary: str,
    current_answer: str,
    current_reasoning: str,
) -> List[dict]:
    history: List[dict] = []
    if str(previous_answer_summary or "").strip():
        history.append(
            {
                "role": "system",
                "content": str(previous_answer_summary).strip(),
            }
        )
    if str(previous_reasoning_summary or "").strip():
        history.append(
            {
                "role": "system",
                "content": str(previous_reasoning_summary).strip(),
            }
        )
    current_parts = _nonempty_parts(
        "Current attempt context for hint generation only:",
        f"Current extracted final answer:\n{current_answer}",
        f"Current reasoning summary:\n{current_reasoning}",
    )
    if current_parts:
        history.append({"role": "system", "content": "\n\n".join(current_parts)})
    return history


class BasePerEvaluator(BaseEvaluator):
    mode_name: ClassVar[str] = "per-evaluator"
    box_required: ClassVar[bool] = False
    generic_failure_hint: str = ""

    def _signal_label(self, passed: bool) -> str:
        return "PASS" if passed else "FAIL"

    def _generic_failure_hint(self) -> str:
        custom_hint = str(getattr(self, "generic_failure_hint", "") or "").strip()
        if custom_hint:
            return custom_hint
        return (
            "Re-check both mathematical equivalence and answer shape. "
            "For Omni-MATH prompts such as find all, for each, all positive integers, "
            "all functions, or explicit form, the expected answer may be a complete "
            "symbolic condition, equation, family, classification, or all cases rather "
            "than isolated numeric examples."
        )

    def _canonicalize_advice(
        self,
        passed: bool,
        advice: str,
        ground_truth: str,
        extra_references: List[str] | None = None,
    ) -> str:
        raw = _sanitize_reference_leaks(advice, ground_truth, extra_references)
        if raw.startswith("Verifier: PASS") or raw.startswith("Verifier: FAIL"):
            return raw

        prefix = f"Verifier: {self._signal_label(passed)} ({self.mode_name})."
        if not raw:
            return prefix
        return f"{prefix} {raw}"

    def _prepare_candidate(
        self,
        reviewer_output: str,
        ground_truth: str,
        agent: EvaluatorAgent | None,
    ) -> Tuple[str, str, str, EvaluatorMessage | None]:
        reviewer_output = (reviewer_output or "").strip()
        ground_truth_final = extract_final_answer(str(ground_truth))

        if self.box_required:
            candidate = _boxed_only_candidate(reviewer_output)
            if not candidate:
                verdict_advice = "System protocol check: FAIL."
                hint_advice = (
                    "System protocol check: FAIL. "
                    "No boxed final answer was found in the submitted solution, so the evaluator was not called. "
                    "Please resubmit with exactly one clear boxed final answer. "
                    "Try to keep the reasoning concise and finish the solution within the available output window."
                )
                return "", "", ground_truth_final, self._build_message(
                    agent=agent,
                    passed=False,
                    final_answer="",
                    ground_truth=ground_truth_final,
                    advice=hint_advice,
                    verdict_advice=verdict_advice,
                    hint_advice=hint_advice,
                    used_hint=True,
                    direct_protocol_fail=True,
                    direct_protocol_fail_reason="missing_boxed_final_answer",
                    judge_student_final_answer="",
                    judge_equivalence_judgement="FALSE",
                )
        else:
            candidate = reviewer_output

        final_answer = extract_final_answer(candidate)
        return candidate, final_answer, ground_truth_final, None

    def _build_message(
        self,
        agent: EvaluatorAgent | None,
        passed: bool,
        final_answer: str,
        ground_truth: str,
        advice: str,
        extra_references: List[str] | None = None,
        **extra: Any,
    ) -> EvaluatorMessage:
        normalized_advice = self._canonicalize_advice(
            passed,
            advice,
            ground_truth,
            extra_references,
        )
        payload = {
            "mode": self.mode_name,
            "passed": passed,
            "signal": self._signal_label(passed),
            "final_answer": final_answer,
            "ground_truth_redacted": True,
            "advice": normalized_advice,
            "verdict_advice": normalized_advice,
            "hint_advice": "",
            "used_hint": False,
        }
        payload.update(extra)
        return EvaluatorMessage(
            sender=getattr(agent, "name", "Evaluator"),
            sender_agent=agent,
            content=payload,
            score=passed,
            advice=normalized_advice,
        )


@evaluator_registry.register("exact")
class ExactPerEvaluator(BasePerEvaluator):
    mode_name: ClassVar[str] = "exact"
    box_required: ClassVar[bool] = True

    async def astep(
        self,
        agent: EvaluatorAgent | None,
        solution: List[SolverMessage],
        result: List[ExecutorMessage],
        task_description: str,
        all_role_description: List[str],
        *args,
        **kwargs,
    ) -> EvaluatorMessage:
        candidate, final_answer, ground_truth_final, failure = self._prepare_candidate(
            kwargs.get("reviewer_output", ""),
            kwargs.get("ground_truth", ""),
            agent,
        )
        if failure is not None:
            return failure

        passed = bool(final_answer) and str(final_answer).strip() == str(ground_truth_final).strip()
        if passed:
            advice = "Verifier: PASS (exact)."
        else:
            advice = f"Verifier: FAIL (exact). {self._generic_failure_hint()}"
        return self._build_message(
            agent=agent,
            passed=passed,
            final_answer=final_answer,
            ground_truth=ground_truth_final,
            advice=advice,
        )


@evaluator_registry.register("numeric-verifier")
class NumericPerEvaluator(BasePerEvaluator):
    mode_name: ClassVar[str] = "numeric-verifier"
    box_required: ClassVar[bool] = True

    numeric_tolerance: str = "0"

    async def astep(
        self,
        agent: EvaluatorAgent | None,
        solution: List[SolverMessage],
        result: List[ExecutorMessage],
        task_description: str,
        all_role_description: List[str],
        *args,
        **kwargs,
    ) -> EvaluatorMessage:
        candidate, final_answer, ground_truth_final, failure = self._prepare_candidate(
            kwargs.get("reviewer_output", ""),
            kwargs.get("ground_truth", ""),
            agent,
        )
        if failure is not None:
            return failure

        passed = bool(final_answer) and _numeric_equal(
            final_answer,
            ground_truth_final,
            tol=self.numeric_tolerance,
        )
        if passed:
            advice = "Verifier: PASS (numeric-verifier)."
        else:
            advice = f"Verifier: FAIL (numeric-verifier). {self._generic_failure_hint()}"
        return self._build_message(
            agent=agent,
            passed=passed,
            final_answer=final_answer,
            ground_truth=ground_truth_final,
            advice=advice,
        )


@evaluator_registry.register("omni-rule")
class OmniRulePerEvaluator(BasePerEvaluator):
    mode_name: ClassVar[str] = "omni-rule"
    box_required: ClassVar[bool] = True

    data_name: str = "omni-math"

    async def astep(
        self,
        agent: EvaluatorAgent | None,
        solution: List[SolverMessage],
        result: List[ExecutorMessage],
        task_description: str,
        all_role_description: List[str],
        *args,
        **kwargs,
    ) -> EvaluatorMessage:
        candidate, final_answer, ground_truth_final, failure = self._prepare_candidate(
            kwargs.get("reviewer_output", ""),
            kwargs.get("ground_truth", ""),
            agent,
        )
        if failure is not None:
            return failure

        evaluator = get_cached_omni_rule_evaluator(self.data_name)
        row = evaluator.evaluate(
            [
                {
                    "problem": task_description,
                    "answer": str(kwargs.get("ground_truth", "")),
                    "model_generation": candidate,
                }
            ]
        )[0]
        passed = bool(row.get("correctness"))
        normalized_ground_truth = row.get("rule_ground_truth", "") or "[empty]"
        if passed:
            advice = "Verifier: PASS (omni-rule)."
        else:
            advice = f"Verifier: FAIL (omni-rule). {self._generic_failure_hint()}"
        return self._build_message(
            agent=agent,
            passed=passed,
            final_answer=final_answer,
            ground_truth=normalized_ground_truth,
            advice=advice,
            rule_prediction=row.get("rule_prediction", ""),
            rule_ground_truth=row.get("rule_ground_truth", ""),
        )


class OmniJudgeBasePerEvaluator(BasePerEvaluator):
    box_required: ClassVar[bool] = False

    omni_judge_model_path: str = DEFAULT_OMNI_JUDGE_MODEL_ID
    omni_judge_max_new_tokens: int = 300
    omni_judge_device: str = "auto"
    omni_judge_dtype: str = "auto"

    async def astep(
        self,
        agent: EvaluatorAgent | None,
        solution: List[SolverMessage],
        result: List[ExecutorMessage],
        task_description: str,
        all_role_description: List[str],
        *args,
        **kwargs,
    ) -> EvaluatorMessage:
        candidate, final_answer, ground_truth_final, _ = self._prepare_candidate(
            kwargs.get("reviewer_output", ""),
            kwargs.get("ground_truth", ""),
            agent,
        )

        evaluator = get_cached_omni_judge_evaluator(
            model_path=self.omni_judge_model_path,
            max_new_tokens=self.omni_judge_max_new_tokens,
            device_preference=self.omni_judge_device,
            dtype_preference=self.omni_judge_dtype,
        )
        row = evaluator.evaluate(
            [
                {
                    "problem": task_description,
                    "answer": str(kwargs.get("ground_truth", "")),
                    "model_generation": candidate,
                }
            ]
        )[0]
        passed = bool(row.get("correctness"))
        parsed = row.get("omni_judge_parsed", {}) or {}
        equivalence = parsed.get("Equivalence Judgement", "").strip() or (
            "TRUE" if passed else "FALSE"
        )

        if passed:
            advice = f"Verifier: PASS ({self.mode_name}; equivalence={equivalence})."
        else:
            advice = f"Verifier: FAIL ({self.mode_name}). {self._generic_failure_hint()}"
        return self._build_message(
            agent=agent,
            passed=passed,
            final_answer=final_answer,
            ground_truth=ground_truth_final,
            advice=advice,
            omni_judge=row.get("omni_judge", ""),
            judge_student_final_answer=row.get("judge_student_final_answer", ""),
            judge_equivalence_judgement=row.get("judge_equivalence_judgement", ""),
            judge_device=row.get("judge_device"),
            judge_dtype=row.get("judge_dtype"),
        )


@evaluator_registry.register("omni-judge")
class OmniJudgePerEvaluator(OmniJudgeBasePerEvaluator):
    mode_name: ClassVar[str] = "omni-judge"


@evaluator_registry.register("omni-verifier")
class OmniVerifierPerEvaluator(OmniJudgeBasePerEvaluator):
    mode_name: ClassVar[str] = "omni-verifier"


@evaluator_registry.register("agent")
@evaluator_registry.register("llm")
@evaluator_registry.register("openai")
class LlmPerEvaluator(BasePerEvaluator):
    mode_name: ClassVar[str] = "llm"
    box_required: ClassVar[bool] = True

    async def astep(
        self,
        agent: EvaluatorAgent | None,
        solution: List[SolverMessage],
        result: List[ExecutorMessage],
        task_description: str,
        all_role_description: List[str],
        *args,
        **kwargs,
    ) -> EvaluatorMessage:
        if agent is None:
            raise ValueError(
                "PER evaluator type 'llm' requires an `agent_type: evaluator` entry in the task config."
            )

        submitted_candidate = str(
            kwargs.get("submitted_candidate", kwargs.get("reviewer_output", "")) or ""
        )
        ground_truth = str(kwargs.get("ground_truth", "") or "")
        reference_solution = str(kwargs.get("reference_solution", "") or "").strip()
        debug_stage_prefix = str(kwargs.get("debug_stage_prefix", "") or "").strip()
        include_hint_on_fail = bool(kwargs.get("include_hint_on_fail", True))
        previous_answer_summary = str(kwargs.get("previous_answer_summary", "") or "").strip()
        previous_reasoning_summary = str(
            kwargs.get("previous_reasoning_summary", "") or ""
        ).strip()
        current_reasoning_summary = str(
            kwargs.get("current_reasoning_summary", "") or ""
        ).strip()
        candidate, final_answer, ground_truth_final, failure = self._prepare_candidate(
            submitted_candidate,
            ground_truth,
            agent,
        )
        if failure is not None:
            return failure

        judge_submission = (
            "Current extracted MAS final answer for this evaluator attempt:\n"
            f"{candidate}"
        )
        result_blocks = [f"Ground truth final answer:\n{ground_truth}"]
        if reference_solution:
            result_blocks.append(
                "Reference solution for evaluator-only use. "
                "Use it to judge equivalence and diagnose mismatch type, but do not reveal, quote, or restate it:\n"
                f"{reference_solution}"
            )
        result_text = "\n\n".join(result_blocks)

        agent.current_debug_prompt_stage = (
            f"{debug_stage_prefix}_verdict" if debug_stage_prefix else "evaluation_verdict"
        )
        evaluation = await agent.astep(
            solution=judge_submission,
            result=result_text,
            task_description=task_description,
            all_role_description="\n".join(all_role_description),
            use_memory=False,
        )
        passed = _coerce_score_to_bool(evaluation.score)
        evaluator_feedback = str(evaluation.advice or "").strip()
        verdict_advice = (
            evaluator_feedback
            or (
                "Answer judged equivalent to the reference answer."
                if passed
                else f"The answer is not equivalent to the reference answer. {self._generic_failure_hint()}"
            )
        )
        hint_advice = ""
        if passed:
            advice = verdict_advice
        elif not include_hint_on_fail:
            advice = verdict_advice
        else:
            hint_submission = (
                f"{judge_submission}\n\n"
                "The boxed final answer above has already been judged incorrect.\n"
                "Generate a more useful non-leaking repair hint.\n"
                "Do NOT change or reconsider the correctness verdict."
            )
            hint_blocks = result_blocks + [
                "Correctness is already fixed to FAIL based on the boxed final answer above. "
                "Use the reference material and the structured context only to diagnose the mismatch type "
                "and produce a brief non-leaking repair hint."
            ]
            hint_history = _build_hint_history(
                previous_answer_summary=previous_answer_summary,
                previous_reasoning_summary=previous_reasoning_summary,
                current_answer=candidate,
                current_reasoning=(
                    current_reasoning_summary
                    or _compact_hint_context(submitted_candidate)
                ),
            )
            agent.current_debug_prompt_stage = (
                f"{debug_stage_prefix}_hint" if debug_stage_prefix else "evaluation_hint"
            )
            hint_evaluation = await agent.astep(
                solution=hint_submission,
                result="\n\n".join(hint_blocks),
                task_description=task_description,
                all_role_description="\n".join(all_role_description),
                use_memory=False,
                forced_history=hint_history,
            )
            hint_feedback = str(hint_evaluation.advice or "").strip()
            hint_advice = (
                hint_feedback
                or evaluator_feedback
                or f"The answer is not equivalent to the reference answer. {self._generic_failure_hint()}"
            )
            advice = hint_advice
        message = self._build_message(
            agent=agent,
            passed=passed,
            final_answer=final_answer,
            ground_truth=ground_truth_final,
            advice=advice,
            verdict_advice=verdict_advice,
            hint_advice=hint_advice,
            used_hint=bool(hint_advice),
            extra_references=[reference_solution],
            raw_feedback_suppressed=not bool(evaluator_feedback),
            reference_solution_used=bool(reference_solution),
            judge_student_final_answer=final_answer,
            judge_equivalence_judgement="TRUE" if passed else "FALSE",
        )
        return message
