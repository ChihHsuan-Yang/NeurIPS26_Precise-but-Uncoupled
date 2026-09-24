from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from agentverse.evaluation.common import extract_boxed
from agentverse.environments.tasksolving_env.rules.evaluator import (
    BaseEvaluator,
    evaluator_registry,
)
from agentverse.environments.tasksolving_env.rules.failure_memory import (
    CandidateRevisionMemory,
    FailedAttemptMemory,
    with_authority_guide,
)
from agentverse.environments.tasksolving_env.rules.per import (
    _append_log,
    _coerce_evaluator_score,
    _format_evaluator_hint_only,
    _format_evaluator_verdict_only,
    _format_reviewer_feedback,
    prepare_per_rule_kwargs,
)
from agentverse.utils import AGENT_TYPES


def prepare_single_agent_reflect_rule_kwargs(
    rule_config: Dict[str, Any] | None,
) -> Dict[str, Any]:
    return prepare_per_rule_kwargs(rule_config)


def _normalize_seed_text(text: str) -> str:
    raw = str(text or "").strip()
    if raw in {"No solution yet.", "No advice yet.", "[No candidate yet]"}:
        return ""
    return raw


def _boxed_only(candidate_text: str) -> str:
    boxed = extract_boxed(candidate_text or "")
    return f"\\boxed{{{boxed}}}" if boxed else "[No boxed answer]"


def _submitted_final_answer_for_trace(candidate_text: str) -> str:
    boxed = extract_boxed(candidate_text or "")
    return f"\\boxed{{{boxed}}}" if boxed else "[No extracted final answer]"


def _compact(text: str, max_chars: int = 320) -> str:
    cleaned = " ".join(str(text or "").split()).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3].rstrip() + "..."


def _build_attempt_repair_summary(submitted_answer: str, eval_feedback: str) -> str:
    answer_text = _compact(submitted_answer, max_chars=180) or "[no extracted answer]"
    feedback_text = _compact(eval_feedback, max_chars=260) or "[no evaluator feedback]"
    return (
        f"Rejected answer: {answer_text}\n"
        f"Evaluator feedback to address: {feedback_text}\n"
        "Next attempt goal: produce a new complete answer that addresses the evaluator "
        "feedback and do not repeat the same rejected answer unless the mismatch has "
        "been resolved."
    )


class SingleAgentReflectRule:
    """Single-Agent Iterative outer-loop solver with shared evaluator and ledgers."""

    def __init__(self, **kwargs):
        evaluator_config = dict(kwargs.pop("evaluator_config", {}) or {})
        self.evaluator_type = str(evaluator_config.pop("type", "numeric-verifier"))
        self.evaluator: BaseEvaluator = evaluator_registry.build(
            self.evaluator_type, **evaluator_config
        )

        feedback_mode = str(
            kwargs.pop("reviewer_feedback_mode", kwargs.pop("evaluator_feedback_mode", "plain"))
        ).strip().lower()
        if feedback_mode not in {"plain", "hint"}:
            feedback_mode = "plain"
        self.reviewer_feedback_mode: str = feedback_mode

        self.failed_attempt_memory_limit: int = int(
            kwargs.pop("failed_attempt_memory_limit", 3)
        )
        self.failed_attempt_memory = FailedAttemptMemory(self.failed_attempt_memory_limit)
        self.candidate_revision_memory_limit: int = int(
            kwargs.pop("candidate_revision_memory_limit", 5)
        )
        self.candidate_revision_memory = CandidateRevisionMemory(
            self.candidate_revision_memory_limit
        )
        self.reset_solver_memory_each_attempt: bool = bool(
            kwargs.pop("reset_solver_memory_each_attempt", True)
        )

    def reset(self) -> None:
        self.failed_attempt_memory.reset()
        self.candidate_revision_memory.reset()
        try:
            self.evaluator.reset()
        except Exception:
            pass

    def _advice_with_question_memory(self, advice: str = "") -> str:
        parts: List[str] = []
        if str(advice or "").strip():
            parts.append(
                "Latest Evaluator Feedback From The Previous Attempt:\n"
                f"{str(advice).strip()}"
            )
        if self.candidate_revision_memory.records:
            parts.append(self.candidate_revision_memory.render())
        if self.failed_attempt_memory.records:
            parts.append(self.failed_attempt_memory.render())
        combined = "\n\n".join(part for part in parts if str(part).strip()).strip()
        return with_authority_guide(combined) if combined else ""

    def _record_candidate_event(
        self,
        logs: List[Dict[str, Any]],
        *,
        round_id: int,
        stage: str,
        source: str,
        action: str,
        candidate_text: str = "",
        rationale_or_review: str = "",
        parent_event_id: Optional[int] = None,
        evaluator_signal: str = "",
        live_log_sink: Any = None,
    ):
        record = self.candidate_revision_memory.add(
            stage=stage,
            source=source,
            action=action,
            candidate_answer=candidate_text,
            rationale_or_review=rationale_or_review,
            parent_event_id=parent_event_id,
            evaluator_signal=evaluator_signal,
        )
        if record is None:
            return None
        _append_log(
            logs,
            {
                "type": "summary",
                "round": round_id,
                "stage": f"{stage}_candidate_memory",
                "sender": "system",
                "content": f"C{record.event_id} {record.action} {record.candidate_answer}",
                "candidate_memory_event": record.to_dict(),
                "trace_visible": False,
            },
            live_log_sink,
        )
        return record

    async def astep(
        self,
        task_description: str,
        agents: Dict[Any, Any],
        advice: str = "No advice yet.",
        previous_plan: str = "No solution yet.",
        ground_truth: str = "",
        reference_solution: str = "",
        round_id: int = 0,
        live_log_sink: Any = None,
        **kwargs,
    ) -> Tuple[str, str, str, List[Dict[str, Any]], bool]:
        logs: List[Dict[str, Any]] = []

        solver = agents.get(AGENT_TYPES.SOLVER, None)
        evaluator_agent = agents.get(AGENT_TYPES.EVALUATION, None)
        if solver is None:
            raise ValueError(
                "single-agent-reflect requires one solver agent and one evaluator config."
            )

        if self.reset_solver_memory_each_attempt:
            try:
                solver.reset()
            except Exception:
                pass
        if evaluator_agent is not None:
            try:
                evaluator_agent.reset()
            except Exception:
                pass

        normalized_advice = _normalize_seed_text(advice)
        former_solution = _normalize_seed_text(previous_plan)
        solver_advice = self._advice_with_question_memory(normalized_advice)
        is_first_attempt = (
            not former_solution
            and not normalized_advice
            and not self.failed_attempt_memory.records
            and not self.candidate_revision_memory.records
        )

        solver_msg = await solver.astep(
            former_solution=former_solution,
            previous_plan=former_solution,
            current_candidate="",
            current_final_answer=_boxed_only(former_solution),
            phase="attempt",
            phase_instruction=(
                "Solve the problem directly. Use question-scoped memory only to avoid "
                "repeating rejected answers and to incorporate the latest evaluator "
                "feedback. Produce a full solution and end with EXACTLY one final line "
                "`\\boxed{<answer>}`."
            ),
            advice=solver_advice,
            task_description=task_description,
            use_first_attempt_prompt=is_first_attempt,
        )
        solver_text = getattr(solver_msg, "content", "") or "[No solver output]"
        current_candidate = solver_text.strip() or "[No solver output]"

        _append_log(
            logs,
            {
                "type": "message",
                "round": round_id,
                "stage": "solver",
                "sender": getattr(solver_msg, "sender", "Solo"),
                "content": solver_text,
            },
            live_log_sink,
        )
        _append_log(
            logs,
            {
                "type": "message",
                "round": round_id,
                "stage": "evaluation_submission",
                "sender": "system",
                "content": (
                    "Actual extracted final answer sent to evaluator: "
                    f"{_submitted_final_answer_for_trace(current_candidate)}"
                ),
            },
            live_log_sink,
        )
        proposal_record = self._record_candidate_event(
            logs,
            round_id=round_id,
            stage="solver",
            source=getattr(solver_msg, "sender", "Solo"),
            action="propose",
            candidate_text=current_candidate,
            rationale_or_review=solver_text,
            live_log_sink=live_log_sink,
        )
        submission_record = self._record_candidate_event(
            logs,
            round_id=round_id,
            stage="submission",
            source="system",
            action="submit",
            candidate_text=current_candidate,
            rationale_or_review="Submitted to evaluator.",
            parent_event_id=(proposal_record.event_id if proposal_record is not None else None),
            live_log_sink=live_log_sink,
        )

        evaluation = await self.evaluator.astep(
            agent=evaluator_agent,
            solution=[],
            result=[],
            task_description=task_description,
            all_role_description=[getattr(solver, "role_description", "SingleAgentIterative")],
            reviewer_output=current_candidate,
            submitted_candidate=current_candidate,
            ground_truth=str(ground_truth),
            reference_solution=str(reference_solution or ""),
            debug_stage_prefix="evaluation",
            previous_answer_summary=self.failed_attempt_memory.render_for_evaluator_hint(),
            previous_reasoning_summary=self.candidate_revision_memory.render_for_evaluator_hint(),
            current_reasoning_summary=solver_text,
        )
        eval_result = (
            dict(evaluation.content) if isinstance(evaluation.content, dict) else {}
        )
        passed = _coerce_evaluator_score(evaluation.score)
        eval_result.setdefault("mode", self.evaluator_type)
        eval_result.setdefault("passed", passed)
        eval_result.setdefault("signal", "PASS" if passed else "FAIL")
        eval_result.setdefault("final_answer", "")
        eval_advice = str(evaluation.advice or "")
        direct_protocol_fail = bool(eval_result.get("direct_protocol_fail"))
        verdict_advice = str(eval_result.get("verdict_advice") or eval_advice or "").strip()
        hint_advice = str(eval_result.get("hint_advice") or "").strip()
        if direct_protocol_fail:
            eval_sender = "system"
        else:
            eval_sender = getattr(evaluation, "sender", "") or getattr(
                evaluator_agent, "name", "Evaluator"
            )
        eval_feedback = _format_reviewer_feedback(
            eval_result["mode"],
            passed,
            hint_advice or verdict_advice or eval_advice,
            self.reviewer_feedback_mode,
        )

        eval_log_content = _format_evaluator_verdict_only(
            eval_result["mode"],
            passed,
        )
        if eval_result.get("judge_device"):
            eval_log_content += (
                f"\nJudge student final answer: {eval_result.get('judge_student_final_answer')}"
                f"\nJudge equivalence judgement: {eval_result.get('judge_equivalence_judgement')}"
                f"\nJudge device: {eval_result.get('judge_device')}"
                f"\nJudge dtype: {eval_result.get('judge_dtype')}"
            )

        _append_log(
            logs,
                {
                    "type": "message",
                    "round": round_id,
                    "stage": "evaluation",
                    "sender": eval_sender,
                    "content": eval_log_content,
                    "force_numbered_system_trace": bool(direct_protocol_fail),
                },
                live_log_sink,
            )
        if not passed and hint_advice:
            _append_log(
                logs,
                {
                    "type": "message",
                    "round": round_id,
                    "stage": "evaluation_hint",
                    "sender": eval_sender,
                    "content": _format_evaluator_hint_only(hint_advice),
                    "force_numbered_system_trace": bool(direct_protocol_fail),
                },
                live_log_sink,
            )
        _append_log(
            logs,
            {
                "type": "meta",
                "round": round_id,
                "stage": "evaluation_result",
                "sender": eval_sender,
                "content": "PASS" if passed else "FAIL",
                "mode": eval_result["mode"],
                "signal": eval_result.get("signal", "PASS" if passed else "FAIL"),
                "correctness": 1 if passed else 0,
                "final_answer": eval_result.get("final_answer", ""),
                "advice": eval_feedback,
                "verdict_advice": verdict_advice,
                "hint_advice": hint_advice,
                "reviewer_feedback_mode": self.reviewer_feedback_mode,
                "reference_solution_used": bool(eval_result.get("reference_solution_used")),
                "judge_student_final_answer": eval_result.get("judge_student_final_answer", ""),
                "judge_equivalence_judgement": eval_result.get("judge_equivalence_judgement", ""),
                "judge_device": eval_result.get("judge_device"),
                "judge_dtype": eval_result.get("judge_dtype"),
                "rule_prediction": eval_result.get("rule_prediction", ""),
            },
            live_log_sink,
        )
        eval_final_answer = str(eval_result.get("final_answer", "") or "").strip()
        self._record_candidate_event(
            logs,
            round_id=round_id,
            stage="evaluation",
            source=eval_sender,
            action="evaluate",
            candidate_text=(
                f"\\boxed{{{eval_final_answer}}}" if eval_final_answer else current_candidate
            ),
            rationale_or_review=eval_feedback,
            parent_event_id=(
                submission_record.event_id if submission_record is not None else None
            ),
            evaluator_signal=eval_result.get("signal", "PASS" if passed else "FAIL"),
            live_log_sink=live_log_sink,
        )

        if not passed:
            submitted_answer = eval_final_answer or current_candidate
            repair_summary = _build_attempt_repair_summary(
                submitted_answer=submitted_answer,
                eval_feedback=eval_feedback,
            )
            self.failed_attempt_memory.add(
                submitted_answer=submitted_answer,
                evaluator_signal=str(eval_result.get("signal", "FAIL") or "FAIL"),
                evaluator_hint=eval_feedback,
                repair_summary=repair_summary,
                source_stage="evaluation",
            )

        _append_log(
            logs,
            {
                "type": "meta",
                "round": round_id,
                "stage": "system",
                "sender": "system",
                "content": "Good score! Accept!" if passed else "Bad score! Reject!",
            },
            live_log_sink,
        )

        final_result = current_candidate
        final_advice = eval_feedback
        final_plan = current_candidate
        return final_result, final_advice, final_plan, logs, passed
