from __future__ import annotations

from typing import Any, Dict, List, Tuple

from agentverse.evaluation.common import extract_boxed
from agentverse.environments.tasksolving_env.rules.evaluator import (
    BaseEvaluator,
    evaluator_registry,
)
from agentverse.environments.tasksolving_env.rules.per import (
    _append_log,
    _coerce_evaluator_score,
    _enable_full_question_history,
    _format_reviewer_feedback,
    prepare_per_rule_kwargs,
)
from agentverse.utils import AGENT_TYPES


def prepare_single_llm_benchmark_rule_kwargs(
    rule_config: Dict[str, Any] | None,
) -> Dict[str, Any]:
    return prepare_per_rule_kwargs(rule_config)


def _submitted_final_answer_for_trace(text: str) -> str:
    boxed = extract_boxed(text or "")
    return f"\\boxed{{{boxed}}}" if boxed else "[No extracted final answer]"


class SingleLLMBenchmarkRule:
    """Single-call solver baseline under the shared evaluator contract."""

    def __init__(self, **kwargs):
        evaluator_config = dict(kwargs.pop("evaluator_config", {}) or {})
        self.evaluator_type = str(evaluator_config.pop("type", "numeric-verifier"))
        self.evaluator: BaseEvaluator = evaluator_registry.build(
            self.evaluator_type, **evaluator_config
        )

        # Baseline LLM is judge-only. Even if a config still passes "hint" for
        # plumbing symmetry, the paper-facing baseline should use plain
        # evaluator feedback because the solver never consumes it.
        self.feedback_mode: str = "plain"

    def reset(self) -> None:
        try:
            self.evaluator.reset()
        except Exception:
            pass

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
                "single-llm-benchmark requires one solver agent and one evaluator config."
            )

        try:
            solver.reset()
        except Exception:
            pass
        if evaluator_agent is not None:
            try:
                evaluator_agent.reset()
            except Exception:
                pass

        if solver is not None:
            _enable_full_question_history(solver)

        solver_advice = str(advice or "").strip()
        if solver_advice in {"No advice yet.", "No solution yet."}:
            solver_advice = ""

        solver_msg = await solver.astep(
            former_solution="",
            previous_plan="",
            current_candidate="",
            current_final_answer="",
            phase="single_shot",
            phase_instruction=(
                "Produce your best one-shot solution. Infer the required answer shape "
                "from the problem statement and end with EXACTLY one final line "
                "`\\boxed{<answer>}`."
            ),
            advice=solver_advice,
            task_description=task_description,
        )
        solver_text = getattr(solver_msg, "content", "") or "[No solver output]"
        submitted_text = solver_text
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
                    f"{_submitted_final_answer_for_trace(submitted_text)}"
                ),
            },
            live_log_sink,
        )

        evaluation = await self.evaluator.astep(
            agent=evaluator_agent,
            solution=[],
            result=[],
            task_description=task_description,
            all_role_description=[getattr(solver, "role_description", "SingleLLMSolver")],
            reviewer_output=submitted_text,
            ground_truth=str(ground_truth),
            reference_solution=str(reference_solution or ""),
            debug_stage_prefix="evaluation",
            include_hint_on_fail=False,
            previous_answer_summary="",
            previous_reasoning_summary="",
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
        if direct_protocol_fail:
            eval_sender = "system"
        else:
            eval_sender = getattr(evaluation, "sender", "") or getattr(
                evaluator_agent, "name", "Evaluator"
            )
        eval_feedback = _format_reviewer_feedback(
            eval_result["mode"],
            passed,
            eval_advice,
            self.feedback_mode,
        )

        eval_log_content = eval_feedback
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
                "reviewer_feedback_mode": self.feedback_mode,
                "reference_solution_used": bool(eval_result.get("reference_solution_used")),
                "judge_student_final_answer": eval_result.get("judge_student_final_answer", ""),
                "judge_equivalence_judgement": eval_result.get("judge_equivalence_judgement", ""),
                "judge_device": eval_result.get("judge_device"),
                "judge_dtype": eval_result.get("judge_dtype"),
                "rule_prediction": eval_result.get("rule_prediction", ""),
            },
            live_log_sink,
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

        final_result = submitted_text.strip() or "[No solver output]"
        final_advice = eval_feedback
        final_plan = final_result
        return final_result, final_advice, final_plan, logs, passed
