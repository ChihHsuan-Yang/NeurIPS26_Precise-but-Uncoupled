#AgentVerse/agentverse/environments/tasksolving_env/rules/evaluator/basic.py
from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple

from agentverse.evaluation.common import extract_final_answer

from . import evaluator_registry
from .base import BaseEvaluator

if TYPE_CHECKING:
    from agentverse.agents import EvaluatorAgent
    from agentverse.message import EvaluatorMessage, SolverMessage, ExecutorMessage


def _flatten_messages(messages: List[SolverMessage] | List[ExecutorMessage]) -> str:
    return "\n".join([getattr(message, "content", "") for message in messages])


def _build_reference_aware_inputs(
    flatten_solution: str,
    flatten_result: str,
    ground_truth: str = "",
    reference_solution: str = "",
) -> Tuple[str, str]:
    """Add evaluator-only references for benchmark modes that provide them.

    We keep the old prompt inputs unchanged unless a reference solution exists.
    This gives Omni-MATH-2 style benchmark modes answer-specific hints without
    perturbing older open-ended task configs.
    """
    reference_solution = str(reference_solution or "").strip()
    if not reference_solution:
        return flatten_solution, flatten_result

    candidate_source = (flatten_result or flatten_solution or "").strip()
    current_final_answer = extract_final_answer(candidate_source)
    solution_for_evaluator = (
        "Current extracted MAS final answer for this evaluator attempt:\n"
        f"{current_final_answer or '[no extracted final answer]'}\n\n"
        "Full current MAS output:\n"
        f"{candidate_source or '[empty]'}\n\n"
        "Planner / solution context:\n"
        f"{flatten_solution or '[empty]'}"
    )

    result_blocks = []
    if flatten_result:
        result_blocks.append(f"Execution/result text:\n{flatten_result}")
    if ground_truth:
        result_blocks.append(f"Ground truth final answer:\n{ground_truth}")
    result_blocks.append(
        "Reference solution for evaluator-only use. "
        "Use it to judge equivalence and diagnose mismatch type, but do not "
        "reveal, quote, or restate it:\n"
        f"{reference_solution}"
    )
    return solution_for_evaluator, "\n\n".join(result_blocks)


@evaluator_registry.register("basic")
class BasicEvaluator(BaseEvaluator):
    cnt_agents: int = 0

    async def astep(
        self,
        agent: EvaluatorAgent,
        solution: List[SolverMessage],
        result: List[ExecutorMessage],
        task_description: str,
        all_role_description: List[str],
        *args,
        **kwargs,
    ) -> EvaluatorMessage:
        flatten_solution = _flatten_messages(solution)
        flatten_result = _flatten_messages(result)
        flatten_solution, flatten_result = _build_reference_aware_inputs(
            flatten_solution,
            flatten_result,
            ground_truth=str(kwargs.get("ground_truth", "") or ""),
            reference_solution=str(kwargs.get("reference_solution", "") or ""),
        )
        flatten_all_role_description = "\n".join(all_role_description)
        evaluation = await agent.astep(
            flatten_solution,
            flatten_result,
            task_description,
            flatten_all_role_description,
        )
        return evaluation


@evaluator_registry.register("basic-message")
class BasicEvaluator(BaseEvaluator):
    cnt_agents: int = 0

    async def astep(
        self,
        agent: EvaluatorAgent,
        solution: List[SolverMessage],
        result: List[ExecutorMessage],
        task_description: str,
        all_role_description: List[str],
        *args,
        **kwargs,
    ) -> EvaluatorMessage:
        flatten_solution = _flatten_messages(solution)
        flatten_result = _flatten_messages(result)
        flatten_solution, flatten_result = _build_reference_aware_inputs(
            flatten_solution,
            flatten_result,
            ground_truth=str(kwargs.get("ground_truth", "") or ""),
            reference_solution=str(kwargs.get("reference_solution", "") or ""),
        )
        flatten_all_role_description = "\n".join(all_role_description)
        agent.add_message_to_memory(result)
        evaluation = await agent.astep(
            flatten_solution,
            flatten_result,
            task_description,
            flatten_all_role_description,
        )
        agent.add_message_to_memory([evaluation])
        return evaluation
