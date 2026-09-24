#AgentVerse/agentverse/environments/tasksolving_env/rules/evaluator/groundtruth.py
from __future__ import annotations

import re
from typing import TYPE_CHECKING, List

from agentverse.evaluation.common import extract_boxed
from . import evaluator_registry
from .base import BaseEvaluator
from agentverse.message import EvaluatorMessage

if TYPE_CHECKING:
    from agentverse.agents import EvaluatorAgent
    from agentverse.message import SolverMessage, ExecutorMessage

_NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")

def _extract_final_number(text: str) -> str:
    if not text:
        return ""
    # Prefer the last \boxed{...}
    boxed = extract_boxed(text)
    if boxed:
        cand = boxed.strip()
        # If boxed contains extra text, grab last number inside it
        nums = _NUM_RE.findall(cand)
        return nums[-1] if nums else cand

    # Fallback: last number anywhere
    nums = _NUM_RE.findall(text)
    return nums[-1] if nums else ""

def _normalize(s: str) -> str:
    return str(s).strip().replace(",", "")

@evaluator_registry.register("groundtruth")
class GroundTruthEvaluator(BaseEvaluator):
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
        # Use solver output as the "final answer" source (your pipeline writes final plan)
        flatten_solution = "\n".join([s.content for s in solution])

        pred = _normalize(_extract_final_number(flatten_solution))
        gt = _normalize(kwargs.get("ground_truth", ""))  # <-- use passed ground truth

        # If agent.environment is not attached, we can also accept kwargs
        if not gt:
            gt = _normalize(kwargs.get("ground_truth", ""))

        correct = 1 if (pred != "" and gt != "" and pred == gt) else 0

        advice = []
        advice.append(f"pred={pred!r} gt={gt!r}")
        if correct == 0:
            advice.append("Mismatch. Check arithmetic and ensure final answer is formatted as \\boxed{answer}.")
        else:
            advice.append("Correct.")

        return EvaluatorMessage(score=correct, advice="\n".join(advice))
