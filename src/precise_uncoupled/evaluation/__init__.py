"""Evaluator / verifier contract -- facade over `agentverse.evaluation`.

The paper's four primary configs use `evaluator: {type: llm}` -- the same
`openai/gpt-oss-120b` checkpoint acting as verifier, held fixed across protocols.
The rule-based and Omni-Judge evaluators below are OPTIONAL alternatives that the
paper does not use on its default path; `OmniJudgeEvaluator` additionally needs
`torch` + `transformers` and downloads the public model `KbsdJames/Omni-Judge`.
They are imported lazily so the base install stays light.

Single definition rule: these names are re-exported, never redefined.
"""

from agentverse.evaluation.common import (  # noqa: F401
    exact_match,
    extract_boxed,
    extract_final_answer,
    parse_omni_judge_report,
)
from agentverse.evaluation.content_based import (  # noqa: F401
    ContentJudgeResult,
    TraceAnalysisLabeler,
    normalize_candidate_text,
)

__all__ = [
    "ContentJudgeResult",
    "TraceAnalysisLabeler",
    "exact_match",
    "extract_boxed",
    "extract_final_answer",
    "normalize_candidate_text",
    "parse_omni_judge_report",
]


def load_omni_rule_evaluator():
    """Optional rule-based Omni-MATH evaluator (needs sympy/latex2sympy2)."""
    from agentverse.evaluation.omni_rule import OmniRuleEvaluator
    return OmniRuleEvaluator


def load_omni_judge_evaluator():
    """Optional Omni-Judge evaluator (needs torch + transformers; public model
    `KbsdJames/Omni-Judge`). NOT used by the paper's primary configs."""
    from agentverse.evaluation.omni_judge import OmniJudgeEvaluator
    return OmniJudgeEvaluator
