"""Main-paper trace metrics.

This package is intentionally narrower than the exploratory trace diagnostics.
It computes only the must-have basic, outer-loop, and inner-loop metrics used
for the paper-facing analysis.
"""

from agentverse.metrics.main_paper.basic import summarize_basic_metrics
from agentverse.metrics.main_paper.evaluator_feedback import (
    extract_evaluator_feedback_episode_bundle,
    extract_evaluator_feedback_episodes,
    extract_system_feedback_episode_bundle,
    extract_system_feedback_episodes,
    summarize_evaluator_feedback_incorporation,
    summarize_system_feedback_incorporation,
)
from agentverse.metrics.main_paper.inputs import load_main_paper_dataset
from agentverse.metrics.main_paper.inner_loop import summarize_inner_loop_metrics
from agentverse.metrics.main_paper.outer_loop import (
    build_outer_loop_problem_scores,
    summarize_hint_comparison,
    summarize_outer_loop_metrics,
)
from agentverse.metrics.main_paper.records import (
    AttemptRecord,
    InnerTransitionRecord,
    MainPaperDataset,
    MetricRecord,
    ProblemRecord,
)

__all__ = [
    "AttemptRecord",
    "InnerTransitionRecord",
    "MainPaperDataset",
    "MetricRecord",
    "ProblemRecord",
    "build_outer_loop_problem_scores",
    "extract_evaluator_feedback_episode_bundle",
    "extract_evaluator_feedback_episodes",
    "extract_system_feedback_episode_bundle",
    "extract_system_feedback_episodes",
    "load_main_paper_dataset",
    "summarize_basic_metrics",
    "summarize_evaluator_feedback_incorporation",
    "summarize_system_feedback_incorporation",
    "summarize_hint_comparison",
    "summarize_inner_loop_metrics",
    "summarize_outer_loop_metrics",
]
