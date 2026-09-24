"""Paper trace metrics -- facade over the vendored `agentverse.metrics` modules.

IMPORTANT (single definition rule): this module only RE-EXPORTS. Every function
below has exactly one definition, in `src/agentverse/metrics/...`, and those files
are byte-identical to the submitted run's base commit be9c47c9. The release does
NOT ship a second copy of these functions. See docs/PROVENANCE.md section
"Duplicate-module hazard".

Main-paper metrics (Table 1, Tables 9/10/11, Figures 2/3):
    summarize_basic_metrics, summarize_inner_loop_metrics, summarize_outer_loop_metrics,
    load_main_paper_dataset, summarize_evaluator_feedback_incorporation

Reviewer-conditioned decomposition (Table 11 -- precision / coupling / repair):
    load_collaboration_decomposition_dataset, summarize_reviewer_conditioned_decomposition

Strict coupling audit (Table 3):
    summarize_strict_coupling_rate, summarize_strict_coupling_by_protocol
"""

from agentverse.metrics.main_paper import (  # noqa: F401
    AttemptRecord,
    InnerTransitionRecord,
    MainPaperDataset,
    MetricRecord,
    ProblemRecord,
    build_outer_loop_problem_scores,
    load_main_paper_dataset,
    summarize_basic_metrics,
    summarize_evaluator_feedback_incorporation,
    summarize_hint_comparison,
    summarize_inner_loop_metrics,
    summarize_outer_loop_metrics,
    summarize_system_feedback_incorporation,
)
from agentverse.metrics.collaboration_decomposition import (  # noqa: F401
    load_collaboration_decomposition_dataset,
    summarize_reviewer_conditioned_decomposition,
)
from agentverse.metrics.collaboration_decomposition.strict_coupling import (  # noqa: F401
    summarize_strict_coupling_by_protocol,
    summarize_strict_coupling_rate,
)
from agentverse.metrics.repair import (  # noqa: F401
    FeedbackEpisode,
    ReviewEpisode,
    TraceExample,
    parse_trace_file,
)

__all__ = [
    "AttemptRecord",
    "FeedbackEpisode",
    "InnerTransitionRecord",
    "MainPaperDataset",
    "MetricRecord",
    "ProblemRecord",
    "ReviewEpisode",
    "TraceExample",
    "build_outer_loop_problem_scores",
    "load_collaboration_decomposition_dataset",
    "load_main_paper_dataset",
    "parse_trace_file",
    "summarize_basic_metrics",
    "summarize_evaluator_feedback_incorporation",
    "summarize_hint_comparison",
    "summarize_inner_loop_metrics",
    "summarize_outer_loop_metrics",
    "summarize_reviewer_conditioned_decomposition",
    "summarize_strict_coupling_by_protocol",
    "summarize_strict_coupling_rate",
    "summarize_system_feedback_incorporation",
]
