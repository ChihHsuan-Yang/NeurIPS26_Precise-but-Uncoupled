"""Records for reviewer/evaluator collaboration decomposition metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from agentverse.metrics.repair import ReviewEpisode, TraceExample


@dataclass
class ReviewerFeedbackIncorporationEpisode:
    """One reviewer-feedback response event.

    This record asks whether agents acted according to the reviewer feedback,
    separately from whether the resulting answer became correct.
    """

    trace_path: str
    task: str
    protocol: str
    example_idx: int
    episode_id: str
    review_turn: int
    reviewer: str
    action: str
    candidate_before: str
    candidate_after: str
    pre_correct: Optional[bool]
    post_correct: Optional[bool]
    incorporated_feedback: Optional[bool]
    incorporated_feedback_source: str
    evidence_turns: list[int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CollaborationDecompositionDataset:
    """Parsed data used by the collaboration-decomposition CLI."""

    examples: list[TraceExample]
    review_episodes: list[ReviewEpisode]
    reviewer_feedback_episodes: list[ReviewerFeedbackIncorporationEpisode]
    trace_files: list[str] = field(default_factory=list)
    correctness_evaluator_type: str = ""
    reviewer_feedback_labeler_mode: str = ""
    analysis_accounting: dict[str, Any] = field(default_factory=dict)
