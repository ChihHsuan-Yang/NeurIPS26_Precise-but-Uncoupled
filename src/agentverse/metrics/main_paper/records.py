"""Shared records for main-paper trace metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from agentverse.metrics.repair import FeedbackEpisode, TraceExample


@dataclass
class MetricRecord:
    section: str
    metric: str
    value: Any
    numerator: Optional[float] = None
    denominator: Optional[float] = None
    source: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AttemptRecord:
    trace_path: str
    task: str
    protocol: str
    example_idx: int
    attempt_idx: int
    evaluator_turn: int
    correct: Optional[bool]
    hint_used: bool
    feedback: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProblemRecord:
    trace_path: str
    task: str
    protocol: str
    example_idx: int
    final_passed: bool
    first_passed: bool
    first_success_attempt: Optional[int]
    correction_loops: int
    hints_used: int
    reflective_rounds: int
    reasoning_turns: int
    evaluator_submission_count: int = 0
    difficulty_tier: Any = None
    domain: Any = None
    source: Any = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model_calls: int = 0
    wall_time_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class InnerTransitionRecord:
    trace_path: str
    task: str
    protocol: str
    example_idx: int
    attempt_idx: int
    start_turn: int
    end_turn: int
    initial_candidate: str
    final_candidate: str
    candidate_count: int
    initial_correct: Optional[bool]
    final_correct: Optional[bool]
    correctness_source: str
    difficulty_tier: Any = None
    domain: Any = None

    @property
    def transition_name(self) -> str:
        if self.initial_correct is None or self.final_correct is None:
            return "unknown"
        start = "correct" if self.initial_correct else "wrong"
        end = "correct" if self.final_correct else "wrong"
        return f"{start}_to_{end}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MainPaperDataset:
    examples: list[TraceExample]
    attempts: list[AttemptRecord]
    problems: list[ProblemRecord]
    inner_transitions: list[InnerTransitionRecord]
    system_feedback_episodes: list[FeedbackEpisode] = field(default_factory=list)
    results_by_idx: dict[int, dict[str, Any]] = field(default_factory=dict)
    metrics_by_idx: dict[int, dict[str, Any]] = field(default_factory=dict)
    trace_files: list[str] = field(default_factory=list)
    evaluator_type: str = ""
    analysis_accounting: dict[str, Any] = field(default_factory=dict)

    @property
    def evaluator_feedback_episodes(self) -> list[FeedbackEpisode]:
        """Backward-compatible alias for system feedback episodes."""
        return self.system_feedback_episodes
