"""Data models for structured metrics collection."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class AgentMeta:
    """Agent metadata for trace headers (duplicated from tracing/)."""

    agent_number: str
    name: str
    model: str
    llm_type: str
    pre_prompt: str


@dataclass
class TokenUsage:
    """Accumulated token counts."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def add(self, prompt: int, completion: int) -> None:
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_tokens += prompt + completion

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )

    def __iadd__(self, other: TokenUsage) -> TokenUsage:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.total_tokens += other.total_tokens
        return self

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass
class AgentMetrics:
    """Per-agent metrics for one example."""

    agent_name: str
    agent_type: str = ""
    model: str = ""
    tokens: TokenUsage = field(default_factory=TokenUsage)
    message_count: int = 0
    invocation_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


@dataclass
class RoundMetrics:
    """Per-round metrics within one example."""

    round_id: int
    evaluation_passed: bool | None = None
    agent_messages: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ConfidencePollRecord:
    """One confidence-poll record for one agent at one discussion turn."""

    outer_round_id: int
    discussion_turn: int
    agent_name: str
    score: int
    reason: str = ""
    intent: str = ""
    candidate_answer: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExampleMetrics:
    """Complete metrics for one example."""

    example_idx: int
    input_text: str
    label: str
    predicted_answer: str = ""
    correct: int = 0  # 0 or 1

    total_tokens: TokenUsage = field(default_factory=TokenUsage)
    total_messages: int = 0
    total_rounds: int = 0
    system_try_count: int = 0
    first_system_try_correct: bool = False
    first_success_system_try: int | None = None
    system_try_results: list[bool] = field(default_factory=list)
    evaluator_attempt_count: int = 0
    first_success_attempt: int | None = None
    evaluation_attempt_results: list[bool] = field(default_factory=list)
    total_model_calls: int = 0

    agent_metrics: dict[str, AgentMetrics] = field(default_factory=dict)
    round_metrics: list[RoundMetrics] = field(default_factory=list)
    confidence_poll_count: int = 0
    confidence_scores: list[ConfidencePollRecord] = field(default_factory=list)

    wall_time_seconds: float = 0.0
    first_round_correct: bool = False
    reviewer_agree_count: int = 0
    reviewer_reroute_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        d = {
            "example_idx": self.example_idx,
            "input_text": self.input_text,
            "label": self.label,
            "predicted_answer": self.predicted_answer,
            "correct": self.correct,
            "total_tokens": self.total_tokens.to_dict(),
            "total_messages": self.total_messages,
            "total_rounds": self.total_rounds,
            "system_try_count": self.system_try_count,
            "first_system_try_correct": self.first_system_try_correct,
            "first_success_system_try": self.first_success_system_try,
            "system_try_results": self.system_try_results,
            "evaluator_attempt_count": self.evaluator_attempt_count,
            "first_success_attempt": self.first_success_attempt,
            "evaluation_attempt_results": self.evaluation_attempt_results,
            "total_model_calls": self.total_model_calls,
            "first_round_correct": self.first_round_correct,
            "reviewer_agree_count": self.reviewer_agree_count,
            "reviewer_reroute_count": self.reviewer_reroute_count,
            "confidence_poll_count": self.confidence_poll_count,
            "confidence_scores": [item.to_dict() for item in self.confidence_scores],
            "agent_metrics": {k: v.to_dict() for k, v in self.agent_metrics.items()},
            "round_metrics": [r.to_dict() for r in self.round_metrics],
            "wall_time_seconds": self.wall_time_seconds,
        }
        return d


@dataclass
class RunSummary:
    """Aggregate metrics over all examples in a benchmark run."""

    run_id: str
    task: str
    model: str = ""
    total_examples: int = 0
    accuracy: float = 0.0
    first_round_accuracy: float = 0.0

    total_tokens: TokenUsage = field(default_factory=TokenUsage)
    avg_rounds: float = 0.0
    avg_messages: float = 0.0
    avg_tokens: TokenUsage = field(default_factory=TokenUsage)
    total_model_calls: int = 0
    avg_model_calls: float = 0.0
    avg_evaluator_attempts: float = 0.0
    avg_attempts_to_success: float = 0.0
    pass_at_attempt: dict[str, float] = field(default_factory=dict)
    solved_by_attempt_count: dict[str, int] = field(default_factory=dict)
    total_confidence_polls: int = 0
    avg_confidence_score: float = 0.0

    per_agent_summary: dict[str, AgentMetrics] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = {
            "run_id": self.run_id,
            "task": self.task,
            "model": self.model,
            "total_examples": self.total_examples,
            "accuracy": round(self.accuracy, 4),
            "first_round_accuracy": round(self.first_round_accuracy, 4),
            "total_tokens": self.total_tokens.to_dict(),
            "avg_rounds": round(self.avg_rounds, 2),
            "avg_messages": round(self.avg_messages, 2),
            "avg_tokens": self.avg_tokens.to_dict(),
            "total_model_calls": self.total_model_calls,
            "avg_model_calls": round(self.avg_model_calls, 2),
            "avg_evaluator_attempts": round(self.avg_evaluator_attempts, 2),
            "avg_attempts_to_success": round(self.avg_attempts_to_success, 2),
            "pass_at_attempt": self.pass_at_attempt,
            "solved_by_attempt_count": self.solved_by_attempt_count,
            "total_confidence_polls": self.total_confidence_polls,
            "avg_confidence_score": round(self.avg_confidence_score, 2),
            "per_agent_summary": {k: v.to_dict() for k, v in self.per_agent_summary.items()},
        }
        return d
