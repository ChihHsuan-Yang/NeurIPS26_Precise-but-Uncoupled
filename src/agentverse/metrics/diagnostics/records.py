"""Records for supplemental trace diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from agentverse.metrics.repair import ReviewEpisode, TraceExample


@dataclass
class ProcessExampleRecord:
    trace_path: str
    task: str
    protocol: str
    example_idx: int
    review_events: int = 0
    agreement_events: int = 0
    agreement_with_later_label: int = 0
    agreement_followed_by_fail: int = 0
    agreement_followed_by_pass: int = 0
    adjacent_submission_pairs: int = 0
    repeated_adjacent_answers: int = 0
    fail_with_next_count: int = 0
    no_change_after_fail: int = 0
    repeated_candidate_signal: bool = False
    repeated_route_signal: bool = False
    looping_signal: bool = False
    distinct_candidate_count: int = 0
    turn_imbalance: float | None = None
    turn_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DiagnosticsDataset:
    examples: list[TraceExample]
    review_episodes: list[ReviewEpisode]
    process_records: list[ProcessExampleRecord]
    trace_files: list[str]
