"""RQ2 system-feedback incorporation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agentverse.evaluation.content_based import TraceAnalysisLabeler
from agentverse.metrics.api_accounting import (
    build_trace_labeler_accounting,
    merge_api_accounting,
)
from agentverse.evaluation.feedback_label_cache import (
    default_feedback_label_cache_path,
)
from agentverse.metrics.main_paper.records import MetricRecord
from agentverse.metrics.repair import FeedbackEpisode, TraceExample, _extract_feedback_episodes


def extract_system_feedback_episodes(
    examples: list[TraceExample],
    *,
    labeler_mode: str = "hybrid",
    llm_model: str = "openai/gpt-oss-120b",
    temperature: float = 0.0,
    max_tokens: int = 400,
    cache_feedback_labels: bool = True,
    feedback_label_cache_path: str | Path | None = None,
    base_dir: str | Path | None = None,
) -> list[FeedbackEpisode]:
    episodes, _ = extract_system_feedback_episode_bundle(
        examples,
        labeler_mode=labeler_mode,
        llm_model=llm_model,
        temperature=temperature,
        max_tokens=max_tokens,
        cache_feedback_labels=cache_feedback_labels,
        feedback_label_cache_path=feedback_label_cache_path,
        base_dir=base_dir,
    )
    return episodes


def extract_system_feedback_episode_bundle(
    examples: list[TraceExample],
    *,
    labeler_mode: str = "hybrid",
    llm_model: str = "openai/gpt-oss-120b",
    temperature: float = 0.0,
    max_tokens: int = 400,
    cache_feedback_labels: bool = True,
    feedback_label_cache_path: str | Path | None = None,
    base_dir: str | Path | None = None,
) -> tuple[list[FeedbackEpisode], dict[str, Any]]:
    resolved_cache_path = _resolve_feedback_label_cache_path(
        cache_feedback_labels=cache_feedback_labels,
        feedback_label_cache_path=feedback_label_cache_path,
        base_dir=base_dir,
    )
    labeler = TraceAnalysisLabeler(
        mode=labeler_mode,
        model=llm_model,
        temperature=temperature,
        max_tokens=max_tokens,
        cache_feedback_labels=cache_feedback_labels,
        feedback_incorporation_cache_path=resolved_cache_path,
    )
    episodes: list[FeedbackEpisode] = []
    for example in examples:
        episodes.extend(_extract_feedback_episodes(example, labeler))
    accounting = merge_api_accounting(
        build_trace_labeler_accounting(
            labeler,
            component_name="system_feedback_labeler",
        )
    )
    return episodes, accounting


def summarize_system_feedback_incorporation(
    episodes: list[FeedbackEpisode],
) -> list[MetricRecord]:
    """Summarize response to system feedback after failed system tries."""

    eligible = [
        episode
        for episode in episodes
        if episode.used_feedback is not None and episode.next_submission_correct is not None
    ]
    denom = len(eligible)
    incorporated = sum(1 for episode in eligible if episode.used_feedback is True)
    repaired = sum(1 for episode in eligible if episode.next_submission_correct is True)
    incorporate_and_fix = sum(
        1
        for episode in eligible
        if episode.used_feedback is True and episode.next_submission_correct is True
    )
    incorporate_but_wrong = sum(
        1
        for episode in eligible
        if episode.used_feedback is True and episode.next_submission_correct is False
    )
    ignore_but_fix = sum(
        1
        for episode in eligible
        if episode.used_feedback is False and episode.next_submission_correct is True
    )
    ignore_and_wrong = sum(
        1
        for episode in eligible
        if episode.used_feedback is False and episode.next_submission_correct is False
    )

    return [
        _count("system_feedback_incorporation", "SystemFeedbackEpisodes", len(episodes)),
        _count(
            "system_feedback_incorporation",
            "EligibleSystemFeedbackEpisodes",
            denom,
            source="trace+feedback_labeler",
        ),
        # SystemFeedbackIncorporationRate = incorporated feedback / eligible failed system tries.
        _rate(
            "system_feedback_incorporation",
            "SystemFeedbackIncorporationRate",
            incorporated,
            denom,
            source="trace+feedback_labeler",
        ),
        # PostSystemFeedbackRepairRate = next system try passes / eligible failed system tries.
        _rate(
            "system_feedback_incorporation",
            "PostSystemFeedbackRepairRate",
            repaired,
            denom,
            source="trace",
        ),
        _rate(
            "system_feedback_incorporation",
            "SystemIncorporateAndFix",
            incorporate_and_fix,
            denom,
            source="trace+feedback_labeler",
        ),
        _rate(
            "system_feedback_incorporation",
            "SystemIncorporateButStillWrong",
            incorporate_but_wrong,
            denom,
            source="trace+feedback_labeler",
        ),
        _rate(
            "system_feedback_incorporation",
            "SystemIgnoreButFix",
            ignore_but_fix,
            denom,
            source="trace+feedback_labeler",
        ),
        _rate(
            "system_feedback_incorporation",
            "SystemIgnoreAndStayWrong",
            ignore_and_wrong,
            denom,
            source="trace+feedback_labeler",
        ),
        # RepairGivenSystemFeedbackIncorporation = incorporate_and_fix / incorporated.
        _rate(
            "system_feedback_incorporation",
            "RepairGivenSystemFeedbackIncorporation",
            incorporate_and_fix,
            incorporated,
            source="trace+feedback_labeler",
        ),
    ]


def extract_evaluator_feedback_episodes(
    examples: list[TraceExample],
    *,
    labeler_mode: str = "hybrid",
    llm_model: str = "openai/gpt-oss-120b",
    temperature: float = 0.0,
    max_tokens: int = 400,
    cache_feedback_labels: bool = True,
    feedback_label_cache_path: str | Path | None = None,
    base_dir: str | Path | None = None,
) -> list[FeedbackEpisode]:
    """Backward-compatible alias for system-feedback episode extraction."""
    return extract_system_feedback_episodes(
        examples,
        labeler_mode=labeler_mode,
        llm_model=llm_model,
        temperature=temperature,
        max_tokens=max_tokens,
        cache_feedback_labels=cache_feedback_labels,
        feedback_label_cache_path=feedback_label_cache_path,
        base_dir=base_dir,
    )


def extract_evaluator_feedback_episode_bundle(
    examples: list[TraceExample],
    *,
    labeler_mode: str = "hybrid",
    llm_model: str = "openai/gpt-oss-120b",
    temperature: float = 0.0,
    max_tokens: int = 400,
    cache_feedback_labels: bool = True,
    feedback_label_cache_path: str | Path | None = None,
    base_dir: str | Path | None = None,
) -> tuple[list[FeedbackEpisode], dict[str, Any]]:
    """Backward-compatible alias for system-feedback bundle extraction."""
    return extract_system_feedback_episode_bundle(
        examples,
        labeler_mode=labeler_mode,
        llm_model=llm_model,
        temperature=temperature,
        max_tokens=max_tokens,
        cache_feedback_labels=cache_feedback_labels,
        feedback_label_cache_path=feedback_label_cache_path,
        base_dir=base_dir,
    )


def summarize_evaluator_feedback_incorporation(
    episodes: list[FeedbackEpisode],
) -> list[MetricRecord]:
    """Backward-compatible alias for system-feedback incorporation metrics."""
    return summarize_system_feedback_incorporation(episodes)


def _resolve_feedback_label_cache_path(
    *,
    cache_feedback_labels: bool,
    feedback_label_cache_path: str | Path | None,
    base_dir: str | Path | None,
) -> Path | None:
    if not cache_feedback_labels:
        return None
    if feedback_label_cache_path:
        return Path(feedback_label_cache_path).expanduser().resolve()
    resolved_base_dir = Path(base_dir).expanduser().resolve() if base_dir else Path.cwd()
    return default_feedback_label_cache_path(resolved_base_dir)


def _count(section: str, metric: str, value: int, *, source: str = "trace") -> MetricRecord:
    return MetricRecord(section=section, metric=metric, value=value, source=source)


def _rate(
    section: str,
    metric: str,
    numerator: int,
    denominator: int,
    *,
    source: str,
) -> MetricRecord:
    value = (numerator / denominator) if denominator else None
    return MetricRecord(
        section=section,
        metric=metric,
        value=value,
        numerator=numerator,
        denominator=denominator,
        source=source,
    )
