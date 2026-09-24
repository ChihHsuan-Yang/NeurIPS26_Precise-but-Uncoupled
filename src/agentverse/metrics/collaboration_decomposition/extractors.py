"""Trace extraction for review-conditioned collaboration decomposition metrics."""

from __future__ import annotations

import re
from pathlib import Path

from agentverse.evaluation.content_based import TraceAnalysisLabeler
from agentverse.metrics.api_accounting import (
    build_candidate_correctness_accounting,
    build_trace_labeler_accounting,
    merge_api_accounting,
)
from agentverse.evaluation.feedback_label_cache import (
    default_feedback_label_cache_path,
)
from agentverse.metrics.collaboration_decomposition.records import (
    CollaborationDecompositionDataset,
    ReviewerFeedbackIncorporationEpisode,
)
from agentverse.metrics.diagnostics.labeling import build_review_correctness_labeler
from agentverse.metrics.repair import ReviewEpisode, TraceExample, _extract_review_episodes, parse_trace_file


def load_collaboration_decomposition_dataset(
    trace_files: list[Path],
    *,
    correctness_evaluator_type: str = "llm",
    numeric_tolerance: str = "0",
    data_name: str = "omni-math",
    omni_judge_model_path: str | None = None,
    omni_judge_max_new_tokens: int = 300,
    omni_judge_device: str = "auto",
    omni_judge_dtype: str = "auto",
    run_config_path: str | Path | None = None,
    reuse_runtime_final_labels: bool = True,
    cache_candidate_labels: bool = True,
    candidate_label_cache_path: str | Path | None = None,
    reviewer_feedback_labeler_mode: str = "heuristic",
    reviewer_feedback_llm_model: str = "openai/gpt-oss-120b",
    reviewer_feedback_temperature: float = 0.0,
    reviewer_feedback_max_tokens: int = 400,
    cache_feedback_labels: bool = True,
    feedback_label_cache_path: str | Path | None = None,
) -> CollaborationDecompositionDataset:
    """Load traces and build the paper-facing decomposition dataset."""

    examples = _load_trace_examples(trace_files)
    correctness_labeler, selected_evaluator = build_review_correctness_labeler(
        examples,
        evaluator_type=correctness_evaluator_type,
        numeric_tolerance=numeric_tolerance,
        data_name=data_name,
        omni_judge_model_path=omni_judge_model_path,
        omni_judge_max_new_tokens=omni_judge_max_new_tokens,
        omni_judge_device=omni_judge_device,
        omni_judge_dtype=omni_judge_dtype,
        run_config_path=run_config_path,
        reuse_runtime_final_labels=reuse_runtime_final_labels,
        cache_candidate_labels=cache_candidate_labels,
        candidate_label_cache_path=candidate_label_cache_path,
    )
    resolved_feedback_cache_path = _resolve_feedback_label_cache_path(
        trace_files=trace_files,
        cache_feedback_labels=cache_feedback_labels,
        feedback_label_cache_path=feedback_label_cache_path,
    )
    reviewer_feedback_labeler = TraceAnalysisLabeler(
        mode=reviewer_feedback_labeler_mode,
        model=reviewer_feedback_llm_model,
        temperature=reviewer_feedback_temperature,
        max_tokens=reviewer_feedback_max_tokens,
        cache_feedback_labels=cache_feedback_labels,
        feedback_incorporation_cache_path=resolved_feedback_cache_path,
    )

    review_episodes: list[ReviewEpisode] = []
    for example in examples:
        if example.protocol not in {"per", "broadcast"}:
            continue
        review_episodes.extend(_extract_review_episodes(example, correctness_labeler))

    reviewer_feedback_episodes = extract_reviewer_feedback_incorporation_episodes(
        review_episodes,
        labeler=reviewer_feedback_labeler,
    )

    return CollaborationDecompositionDataset(
        examples=examples,
        review_episodes=review_episodes,
        reviewer_feedback_episodes=reviewer_feedback_episodes,
        trace_files=[str(path) for path in trace_files],
        correctness_evaluator_type=selected_evaluator,
        reviewer_feedback_labeler_mode=reviewer_feedback_labeler.mode,
        analysis_accounting=merge_api_accounting(
            build_candidate_correctness_accounting(
                correctness_labeler.evaluator,
                extra={
                    "runtime_label_hits": int(getattr(correctness_labeler, "runtime_hits", 0) or 0),
                },
            ),
            build_trace_labeler_accounting(
                reviewer_feedback_labeler,
                component_name="reviewer_feedback_labeler",
            ),
        ),
    )


def extract_reviewer_feedback_incorporation_episodes(
    episodes: list[ReviewEpisode],
    *,
    labeler: TraceAnalysisLabeler,
) -> list[ReviewerFeedbackIncorporationEpisode]:
    """Label whether agents acted according to reviewer feedback.

    The default path is deterministic and avoids LLM calls:
    - revise feedback is incorporated when the candidate changes;
    - agree feedback is incorporated when the candidate is preserved.
    If revise feedback produces no detectable answer change and the reviewer
    labeler is configured as `hybrid` or `llm`, a small local window can be sent
    to the feedback-incorporation judge.
    """

    records: list[ReviewerFeedbackIncorporationEpisode] = []
    for episode in episodes:
        incorporated, source = _judge_reviewer_feedback_incorporation(
            episode,
            labeler=labeler,
        )
        records.append(
            ReviewerFeedbackIncorporationEpisode(
                trace_path=episode.trace_path,
                task=episode.task,
                protocol=episode.protocol,
                example_idx=episode.example_idx,
                episode_id=episode.episode_id,
                review_turn=episode.review_turn,
                reviewer=episode.reviewer,
                action=episode.action,
                candidate_before=episode.candidate_before,
                candidate_after=episode.candidate_after,
                pre_correct=episode.pre_correct,
                post_correct=episode.post_correct,
                incorporated_feedback=incorporated,
                incorporated_feedback_source=source,
                evidence_turns=list(episode.evidence_turns),
            )
        )
    return records


def _judge_reviewer_feedback_incorporation(
    episode: ReviewEpisode,
    *,
    labeler: TraceAnalysisLabeler,
) -> tuple[bool | None, str]:
    same_candidate = _same_candidate(episode.candidate_before, episode.candidate_after)

    if episode.action == "agree":
        return same_candidate, "candidate_preservation_heuristic"

    if episode.action != "revise":
        return None, "unsupported_review_action"

    if not same_candidate:
        return True, "candidate_change_heuristic"

    if labeler.mode not in {"hybrid", "llm"} or not labeler.llm_available:
        return False, "candidate_no_change_heuristic"

    response = (
        f"Review action: {episode.action}\n"
        f"Candidate before review:\n{episode.candidate_before}\n\n"
        f"Candidate after review:\n{episode.candidate_after}\n"
    )
    result = labeler.judge_feedback_incorporation(
        feedback_text=episode.review_feedback,
        response_path=response,
        feedback_source="reviewer",
    )
    return result.value, result.source


def _load_trace_examples(trace_files: list[Path]) -> list[TraceExample]:
    examples: list[TraceExample] = []
    for trace_file in trace_files:
        examples.extend(parse_trace_file(trace_file))
    return examples


def _same_candidate(left: str, right: str) -> bool:
    left_key = _candidate_key(left)
    right_key = _candidate_key(right)
    return bool(left_key and right_key and left_key == right_key)


def _candidate_key(candidate: str) -> str:
    return re.sub(r"\s+", "", str(candidate or "").strip()).lower()


def _resolve_feedback_label_cache_path(
    *,
    trace_files: list[Path],
    cache_feedback_labels: bool,
    feedback_label_cache_path: str | Path | None,
) -> Path | None:
    if not cache_feedback_labels:
        return None
    if feedback_label_cache_path:
        return Path(feedback_label_cache_path).expanduser().resolve()
    base_dir = trace_files[0].parent if trace_files else Path.cwd()
    return default_feedback_label_cache_path(base_dir)
