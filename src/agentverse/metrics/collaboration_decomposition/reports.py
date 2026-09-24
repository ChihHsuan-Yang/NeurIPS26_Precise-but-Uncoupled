"""CSV and concise text reports for collaboration decomposition."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from agentverse.metrics.collaboration_decomposition.records import (
    ReviewerFeedbackIncorporationEpisode,
)
from agentverse.metrics.main_paper.records import MetricRecord
from agentverse.metrics.main_paper.reports import metric_explanation, write_metric_csv
from agentverse.metrics.repair import ReviewEpisode


def render_reviewer_conditioned_text(
    records: list[MetricRecord],
    episodes: list[ReviewEpisode],
) -> str:
    lines = ["Reviewer-Conditioned Decomposition", ""]
    for section, title in (
        ("reviewer_detection", "Reviewer Detection"),
        ("reviewer_conditioned_response", "Agent Response After Reviewer Detection"),
        ("reviewer_conditioned_tensor", "Count Tensor"),
    ):
        section_records = [record for record in records if record.section == section]
        if not section_records:
            continue
        lines.extend([title, ""])
        lines.extend(_metric_lines(section_records))
        lines.append("")

    lines.extend(["Per Problem", ""])
    for key, group in _group_review_episodes(episodes).items():
        example_idx = key[1]
        counts = _review_counts(group)
        lines.append(
            (
                f"Example {example_idx}: reviews={len(group)}; "
                f"TP={counts['tp']}; FP={counts['fp']}; "
                f"FN={counts['fn']}; TN={counts['tn']}; "
                f"guided_fix={counts['wrong_revise_correct']}; "
                f"detected_not_fixed={counts['wrong_revise_wrong']}; "
                f"misleading_harm={counts['correct_revise_wrong']}; "
                f"misleading_resist={counts['correct_revise_correct']}"
            )
        )
    return "\n".join(lines).rstrip() + "\n"


def render_reviewer_feedback_text(
    records: list[MetricRecord],
    episodes: list[ReviewerFeedbackIncorporationEpisode],
) -> str:
    lines = ["Reviewer Feedback Incorporation", ""]
    lines.extend(_metric_lines(records))
    lines.extend(["", "Per Problem", ""])
    for key, group in _group_reviewer_feedback_episodes(episodes).items():
        example_idx = key[1]
        counts = _reviewer_feedback_counts(group)
        lines.append(
            (
                f"Example {example_idx}: feedback_episodes={len(group)}; "
                f"eligible={counts['eligible']}; incorporated={counts['incorporated']}; "
                f"useful_revise_incorporated={counts['useful_revise_incorporated']}; "
                f"misleading_review_incorporated={counts['misleading_review_incorporated']}"
            )
        )
    return "\n".join(lines).rstrip() + "\n"


def _metric_lines(records: list[MetricRecord]) -> list[str]:
    lines: list[str] = []
    for record in records:
        detail = f"{record.metric}: {_fmt(record.value)}"
        if record.numerator is not None or record.denominator is not None:
            detail += f" ({_fmt(record.numerator)}/{_fmt(record.denominator)})"
        if record.notes:
            detail += f"; {record.notes}"
        explanation = metric_explanation(record.metric, notes=record.notes)
        if explanation:
            detail += f" # {explanation}"
        lines.append(detail)
    return lines


def _group_review_episodes(
    episodes: list[ReviewEpisode],
) -> dict[tuple[str, int], list[ReviewEpisode]]:
    grouped: dict[tuple[str, int], list[ReviewEpisode]] = defaultdict(list)
    for episode in episodes:
        grouped[(episode.trace_path, episode.example_idx)].append(episode)
    return dict(sorted(grouped.items(), key=lambda item: item[0]))


def _group_reviewer_feedback_episodes(
    episodes: list[ReviewerFeedbackIncorporationEpisode],
) -> dict[tuple[str, int], list[ReviewerFeedbackIncorporationEpisode]]:
    grouped: dict[tuple[str, int], list[ReviewerFeedbackIncorporationEpisode]] = defaultdict(list)
    for episode in episodes:
        grouped[(episode.trace_path, episode.example_idx)].append(episode)
    return dict(sorted(grouped.items(), key=lambda item: item[0]))


def _review_counts(episodes: list[ReviewEpisode]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for episode in episodes:
        if episode.pre_correct is None:
            continue
        if episode.action == "revise" and episode.pre_correct is False:
            counts["tp"] += 1
        if episode.action == "revise" and episode.pre_correct is True:
            counts["fp"] += 1
        if episode.action == "agree" and episode.pre_correct is False:
            counts["fn"] += 1
        if episode.action == "agree" and episode.pre_correct is True:
            counts["tn"] += 1
        if episode.post_correct is None:
            continue
        pre = "correct" if episode.pre_correct else "wrong"
        post = "correct" if episode.post_correct else "wrong"
        counts[f"{pre}_{episode.action}_{post}"] += 1

    for key in (
        "tp",
        "fp",
        "fn",
        "tn",
        "wrong_revise_correct",
        "wrong_revise_wrong",
        "correct_revise_correct",
        "correct_revise_wrong",
    ):
        counts.setdefault(key, 0)
    return dict(counts)


def _reviewer_feedback_counts(
    episodes: list[ReviewerFeedbackIncorporationEpisode],
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for episode in episodes:
        if episode.incorporated_feedback is None:
            continue
        counts["eligible"] += 1
        if episode.incorporated_feedback is True:
            counts["incorporated"] += 1
        if (
            episode.action == "revise"
            and episode.pre_correct is False
            and episode.incorporated_feedback is True
        ):
            counts["useful_revise_incorporated"] += 1
        if (
            episode.action == "revise"
            and episode.pre_correct is True
            and episode.incorporated_feedback is True
        ):
            counts["misleading_review_incorporated"] += 1
    for key in (
        "eligible",
        "incorporated",
        "useful_revise_incorporated",
        "misleading_review_incorporated",
    ):
        counts.setdefault(key, 0)
    return dict(counts)


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)
