"""Must-have inner-loop reflective-refinement transition metrics."""

from __future__ import annotations

from collections import Counter, defaultdict
import re
from typing import Optional

from agentverse.metrics.main_paper.records import InnerTransitionRecord, MetricRecord


TRANSITION_ORDER = (
    "wrong_to_wrong",
    "wrong_to_correct",
    "correct_to_correct",
    "correct_to_wrong",
)


def summarize_inner_loop_metrics(
    transitions: list[InnerTransitionRecord],
) -> list[MetricRecord]:
    eligible = _eligible(transitions)
    counts = Counter(transition.transition_name for transition in eligible)
    rows: list[MetricRecord] = []

    for transition_name in TRANSITION_ORDER:
        count = counts.get(transition_name, 0)
        # Transition_* value = transition count / all labeled inner transitions.
        rows.append(
            MetricRecord(
                section="inner_loop",
                metric=f"Transition_{transition_name}",
                value=(count / len(eligible)) if eligible else None,
                numerator=count,
                denominator=len(eligible),
                source="trace+benchmark_evaluator",
            )
        )

    wrong_initial = counts["wrong_to_wrong"] + counts["wrong_to_correct"]
    correct_initial = counts["correct_to_correct"] + counts["correct_to_wrong"]
    wrong_to_wrong_same = sum(
        1
        for transition in eligible
        if transition.transition_name == "wrong_to_wrong"
        and _same_candidate(transition.initial_candidate, transition.final_candidate)
    )
    wrong_to_wrong_changed = counts["wrong_to_wrong"] - wrong_to_wrong_same

    # ReflectiveRepairRate = #(wrong -> correct) / #(initially wrong).
    rows.append(
        MetricRecord(
            section="inner_loop",
            metric="ReflectiveRepairRate",
            value=_rate(counts["wrong_to_correct"], wrong_initial),
            numerator=counts["wrong_to_correct"],
            denominator=wrong_initial,
            source="trace+benchmark_evaluator",
        )
    )
    # ReflectiveStillWrongRate = #(wrong -> wrong) / #(initially wrong).
    rows.append(
        MetricRecord(
            section="inner_loop",
            metric="ReflectiveStillWrongRate",
            value=_rate(counts["wrong_to_wrong"], wrong_initial),
            numerator=counts["wrong_to_wrong"],
            denominator=wrong_initial,
            source="trace+benchmark_evaluator",
        )
    )
    # ReflectiveNeglectRate = #(wrong -> same wrong answer) / #(initially wrong).
    rows.append(
        MetricRecord(
            section="inner_loop",
            metric="ReflectiveNeglectRate",
            value=_rate(wrong_to_wrong_same, wrong_initial),
            numerator=wrong_to_wrong_same,
            denominator=wrong_initial,
            source="trace+benchmark_evaluator",
        )
    )
    # ReflectiveTryButFailRate = #(wrong -> changed-but-still-wrong answer) / #(initially wrong).
    rows.append(
        MetricRecord(
            section="inner_loop",
            metric="ReflectiveTryButFailRate",
            value=_rate(wrong_to_wrong_changed, wrong_initial),
            numerator=wrong_to_wrong_changed,
            denominator=wrong_initial,
            source="trace+benchmark_evaluator",
        )
    )
    # ReflectivePreservationRate = #(correct -> correct) / #(initially correct).
    rows.append(
        MetricRecord(
            section="inner_loop",
            metric="ReflectivePreservationRate",
            value=_rate(counts["correct_to_correct"], correct_initial),
            numerator=counts["correct_to_correct"],
            denominator=correct_initial,
            source="trace+benchmark_evaluator",
        )
    )
    # ReflectiveHarmRate = #(correct -> wrong) / #(initially correct).
    rows.append(
        MetricRecord(
            section="inner_loop",
            metric="ReflectiveHarmRate",
            value=_rate(counts["correct_to_wrong"], correct_initial),
            numerator=counts["correct_to_wrong"],
            denominator=correct_initial,
            source="trace+benchmark_evaluator",
        )
    )
    # ReflectiveNetRepairGain = (#(wrong -> correct) - #(correct -> wrong)) / labeled transitions.
    rows.append(
        MetricRecord(
            section="inner_loop",
            metric="ReflectiveNetRepairGain",
            value=(
                (counts["wrong_to_correct"] - counts["correct_to_wrong"]) / len(eligible)
                if eligible
                else None
            ),
            numerator=counts["wrong_to_correct"] - counts["correct_to_wrong"],
            denominator=len(eligible),
            source="trace+benchmark_evaluator",
        )
    )
    rows.extend(_difficulty_transition_matrix(eligible))
    return rows


def _difficulty_transition_matrix(
    transitions: list[InnerTransitionRecord],
) -> list[MetricRecord]:
    grouped: dict[str, list[InnerTransitionRecord]] = defaultdict(list)
    for transition in transitions:
        if transition.difficulty_tier is None:
            continue
        grouped[str(transition.difficulty_tier)].append(transition)

    rows: list[MetricRecord] = []
    for tier, tier_transitions in sorted(grouped.items()):
        counts = Counter(transition.transition_name for transition in tier_transitions)
        for transition_name in TRANSITION_ORDER:
            count = counts.get(transition_name, 0)
            # Difficulty-sliced matrix cell = tier transition count / tier labeled transitions.
            rows.append(
                MetricRecord(
                    section="inner_loop_by_difficulty",
                    metric=f"{tier}:Transition_{transition_name}",
                    value=(count / len(tier_transitions)) if tier_transitions else None,
                    numerator=count,
                    denominator=len(tier_transitions),
                    source="trace+benchmark_evaluator+results_jsonl",
                )
            )
    return rows


def _eligible(
    transitions: list[InnerTransitionRecord],
) -> list[InnerTransitionRecord]:
    return [
        transition
        for transition in transitions
        if transition.initial_correct is not None and transition.final_correct is not None
    ]


def _rate(numerator: int, denominator: int) -> Optional[float]:
    if denominator <= 0:
        return None
    return numerator / denominator


def _same_candidate(left: str, right: str) -> bool:
    left_key = _candidate_key(left)
    right_key = _candidate_key(right)
    return bool(left_key and right_key and left_key == right_key)


def _candidate_key(candidate: str) -> str:
    return re.sub(r"\s+", "", str(candidate or "").strip()).lower()
