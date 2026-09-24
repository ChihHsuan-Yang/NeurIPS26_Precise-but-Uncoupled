"""Metric summaries for supplemental diagnostics."""

from __future__ import annotations

from collections import Counter
from typing import Iterable

from agentverse.metrics.diagnostics.records import ProcessExampleRecord
from agentverse.metrics.main_paper.records import MetricRecord


def summarize_process_diagnostics(records: list[ProcessExampleRecord]) -> list[MetricRecord]:
    n = len(records)
    review_events = sum(record.review_events for record in records)
    agreement_events = sum(record.agreement_events for record in records)
    agreement_with_later = sum(record.agreement_with_later_label for record in records)
    agreement_fail = sum(record.agreement_followed_by_fail for record in records)
    agreement_pass = sum(record.agreement_followed_by_pass for record in records)
    adjacent_pairs = sum(record.adjacent_submission_pairs for record in records)
    repeated_adjacent = sum(record.repeated_adjacent_answers for record in records)
    fail_with_next = sum(record.fail_with_next_count for record in records)
    no_change_after_fail = sum(record.no_change_after_fail for record in records)
    wrong_convergence_examples = sum(
        1 for record in records if record.agreement_followed_by_fail > 0
    )
    looping_examples = sum(1 for record in records if record.looping_signal)

    metric_records = [
        _count("process_diagnostics", "ExamplesAnalyzed", n),
        _rate("process_diagnostics", "AgreementEventRate", agreement_events, review_events),
        _rate(
            "process_diagnostics",
            "WrongConvergenceRate",
            agreement_fail,
            agreement_with_later,
        ),
        _rate(
            "process_diagnostics",
            "ProductiveConvergenceRate",
            agreement_pass,
            agreement_with_later,
        ),
        _rate(
            "process_diagnostics",
            "PrematureConvergenceSignalRate",
            wrong_convergence_examples,
            n,
        ),
        _rate(
            "process_diagnostics",
            "RepeatedAnswerRate",
            repeated_adjacent,
            adjacent_pairs,
        ),
        _rate(
            "process_diagnostics",
            "NoChangeAfterFailRate",
            no_change_after_fail,
            fail_with_next,
        ),
        _rate("process_diagnostics", "LoopingSignalRate", looping_examples, n),
        MetricRecord(
            section="process_diagnostics",
            metric="AvgDistinctCandidateCount",
            value=_mean(record.distinct_candidate_count for record in records),
            numerator=sum(record.distinct_candidate_count for record in records),
            denominator=n,
            source="trace",
        ),
        MetricRecord(
            section="process_diagnostics",
            metric="AvgTurnImbalance",
            value=_mean(
                record.turn_imbalance
                for record in records
                if record.turn_imbalance is not None
            ),
            numerator=sum(
                record.turn_imbalance or 0.0
                for record in records
                if record.turn_imbalance is not None
            ),
            denominator=sum(1 for record in records if record.turn_imbalance is not None),
            source="trace",
        ),
    ]
    metric_records.extend(_turn_share_records(records))
    return metric_records


def _turn_share_records(records: list[ProcessExampleRecord]) -> list[MetricRecord]:
    counts: Counter[str] = Counter()
    for record in records:
        counts.update(record.turn_counts)
    total = sum(counts.values())
    rows: list[MetricRecord] = []
    for role, count in sorted(counts.items()):
        rows.append(
            MetricRecord(
                section="process_turn_share",
                metric=f"TurnShare_{_metric_safe_role(role)}",
                value=(count / total) if total else None,
                numerator=count,
                denominator=total,
                source="trace",
                notes=role,
            )
        )
    return rows


def _count(section: str, metric: str, value: int, *, source: str = "trace") -> MetricRecord:
    return MetricRecord(section=section, metric=metric, value=value, source=source)


def _rate(
    section: str,
    metric: str,
    numerator: int,
    denominator: int,
    *,
    source: str = "trace",
) -> MetricRecord:
    return MetricRecord(
        section=section,
        metric=metric,
        value=(numerator / denominator) if denominator else None,
        numerator=numerator,
        denominator=denominator,
        source=source,
    )


def _mean(values: Iterable[float | int | None]) -> float | None:
    filtered = [float(value) for value in values if value is not None]
    if not filtered:
        return None
    return sum(filtered) / len(filtered)


def _metric_safe_role(role: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in str(role or "unknown"))
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return cleaned or "unknown"
