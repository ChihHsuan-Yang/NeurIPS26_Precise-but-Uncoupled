"""Must-have basic metrics for the main paper."""

from __future__ import annotations

from typing import Optional

from agentverse.metrics.main_paper.records import MetricRecord, ProblemRecord


def summarize_basic_metrics(records: list[ProblemRecord]) -> list[MetricRecord]:
    n = len(records)
    solved = [record for record in records if record.final_passed]
    return [
        # FinalPassRate = #(eventually solved) / N
        _rate("basic", "FinalPassRate", _sum(records, "final_passed"), n),
        # Pass@1 = #(solved on first system-level try) / N
        _rate("basic", "Pass@1", _sum(records, "first_passed"), n),
        # AvgCorrectionAttemptsToSuccess = mean first successful system try over solved problems.
        MetricRecord(
            section="basic",
            metric="AvgCorrectionAttemptsToSuccess",
            value=_mean([record.first_success_attempt for record in solved]),
            numerator=sum(record.first_success_attempt or 0 for record in solved),
            denominator=len(solved),
            source="runtime_metrics_or_trace",
        ),
        # TotalCorrectionAttemptsToSuccess = sum first successful attempt over solved problems.
        MetricRecord(
            section="basic",
            metric="TotalCorrectionAttemptsToSuccess",
            value=sum(record.first_success_attempt or 0 for record in solved),
            source="runtime_metrics_or_trace",
        ),
        # AvgCorrectionLoops = mean system tries used per problem.
        MetricRecord(
            section="basic",
            metric="AvgCorrectionLoops",
            value=_mean([record.correction_loops for record in records]),
            numerator=sum(record.correction_loops for record in records),
            denominator=n,
            source="runtime_metrics_or_trace",
        ),
        # TotalEvaluatorAttempts = sum evaluator submissions over all problems.
        MetricRecord(
            section="basic",
            metric="TotalEvaluatorAttempts",
            value=sum(record.evaluator_submission_count for record in records),
            source="runtime_metrics_or_trace",
        ),
        # AvgReflectiveRefinementRounds = mean internal candidate revisions before submission.
        MetricRecord(
            section="basic",
            metric="AvgReflectiveRefinementRounds",
            value=_mean([record.reflective_rounds for record in records]),
            numerator=sum(record.reflective_rounds for record in records),
            denominator=n,
            source="trace",
        ),
        # AvgTotalTokens = mean total tokens per problem.
        MetricRecord(
            section="basic",
            metric="AvgTotalTokens",
            value=_mean([record.total_tokens for record in records]),
            numerator=sum(record.total_tokens for record in records),
            denominator=n,
            source="metrics_jsonl",
        ),
        # AvgModelCalls = mean model/API calls per problem.
        MetricRecord(
            section="basic",
            metric="AvgModelCalls",
            value=_mean([record.model_calls for record in records]),
            numerator=sum(record.model_calls for record in records),
            denominator=n,
            source="metrics_jsonl",
        ),
        # TotalModelCalls = sum model/API calls over all problems.
        MetricRecord(
            section="basic",
            metric="TotalModelCalls",
            value=sum(record.model_calls for record in records),
            source="metrics_jsonl",
        ),
        # AvgReasoningTurns = mean non-evaluator turns per problem.
        MetricRecord(
            section="basic",
            metric="AvgReasoningTurns",
            value=_mean([record.reasoning_turns for record in records]),
            numerator=sum(record.reasoning_turns for record in records),
            denominator=n,
            source="trace",
        ),
    ]


def _rate(section: str, metric: str, numerator: int, denominator: int) -> MetricRecord:
    return MetricRecord(
        section=section,
        metric=metric,
        value=(numerator / denominator) if denominator else None,
        numerator=numerator,
        denominator=denominator,
        source="runtime_metrics_or_trace",
    )


def _mean(values: list[Optional[float]]) -> Optional[float]:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def _sum(records: list[ProblemRecord], field: str) -> int:
    return sum(1 for record in records if bool(getattr(record, field)))
