"""Must-have outer-loop verifier-guided correction metrics."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Optional

from agentverse.metrics.main_paper.records import MetricRecord, ProblemRecord


def summarize_outer_loop_metrics(
    records: list[ProblemRecord],
    *,
    hint_lambda: float = 1.0,
    max_k: int | None = None,
) -> tuple[list[MetricRecord], dict[int, dict[str, Any]]]:
    scores = build_outer_loop_problem_scores(records, hint_lambda=hint_lambda)
    n = len(records)
    solved = [record for record in records if record.final_passed]
    metric_records: list[MetricRecord] = []
    resolved_max_k = _resolve_max_k(records, max_k=max_k)

    # MRRAttemptAccuracy_i = 1 / first_success_try_i if solved, else 0.
    metric_records.append(
        MetricRecord(
            section="outer_loop",
            metric="MRRAttemptAccuracy",
            value=_mean([row["mrr_attempt_accuracy"] for row in scores.values()]),
            numerator=sum(row["mrr_attempt_accuracy"] for row in scores.values()),
            denominator=n,
            source="runtime_metrics_or_trace",
        )
    )
    # PassFailCorrectionScore_i = 1 / (1 + failed_before_success_i) if solved, else 0.
    metric_records.append(
        MetricRecord(
            section="outer_loop",
            metric="PassFailCorrectionScore",
            value=_mean([row["pass_fail_correction_score"] for row in scores.values()]),
            numerator=sum(row["pass_fail_correction_score"] for row in scores.values()),
            denominator=n,
            source="runtime_metrics_or_trace",
            notes="Same reciprocal-attempt value as MRRAttemptAccuracy under the current definition.",
        )
    )
    # HintAwareCorrectionScore_i = 1 / (1 + failed_before_success_i + lambda * hints_i) if solved, else 0.
    metric_records.append(
        MetricRecord(
            section="outer_loop",
            metric="HintAwareCorrectionScore",
            value=_mean([row["hint_aware_correction_score"] for row in scores.values()]),
            numerator=sum(row["hint_aware_correction_score"] for row in scores.values()),
            denominator=n,
            source="runtime_metrics_or_trace",
            notes=f"lambda={hint_lambda}",
        )
    )
    # DMRAS = macro-average HintAwareCorrectionScore over difficulty tiers.
    metric_records.append(
        MetricRecord(
            section="outer_loop",
            metric="DMRAS",
            value=_macro_score_by_key(records, scores, "difficulty_tier"),
            source="results_jsonl+trace",
        )
    )
    # AvgCorrectionAttemptsToSuccess = mean first successful system try over solved problems.
    metric_records.append(
        MetricRecord(
            section="outer_loop",
            metric="AvgCorrectionAttemptsToSuccess",
            value=_mean([record.first_success_attempt for record in solved]),
            numerator=sum(record.first_success_attempt or 0 for record in solved),
            denominator=len(solved),
            source="runtime_metrics_or_trace",
        )
    )

    for k in range(1, resolved_max_k + 1):
        pass_k = _count_pass_at_k(records, k)
        reached_k = _count_reached_try_k(records, k)
        first_success_at_k = _count_first_success_at_k(records, k)
        # OuterPass@k = #(solved within k system tries) / N.
        metric_records.append(_rate("outer_loop", f"OuterPass@{k}", pass_k, n))
        # ConditionalRecoveryRate@k = #(first solved exactly at try k) / #(problems that reached try k).
        metric_records.append(
            MetricRecord(
                section="outer_loop",
                metric=f"ConditionalRecoveryRate@{k}",
                value=(first_success_at_k / reached_k) if reached_k else None,
                numerator=first_success_at_k,
                denominator=reached_k,
                source="runtime_metrics_or_trace",
                notes=(
                    "Among problems that consumed system try "
                    f"{k}, fraction first solved at that try."
                ),
            )
        )
        if k >= 2:
            prev = _count_pass_at_k(records, k - 1)
            # MarginalRecoveryGain_k = OuterPass@k - OuterPass@(k - 1).
            metric_records.append(
                MetricRecord(
                    section="outer_loop",
                    metric=f"MarginalRecoveryGain_{k}",
                    value=((pass_k - prev) / n) if n else None,
                    numerator=pass_k - prev,
                    denominator=n,
                    source="runtime_metrics_or_trace",
                )
            )

    metric_records.extend(_domain_score_records(records, scores))
    return metric_records, scores


def build_outer_loop_problem_scores(
    records: list[ProblemRecord],
    *,
    hint_lambda: float = 1.0,
) -> dict[int, dict[str, Any]]:
    scores: dict[int, dict[str, Any]] = {}
    for record in records:
        failed_before_success = (
            max(0, (record.first_success_attempt or 0) - 1)
            if record.final_passed
            else record.correction_loops
        )
        mrr_score = (
            1.0 / float(record.first_success_attempt)
            if record.final_passed and record.first_success_attempt
            else 0.0
        )
        pass_fail_score = (
            1.0 / (1.0 + failed_before_success) if record.final_passed else 0.0
        )
        hint_score = (
            1.0 / (1.0 + failed_before_success + hint_lambda * record.hints_used)
            if record.final_passed
            else 0.0
        )
        scores[record.example_idx] = {
            "failed_before_success": failed_before_success,
            "mrr_attempt_accuracy": mrr_score,
            "pass_fail_correction_score": pass_fail_score,
            "hint_aware_correction_score": hint_score,
        }
    return scores


def summarize_hint_comparison(
    plain_records: list[ProblemRecord],
    hint_records: list[ProblemRecord],
    *,
    hint_lambda: float = 1.0,
) -> list[MetricRecord]:
    plain_by_idx = {record.example_idx: record for record in plain_records}
    hint_by_idx = {record.example_idx: record for record in hint_records}
    matched = sorted(set(plain_by_idx) & set(hint_by_idx))
    if not matched:
        return [
            MetricRecord(
                section="outer_loop_hint_comparison",
                metric="MatchedSubsetSize",
                value=0,
                source="paired_trace",
                notes="No matched example_idx values found.",
            )
        ]

    plain = [plain_by_idx[idx] for idx in matched]
    hint = [hint_by_idx[idx] for idx in matched]
    plain_scores = build_outer_loop_problem_scores(plain, hint_lambda=hint_lambda)
    hint_scores = build_outer_loop_problem_scores(hint, hint_lambda=hint_lambda)
    plain_summary = _comparison_summary(plain, plain_scores)
    hint_summary = _comparison_summary(hint, hint_scores)
    avg_hint_delta = hint_summary["AvgHints"] - plain_summary["AvgHints"]

    rows = [
        MetricRecord(
            section="outer_loop_hint_comparison",
            metric="MatchedSubsetSize",
            value=len(matched),
            source="paired_trace",
        )
    ]
    for metric in (
        "FinalPassRate",
        "Pass@1",
        "MRRAttemptAccuracy",
        "PassFailCorrectionScore",
        "HintAwareCorrectionScore",
        "AvgCorrectionLoops",
        "AvgModelCalls",
    ):
        rows.append(
            MetricRecord(
                section="outer_loop_hint_comparison",
                metric=f"HintMinusPlain_{metric}",
                value=hint_summary[metric] - plain_summary[metric],
                source="paired_trace",
            )
        )
    # MarginalHintGain = added final-pass rate per added average hint.
    rows.append(
        MetricRecord(
            section="outer_loop_hint_comparison",
            metric="MarginalHintGain",
            value=(
                (hint_summary["FinalPassRate"] - plain_summary["FinalPassRate"])
                / avg_hint_delta
                if avg_hint_delta
                else None
            ),
            denominator=avg_hint_delta,
            source="paired_trace",
            notes="FinalPassRate lift per additional average hint.",
        )
    )
    return rows


def _comparison_summary(
    records: list[ProblemRecord],
    scores: dict[int, dict[str, Any]],
) -> dict[str, float]:
    n = len(records)
    return {
        "FinalPassRate": _count(records, "final_passed") / n if n else 0.0,
        "Pass@1": _count(records, "first_passed") / n if n else 0.0,
        "MRRAttemptAccuracy": _mean(
            [scores[record.example_idx]["mrr_attempt_accuracy"] for record in records]
        )
        or 0.0,
        "PassFailCorrectionScore": _mean(
            [scores[record.example_idx]["pass_fail_correction_score"] for record in records]
        )
        or 0.0,
        "HintAwareCorrectionScore": _mean(
            [scores[record.example_idx]["hint_aware_correction_score"] for record in records]
        )
        or 0.0,
            "AvgCorrectionLoops": _mean([record.correction_loops for record in records]) or 0.0,
        "AvgModelCalls": _mean([record.model_calls for record in records]) or 0.0,
        "AvgHints": _mean([record.hints_used for record in records]) or 0.0,
    }


def _rate(section: str, metric: str, numerator: int, denominator: int) -> MetricRecord:
    return MetricRecord(
        section=section,
        metric=metric,
        value=(numerator / denominator) if denominator else None,
        numerator=numerator,
        denominator=denominator,
        source="runtime_metrics_or_trace",
    )


def _count_pass_at_k(records: list[ProblemRecord], k: int) -> int:
    return sum(
        1
        for record in records
        if record.first_success_attempt is not None and record.first_success_attempt <= k
    )


def _count_first_success_at_k(records: list[ProblemRecord], k: int) -> int:
    return sum(1 for record in records if record.first_success_attempt == k)


def _count_reached_try_k(records: list[ProblemRecord], k: int) -> int:
    return sum(1 for record in records if int(record.correction_loops or 0) >= k)


def _resolve_max_k(
    records: list[ProblemRecord],
    *,
    max_k: int | None,
) -> int:
    if max_k is not None and int(max_k) > 0:
        return int(max_k)

    observed = max(
        [
            int(record.correction_loops or 0)
            for record in records
        ]
        + [
            int(record.first_success_attempt or 0)
            for record in records
        ]
        + [1]
    )
    return observed


def _macro_score_by_key(
    records: list[ProblemRecord],
    scores: dict[int, dict[str, Any]],
    key: str,
) -> Optional[float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for record in records:
        value = getattr(record, key, None)
        if value is None:
            continue
        grouped[str(value)].append(
            float(scores[record.example_idx]["hint_aware_correction_score"])
        )
    if not grouped:
        return None
    return sum(_mean(values) or 0.0 for values in grouped.values()) / len(grouped)


def _domain_score_records(
    records: list[ProblemRecord],
    scores: dict[int, dict[str, Any]],
) -> list[MetricRecord]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for record in records:
        domains = record.domain if isinstance(record.domain, list) else [record.domain]
        for domain in domains:
            if domain is None:
                continue
            grouped[str(domain)].append(
                float(scores[record.example_idx]["hint_aware_correction_score"])
            )
    return [
        MetricRecord(
            section="outer_loop_by_domain",
            metric=str(domain),
            value=_mean(values),
            numerator=sum(values),
            denominator=len(values),
            source="results_jsonl+trace",
        )
        for domain, values in sorted(grouped.items())
    ]


def _mean(values: list[Any]) -> Optional[float]:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def _count(records: list[ProblemRecord], field: str) -> int:
    return sum(1 for record in records if bool(getattr(record, field)))
