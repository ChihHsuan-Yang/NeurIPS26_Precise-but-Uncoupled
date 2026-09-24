"""Metric summaries for collaboration decomposition."""

from __future__ import annotations

from collections import Counter
from typing import Optional

from agentverse.metrics.collaboration_decomposition.records import (
    ReviewerFeedbackIncorporationEpisode,
)
from agentverse.metrics.main_paper.records import MetricRecord
from agentverse.metrics.repair import ReviewEpisode


def summarize_reviewer_conditioned_decomposition(
    episodes: list[ReviewEpisode],
) -> list[MetricRecord]:
    """Summarize reviewer detection and reviewer-conditioned response.

    We count a three-way tensor C[pre_correctness, review_action, post_correctness].
    Reviewer detection uses the first two axes. Agent response uses all three.
    """

    eligible_detection = [
        episode
        for episode in episodes
        if episode.pre_correct is not None and episode.action in {"revise", "agree"}
    ]
    tensor = _review_tensor_counts(eligible_detection)

    # TP_review = reviewer says revise when the candidate is wrong.
    tp = sum(
        1
        for episode in eligible_detection
        if episode.pre_correct is False and episode.action == "revise"
    )
    # FP_review = reviewer says revise when the candidate is already correct.
    fp = sum(
        1
        for episode in eligible_detection
        if episode.pre_correct is True and episode.action == "revise"
    )
    # FN_review = reviewer agrees with a wrong candidate.
    fn = sum(
        1
        for episode in eligible_detection
        if episode.pre_correct is False and episode.action == "agree"
    )
    # TN_review = reviewer agrees with a correct candidate.
    tn = sum(
        1
        for episode in eligible_detection
        if episode.pre_correct is True and episode.action == "agree"
    )

    wrong_revise_correct = tensor["wrong_revise_correct"]
    wrong_revise_wrong = tensor["wrong_revise_wrong"]
    correct_revise_correct = tensor["correct_revise_correct"]
    correct_revise_wrong = tensor["correct_revise_wrong"]
    response_tp = wrong_revise_correct + wrong_revise_wrong
    response_fp = correct_revise_correct + correct_revise_wrong

    rows: list[MetricRecord] = [
        _count("reviewer_detection", "ReviewEpisodes", len(episodes)),
        _count(
            "reviewer_detection",
            "EligibleReviewEpisodes",
            len(eligible_detection),
            source="trace+benchmark_evaluator",
        ),
        _count("reviewer_detection", "TP_review", tp, source="trace+benchmark_evaluator"),
        _count("reviewer_detection", "FP_review", fp, source="trace+benchmark_evaluator"),
        _count("reviewer_detection", "FN_review", fn, source="trace+benchmark_evaluator"),
        _count("reviewer_detection", "TN_review", tn, source="trace+benchmark_evaluator"),
        # Prec_review = TP_review / (TP_review + FP_review).
        _rate("reviewer_detection", "Prec_review", tp, tp + fp, source="trace+benchmark_evaluator"),
        # Rec_review = TP_review / (TP_review + FN_review).
        _rate("reviewer_detection", "Rec_review", tp, tp + fn, source="trace+benchmark_evaluator"),
        # FAR_review = FP_review / (FP_review + TN_review).
        _rate("reviewer_detection", "FAR_review", fp, fp + tn, source="trace+benchmark_evaluator"),
        MetricRecord(
            section="reviewer_detection",
            metric="BalAcc_review",
            value=_balanced_accuracy(tp, fp, fn, tn),
            source="trace+benchmark_evaluator",
        ),
    ]

    rows.extend(
        _count("reviewer_conditioned_tensor", f"C_{cell}", count, source="trace+benchmark_evaluator")
        for cell, count in sorted(tensor.items())
    )
    rows.extend(
        [
            _count(
                "reviewer_conditioned_response",
                "EligibleCorrectReviewEpisodes",
                response_tp,
                source="trace+benchmark_evaluator",
            ),
            _count(
                "reviewer_conditioned_response",
                "EligibleMisleadingReviewEpisodes",
                response_fp,
                source="trace+benchmark_evaluator",
            ),
            # ReviewerGuidedRepairRate = C[wrong, revise, correct] / C[wrong, revise, *].
            _rate(
                "reviewer_conditioned_response",
                "ReviewerGuidedRepairRate",
                wrong_revise_correct,
                response_tp,
                source="trace+benchmark_evaluator",
            ),
            # ReviewerDetectedButNotFixedRate = C[wrong, revise, wrong] / C[wrong, revise, *].
            _rate(
                "reviewer_conditioned_response",
                "ReviewerDetectedButNotFixedRate",
                wrong_revise_wrong,
                response_tp,
                source="trace+benchmark_evaluator",
            ),
            # MisleadingReviewHarmRate = C[correct, revise, wrong] / C[correct, revise, *].
            _rate(
                "reviewer_conditioned_response",
                "MisleadingReviewHarmRate",
                correct_revise_wrong,
                response_fp,
                source="trace+benchmark_evaluator",
            ),
            # MisleadingReviewResistanceRate = C[correct, revise, correct] / C[correct, revise, *].
            _rate(
                "reviewer_conditioned_response",
                "MisleadingReviewResistanceRate",
                correct_revise_correct,
                response_fp,
                source="trace+benchmark_evaluator",
            ),
        ]
    )
    return rows


def summarize_reviewer_feedback_incorporation(
    episodes: list[ReviewerFeedbackIncorporationEpisode],
) -> list[MetricRecord]:
    """Summarize whether agents follow reviewer feedback.

    This is distinct from whether following that feedback improves correctness.
    """

    eligible = [episode for episode in episodes if episode.incorporated_feedback is not None]
    correct_negative_feedback = [
        episode
        for episode in eligible
        if episode.action == "revise" and episode.pre_correct is False
    ]
    misleading_negative_feedback = [
        episode
        for episode in eligible
        if episode.action == "revise" and episode.pre_correct is True
    ]
    incorporated = sum(1 for episode in eligible if episode.incorporated_feedback is True)
    correct_incorporated = sum(
        1 for episode in correct_negative_feedback if episode.incorporated_feedback is True
    )
    misleading_incorporated = sum(
        1 for episode in misleading_negative_feedback if episode.incorporated_feedback is True
    )

    return [
        _count("reviewer_feedback_incorporation", "ReviewerFeedbackEpisodes", len(episodes)),
        _count(
            "reviewer_feedback_incorporation",
            "EligibleReviewerFeedbackEpisodes",
            len(eligible),
            source="trace+feedback_labeler",
        ),
        # ReviewerFeedbackIncorporationRate = feedback-following events / eligible review events.
        _rate(
            "reviewer_feedback_incorporation",
            "ReviewerFeedbackIncorporationRate",
            incorporated,
            len(eligible),
            source="trace+feedback_labeler",
        ),
        # UsefulReviseFeedbackIncorporationRate = followed useful revise feedback / useful revise feedback.
        _rate(
            "reviewer_feedback_incorporation",
            "UsefulReviseFeedbackIncorporationRate",
            correct_incorporated,
            len(correct_negative_feedback),
            source="trace+feedback_labeler+benchmark_evaluator",
        ),
        # MisleadingReviewSusceptibilityRate = followed misleading revise feedback / misleading revise feedback.
        _rate(
            "reviewer_feedback_incorporation",
            "MisleadingReviewSusceptibilityRate",
            misleading_incorporated,
            len(misleading_negative_feedback),
            source="trace+feedback_labeler+benchmark_evaluator",
        ),
    ]


def _review_tensor_counts(episodes: list[ReviewEpisode]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for episode in episodes:
        if episode.pre_correct is None or episode.post_correct is None:
            continue
        pre = "correct" if episode.pre_correct else "wrong"
        post = "correct" if episode.post_correct else "wrong"
        counts[f"{pre}_{episode.action}_{post}"] += 1

    for pre in ("wrong", "correct"):
        for action in ("revise", "agree"):
            for post in ("wrong", "correct"):
                counts.setdefault(f"{pre}_{action}_{post}", 0)
    return counts


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


def _balanced_accuracy(tp: int, fp: int, fn: int, tn: int) -> Optional[float]:
    recall = _safe_rate(tp, tp + fn)
    specificity = _safe_rate(tn, tn + fp)
    if recall is None or specificity is None:
        return None
    return (recall + specificity) / 2.0


def _safe_rate(numerator: int, denominator: int) -> Optional[float]:
    if denominator <= 0:
        return None
    return numerator / denominator
