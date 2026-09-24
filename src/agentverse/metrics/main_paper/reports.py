"""CSV and concise text reports for main-paper trace metrics."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

from agentverse.metrics.main_paper.records import (
    InnerTransitionRecord,
    MetricRecord,
    ProblemRecord,
)
from agentverse.metrics.repair import FeedbackEpisode


def write_metric_csv(path: Path, records: list[MetricRecord]) -> None:
    fieldnames = ["section", "metric", "value", "numerator", "denominator", "source", "notes"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = record.to_dict()
            for key in ("value", "numerator", "denominator"):
                row[key] = _cell(row.get(key))
            writer.writerow(row)


def render_basic_text(records: list[MetricRecord], problems: list[ProblemRecord]) -> str:
    lines = ["Basic Metrics", ""]
    lines.extend(_metric_lines(records))
    lines.extend(["", "Per Problem", ""])
    for problem in sorted(problems, key=lambda item: (item.trace_path, item.example_idx)):
        final = "PASS" if problem.final_passed else "FAIL"
        first = "yes" if problem.first_passed else "no"
        lines.append(
            (
                f"Example {problem.example_idx}: {final}; first_pass={first}; "
                f"system_tries={problem.correction_loops}; "
                f"success_try={_fmt(problem.first_success_attempt)}; "
                f"evaluator_submissions={problem.evaluator_submission_count}; "
                f"reflective_rounds={problem.reflective_rounds}; "
                f"turns={problem.reasoning_turns}; "
                f"tokens={problem.total_tokens}; model_calls={problem.model_calls}"
            )
        )
    return "\n".join(lines).rstrip() + "\n"


def render_outer_text(
    records: list[MetricRecord],
    problems: list[ProblemRecord],
    outer_scores: dict[int, dict[str, Any]],
) -> str:
    lines = ["Outer-Loop Metrics", ""]
    lines.extend(_metric_lines(records))
    lines.extend(["", "Per Problem", ""])
    for problem in sorted(problems, key=lambda item: (item.trace_path, item.example_idx)):
        score = outer_scores.get(problem.example_idx, {})
        final = "PASS" if problem.final_passed else "FAIL"
        lines.append(
            (
                f"Example {problem.example_idx}: {final}; "
                f"system_tries={problem.correction_loops}; "
                f"success_try={_fmt(problem.first_success_attempt)}; "
                f"evaluator_submissions={problem.evaluator_submission_count}; "
                f"failed_before_success={_fmt(score.get('failed_before_success'))}; "
                f"evaluator_hints={problem.hints_used}; "
                f"mrr={_fmt(score.get('mrr_attempt_accuracy'))}; "
                f"hint_score={_fmt(score.get('hint_aware_correction_score'))}"
            )
        )
    return "\n".join(lines).rstrip() + "\n"


def render_inner_text(
    records: list[MetricRecord],
    transitions: list[InnerTransitionRecord],
) -> str:
    lines = ["Inner-Loop Metrics", ""]
    main_records = [record for record in records if record.section == "inner_loop"]
    difficulty_records = [
        record for record in records if record.section == "inner_loop_by_difficulty"
    ]
    lines.extend(_metric_lines(main_records))
    if difficulty_records:
        lines.extend(["", "Difficulty-Sliced Transition Matrix", ""])
        lines.extend(_metric_lines(difficulty_records))
    lines.extend(["", "Per Inner Transition", ""])
    for transition in sorted(
        transitions,
        key=lambda item: (item.trace_path, item.example_idx, item.attempt_idx),
    ):
        lines.append(
            (
                f"Example {transition.example_idx} attempt {transition.attempt_idx}: "
                f"{transition.transition_name}; "
                f"initial={_bool_label(transition.initial_correct)}; "
                f"final={_bool_label(transition.final_correct)}; "
                f"answer_change={_candidate_change_label(transition.initial_candidate, transition.final_candidate)}; "
                f"candidates={transition.candidate_count}; "
                f"source={transition.correctness_source or 'unknown'}"
            )
        )
    return "\n".join(lines).rstrip() + "\n"


def render_system_feedback_text(
    records: list[MetricRecord],
    episodes: list[FeedbackEpisode],
) -> str:
    lines = ["System Feedback Incorporation", ""]
    lines.extend(_metric_lines(records))
    lines.extend(["", "Per Problem", ""])
    for key, group in _group_feedback_episodes(episodes).items():
        example_idx = key[1]
        counts = _system_feedback_counts(group)
        lines.append(
            (
                f"Example {example_idx}: feedback_episodes={len(group)}; "
                f"eligible={counts['eligible']}; incorporated={counts['incorporated']}; "
                f"next_pass={counts['fixed']}; "
                f"incorporate_and_fix={counts['incorporate_and_fix']}; "
                f"incorporate_but_wrong={counts['incorporate_but_wrong']}; "
                f"ignore_but_fix={counts['ignore_but_fix']}; "
                f"ignore_and_stay_wrong={counts['ignore_and_wrong']}"
            )
        )
    return "\n".join(lines).rstrip() + "\n"


def render_evaluator_feedback_text(
    records: list[MetricRecord],
    episodes: list[FeedbackEpisode],
) -> str:
    """Backward-compatible alias for the system-feedback report renderer."""
    return render_system_feedback_text(records, episodes)


def metrics_to_summary(records: list[MetricRecord]) -> dict[str, Any]:
    summary: dict[str, dict[str, Any]] = {}
    for record in records:
        summary.setdefault(record.section, {})[record.metric] = record.value
    return summary


def write_manifest(
    path: Path,
    *,
    input_path: Path | None,
    trace_files: list[Path],
    output_dir: Path,
    config_path: str | None,
    evaluator_type: str,
    examples_analyzed: int,
    outputs: dict[str, str],
) -> None:
    payload = {
        "input_path": str(input_path) if input_path is not None else None,
        "trace_files": [str(path) for path in trace_files],
        "output_dir": str(output_dir),
        "config_path": config_path,
        "evaluator_type": evaluator_type,
        "examples_analyzed": examples_analyzed,
        "output_files": outputs,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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


def metric_explanation(metric: str, *, notes: str | None = None) -> str | None:
    if metric == "FinalPassRate":
        return "Fraction of problems that eventually end in PASS."
    if metric == "Pass@1":
        return "Fraction of problems that pass on the first system-level try."
    if metric == "AvgCorrectionAttemptsToSuccess":
        return "Average first successful system-try index, computed only over solved problems."
    if metric == "TotalCorrectionAttemptsToSuccess":
        return "Sum of first successful attempt indices over solved problems."
    if metric == "AvgCorrectionLoops":
        return "Average number of system-level tries per problem."
    if metric == "TotalEvaluatorAttempts":
        return "Total number of evaluator submissions across all problems, counted separately from system-level tries."
    if metric == "AvgReflectiveRefinementRounds":
        return "Average number of within-attempt candidate-revision rounds before evaluator judgment."
    if metric == "AvgTotalTokens":
        return "Average total token usage per problem from the saved runtime metrics."
    if metric == "AvgModelCalls":
        return "Average number of model invocations per problem, using runtime counts with a trace-based lower bound when needed."
    if metric == "TotalModelCalls":
        return "Total number of model invocations across all problems, using runtime counts with a trace-based lower bound when needed."
    if metric == "AvgReasoningTurns":
        return "Average non-evaluator message count per problem."
    if metric == "MRRAttemptAccuracy":
        return "Attempt-sensitive accuracy: solved problems get 1 divided by the attempt where they first pass; unsolved problems get 0."
    if metric == "PassFailCorrectionScore":
        return "Same reciprocal-attempt recovery score under the current definition."
    if metric == "HintAwareCorrectionScore":
        return "Attempt-sensitive recovery score that also discounts problems requiring more evaluator hints before success."
    if metric == "DMRAS":
        return "Macro-average HintAwareCorrectionScore across difficulty tiers; n/a when tier metadata is unavailable."
    if metric.startswith("OuterPass@"):
        k = metric.split("@", 1)[1]
        return f"Fraction of problems solved within the first {k} system-level tries."
    if metric.startswith("ConditionalRecoveryRate@"):
        k = metric.split("@", 1)[1]
        return (
            f"Among problems that reached system try {k}, fraction first solved on that try."
        )
    if metric.startswith("MarginalRecoveryGain_"):
        k = metric.split("_", 1)[1]
        return f"Additional solve rate gained exactly when allowing system try {k}."
    if " -> " in metric:
        return "Attempt-sensitive recovery score on this domain slice."
    if metric in {"SystemFeedbackEpisodes", "EvaluatorFeedbackEpisodes"}:
        return "Number of failed system-feedback episodes that are followed by an actual next repair path."
    if metric in {"EligibleSystemFeedbackEpisodes", "EligibleEvaluatorFeedbackEpisodes"}:
        return "Number of failed system-feedback episodes with enough information to judge both uptake and next-try outcome."
    if metric in {"SystemFeedbackIncorporationRate", "EvaluatorFeedbackIncorporationRate"}:
        return "Among eligible system-feedback episodes, fraction where the repair path visibly follows the system feedback."
    if metric in {"PostSystemFeedbackRepairRate", "PostEvaluatorRepairRate"}:
        return "Among eligible system-feedback episodes, fraction where the next system try passes."
    if metric in {"SystemIncorporateAndFix", "EvaluatorIncorporateAndFix"}:
        return "Fraction of eligible system-feedback episodes where the system both follows the feedback and fixes the answer."
    if metric in {"SystemIncorporateButStillWrong", "EvaluatorIncorporateButStillWrong"}:
        return "Fraction of eligible system-feedback episodes where the system follows the feedback but the next try still fails."
    if metric in {"SystemIgnoreButFix", "EvaluatorIgnoreButFix"}:
        return "Fraction of eligible system-feedback episodes where the system does not visibly follow the feedback but still fixes the answer."
    if metric in {"SystemIgnoreAndStayWrong", "EvaluatorIgnoreAndStayWrong"}:
        return "Fraction of eligible system-feedback episodes where the system neither follows the feedback nor fixes the answer."
    if metric in {
        "RepairGivenSystemFeedbackIncorporation",
        "RepairGivenEvaluatorFeedbackIncorporation",
    }:
        return "Conditional repair rate after system feedback is actually incorporated."
    if metric == "Transition_wrong_to_wrong":
        return "Fraction of inner-loop candidate transitions that start wrong and end wrong."
    if metric == "Transition_wrong_to_correct":
        return "Fraction of inner-loop candidate transitions that start wrong and end correct."
    if metric == "Transition_correct_to_correct":
        return "Fraction of inner-loop candidate transitions that start correct and stay correct."
    if metric == "Transition_correct_to_wrong":
        return "Fraction of inner-loop candidate transitions that start correct and become wrong."
    if metric == "ReflectiveRepairRate":
        return "Among wrong starting candidates, fraction repaired to correct by the end of the same inner loop."
    if metric == "ReflectiveStillWrongRate":
        return "Among wrong starting candidates, fraction that are still wrong at the end of the same inner loop, whether unchanged or revised."
    if metric == "ReflectiveNeglectRate":
        return "Among wrong starting candidates, fraction that keep the same wrong answer by the end of the same inner loop."
    if metric == "ReflectiveTryButFailRate":
        return "Among wrong starting candidates, fraction that change the answer but still end wrong by the end of the same inner loop."
    if metric == "ReflectivePreservationRate":
        return "Among correct starting candidates, fraction preserved as correct by the end of the same inner loop."
    if metric == "ReflectiveHarmRate":
        return "Among correct starting candidates, fraction harmed into wrong answers by the end of the same inner loop."
    if metric == "ReflectiveNetRepairGain":
        return "Net correctness gain across all inner-loop transitions: repairs minus harms, normalized by all transitions."
    if metric == "ReviewEpisodes":
        return "Number of extracted review events where a reviewer explicitly agrees or asks for revision."
    if metric == "EligibleReviewEpisodes":
        return "Number of review events whose pre-review candidate correctness could be judged."
    if metric == "TP_review":
        return "Reviewer correctly asked for revision on a wrong answer."
    if metric == "FP_review":
        return "Reviewer incorrectly asked for revision on a correct answer."
    if metric == "FN_review":
        return "Reviewer incorrectly agreed with a wrong answer."
    if metric == "TN_review":
        return "Reviewer correctly agreed with a correct answer."
    if metric == "Prec_review":
        return "When the reviewer says revise, how often that revise signal is correct."
    if metric == "Rec_review":
        return "When the answer is actually wrong, how often the reviewer catches it with revise."
    if metric == "FAR_review":
        return "False-alarm rate: when the answer is actually correct, how often the reviewer still says revise."
    if metric == "BalAcc_review":
        return "Balanced reviewer detection accuracy, averaging true-positive rate and true-negative rate."
    if metric == "EligibleCorrectReviewEpisodes":
        return "Number of useful revise episodes: reviewer said revise and the pre-review answer was wrong."
    if metric == "EligibleMisleadingReviewEpisodes":
        return "Number of misleading revise episodes: reviewer said revise even though the pre-review answer was correct."
    if metric == "ReviewerGuidedRepairRate":
        return "Among useful revise episodes, fraction where the next answer becomes correct."
    if metric == "ReviewerDetectedButNotFixedRate":
        return "Among useful revise episodes, fraction where the answer stays wrong despite correct detection."
    if metric == "MisleadingReviewHarmRate":
        return "Among misleading revise episodes, fraction where the correct answer gets harmed into a wrong one."
    if metric == "MisleadingReviewResistanceRate":
        return "Among misleading revise episodes, fraction where the answer stays correct despite bad review advice."
    if metric.startswith("C_"):
        return _tensor_explanation(metric)
    if metric == "ReviewerFeedbackEpisodes":
        return "Number of review episodes whose downstream answer can be checked for follow-through."
    if metric == "EligibleReviewerFeedbackEpisodes":
        return "Number of reviewer feedback episodes with enough information to judge whether the system followed the review signal."
    if metric == "ReviewerFeedbackIncorporationRate":
        return "Fraction of eligible review episodes where the system follows the review signal: change after revise, or preserve after agree."
    if metric == "UsefulReviseFeedbackIncorporationRate":
        return "Among useful revise episodes, fraction where the system actually follows that revise signal."
    if metric == "MisleadingReviewSusceptibilityRate":
        return "Among misleading revise episodes, fraction where the system still follows the bad revise signal."
    if metric == "ExamplesAnalyzed":
        return "Number of problems included in this diagnostics summary."
    if metric == "AgreementEventRate":
        return "Fraction of review events that end in explicit agreement."
    if metric == "WrongConvergenceRate":
        return "Among agreement events with a later evaluator label, fraction that are later proven wrong."
    if metric == "ProductiveConvergenceRate":
        return "Among agreement events with a later evaluator label, fraction that are later validated as correct."
    if metric == "PrematureConvergenceSignalRate":
        return "Fraction of problems showing at least one agreement event that is later followed by evaluator FAIL."
    if metric == "RepeatedAnswerRate":
        return "Fraction of adjacent submission pairs where the boxed candidate answer does not change."
    if metric == "NoChangeAfterFailRate":
        return "Fraction of evaluator FAIL episodes followed by no candidate change before the next attempt."
    if metric == "LoopingSignalRate":
        return "Fraction of problems flagged by the deterministic looping heuristic."
    if metric == "AvgDistinctCandidateCount":
        return "Average number of distinct candidate answers proposed per problem."
    if metric == "AvgTurnImbalance":
        return "Average imbalance in how unevenly turns are distributed across participating agents."
    if metric.startswith("TurnShare_"):
        role = notes or metric.split("TurnShare_", 1)[-1]
        return f"Fraction of all non-evaluator turns contributed by role or agent `{role}`."
    if metric == "MatchedSubsetSize":
        return "Number of example IDs present in both the plain and hint traces for paired comparison."
    if metric.startswith("HintMinusPlain_"):
        base_metric = metric.split("HintMinusPlain_", 1)[1]
        return f"Hint-run minus plain-run difference for {base_metric} on matched examples."
    if metric == "MarginalHintGain":
        return "Final-pass-rate lift per additional average hint on the matched hint-versus-plain subset."
    return None


def _tensor_explanation(metric: str) -> str | None:
    match = re.fullmatch(r"C_(correct|wrong)_(agree|revise)_(correct|wrong)", metric)
    if not match:
        return None
    pre, action, post = match.groups()
    return (
        "Count of review episodes with "
        f"pre-review answer `{pre}`, reviewer action `{action}`, and post-review answer `{post}`."
    )


def _group_feedback_episodes(
    episodes: list[FeedbackEpisode],
) -> dict[tuple[str, int], list[FeedbackEpisode]]:
    grouped: dict[tuple[str, int], list[FeedbackEpisode]] = {}
    for episode in episodes:
        key = (episode.trace_path, episode.example_idx)
        grouped.setdefault(key, []).append(episode)
    return dict(sorted(grouped.items(), key=lambda item: item[0]))


def _system_feedback_counts(episodes: list[FeedbackEpisode]) -> dict[str, int]:
    counts = {
        "eligible": 0,
        "incorporated": 0,
        "fixed": 0,
        "incorporate_and_fix": 0,
        "incorporate_but_wrong": 0,
        "ignore_but_fix": 0,
        "ignore_and_wrong": 0,
    }
    for episode in episodes:
        if episode.used_feedback is None or episode.next_submission_correct is None:
            continue
        counts["eligible"] += 1
        if episode.used_feedback is True:
            counts["incorporated"] += 1
        if episode.next_submission_correct is True:
            counts["fixed"] += 1
        if episode.used_feedback is True and episode.next_submission_correct is True:
            counts["incorporate_and_fix"] += 1
        if episode.used_feedback is True and episode.next_submission_correct is False:
            counts["incorporate_but_wrong"] += 1
        if episode.used_feedback is False and episode.next_submission_correct is True:
            counts["ignore_but_fix"] += 1
        if episode.used_feedback is False and episode.next_submission_correct is False:
            counts["ignore_and_wrong"] += 1
    return counts


def _evaluator_feedback_counts(episodes: list[FeedbackEpisode]) -> dict[str, int]:
    """Backward-compatible alias for system-feedback count summaries."""
    return _system_feedback_counts(episodes)


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _bool_label(value: bool | None) -> str:
    if value is True:
        return "correct"
    if value is False:
        return "wrong"
    return "unknown"


def _candidate_change_label(left: str, right: str) -> str:
    left_key = _candidate_key(left)
    right_key = _candidate_key(right)
    if not left_key or not right_key:
        return "unknown"
    return "same" if left_key == right_key else "changed"


def _candidate_key(candidate: str) -> str:
    return re.sub(r"\s+", "", str(candidate or "").strip()).lower()
