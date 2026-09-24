"""Offline trace-analysis helpers for repair, review, and system-feedback behavior."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from agentverse.evaluation.common import extract_boxed
from agentverse.evaluation.content_based import TraceAnalysisLabeler, normalize_candidate_text


_AGENT_LINE_RE = re.compile(r"^(A\d+): name=(.*?) llm_type=.* model=.*$")
_TURN_RE = re.compile(r"^(T\d+(?:_\d+)?): ([^:]+): (.*)$")
_APPROVE_RE = re.compile(r"\bApprove:\s*(True|False)\b", re.IGNORECASE)
_REVIEW_POSITION_RE = re.compile(r"\bReview Position:\s*([A-Za-z0-9_-]+)\b", re.IGNORECASE)
_ROUTE_RE = re.compile(r"\[Route:[^\]]+\]", re.IGNORECASE)
_AGREE_RE = re.compile(r"\[Agree\]", re.IGNORECASE)
_SELF_REVISE_RE = re.compile(r"\[(?:Self-Revise|Revise)\]", re.IGNORECASE)
_SCORE_RE = re.compile(r"\bScore:\s*(True|False)\b", re.IGNORECASE)
_EVAL_SIGNAL_RE = re.compile(r"\bEvaluation signal:\s*(PASS|FAIL)\b", re.IGNORECASE)
_VERIFIER_RE = re.compile(r"\bVerifier:\s*(PASS|FAIL)\b", re.IGNORECASE)
_EVAL_HINT_RE = re.compile(r"Evaluation hint:", re.IGNORECASE)
_CANDIDATE_UNDER_REVIEW_RE = re.compile(
    r"Candidate under review:\s*(.*?)(?=\s+(?:Approve:|Review Position:|Feedback:|Peer Review:|Proposed Correction:|Suggested Revision:)|$)",
    re.IGNORECASE,
)
_GROUP_CANDIDATE_RE = re.compile(
    r"Candidate answer under group review:\s*Source:.*?Candidate Answer:\s*(.*)$",
    re.IGNORECASE,
)
_GROUP_CANDIDATE_UPDATE_RE = re.compile(
    r"Current Candidate For Further Review:\s*(.*?)(?=\s+Instruction:|$)",
    re.IGNORECASE,
)
_PROPOSED_CORRECTION_RE = re.compile(
    r"(?:Proposed Correction|Suggested Revision):\s*(.*)$",
    re.IGNORECASE,
)
_JSON_REVISED_RE = re.compile(r'"revised_answer"\s*:\s*"([^"]*)"', re.IGNORECASE)


@dataclass
class TraceTurn:
    turn_idx: int
    turn_label: str
    agent_id: str
    agent_name: str
    content: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TraceExample:
    trace_path: str
    task: str
    protocol: str
    example_idx: int
    input_text: str
    label: str
    agent_map: dict[str, str]
    turns: list[TraceTurn]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["turns"] = [turn.to_dict() for turn in self.turns]
        return payload


@dataclass
class ReviewEpisode:
    trace_path: str
    task: str
    protocol: str
    example_idx: int
    episode_id: str
    review_turn: int
    reviewer: str
    action: str
    candidate_before: str
    candidate_after: str
    pre_correct: Optional[bool]
    post_correct: Optional[bool]
    pre_correct_source: str
    post_correct_source: str
    after_evaluator_fail: bool
    evidence_turns: list[int]
    review_feedback: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FeedbackEpisode:
    trace_path: str
    task: str
    protocol: str
    example_idx: int
    episode_id: str
    evaluator_turn: int
    evaluator_feedback: str
    candidate_before_feedback: str
    repair_path_turns: list[int]
    used_feedback: Optional[bool]
    used_feedback_source: str
    next_submission_correct: Optional[bool]
    evidence_turns: list[int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _turn_ref(turn: TraceTurn) -> str:
    return str(turn.turn_label or f"T{turn.turn_idx}")


def parse_trace_file(path: str | Path) -> list[TraceExample]:
    """Parse a `.trace.txt` file into example-level records."""
    trace_path = Path(path)
    text = trace_path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    agent_map = _parse_agent_map(lines)
    task = _parse_task_name(text)
    protocol = _infer_protocol(task, agent_map)

    chunks = re.split(r"\n={80}\n", text)
    examples: list[TraceExample] = []
    for chunk in chunks:
        stripped = chunk.strip()
        if not stripped.startswith("EXAMPLE "):
            continue

        example = _parse_example_chunk(
            chunk=stripped,
            trace_path=str(trace_path),
            task=task,
            protocol=protocol,
            agent_map=agent_map,
        )
        if example is not None:
            examples.append(example)

    return examples


def analyze_trace_examples(
    examples: list[TraceExample],
    *,
    labeler: TraceAnalysisLabeler,
) -> dict[str, Any]:
    """Analyze parsed examples and compute summary metrics."""
    review_episodes: list[ReviewEpisode] = []
    feedback_episodes: list[FeedbackEpisode] = []

    for example in examples:
        review_episodes.extend(_extract_review_episodes(example, labeler))
        feedback_episodes.extend(_extract_feedback_episodes(example, labeler))

    summary = {
        "trace_files": sorted({example.trace_path for example in examples}),
        "tasks": sorted({example.task for example in examples if example.task}),
        "protocols": _count_protocols(examples),
        "examples_analyzed": len(examples),
        "review_behavior": _summarize_review_behavior(review_episodes),
        "review_behavior_after_fail": _summarize_review_behavior(
            [episode for episode in review_episodes if episode.after_evaluator_fail]
        ),
        "system_feedback_learning": _summarize_system_feedback_learning(feedback_episodes),
        "evaluator_feedback_learning": _summarize_system_feedback_learning(feedback_episodes),
    }

    return {
        "review_episodes": [episode.to_dict() for episode in review_episodes],
        "feedback_episodes": [episode.to_dict() for episode in feedback_episodes],
        "summary": summary,
    }


def render_trace_analysis_report(summary: dict[str, Any]) -> str:
    """Render a short Markdown report for the trace analysis."""
    review = summary.get("review_behavior", {})
    review_fail = summary.get("review_behavior_after_fail", {})
    feedback = summary.get("system_feedback_learning") or summary.get("evaluator_feedback_learning", {})

    lines = [
        "# Trace Analysis Report",
        "",
        f"- Examples analyzed: {summary.get('examples_analyzed', 0)}",
        f"- Protocol counts: {json.dumps(summary.get('protocols', {}), ensure_ascii=False)}",
        "",
        "## Reviewer Behavior",
        "",
        f"- Review episodes: {review.get('episodes', 0)}",
        f"- `TP_review`: {review.get('TP_review', 0)}",
        f"- `FP_review`: {review.get('FP_review', 0)}",
        f"- `FN_review`: {review.get('FN_review', 0)}",
        f"- `TN_review`: {review.get('TN_review', 0)}",
        f"- `Prec_review`: {_fmt_rate(review.get('Prec_review'))}",
        f"- `Rec_review`: {_fmt_rate(review.get('Rec_review'))}",
        f"- `FAR_review`: {_fmt_rate(review.get('FAR_review'))}",
        f"- `BalAcc_review`: {_fmt_rate(review.get('BalAcc_review'))}",
        f"- `FixWrong`: {_fmt_rate(review.get('FixWrong'))}",
        f"- `CorruptRight`: {_fmt_rate(review.get('CorruptRight'))}",
        f"- `AgreeRight`: {_fmt_rate(review.get('AgreeRight'))}",
        f"- `AgreeWrong`: {_fmt_rate(review.get('AgreeWrong'))}",
        "",
        "## Reviewer Behavior After Evaluator FAIL",
        "",
        f"- Review episodes: {review_fail.get('episodes', 0)}",
        f"- `FixWrong`: {_fmt_rate(review_fail.get('FixWrong'))}",
        f"- `CorruptRight`: {_fmt_rate(review_fail.get('CorruptRight'))}",
        f"- `AgreeRight`: {_fmt_rate(review_fail.get('AgreeRight'))}",
        f"- `AgreeWrong`: {_fmt_rate(review_fail.get('AgreeWrong'))}",
        "",
        "## System-Feedback Learning",
        "",
        f"- Eligible system-feedback episodes: {feedback.get('eligible_feedback_episodes', feedback.get('eligible_failures', 0))}",
        f"- `SystemFeedbackIncorporationRate`: {_fmt_rate(feedback.get('SystemFeedbackIncorporationRate', feedback.get('FeedbackUptake')))}",
        f"- `PostSystemFeedbackRepairRate`: {_fmt_rate(feedback.get('PostSystemFeedbackRepairRate', feedback.get('LearnFromFeedback')))}",
        f"- `SystemIncorporateAndFix`: {_fmt_rate(feedback.get('SystemIncorporateAndFix', feedback.get('UseAndFix')))}",
        f"- `SystemIncorporateButStillWrong`: {_fmt_rate(feedback.get('SystemIncorporateButStillWrong', feedback.get('UseButStillWrong')))}",
        f"- `SystemIgnoreButFix`: {_fmt_rate(feedback.get('SystemIgnoreButFix', feedback.get('IgnoreButFix')))}",
        f"- `SystemIgnoreAndStayWrong`: {_fmt_rate(feedback.get('SystemIgnoreAndStayWrong', feedback.get('IgnoreAndStayWrong')))}",
        "",
    ]
    return "\n".join(lines)


def _parse_agent_map(lines: list[str]) -> dict[str, str]:
    agent_map: dict[str, str] = {}
    for line in lines:
        match = _AGENT_LINE_RE.match(line.strip())
        if match:
            agent_map[match.group(1)] = match.group(2).strip()
    return agent_map


def _parse_task_name(text: str) -> str:
    match = re.search(r'"task"\s*:\s*"([^"]+)"', text)
    return match.group(1).strip() if match else ""


def _infer_protocol(task: str, agent_map: dict[str, str]) -> str:
    normalized_task = str(task or "").lower()
    names = " ".join(agent_map.values()).lower()
    if "broadcast" in normalized_task or "deliberation" in normalized_task:
        return "broadcast"
    if "single_llm" in normalized_task or "single-llm" in normalized_task:
        return "baseline_llm"
    if "baseline" in normalized_task and "llm" in normalized_task:
        return "baseline_llm"
    if "single_agent" in normalized_task or "single-agent" in normalized_task:
        return "single_agent"
    if "single" in normalized_task and "reflect" in normalized_task:
        return "single_agent"
    if "planner" in names and "executor" in names and "reviewer" in names:
        return "per"
    if "per" in normalized_task:
        return "per"
    return "unknown"


def _parse_example_chunk(
    *,
    chunk: str,
    trace_path: str,
    task: str,
    protocol: str,
    agent_map: dict[str, str],
) -> Optional[TraceExample]:
    lines = chunk.splitlines()
    if not lines:
        return None

    header = lines[0].strip()
    match = re.match(r"EXAMPLE\s+(\d+)", header)
    if not match:
        return None
    example_idx = int(match.group(1))

    input_text = ""
    label = ""
    turns: list[TraceTurn] = []
    for line in lines[1:]:
        if line.startswith("input: "):
            input_text = line[len("input: ") :].strip()
            continue
        if line.startswith("label: "):
            label = line[len("label: ") :].strip()
            continue
        turn_match = _TURN_RE.match(line.strip())
        if not turn_match:
            continue
        agent_id = turn_match.group(2).strip()
        turns.append(
            TraceTurn(
                turn_idx=len(turns) + 1,
                turn_label=turn_match.group(1),
                agent_id=agent_id,
                agent_name=agent_map.get(agent_id, agent_id),
                content=turn_match.group(3).strip(),
            )
        )

    return TraceExample(
        trace_path=trace_path,
        task=task,
        protocol=protocol,
        example_idx=example_idx,
        input_text=input_text,
        label=label,
        agent_map=dict(agent_map),
        turns=turns,
    )


def _extract_review_episodes(
    example: TraceExample,
    labeler: TraceAnalysisLabeler,
) -> list[ReviewEpisode]:
    if example.protocol == "broadcast":
        return _extract_broadcast_review_episodes(example, labeler)
    if example.protocol == "single_agent":
        return _extract_single_agent_review_episodes(example, labeler)
    return _extract_per_review_episodes(example, labeler)


def _extract_per_review_episodes(
    example: TraceExample,
    labeler: TraceAnalysisLabeler,
) -> list[ReviewEpisode]:
    reviewer_names = {
        name for name in example.agent_map.values() if "reviewer" in name.lower() or "critic" in name.lower()
    }
    evaluator_names = {name for name in example.agent_map.values() if "evaluator" in name.lower()}
    episodes: list[ReviewEpisode] = []
    last_candidate = ""
    after_fail = False

    for idx, turn in enumerate(example.turns):
        if _is_evaluator_turn(turn, evaluator_names):
            passed = _parse_evaluator_pass_fail(turn.content)
            if passed is not None:
                after_fail = not passed
            continue

        if turn.agent_name not in reviewer_names:
            turn_candidate = _extract_any_candidate(turn.content)
            if turn_candidate:
                last_candidate = turn_candidate
            continue

        action = _per_review_action(turn.content)
        reviewer_candidate = _extract_any_candidate(turn.content)
        if action is None:
            if reviewer_candidate:
                last_candidate = reviewer_candidate
            continue

        candidate_before = _extract_candidate_under_review(turn.content) or last_candidate
        candidate_after = candidate_before
        if reviewer_candidate:
            candidate_after = reviewer_candidate
        elif action == "revise":
            candidate_after = _find_next_candidate(example.turns, idx + 1) or candidate_before

        pre = labeler.judge_correctness(
            problem=example.input_text,
            gold_answer=example.label,
            candidate_answer=candidate_before,
        )
        post = labeler.judge_correctness(
            problem=example.input_text,
            gold_answer=example.label,
            candidate_answer=candidate_after,
        )

        episodes.append(
            ReviewEpisode(
                trace_path=example.trace_path,
                task=example.task,
                protocol=example.protocol,
                example_idx=example.example_idx,
                episode_id=f"{example.example_idx}.review.{len(episodes) + 1}",
                review_turn=turn.turn_idx,
                reviewer=turn.agent_name,
                action=action,
                candidate_before=candidate_before,
                candidate_after=candidate_after,
                pre_correct=pre.value,
                post_correct=post.value,
                pre_correct_source=pre.source,
                post_correct_source=post.source,
                after_evaluator_fail=after_fail,
                evidence_turns=[turn.turn_idx],
                review_feedback=turn.content,
            )
        )

        if reviewer_candidate:
            last_candidate = reviewer_candidate

    return episodes


def _extract_broadcast_review_episodes(
    example: TraceExample,
    labeler: TraceAnalysisLabeler,
) -> list[ReviewEpisode]:
    evaluator_names = {name for name in example.agent_map.values() if "evaluator" in name.lower()}
    episodes: list[ReviewEpisode] = []
    after_fail = False
    turns = example.turns
    idx = 0

    while idx < len(turns):
        turn = turns[idx]
        if _is_evaluator_turn(turn, evaluator_names):
            passed = _parse_evaluator_pass_fail(turn.content)
            if passed is not None:
                after_fail = not passed
            idx += 1
            continue

        candidate_under_review = _extract_candidate_under_review(turn.content)
        approve = _parse_approve_flag(turn.content)
        if not candidate_under_review or approve is None:
            idx += 1
            continue
        if idx > 0:
            prev_turn = turns[idx - 1]
            prev_candidate = _extract_candidate_under_review(prev_turn.content)
            prev_approve = _parse_approve_flag(prev_turn.content)
            if prev_candidate == candidate_under_review and prev_approve is not None:
                idx += 1
                continue

        episode, next_idx = _aggregate_broadcast_group_review_event(
            example=example,
            labeler=labeler,
            start_idx=idx,
            candidate_before=candidate_under_review,
            after_fail=after_fail,
            evaluator_names=evaluator_names,
            episode_number=len(episodes) + 1,
        )
        if episode is not None:
            episodes.append(episode)
        idx = next_idx

    return episodes


def _extract_single_agent_review_episodes(
    example: TraceExample,
    labeler: TraceAnalysisLabeler,
) -> list[ReviewEpisode]:
    evaluator_names = {name for name in example.agent_map.values() if "evaluator" in name.lower()}
    episodes: list[ReviewEpisode] = []
    last_candidate = ""
    after_fail = False

    for idx, turn in enumerate(example.turns):
        if _is_evaluator_turn(turn, evaluator_names):
            passed = _parse_evaluator_pass_fail(turn.content)
            if passed is not None:
                after_fail = not passed
            continue

        turn_candidate = _extract_any_candidate(turn.content)
        action = _single_agent_review_action(turn.content)
        if action is None:
            if turn_candidate:
                last_candidate = turn_candidate
            continue

        candidate_before = _extract_candidate_under_review(turn.content) or last_candidate
        candidate_after = candidate_before
        if action == "revise":
            candidate_after = _find_next_candidate(example.turns, idx + 1) or candidate_before

        pre = labeler.judge_correctness(
            problem=example.input_text,
            gold_answer=example.label,
            candidate_answer=candidate_before,
        )
        post = labeler.judge_correctness(
            problem=example.input_text,
            gold_answer=example.label,
            candidate_answer=candidate_after,
        )

        episodes.append(
            ReviewEpisode(
                trace_path=example.trace_path,
                task=example.task,
                protocol=example.protocol,
                example_idx=example.example_idx,
                episode_id=f"{example.example_idx}.review.{len(episodes) + 1}",
                review_turn=turn.turn_idx,
                reviewer=turn.agent_name,
                action=action,
                candidate_before=candidate_before,
                candidate_after=candidate_after,
                pre_correct=pre.value,
                post_correct=post.value,
                pre_correct_source=pre.source,
                post_correct_source=post.source,
                after_evaluator_fail=after_fail,
                evidence_turns=[turn.turn_idx],
                review_feedback=turn.content,
            )
        )

        if candidate_after:
            last_candidate = candidate_after

    return episodes


def _extract_feedback_episodes(
    example: TraceExample,
    labeler: TraceAnalysisLabeler,
) -> list[FeedbackEpisode]:
    evaluator_names = {name for name in example.agent_map.values() if "evaluator" in name.lower()}
    episodes: list[FeedbackEpisode] = []
    last_candidate = ""

    evaluator_turn_positions = [
        idx
        for idx, turn in enumerate(example.turns)
        if _is_evaluator_turn(turn, evaluator_names)
        and _parse_evaluator_pass_fail(turn.content) is not None
    ]
    if not evaluator_turn_positions:
        return episodes

    for pos_idx, turn_pos in enumerate(evaluator_turn_positions):
        turn = example.turns[turn_pos]
        passed = _parse_evaluator_pass_fail(turn.content)
        if passed is not False:
            candidate = _extract_any_candidate(turn.content)
            if candidate:
                last_candidate = candidate
            continue

        candidate_before_feedback = last_candidate or _find_previous_candidate(example.turns, turn_pos)
        feedback_turns = [turn]
        repair_start_pos = turn_pos + 1
        while repair_start_pos < len(example.turns) and _is_followup_feedback_turn(
            example.turns[repair_start_pos],
            evaluator_names,
        ):
            feedback_turns.append(example.turns[repair_start_pos])
            repair_start_pos += 1

        next_eval_pos = (
            evaluator_turn_positions[pos_idx + 1] if pos_idx + 1 < len(evaluator_turn_positions) else None
        )
        repair_turns = (
            example.turns[repair_start_pos:next_eval_pos]
            if next_eval_pos is not None
            else example.turns[repair_start_pos:]
        )
        if not repair_turns:
            continue

        repair_path = "\n".join(
            f"{_turn_ref(item)} {item.agent_name}: {item.content}" for item in repair_turns
        )
        uptake = labeler.judge_feedback_incorporation(
            feedback_text="\n".join(item.content for item in feedback_turns),
            response_path=repair_path,
            feedback_source="system",
        )

        next_submission_correct: Optional[bool] = None
        if next_eval_pos is not None:
            next_submission_correct = _parse_evaluator_pass_fail(example.turns[next_eval_pos].content)

        episodes.append(
            FeedbackEpisode(
                trace_path=example.trace_path,
                task=example.task,
                protocol=example.protocol,
                example_idx=example.example_idx,
                episode_id=f"{example.example_idx}.feedback.{len(episodes) + 1}",
                evaluator_turn=turn.turn_idx,
                evaluator_feedback="\n".join(item.content for item in feedback_turns),
                candidate_before_feedback=candidate_before_feedback,
                repair_path_turns=[item.turn_idx for item in repair_turns],
                used_feedback=uptake.value,
                used_feedback_source=uptake.source,
                next_submission_correct=next_submission_correct,
                evidence_turns=[item.turn_idx for item in feedback_turns] + [item.turn_idx for item in repair_turns],
            )
        )

        later_candidate = _find_previous_candidate(example.turns, next_eval_pos) if next_eval_pos is not None else ""
        if later_candidate:
            last_candidate = later_candidate

    return episodes


def _summarize_review_behavior(episodes: list[ReviewEpisode]) -> dict[str, Any]:
    summary: dict[str, Any] = {"episodes": len(episodes)}
    eligible = [episode for episode in episodes if episode.pre_correct is not None]
    if not eligible:
        summary.update(
            {
                "TP_review": 0,
                "FP_review": 0,
                "FN_review": 0,
                "TN_review": 0,
                "Prec_review": None,
                "Rec_review": None,
                "FAR_review": None,
                "BalAcc_review": None,
                "FixWrong": None,
                "CorruptRight": None,
                "AgreeRight": None,
                "AgreeWrong": None,
            }
        )
        return summary

    tp = sum(1 for episode in eligible if episode.action == "revise" and episode.pre_correct is False)
    fp = sum(1 for episode in eligible if episode.action == "revise" and episode.pre_correct is True)
    fn = sum(1 for episode in eligible if episode.action == "agree" and episode.pre_correct is False)
    tn = sum(1 for episode in eligible if episode.action == "agree" and episode.pre_correct is True)

    transitions = [episode for episode in eligible if episode.post_correct is not None]
    wrong_inputs = sum(1 for episode in transitions if episode.pre_correct is False)
    right_inputs = sum(1 for episode in transitions if episode.pre_correct is True)
    wrong_to_right = sum(
        1 for episode in transitions if episode.pre_correct is False and episode.post_correct is True
    )
    right_to_wrong = sum(
        1 for episode in transitions if episode.pre_correct is True and episode.post_correct is False
    )

    summary.update(
        {
            "TP_review": tp,
            "FP_review": fp,
            "FN_review": fn,
            "TN_review": tn,
            "Prec_review": _rate(tp, tp + fp),
            "Rec_review": _rate(tp, tp + fn),
            "FAR_review": _rate(fp, fp + tn),
            "BalAcc_review": _balanced_accuracy(tp, fp, fn, tn),
            "FixWrong": _rate(wrong_to_right, wrong_inputs),
            "CorruptRight": _rate(right_to_wrong, right_inputs),
            "AgreeRight": _rate(tn, tn + fp),
            "AgreeWrong": _rate(fn, fn + tp),
        }
    )
    return summary


def _summarize_system_feedback_learning(episodes: list[FeedbackEpisode]) -> dict[str, Any]:
    eligible = [
        episode
        for episode in episodes
        if episode.used_feedback is not None and episode.next_submission_correct is not None
    ]
    summary: dict[str, Any] = {
        "episodes": len(episodes),
        "system_feedback_episodes": len(episodes),
        "eligible_failures": len(eligible),
        "eligible_feedback_episodes": len(eligible),
    }
    if not eligible:
        summary.update(
            {
                "FeedbackUptake": None,
                "LearnFromFeedback": None,
                "UseAndFix": None,
                "UseButStillWrong": None,
                "IgnoreButFix": None,
                "IgnoreAndStayWrong": None,
                "SystemFeedbackIncorporationRate": None,
                "PostSystemFeedbackRepairRate": None,
                "SystemIncorporateAndFix": None,
                "SystemIncorporateButStillWrong": None,
                "SystemIgnoreButFix": None,
                "SystemIgnoreAndStayWrong": None,
            }
        )
        return summary

    used = sum(1 for episode in eligible if episode.used_feedback is True)
    learned = sum(1 for episode in eligible if episode.next_submission_correct is True)
    use_and_fix = sum(
        1 for episode in eligible if episode.used_feedback is True and episode.next_submission_correct is True
    )
    use_but_still_wrong = sum(
        1 for episode in eligible if episode.used_feedback is True and episode.next_submission_correct is False
    )
    ignore_but_fix = sum(
        1 for episode in eligible if episode.used_feedback is False and episode.next_submission_correct is True
    )
    ignore_and_stay_wrong = sum(
        1 for episode in eligible if episode.used_feedback is False and episode.next_submission_correct is False
    )

    denom = len(eligible)
    summary.update(
        {
            "FeedbackUptake": _rate(used, denom),
            "LearnFromFeedback": _rate(learned, denom),
            "UseAndFix": _rate(use_and_fix, denom),
            "UseButStillWrong": _rate(use_but_still_wrong, denom),
            "IgnoreButFix": _rate(ignore_but_fix, denom),
            "IgnoreAndStayWrong": _rate(ignore_and_stay_wrong, denom),
            "SystemFeedbackIncorporationRate": _rate(used, denom),
            "PostSystemFeedbackRepairRate": _rate(learned, denom),
            "SystemIncorporateAndFix": _rate(use_and_fix, denom),
            "SystemIncorporateButStillWrong": _rate(use_but_still_wrong, denom),
            "SystemIgnoreButFix": _rate(ignore_but_fix, denom),
            "SystemIgnoreAndStayWrong": _rate(ignore_and_stay_wrong, denom),
        }
    )
    return summary


def _summarize_feedback_learning(episodes: list[FeedbackEpisode]) -> dict[str, Any]:
    """Backward-compatible alias for system-feedback summaries."""
    return _summarize_system_feedback_learning(episodes)


def _count_protocols(examples: list[TraceExample]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for example in examples:
        counts[example.protocol] = counts.get(example.protocol, 0) + 1
    return counts


def _per_review_action(content: str) -> Optional[str]:
    text = str(content or "")
    if _AGREE_RE.search(text):
        return "agree"
    if _ROUTE_RE.search(text) or "Fix Instruction:" in text or "Diagnosis:" in text:
        return "revise"
    return None


def _single_agent_review_action(content: str) -> Optional[str]:
    text = str(content or "")
    if "Candidate under review:" not in text:
        return None
    if _AGREE_RE.search(text):
        return "agree"
    if _SELF_REVISE_RE.search(text):
        return "revise"
    match = _REVIEW_POSITION_RE.search(text)
    if match:
        value = match.group(1).strip().lower()
        if "agree" in value or "approve" in value or "accept" in value:
            return "agree"
        if "revise" in value or "reject" in value or "correction" in value:
            return "revise"
    if "Fix Instruction:" in text or "Diagnosis:" in text:
        return "revise"
    return None


def _parse_approve_flag(content: str) -> Optional[bool]:
    text = str(content or "")
    match = _APPROVE_RE.search(text)
    if match:
        return match.group(1).lower() == "true"

    match = _REVIEW_POSITION_RE.search(text)
    if match:
        value = match.group(1).strip().lower()
        if "approve" in value or "accept" in value:
            return True
        if (
            "reject" in value
            or "revise" in value
            or "revision" in value
            or "objection" in value
            or "correction" in value
            or "propose" in value
        ):
            return False
    return None


def _is_evaluator_turn(turn: TraceTurn, evaluator_names: set[str]) -> bool:
    if turn.agent_name in evaluator_names:
        return True
    if turn.agent_id.lower() == "evaluator":
        return True
    text = turn.content
    return "Evaluation result:" in text or "Evaluation signal:" in text or "Verifier:" in text


def _is_followup_feedback_turn(turn: TraceTurn, evaluator_names: set[str]) -> bool:
    text = str(turn.content or "")
    if _parse_evaluator_pass_fail(text) is not None:
        return False
    if _EVAL_HINT_RE.search(text):
        return True
    if turn.agent_name in evaluator_names or turn.agent_id.lower() == "evaluator":
        return "Hint:" in text or "Advice:" in text
    return False


def _parse_evaluator_pass_fail(content: str) -> Optional[bool]:
    text = str(content or "")
    if _EVAL_HINT_RE.search(text):
        return None
    for pattern in (_SCORE_RE, _EVAL_SIGNAL_RE, _VERIFIER_RE):
        match = pattern.search(text)
        if not match:
            continue
        value = match.group(1).strip().lower()
        if value in {"true", "pass"}:
            return True
        if value in {"false", "fail"}:
            return False
    return None


def _extract_group_review_candidate(text: str) -> str:
    match = _GROUP_CANDIDATE_RE.search(str(text or ""))
    if not match:
        return ""
    return _normalize_candidate_fragment(match.group(1))


def _extract_group_candidate_update(text: str) -> str:
    match = _GROUP_CANDIDATE_UPDATE_RE.search(str(text or ""))
    if not match:
        return ""
    return _normalize_candidate_fragment(match.group(1))


def _extract_candidate_under_review(text: str) -> str:
    match = _CANDIDATE_UNDER_REVIEW_RE.search(str(text or ""))
    if not match:
        return ""
    return _normalize_candidate_fragment(match.group(1))


def _extract_revised_candidate(text: str) -> str:
    raw = str(text or "")
    match = _PROPOSED_CORRECTION_RE.search(raw)
    if match:
        return _normalize_candidate_fragment(match.group(1))

    match = _JSON_REVISED_RE.search(raw)
    if match:
        fragment = match.group(1).replace("\\\\", "\\")
        return _normalize_candidate_fragment(fragment)

    return ""


def _extract_any_candidate(text: str) -> str:
    for extractor in (
        _extract_revised_candidate,
        _extract_candidate_under_review,
        _extract_group_review_candidate,
        _extract_group_candidate_update,
    ):
        value = extractor(text)
        if value:
            return value

    raw = str(text or "")
    boxed = extract_boxed(raw)
    if boxed:
        return normalize_candidate_text(boxed)

    if "Candidate Answer:" in raw or "Final Answer:" in raw:
        return normalize_candidate_text(raw)
    return ""


def extract_protocol_candidate_sequence(
    example: TraceExample,
    turns: Optional[list[TraceTurn]] = None,
) -> list[str]:
    """Extract meaningful candidate changes for PER, broadcast, and single-agent traces."""
    selected_turns = turns if turns is not None else example.turns
    evaluator_names = {name for name in example.agent_map.values() if "evaluator" in name.lower()}

    if example.protocol == "broadcast":
        # Broadcast traces repeat candidates in every peer-review message. For
        # refinement counts, use committed group-review candidates first.
        group_candidates = [
            candidate
            for candidate in (_extract_group_review_candidate(turn.content) for turn in selected_turns)
            if _is_real_candidate(candidate)
        ]
        if group_candidates:
            return _dedupe_consecutive_candidates(group_candidates)

    candidates: list[str] = []
    for turn in selected_turns:
        if _is_evaluator_turn(turn, evaluator_names):
            continue
        candidate = _extract_any_candidate(turn.content)
        if _is_real_candidate(candidate):
            candidates.append(candidate)
    return _dedupe_consecutive_candidates(candidates)


def extract_attempt_candidate_sequence(example: TraceExample) -> list[str]:
    """Return the final meaningful candidate before each evaluator attempt."""
    evaluator_names = {name for name in example.agent_map.values() if "evaluator" in name.lower()}
    eval_positions = [
        idx
        for idx, turn in enumerate(example.turns)
        if _is_evaluator_turn(turn, evaluator_names)
        and _parse_evaluator_pass_fail(turn.content) is not None
    ]
    candidates: list[str] = []
    start_pos = 0
    for eval_pos in eval_positions:
        interval_candidates = extract_protocol_candidate_sequence(
            example,
            example.turns[start_pos:eval_pos],
        )
        candidates.append(interval_candidates[-1] if interval_candidates else "")
        start_pos = eval_pos + 1
    return candidates


def _normalize_candidate_fragment(fragment: str) -> str:
    text = str(fragment or "").strip().strip("`")
    text = text.replace("\\\\", "\\")
    boxed = extract_boxed(text)
    if boxed:
        return normalize_candidate_text(boxed)
    return normalize_candidate_text(text)


def _find_next_candidate(turns: list[TraceTurn], start_idx: int) -> str:
    for turn in turns[start_idx:]:
        candidate = _extract_any_candidate(turn.content)
        if candidate:
            return candidate
        if _is_evaluator_turn(turn, set()):
            break
    return ""


def _find_next_group_candidate(turns: list[TraceTurn], start_idx: int) -> str:
    for turn in turns[start_idx:]:
        candidate = _extract_group_review_candidate(turn.content)
        if _is_real_candidate(candidate):
            return candidate
        if _is_evaluator_turn(turn, set()):
            break
    return ""


def _find_previous_candidate(turns: list[TraceTurn], turn_pos: Optional[int]) -> str:
    if turn_pos is None:
        return ""
    for idx in range(turn_pos - 1, -1, -1):
        candidate = _extract_any_candidate(turns[idx].content)
        if candidate:
            return candidate
    return ""


def _dedupe_consecutive_candidates(candidates: list[str]) -> list[str]:
    deduped: list[str] = []
    last_key = ""
    for candidate in candidates:
        if not _is_real_candidate(candidate):
            continue
        key = _candidate_key(candidate)
        if key == last_key:
            continue
        deduped.append(candidate)
        last_key = key
    return deduped


def _candidate_key(candidate: str) -> str:
    return re.sub(r"\s+", "", str(candidate or "").strip()).lower()


def _is_real_candidate(candidate: str) -> bool:
    value = str(candidate or "").strip()
    if not value:
        return False
    return value.lower() not in {"[none]", "none", "null", "n/a"}


def _aggregate_broadcast_group_review_event(
    *,
    example: TraceExample,
    labeler: TraceAnalysisLabeler,
    start_idx: int,
    candidate_before: str,
    after_fail: bool,
    evaluator_names: set[str],
    episode_number: int,
) -> tuple[ReviewEpisode | None, int]:
    """Aggregate one broadcast approval step into one group review event.

    The paper-facing broadcast unit is one candidate under review in one
    approval step, not one peer-review message. Under unanimous approval,
    multiple peer reactions to the same candidate belong to the same group
    review event.
    """

    turns = example.turns
    idx = start_idx
    evidence_turns: list[int] = []
    review_chunks: list[str] = []
    saw_review = False
    any_revise = False
    any_agree = False
    candidate_after_update = ""

    while idx < len(turns):
        turn = turns[idx]
        if _is_evaluator_turn(turn, evaluator_names):
            break

        next_group_candidate = _extract_group_review_candidate(turn.content)
        if next_group_candidate:
            break

        candidate_update = _extract_group_candidate_update(turn.content)
        if _is_real_candidate(candidate_update):
            candidate_after_update = candidate_update
            evidence_turns.append(turn.turn_idx)
            idx += 1
            break

        candidate_under_review = _extract_candidate_under_review(turn.content)
        approve = _parse_approve_flag(turn.content)
        if candidate_under_review and approve is not None:
            if _candidate_key(candidate_under_review) != _candidate_key(candidate_before):
                break
            saw_review = True
            if approve:
                any_agree = True
            else:
                any_revise = True
            evidence_turns.append(turn.turn_idx)
            review_chunks.append(f"{turn.agent_name}: {turn.content}")

        idx += 1

    if not saw_review:
        return None, idx

    action = "revise" if any_revise else "agree"
    candidate_after = candidate_before
    if _is_real_candidate(candidate_after_update):
        candidate_after = candidate_after_update
    elif action == "revise":
        candidate_after = (
            _find_next_distinct_review_candidate(turns, idx, candidate_before)
            or _find_next_group_candidate(turns, start_idx + 1)
            or _find_next_candidate(turns, start_idx + 1)
            or candidate_before
        )

    pre = labeler.judge_correctness(
        problem=example.input_text,
        gold_answer=example.label,
        candidate_answer=candidate_before,
    )
    post = labeler.judge_correctness(
        problem=example.input_text,
        gold_answer=example.label,
        candidate_answer=candidate_after,
    )

    episode = ReviewEpisode(
        trace_path=example.trace_path,
        task=example.task,
        protocol=example.protocol,
        example_idx=example.example_idx,
        episode_id=f"{example.example_idx}.review.{episode_number}",
        review_turn=turns[start_idx].turn_idx,
        reviewer="broadcast_group",
        action=action,
        candidate_before=candidate_before,
        candidate_after=candidate_after,
        pre_correct=pre.value,
        post_correct=post.value,
        pre_correct_source=pre.source,
        post_correct_source=post.source,
        after_evaluator_fail=after_fail,
        evidence_turns=evidence_turns,
        review_feedback="\n\n".join(review_chunks),
    )
    return episode, idx


def _find_next_distinct_review_candidate(
    turns: list[TraceTurn],
    start_idx: int,
    current_candidate: str,
) -> str:
    current_key = _candidate_key(current_candidate)
    for turn in turns[start_idx:]:
        candidate_under_review = _extract_candidate_under_review(turn.content)
        approve = _parse_approve_flag(turn.content)
        if candidate_under_review and approve is not None:
            if _candidate_key(candidate_under_review) != current_key:
                return candidate_under_review
        candidate_update = _extract_group_candidate_update(turn.content)
        if _is_real_candidate(candidate_update) and _candidate_key(candidate_update) != current_key:
            return candidate_update
    return ""


def _rate(numerator: int, denominator: int) -> Optional[float]:
    if denominator <= 0:
        return None
    return numerator / denominator


def _balanced_accuracy(tp: int, fp: int, fn: int, tn: int) -> Optional[float]:
    recall = _rate(tp, tp + fn)
    specificity = _rate(tn, tn + fp)
    if recall is None or specificity is None:
        return None
    return (recall + specificity) / 2


def _fmt_rate(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2%}"
