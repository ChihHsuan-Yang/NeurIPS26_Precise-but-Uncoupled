"""Trace extraction for supplemental diagnostics."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

from agentverse.evaluation.content_based import ContentJudgeResult
from agentverse.metrics.diagnostics.records import DiagnosticsDataset, ProcessExampleRecord
from agentverse.metrics.repair import (
    ReviewEpisode,
    TraceExample,
    _extract_review_episodes,
    _is_evaluator_turn,
    _parse_evaluator_pass_fail,
    extract_attempt_candidate_sequence,
    extract_protocol_candidate_sequence,
    parse_trace_file,
)


_ROUTE_RE = re.compile(r"\[Route:[^\]]+\]", re.IGNORECASE)


def load_diagnostics_dataset(
    trace_files: list[Path],
    *,
    repeated_candidate_threshold: int = 3,
    repeated_route_threshold: int = 3,
) -> DiagnosticsDataset:
    examples = _load_trace_examples(trace_files)

    review_episodes: list[ReviewEpisode] = []
    null_labeler = _ProcessOnlyCorrectnessLabeler()
    for example in examples:
        review_episodes.extend(_extract_review_episodes(example, null_labeler))

    process_records = extract_process_records(
        examples,
        review_episodes=review_episodes,
        repeated_candidate_threshold=repeated_candidate_threshold,
        repeated_route_threshold=repeated_route_threshold,
    )

    return DiagnosticsDataset(
        examples=examples,
        review_episodes=review_episodes,
        process_records=process_records,
        trace_files=[str(path) for path in trace_files],
    )


def extract_process_records(
    examples: list[TraceExample],
    *,
    review_episodes: list[ReviewEpisode],
    repeated_candidate_threshold: int = 3,
    repeated_route_threshold: int = 3,
) -> list[ProcessExampleRecord]:
    reviews_by_example: dict[tuple[str, int], list[ReviewEpisode]] = defaultdict(list)
    for episode in review_episodes:
        reviews_by_example[(episode.trace_path, episode.example_idx)].append(episode)

    records: list[ProcessExampleRecord] = []
    for example in examples:
        candidates = _candidate_sequence(example)
        attempt_candidates = _attempt_candidate_sequence(example)
        eval_results = _evaluator_results(example)
        review_events = reviews_by_example.get((example.trace_path, example.example_idx), [])

        agreement_followed_by_fail = 0
        agreement_followed_by_pass = 0
        agreement_with_later_label = 0
        for episode in review_events:
            if episode.action != "agree":
                continue
            later = _next_evaluator_label_after_turn(example, episode.review_turn)
            if later is None:
                continue
            agreement_with_later_label += 1
            if later is True:
                agreement_followed_by_pass += 1
            else:
                agreement_followed_by_fail += 1

        adjacent_pairs = max(0, len(attempt_candidates) - 1)
        repeated_adjacent = sum(
            1
            for left, right in zip(attempt_candidates, attempt_candidates[1:])
            if _same_candidate(left, right)
        )

        fail_with_next = 0
        no_change_after_fail = 0
        for idx, passed in enumerate(eval_results[:-1]):
            if passed is not False:
                continue
            fail_with_next += 1
            if idx + 1 < len(attempt_candidates) and _same_candidate(
                attempt_candidates[idx],
                attempt_candidates[idx + 1],
            ):
                no_change_after_fail += 1

        # Repeated-answer looping should be driven by judged submission
        # candidates; otherwise normal review prompts can repeat a candidate and
        # create false loop signals.
        attempt_candidate_counts = Counter(
            candidate for candidate in attempt_candidates if candidate
        )
        candidate_counts = Counter(candidate for candidate in candidates if candidate)
        route_counts = Counter(_route_sequence(example))
        repeated_candidate_signal = any(
            count >= repeated_candidate_threshold
            for count in attempt_candidate_counts.values()
        )
        repeated_route_signal = any(
            count >= repeated_route_threshold for count in route_counts.values()
        )
        looping_signal = bool(
            repeated_candidate_signal or repeated_route_signal or no_change_after_fail > 0
        )

        turn_counts = _turn_counts(example)
        records.append(
            ProcessExampleRecord(
                trace_path=example.trace_path,
                task=example.task,
                protocol=example.protocol,
                example_idx=example.example_idx,
                review_events=len(review_events),
                agreement_events=sum(1 for episode in review_events if episode.action == "agree"),
                agreement_with_later_label=agreement_with_later_label,
                agreement_followed_by_fail=agreement_followed_by_fail,
                agreement_followed_by_pass=agreement_followed_by_pass,
                adjacent_submission_pairs=adjacent_pairs,
                repeated_adjacent_answers=repeated_adjacent,
                fail_with_next_count=fail_with_next,
                no_change_after_fail=no_change_after_fail,
                repeated_candidate_signal=repeated_candidate_signal,
                repeated_route_signal=repeated_route_signal,
                looping_signal=looping_signal,
                distinct_candidate_count=len(candidate_counts),
                turn_imbalance=_turn_imbalance(turn_counts),
                turn_counts=turn_counts,
            )
        )
    return records


def _load_trace_examples(trace_files: list[Path]) -> list[TraceExample]:
    examples: list[TraceExample] = []
    for trace_file in trace_files:
        examples.extend(parse_trace_file(trace_file))
    return examples


def _candidate_sequence(example: TraceExample) -> list[str]:
    return extract_protocol_candidate_sequence(example)


def _attempt_candidate_sequence(example: TraceExample) -> list[str]:
    return extract_attempt_candidate_sequence(example)


def _evaluator_results(example: TraceExample) -> list[Optional[bool]]:
    evaluator_names = {
        name for name in example.agent_map.values() if "evaluator" in name.lower()
    }
    results: list[Optional[bool]] = []
    for turn in example.turns:
        if not _is_evaluator_turn(turn, evaluator_names):
            continue
        passed = _parse_evaluator_pass_fail(turn.content)
        if passed is not None:
            results.append(passed)
    return results


def _next_evaluator_label_after_turn(
    example: TraceExample,
    turn_idx: int,
) -> Optional[bool]:
    evaluator_names = {
        name for name in example.agent_map.values() if "evaluator" in name.lower()
    }
    for turn in example.turns:
        if turn.turn_idx <= turn_idx:
            continue
        if not _is_evaluator_turn(turn, evaluator_names):
            continue
        passed = _parse_evaluator_pass_fail(turn.content)
        if passed is not None:
            return passed
    return None


def _route_sequence(example: TraceExample) -> list[str]:
    routes: list[str] = []
    for turn in example.turns:
        routes.extend(match.group(0).lower() for match in _ROUTE_RE.finditer(turn.content))
    return routes


def _turn_counts(example: TraceExample) -> dict[str, int]:
    evaluator_names = {
        name for name in example.agent_map.values() if "evaluator" in name.lower()
    }
    counts: dict[str, int] = {}
    for turn in example.turns:
        if turn.agent_id.lower() == "system" or _is_evaluator_turn(turn, evaluator_names):
            continue
        role = turn.agent_name or turn.agent_id
        counts[role] = counts.get(role, 0) + 1
    return counts


def _turn_imbalance(turn_counts: dict[str, int]) -> float | None:
    total = sum(turn_counts.values())
    if total <= 0 or len(turn_counts) <= 1:
        return 0.0 if total > 0 else None
    shares = [count / total for count in turn_counts.values()]
    return max(shares) - min(shares)


def _same_candidate(left: str, right: str) -> bool:
    return bool(left and right and left.strip() == right.strip())


class _ProcessOnlyCorrectnessLabeler:
    """Cheap labeler for process diagnostics that do not need correctness."""

    def judge_correctness(
        self,
        *,
        problem: str,
        gold_answer: str,
        candidate_answer: str,
    ) -> ContentJudgeResult:
        return ContentJudgeResult(
            None,
            "not_needed",
            "Process diagnostics use review actions but not answer correctness.",
        )
