#!/usr/bin/env python3
"""Run the symmetric transition audit across release-v2 PER/Broadcast traces.

This is a diagnostic pipeline, not a paper-result generator. It:

1. streams each release trajectory without loading the message table in memory;
2. applies ``symmetric_transition_extractor`` unchanged;
3. attaches correctness only when the same answer has a frozen in-trajectory
   evaluator PASS/FAIL signal or matches the trajectory's frozen final answer;
4. reports strict, gate-confounded, parser-incomplete, no-after, and source-link
   denominators separately; and
5. refuses to label a PER/Broadcast cell comparison ready when either protocol
   has zero strict observations.

No legacy transition label, legacy answer_after, or process aggregate is read.
"""

from __future__ import annotations

import argparse
import os
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, MutableMapping, Sequence

from symmetric_transition_extractor import (
    extract_explicit_answer,
    extract_symmetric_transitions,
    normalize_answer,
)


HERE = Path(__file__).resolve().parent
# RELEASE PATCH (NeurIPS26_Precise-but-Uncoupled).
# Upstream this file defaulted to an author-private cache directory. In this
# release the defaults are repository-relative and overridable by environment
# variable, so the script runs from a fresh clone with no edits:
#   PU_RELEASE_ROOT -> where the downloaded HF dataset lives
#                      (must contain data/trajectories/ and data/messages/)
#   PU_OUTPUT_DIR   -> where derived CSV/JSON are written
# See docs/DATA.md for the download command and docs/REPRODUCIBILITY.md.
REPO_ROOT = HERE.parents[2]   # src/precise_uncoupled/process -> repo root
DEFAULT_RELEASE_ROOT = Path(
    os.environ.get("PU_RELEASE_ROOT", str(REPO_ROOT / "data" / "hf" / "NeurIPS26_Precise-but-Uncoupled"))
)
DEFAULT_OUTPUT_DIR = Path(
    os.environ.get("PU_OUTPUT_DIR", str(REPO_ROOT / "results" / "derived_tables"))
)

CORE_MATRIX_SCOPE = {
    ("jeebench", "text_only", "gemma_4_31b"),
    ("jeebench", "text_only", "gpt_oss_120b"),
    ("labbench", "llm_strict", "gemma_4_31b"),
    ("labbench", "llm_strict", "gpt_oss_120b"),
    ("mascqa", "text_only", "gemma_4_31b"),
    ("mascqa", "text_only", "gpt_oss_120b"),
    ("omnimath2", "competition_math_4181", "gemma_4_31b"),
    ("omnimath2", "competition_math_4181", "gpt_oss_120b"),
    ("scibench", "text_only", "gemma_4_31b"),
    ("scibench", "text_only", "gpt_oss_120b"),
}

REVISE_ACTION_DEFINITION = (
    "review_action=revise is a reviewer-issued revise decision; it is not an "
    "independently evaluator-verified measure of critique usefulness"
)
PENDING_UNKNOWN_ACTION_RECOVERY_NOTE = (
    "Unknown review actions are excluded from the explicit-revise estimand. "
    "In the core matrix, 326 of 327 strict PER unknown-action records have a "
    "source-linked review_route candidate-memory event, but review_route also "
    "occurs with text-explicit agree and revise decisions and therefore does "
    "not semantically recover critique stance. A separately labeled "
    "route-observed sensitivity includes these records without relabeling "
    "review_action."
)
PER_ROUTE_SENSITIVITY_DEFINITION = (
    "PER strict-pre-gate sensitivity only: include text-explicit "
    "review_action=revise plus review_action=unknown records with a "
    "source-linked review_memory_action=review_route; never include explicit "
    "agree and never reinterpret unknown as revise or useful critique"
)

EVALUATION_SIGNAL_RE = re.compile(
    r"^\s*Evaluation\s+signal\s*:\s*(PASS|FAIL)\b",
    re.IGNORECASE | re.MULTILINE,
)

DETAIL_FIELDS = [
    "dataset",
    "slice_id",
    "actor_family",
    "evaluator_model_id",
    "protocol",
    "release_status",
    "provenance_status",
    "matrix_4x2x5",
    "trajectory_uid",
    "problem_uid",
    "review_id",
    "review_action",
    "review_memory_event_uid",
    "review_memory_sequence_index",
    "review_memory_action",
    "review_memory_source",
    "review_memory_authority",
    "review_memory_stage",
    "review_memory_previous_event_uid",
    "review_memory_immediate",
    "review_memory_event_type_valid",
    "review_memory_system_authored",
    "review_memory_stage_matches_review",
    "review_memory_previous_event_uid_matches_review",
    "review_memory_source_matches_reviewer",
    "review_memory_authority_matches_reviewer",
    "review_memory_provenance_valid",
    "review_memory_link_status",
    "review_route_observed",
    "review_sequence_start",
    "review_sequence_end",
    "before_sequence_index",
    "before_sender",
    "before_stage",
    "answer_before",
    "after_sequence_index",
    "after_sender",
    "after_stage",
    "answer_after",
    "before_candidate_link_method",
    "after_candidate_link_method",
    "candidate_alignment",
    "actor_turn_observed",
    "actor_response_observed",
    "answer_pair_parsed",
    "strict_pre_gate_eligible",
    "intervening_system_candidate_updates",
    "system_selected_answers_json",
    "system_selected_answer_used_as_after",
    "status",
    "exclusion_reason",
    "answer_changed",
    "before_correct",
    "before_correct_source",
    "after_correct",
    "after_correct_source",
    "correctness_pair_labeled",
    "transition_outcome",
    "primary_eligibility_bucket",
]

SUMMARY_COUNT_FIELDS = [
    "review_opportunities",
    "actor_turn_observed",
    "actor_response_observed",
    "answer_pair_parsed",
    "strict_pre_gate_eligible",
    "gate_confounded",
    "parser_incomplete",
    "no_after",
    "source_link_failure",
    "other_non_strict",
    "review_action_revise",
    "review_action_agree",
    "review_action_unknown",
    "review_memory_provenance_valid",
    "review_route_observed",
    "review_route_observed_action_revise",
    "review_route_observed_action_agree",
    "review_route_observed_action_unknown",
    "before_frozen_label_conflict",
    "after_frozen_label_conflict",
    "any_frozen_label_conflict",
    "correctness_pair_labeled",
    "outcome_unchanged",
    "outcome_changed_still_wrong",
    "outcome_repaired",
    "outcome_harmed",
    "outcome_changed_still_correct",
    "outcome_changed_correctness_unlabeled",
    "strict_correctness_pair_labeled",
    "strict_outcome_unchanged",
    "strict_outcome_changed_still_wrong",
    "strict_outcome_repaired",
    "strict_outcome_harmed",
    "strict_outcome_changed_still_correct",
    "strict_outcome_changed_correctness_unlabeled",
    "gate_correctness_pair_labeled",
    "gate_outcome_unchanged",
    "gate_outcome_changed_still_wrong",
    "gate_outcome_repaired",
    "gate_outcome_harmed",
    "gate_outcome_changed_still_correct",
    "gate_outcome_changed_correctness_unlabeled",
]

REVISE_CONDITIONED_COUNT_FIELDS = [
    "strict_explicit_revise_pre_wrong_actor_pairs",
    "strict_explicit_revise_answer_unchanged",
    "strict_explicit_revise_answer_changed",
    "strict_explicit_revise_after_correct_labeled_pairs",
    "strict_explicit_revise_labeled_unchanged",
    "strict_explicit_revise_labeled_changed_still_wrong",
    "strict_explicit_revise_repaired",
    "strict_explicit_revise_changed_after_correct_labeled",
    "gate_explicit_revise_pre_wrong_actor_pairs",
    "gate_explicit_revise_answer_unchanged",
    "gate_explicit_revise_answer_changed",
    "gate_explicit_revise_after_correct_labeled_pairs",
    "gate_explicit_revise_labeled_unchanged",
    "gate_explicit_revise_labeled_changed_still_wrong",
    "gate_explicit_revise_repaired",
    "gate_explicit_revise_changed_after_correct_labeled",
]

PER_ROUTE_SENSITIVITY_COUNT_FIELDS = [
    "per_route_sensitivity_pre_wrong_actor_pairs",
    "per_route_sensitivity_explicit_revise_pairs",
    "per_route_sensitivity_unknown_linked_route_pairs",
    "per_route_sensitivity_answer_unchanged",
    "per_route_sensitivity_answer_changed",
    "per_route_sensitivity_after_correct_labeled_pairs",
    "per_route_sensitivity_labeled_unchanged",
    "per_route_sensitivity_labeled_changed_still_wrong",
    "per_route_sensitivity_repaired",
    "per_route_sensitivity_changed_after_correct_labeled",
]


@dataclass
class FrozenAnswerLabel:
    value: bool | None
    sources: set[str]
    conflict: bool = False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def _parse_pass_fail(content: Any) -> bool | None:
    text = str(content or "").strip()
    match = EVALUATION_SIGNAL_RE.search(text)
    if match:
        return match.group(1).upper() == "PASS"
    if re.fullmatch(r"PASS", text, re.IGNORECASE):
        return True
    if re.fullmatch(r"FAIL", text, re.IGNORECASE):
        return False
    return None


def _submitted_answer(event: Mapping[str, Any]) -> str:
    stage = str(event.get("stage") or "").lower()
    if "submission" not in stage or "candidate_memory" in stage:
        return ""
    answer = extract_explicit_answer(event.get("content"))
    if answer:
        return answer
    return ""


def _add_frozen_label(
    labels: MutableMapping[str, FrozenAnswerLabel],
    answer: Any,
    value: bool | None,
    source: str,
) -> None:
    key = normalize_answer(answer)
    if not key or value is None:
        return
    current = labels.get(key)
    if current is None:
        labels[key] = FrozenAnswerLabel(bool(value), {source}, False)
        return
    current.sources.add(source)
    if current.value is not None and current.value != bool(value):
        current.value = None
        current.conflict = True


def build_frozen_answer_labels(
    events: Sequence[Mapping[str, Any]],
    *,
    final_answer: Any = "",
    final_correct: bool | None = None,
) -> dict[str, FrozenAnswerLabel]:
    """Map answer keys to frozen labels without using legacy transitions."""

    labels: dict[str, FrozenAnswerLabel] = {}
    pending_answer = ""
    pending_sequence = -1
    for fallback, event in enumerate(events):
        submitted = _submitted_answer(event)
        if submitted:
            pending_answer = submitted
            try:
                pending_sequence = int(event.get("sequence_index"))
            except (TypeError, ValueError):
                pending_sequence = fallback
            continue

        stage = str(event.get("stage") or "").lower()
        if (
            not pending_answer
            or "evaluation" not in stage
            or "hint" in stage
            or "candidate_memory" in stage
            or "submission" in stage
        ):
            continue
        value = _parse_pass_fail(event.get("content"))
        if value is None:
            continue
        _add_frozen_label(
            labels,
            pending_answer,
            value,
            f"runtime_evaluator_submission_sequence_{pending_sequence}",
        )
        pending_answer = ""
        pending_sequence = -1

    if final_answer not in (None, "") and final_correct is not None:
        _add_frozen_label(
            labels,
            final_answer,
            bool(final_correct),
            "trajectory_final_correct",
        )
    return labels


def _lookup_frozen_label(
    labels: Mapping[str, FrozenAnswerLabel], answer: Any
) -> tuple[bool | None, str]:
    key = normalize_answer(answer)
    item = labels.get(key)
    if item is None:
        return None, ""
    source = "+".join(sorted(item.sources))
    if item.conflict:
        return None, f"conflict:{source}"
    return item.value, source


def classify_transition_outcome(
    answer_before: Any,
    answer_after: Any,
    before_correct: bool | None,
    after_correct: bool | None,
) -> tuple[bool | None, str]:
    before_key = normalize_answer(answer_before)
    after_key = normalize_answer(answer_after)
    if not before_key or not after_key:
        return None, "not_evaluable_unparsed_actor_pair"
    changed = before_key != after_key
    if not changed:
        return False, "unchanged"
    if before_correct is False and after_correct is True:
        return True, "repaired"
    if before_correct is False and after_correct is False:
        return True, "changed_still_wrong"
    if before_correct is True and after_correct is False:
        return True, "harmed"
    if before_correct is True and after_correct is True:
        return True, "changed_still_correct"
    return True, "changed_correctness_unlabeled"


def primary_eligibility_bucket(record: Mapping[str, Any]) -> str:
    status = str(record.get("status") or "")
    if status.startswith("missing_actor") or "unlinked" in status:
        return "source_link_failure"
    if not bool(record.get("actor_turn_observed")):
        return "no_after"
    if not bool(record.get("answer_pair_parsed")):
        return "parser_incomplete"
    if int(record.get("intervening_system_candidate_updates") or 0) > 0:
        return "gate_confounded"
    if bool(record.get("strict_pre_gate_eligible")):
        return "strict_pre_gate"
    return "other_non_strict"


def attach_frozen_correctness(
    record: Mapping[str, Any],
    labels: Mapping[str, FrozenAnswerLabel],
) -> dict[str, Any]:
    output = dict(record)
    before_correct, before_source = _lookup_frozen_label(
        labels, output.get("answer_before")
    )
    after_correct, after_source = _lookup_frozen_label(
        labels, output.get("answer_after")
    )
    changed, outcome = classify_transition_outcome(
        output.get("answer_before"),
        output.get("answer_after"),
        before_correct,
        after_correct,
    )
    output.update(
        {
            "answer_changed": changed,
            "before_correct": before_correct,
            "before_correct_source": before_source,
            "after_correct": after_correct,
            "after_correct_source": after_source,
            "correctness_pair_labeled": (
                before_correct is not None and after_correct is not None
            ),
            "transition_outcome": outcome,
            "primary_eligibility_bucket": primary_eligibility_bucket(output),
        }
    )
    return output


def _lazy_pyarrow() -> tuple[Any, Any]:
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "PyArrow is required for the release aggregation. Use the full-"
            "Python + PYTHONPATH command documented in process/README.md."
        ) from exc
    return pq, None


def load_trajectory_metadata(
    release_root: Path,
    *,
    scope: str,
) -> dict[str, dict[str, Any]]:
    pq, _ = _lazy_pyarrow()
    path = release_root / "data" / "trajectories" / "part-00000.parquet"
    columns = [
        "trajectory_uid",
        "problem_uid",
        "benchmark_id",
        "slice_id",
        "protocol_id",
        "actor_model_id",
        "evaluator_model_id",
        "run_id",
        "final_answer",
        "final_correct",
        "release_status",
        "provenance_status",
    ]
    rows = pq.read_table(path, columns=columns).to_pylist()
    selected: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row["protocol_id"] not in {"per", "broadcast"}:
            continue
        matrix_member = (
            row["benchmark_id"],
            row["slice_id"],
            row["actor_model_id"],
        ) in CORE_MATRIX_SCOPE
        if scope == "matrix-4x2x5" and not matrix_member:
            continue
        row = dict(row)
        row["matrix_4x2x5"] = matrix_member
        trajectory_uid = str(row["trajectory_uid"])
        if trajectory_uid in selected:
            raise ValueError(f"Duplicate trajectory_uid: {trajectory_uid}")
        selected[trajectory_uid] = row
    return selected


def iter_selected_trajectory_events(
    messages_root: Path,
    selected_uids: set[str],
) -> Iterator[tuple[str, list[dict[str, Any]]]]:
    """Stream globally contiguous release messages one trajectory at a time."""

    pq, _ = _lazy_pyarrow()
    columns = [
        "event_uid",
        "message_uid",
        "previous_event_uid",
        "trajectory_uid",
        "sequence_index",
        "event_type",
        "round_index",
        "stage",
        "sender",
        "role",
        "content",
        "extra_json",
    ]
    current_uid = ""
    current_selected = False
    current_events: list[dict[str, Any]] = []
    closed_uids: set[str] = set()

    for path in sorted(messages_root.glob("*.parquet")):
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=32_768, columns=columns):
            for row in batch.to_pylist():
                trajectory_uid = str(row["trajectory_uid"])
                if trajectory_uid != current_uid:
                    if current_uid:
                        closed_uids.add(current_uid)
                        if current_selected:
                            yield current_uid, current_events
                    if trajectory_uid in closed_uids:
                        raise ValueError(
                            "Messages are not trajectory-contiguous: "
                            f"{trajectory_uid}"
                        )
                    current_uid = trajectory_uid
                    current_selected = trajectory_uid in selected_uids
                    current_events = []
                if current_selected:
                    current_events.append(row)

    if current_uid and current_selected:
        yield current_uid, current_events


def _group_key(metadata: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(metadata["benchmark_id"]),
        str(metadata["slice_id"]),
        str(metadata["actor_model_id"]),
        str(metadata["protocol_id"]),
    )


def _new_summary(metadata: Mapping[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "dataset": metadata["benchmark_id"],
        "slice_id": metadata["slice_id"],
        "actor_family": metadata["actor_model_id"],
        "protocol": metadata["protocol_id"],
        "evaluator_model_id": metadata["evaluator_model_id"],
        "release_statuses_json": json.dumps(
            [metadata["release_status"]], ensure_ascii=False
        ),
        "provenance_statuses_json": json.dumps(
            [metadata["provenance_status"]], ensure_ascii=False
        ),
        "matrix_4x2x5": bool(metadata["matrix_4x2x5"]),
        "trajectories": 0,
        "trajectories_with_review": 0,
    }
    row.update({field: 0 for field in SUMMARY_COUNT_FIELDS})
    return row


def _increment_summary(
    summary: MutableMapping[str, Any], record: Mapping[str, Any]
) -> None:
    summary["review_opportunities"] += 1
    for field in (
        "actor_turn_observed",
        "actor_response_observed",
        "answer_pair_parsed",
        "strict_pre_gate_eligible",
        "correctness_pair_labeled",
    ):
        summary[field] += int(bool(record.get(field)))

    bucket = str(record["primary_eligibility_bucket"])
    if bucket == "gate_confounded":
        summary["gate_confounded"] += 1
    elif bucket == "parser_incomplete":
        summary["parser_incomplete"] += 1
    elif bucket == "no_after":
        summary["no_after"] += 1
    elif bucket == "source_link_failure":
        summary["source_link_failure"] += 1
    elif bucket == "other_non_strict":
        summary["other_non_strict"] += 1

    action = str(record.get("review_action") or "unknown")
    action_field = (
        f"review_action_{action}"
        if action in {"revise", "agree"}
        else "review_action_unknown"
    )
    summary[action_field] += 1
    memory_valid = bool(record.get("review_memory_provenance_valid"))
    route_observed = bool(record.get("review_route_observed"))
    summary["review_memory_provenance_valid"] += int(memory_valid)
    summary["review_route_observed"] += int(route_observed)
    if route_observed:
        route_action = (
            action if action in {"revise", "agree"} else "unknown"
        )
        summary[f"review_route_observed_action_{route_action}"] += 1

    before_conflict = str(record.get("before_correct_source") or "").startswith(
        "conflict:"
    )
    after_conflict = str(record.get("after_correct_source") or "").startswith(
        "conflict:"
    )
    summary["before_frozen_label_conflict"] += int(before_conflict)
    summary["after_frozen_label_conflict"] += int(after_conflict)
    summary["any_frozen_label_conflict"] += int(
        before_conflict or after_conflict
    )

    outcome = str(record.get("transition_outcome") or "")
    outcome_field = {
        "unchanged": "outcome_unchanged",
        "changed_still_wrong": "outcome_changed_still_wrong",
        "repaired": "outcome_repaired",
        "harmed": "outcome_harmed",
        "changed_still_correct": "outcome_changed_still_correct",
        "changed_correctness_unlabeled": "outcome_changed_correctness_unlabeled",
    }.get(outcome)
    if outcome_field:
        summary[outcome_field] += 1

    if bool(record.get("strict_pre_gate_eligible")):
        if bool(record.get("correctness_pair_labeled")):
            summary["strict_correctness_pair_labeled"] += 1
        if outcome_field:
            summary[f"strict_{outcome_field}"] += 1

    if bucket == "gate_confounded":
        if bool(record.get("correctness_pair_labeled")):
            summary["gate_correctness_pair_labeled"] += 1
        if outcome_field:
            summary[f"gate_{outcome_field}"] += 1


def _safe_rate(numerator: int, denominator: int) -> float | None:
    return float(numerator) / float(denominator) if denominator else None


def _validate_detail_record(record: Mapping[str, Any]) -> None:
    if bool(record.get("system_selected_answer_used_as_after")):
        raise AssertionError("System-selected answer leaked into answer_after")
    if bool(record.get("actor_response_observed")) and not bool(
        record.get("actor_turn_observed")
    ):
        raise AssertionError("Actor response recorded without an actor turn")
    if bool(record.get("answer_pair_parsed")) and not bool(
        record.get("actor_response_observed")
    ):
        raise AssertionError("Parsed pair recorded without an actor response")
    if bool(record.get("correctness_pair_labeled")) and not bool(
        record.get("answer_pair_parsed")
    ):
        raise AssertionError("Correctness pair recorded without a parsed pair")
    bucket = str(record.get("primary_eligibility_bucket") or "")
    if bool(record.get("strict_pre_gate_eligible")) and bucket != "strict_pre_gate":
        raise AssertionError("Strict record assigned to a non-strict bucket")
    if bucket == "gate_confounded" and bool(
        record.get("strict_pre_gate_eligible")
    ):
        raise AssertionError("Gate-confounded record marked strict")
    memory_valid = bool(record.get("review_memory_provenance_valid"))
    route_observed = bool(record.get("review_route_observed"))
    if route_observed and not memory_valid:
        raise AssertionError(
            "Review route observed without valid candidate-memory provenance"
        )
    if route_observed and record.get("review_memory_action") != "review_route":
        raise AssertionError(
            "Review route observed with a different raw memory action"
        )
    if memory_valid:
        required_links = (
            "review_memory_immediate",
            "review_memory_event_type_valid",
            "review_memory_system_authored",
            "review_memory_stage_matches_review",
            "review_memory_previous_event_uid_matches_review",
            "review_memory_source_matches_reviewer",
            "review_memory_authority_matches_reviewer",
        )
        if not all(bool(record.get(field)) for field in required_links):
            raise AssertionError(
                "Valid review-memory provenance is missing a required link"
            )
    if str(record.get("protocol") or "") != "per" and (
        memory_valid or route_observed
    ):
        raise AssertionError(
            "PER review-memory provenance appeared in another protocol"
        )


def _validate_summary_row(row: Mapping[str, Any]) -> None:
    reviews = int(row["review_opportunities"])
    primary_partition = sum(
        int(row[field])
        for field in (
            "strict_pre_gate_eligible",
            "gate_confounded",
            "parser_incomplete",
            "no_after",
            "source_link_failure",
            "other_non_strict",
        )
    )
    if primary_partition != reviews:
        raise AssertionError(
            f"Primary eligibility buckets do not partition reviews: "
            f"{primary_partition} != {reviews}"
        )
    if (
        int(row["review_action_revise"])
        + int(row["review_action_agree"])
        + int(row["review_action_unknown"])
        != reviews
    ):
        raise AssertionError("Review-action counts do not partition reviews")
    parsed = int(row["answer_pair_parsed"])
    all_outcomes = sum(
        int(row[field])
        for field in (
            "outcome_unchanged",
            "outcome_changed_still_wrong",
            "outcome_repaired",
            "outcome_harmed",
            "outcome_changed_still_correct",
            "outcome_changed_correctness_unlabeled",
        )
    )
    if all_outcomes != parsed:
        raise AssertionError(
            f"Transition outcomes do not partition parsed pairs: "
            f"{all_outcomes} != {parsed}"
        )
    for prefix, eligible_field in (
        ("strict_", "strict_pre_gate_eligible"),
        ("gate_", "gate_confounded"),
    ):
        eligible = int(row[eligible_field])
        outcomes = sum(
            int(row[f"{prefix}{field}"])
            for field in (
                "outcome_unchanged",
                "outcome_changed_still_wrong",
                "outcome_repaired",
                "outcome_harmed",
                "outcome_changed_still_correct",
                "outcome_changed_correctness_unlabeled",
            )
        )
        if outcomes != eligible:
            raise AssertionError(
                f"{prefix} outcomes do not partition eligible pairs: "
                f"{outcomes} != {eligible}"
            )
    if int(row["actor_response_observed"]) > int(row["actor_turn_observed"]):
        raise AssertionError("Actor responses exceed actor turns")
    if parsed > int(row["actor_response_observed"]):
        raise AssertionError("Parsed pairs exceed actor responses")
    if int(row["correctness_pair_labeled"]) > parsed:
        raise AssertionError("Labeled correctness pairs exceed parsed pairs")
    if int(row["trajectories_with_review"]) > int(row["trajectories"]):
        raise AssertionError("Reviewed trajectories exceed trajectories")
    for prefix in ("strict_", "gate_"):
        uptake_denominator = int(
            row[f"{prefix}explicit_revise_pre_wrong_actor_pairs"]
        )
        answer_unchanged = int(
            row[f"{prefix}explicit_revise_answer_unchanged"]
        )
        answer_changed = int(
            row[f"{prefix}explicit_revise_answer_changed"]
        )
        if answer_unchanged + answer_changed != uptake_denominator:
            raise AssertionError(
                f"{prefix} explicit-revise answer-change outcomes do not "
                "partition the observable pre-wrong actor-pair denominator"
            )

        labeled_denominator = int(
            row[f"{prefix}explicit_revise_after_correct_labeled_pairs"]
        )
        labeled_unchanged = int(
            row[f"{prefix}explicit_revise_labeled_unchanged"]
        )
        labeled_changed_still_wrong = int(
            row[f"{prefix}explicit_revise_labeled_changed_still_wrong"]
        )
        repaired = int(
            row[f"{prefix}explicit_revise_repaired"]
        )
        changed_labeled = int(
            row[
                f"{prefix}explicit_revise_changed_after_correct_labeled"
            ]
        )
        if (
            labeled_unchanged + labeled_changed_still_wrong + repaired
            != labeled_denominator
        ):
            raise AssertionError(
                f"{prefix} explicit-revise complete-case outcomes do not "
                "partition the after-correct-labeled denominator"
            )
        if labeled_changed_still_wrong + repaired != changed_labeled:
            raise AssertionError(
                f"{prefix} explicit-revise changed-label total is inconsistent"
            )
        if labeled_denominator > uptake_denominator:
            raise AssertionError(
                f"{prefix} after-correct-labeled pairs exceed uptake pairs"
            )
        if changed_labeled > answer_changed:
            raise AssertionError(
                f"{prefix} labeled changed pairs exceed observable changes"
            )

    sensitivity_total = int(
        row["per_route_sensitivity_pre_wrong_actor_pairs"]
    )
    sensitivity_explicit = int(
        row["per_route_sensitivity_explicit_revise_pairs"]
    )
    sensitivity_unknown_route = int(
        row["per_route_sensitivity_unknown_linked_route_pairs"]
    )
    sensitivity_unchanged = int(
        row["per_route_sensitivity_answer_unchanged"]
    )
    sensitivity_changed = int(
        row["per_route_sensitivity_answer_changed"]
    )
    sensitivity_labeled = int(
        row["per_route_sensitivity_after_correct_labeled_pairs"]
    )
    sensitivity_labeled_unchanged = int(
        row["per_route_sensitivity_labeled_unchanged"]
    )
    sensitivity_labeled_wrong = int(
        row["per_route_sensitivity_labeled_changed_still_wrong"]
    )
    sensitivity_repaired = int(
        row["per_route_sensitivity_repaired"]
    )
    sensitivity_changed_labeled = int(
        row["per_route_sensitivity_changed_after_correct_labeled"]
    )
    if sensitivity_explicit + sensitivity_unknown_route != sensitivity_total:
        raise AssertionError(
            "PER route sensitivity source strata do not partition its total"
        )
    if sensitivity_unchanged + sensitivity_changed != sensitivity_total:
        raise AssertionError(
            "PER route sensitivity answer-change outcomes do not partition"
        )
    if (
        sensitivity_labeled_unchanged
        + sensitivity_labeled_wrong
        + sensitivity_repaired
        != sensitivity_labeled
    ):
        raise AssertionError(
            "PER route sensitivity labeled outcomes do not partition"
        )
    if (
        sensitivity_labeled_wrong + sensitivity_repaired
        != sensitivity_changed_labeled
    ):
        raise AssertionError(
            "PER route sensitivity changed-label total is inconsistent"
        )
    if sensitivity_labeled > sensitivity_total:
        raise AssertionError(
            "PER route sensitivity labels exceed observable actor pairs"
        )
    if sensitivity_changed_labeled > sensitivity_changed:
        raise AssertionError(
            "PER route sensitivity labeled changes exceed observed changes"
        )
    if row["protocol"] != "per" and sensitivity_total:
        raise AssertionError(
            "PER route sensitivity records appeared in another protocol"
        )


def _finalize_summary(row: MutableMapping[str, Any]) -> None:
    reviews = int(row["review_opportunities"])
    responses = int(row["actor_response_observed"])
    parsed = int(row["answer_pair_parsed"])
    row["actor_response_rate_over_review_opportunities"] = _safe_rate(
        responses, reviews
    )
    row["strict_rate_over_review_opportunities"] = _safe_rate(
        int(row["strict_pre_gate_eligible"]), reviews
    )
    row["correctness_coverage_over_parsed_actor_pairs"] = _safe_rate(
        int(row["correctness_pair_labeled"]), parsed
    )
    # Unchanged includes correct, wrong, and unlabeled rows, so the explicit
    # repair denominators are reconstructed from the detailed frozen labels.
    row["repair_rate_labeled_pre_wrong"] = None
    row["strict_repair_rate_labeled_pre_wrong"] = None
    row["gate_repair_rate_labeled_pre_wrong"] = None
    row["repair_rate_denominator_note"] = (
        "within this dataset x slice x actor x protocol cell: before_correct="
        "false and after_correct labeled; strict and gate-confounded "
        "denominators are reported separately"
    )
    row["explicit_revise_uptake_denominator_note"] = (
        "review_action=revise, before_correct=false, and a parsed actor pair "
        "in the stated strict or gate-confounded bucket; after_correct is not "
        "required to observe answer uptake"
    )
    row["explicit_revise_repair_denominator_note"] = (
        "the explicit-revise uptake denominator restricted to pairs with a "
        "frozen after_correct label"
    )
    row["review_action_revise_definition"] = REVISE_ACTION_DEFINITION
    row["pending_unknown_action_recovery_note"] = (
        PENDING_UNKNOWN_ACTION_RECOVERY_NOTE
    )
    row["per_route_sensitivity_definition"] = (
        PER_ROUTE_SENSITIVITY_DEFINITION
    )


def _detail_row(
    metadata: Mapping[str, Any], record: Mapping[str, Any]
) -> dict[str, Any]:
    row = {
        "dataset": metadata["benchmark_id"],
        "slice_id": metadata["slice_id"],
        "actor_family": metadata["actor_model_id"],
        "evaluator_model_id": metadata["evaluator_model_id"],
        "protocol": metadata["protocol_id"],
        "release_status": metadata["release_status"],
        "provenance_status": metadata["provenance_status"],
        "matrix_4x2x5": bool(metadata["matrix_4x2x5"]),
        "trajectory_uid": metadata["trajectory_uid"],
        "problem_uid": metadata["problem_uid"],
    }
    row.update({field: record.get(field) for field in DETAIL_FIELDS if field not in row})
    return row


def _summary_fields() -> list[str]:
    preferred = [
        "dataset",
        "slice_id",
        "actor_family",
        "protocol",
        "evaluator_model_id",
        "release_statuses_json",
        "provenance_statuses_json",
        "matrix_4x2x5",
        "trajectories",
        "trajectories_with_review",
    ] + SUMMARY_COUNT_FIELDS
    trailing = [
        "actor_response_rate_over_review_opportunities",
        "strict_rate_over_review_opportunities",
        "correctness_coverage_over_parsed_actor_pairs",
        "labeled_pre_wrong_pairs",
        "repaired_over_labeled_pre_wrong",
        "repair_rate_labeled_pre_wrong",
        "strict_labeled_pre_wrong_pairs",
        "strict_repaired_over_labeled_pre_wrong",
        "strict_repair_rate_labeled_pre_wrong",
        "gate_labeled_pre_wrong_pairs",
        "gate_repaired_over_labeled_pre_wrong",
        "gate_repair_rate_labeled_pre_wrong",
        "repair_rate_denominator_note",
        *REVISE_CONDITIONED_COUNT_FIELDS,
        "strict_explicit_revise_change_response_rate",
        "strict_explicit_revise_after_correct_label_coverage_over_uptake",
        "strict_explicit_revise_after_correct_label_coverage_over_changed",
        "strict_explicit_revise_repair_rate_over_after_correct_labeled_pairs",
        "gate_explicit_revise_change_response_rate",
        "gate_explicit_revise_after_correct_label_coverage_over_uptake",
        "gate_explicit_revise_after_correct_label_coverage_over_changed",
        "gate_explicit_revise_repair_rate_over_after_correct_labeled_pairs",
        "explicit_revise_uptake_denominator_note",
        "explicit_revise_repair_denominator_note",
        "review_action_revise_definition",
        "pending_unknown_action_recovery_note",
        *PER_ROUTE_SENSITIVITY_COUNT_FIELDS,
        "per_route_sensitivity_change_response_rate",
        "per_route_sensitivity_after_correct_label_coverage_over_uptake",
        "per_route_sensitivity_after_correct_label_coverage_over_changed",
        "per_route_sensitivity_repair_rate_over_after_correct_labeled_pairs",
        "per_route_sensitivity_definition",
    ]
    return preferred + trailing


def _comparison_gate(
    summary_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in summary_rows:
        key = (
            str(row["dataset"]),
            str(row["slice_id"]),
            str(row["actor_family"]),
        )
        grouped[key][str(row["protocol"])] = row
    results: list[dict[str, Any]] = []
    for (dataset, slice_id, actor), protocols in sorted(grouped.items()):
        per = protocols.get("per")
        broadcast = protocols.get("broadcast")
        per_n = int(per["strict_pre_gate_eligible"]) if per else 0
        broadcast_n = (
            int(broadcast["strict_pre_gate_eligible"]) if broadcast else 0
        )
        if not per or not broadcast:
            status = "missing_protocol"
        elif per_n == 0 or broadcast_n == 0:
            status = "not_comparable_zero_strict_observations"
        else:
            status = "descriptive_strict_pairs_exist_not_causal"
        results.append(
            {
                "dataset": dataset,
                "slice_id": slice_id,
                "actor_family": actor,
                "per_strict_n": per_n,
                "broadcast_strict_n": broadcast_n,
                "comparison_status": status,
            }
        )
    return results


def _update_correctness_denominators(
    summaries: Mapping[tuple[str, str, str, str], MutableMapping[str, Any]],
    denominator_counts: Mapping[tuple[str, str, str, str], Counter[str]],
) -> None:
    for key, row in summaries.items():
        counts = denominator_counts.get(key, Counter())
        for prefix in ("", "strict_", "gate_"):
            denominator_field = f"{prefix}labeled_pre_wrong_pairs"
            repaired_field = f"{prefix}repaired_over_labeled_pre_wrong"
            rate_field = f"{prefix}repair_rate_labeled_pre_wrong"
            denominator = int(counts[denominator_field])
            repaired = int(counts[repaired_field])
            row[denominator_field] = denominator
            row[repaired_field] = repaired
            row[rate_field] = _safe_rate(repaired, denominator)
        for prefix in ("strict_", "gate_"):
            for field in REVISE_CONDITIONED_COUNT_FIELDS:
                if field.startswith(f"{prefix}explicit_revise_"):
                    row[field] = int(counts[field])
            uptake_denominator = int(
                counts[f"{prefix}explicit_revise_pre_wrong_actor_pairs"]
            )
            answer_changed = int(
                counts[
                    f"{prefix}explicit_revise_answer_changed"
                ]
            )
            labeled_denominator = int(
                counts[
                    f"{prefix}explicit_revise_after_correct_labeled_pairs"
                ]
            )
            changed_labeled = int(
                counts[
                    f"{prefix}explicit_revise_changed_after_correct_labeled"
                ]
            )
            repaired = int(
                counts[f"{prefix}explicit_revise_repaired"]
            )
            row[
                f"{prefix}explicit_revise_change_response_rate"
            ] = _safe_rate(answer_changed, uptake_denominator)
            row[
                f"{prefix}explicit_revise_after_correct_label_coverage_over_uptake"
            ] = _safe_rate(labeled_denominator, uptake_denominator)
            row[
                f"{prefix}explicit_revise_after_correct_label_coverage_over_changed"
            ] = _safe_rate(changed_labeled, answer_changed)
            row[
                f"{prefix}explicit_revise_repair_rate_over_after_correct_labeled_pairs"
            ] = _safe_rate(
                repaired, labeled_denominator
            )
        for field in PER_ROUTE_SENSITIVITY_COUNT_FIELDS:
            row[field] = int(counts[field])
        sensitivity_total = int(
            counts["per_route_sensitivity_pre_wrong_actor_pairs"]
        )
        sensitivity_changed = int(
            counts["per_route_sensitivity_answer_changed"]
        )
        sensitivity_labeled = int(
            counts["per_route_sensitivity_after_correct_labeled_pairs"]
        )
        sensitivity_changed_labeled = int(
            counts[
                "per_route_sensitivity_changed_after_correct_labeled"
            ]
        )
        sensitivity_repaired = int(
            counts["per_route_sensitivity_repaired"]
        )
        row["per_route_sensitivity_change_response_rate"] = _safe_rate(
            sensitivity_changed, sensitivity_total
        )
        row[
            "per_route_sensitivity_after_correct_label_coverage_over_uptake"
        ] = _safe_rate(sensitivity_labeled, sensitivity_total)
        row[
            "per_route_sensitivity_after_correct_label_coverage_over_changed"
        ] = _safe_rate(sensitivity_changed_labeled, sensitivity_changed)
        row[
            "per_route_sensitivity_repair_rate_over_after_correct_labeled_pairs"
        ] = _safe_rate(sensitivity_repaired, sensitivity_labeled)


def _update_revise_conditioned_denominators(
    counts: Counter[str],
    detail: Mapping[str, Any],
) -> None:
    """Separate observable answer uptake from label-complete repair."""

    if str(detail.get("review_action") or "") != "revise":
        return
    if bool(detail.get("strict_pre_gate_eligible")):
        prefix = "strict_"
    elif detail.get("primary_eligibility_bucket") == "gate_confounded":
        prefix = "gate_"
    else:
        return
    if detail.get("before_correct") is not False:
        return

    answer_changed = detail.get("answer_changed")
    if answer_changed not in {True, False}:
        raise AssertionError(
            "An eligible parsed actor pair has no observable answer_changed"
        )
    counts[f"{prefix}explicit_revise_pre_wrong_actor_pairs"] += 1
    if answer_changed:
        counts[f"{prefix}explicit_revise_answer_changed"] += 1
    else:
        counts[f"{prefix}explicit_revise_answer_unchanged"] += 1

    if detail.get("after_correct") is None:
        return
    outcome = str(detail.get("transition_outcome") or "")
    outcome_field = {
        "unchanged": f"{prefix}explicit_revise_labeled_unchanged",
        "changed_still_wrong": (
            f"{prefix}explicit_revise_labeled_changed_still_wrong"
        ),
        "repaired": f"{prefix}explicit_revise_repaired",
    }.get(outcome)
    if outcome_field is None:
        raise AssertionError(
            "An after-correct-labeled pre-wrong transition after explicit "
            f"review_action=revise has an impossible outcome: {outcome}"
        )

    counts[f"{prefix}explicit_revise_after_correct_labeled_pairs"] += 1
    counts[outcome_field] += 1
    if outcome in {"changed_still_wrong", "repaired"}:
        counts[
            f"{prefix}explicit_revise_changed_after_correct_labeled"
        ] += 1


def _update_per_route_sensitivity_counts(
    counts: Counter[str],
    detail: Mapping[str, Any],
) -> None:
    """Accumulate a route-observed sensitivity without relabeling stance."""

    if str(detail.get("protocol") or "") != "per":
        return
    if not bool(detail.get("strict_pre_gate_eligible")):
        return
    if detail.get("before_correct") is not False:
        return

    action = str(detail.get("review_action") or "unknown")
    if action == "revise":
        source_field = "per_route_sensitivity_explicit_revise_pairs"
    elif action == "unknown" and bool(detail.get("review_route_observed")):
        source_field = (
            "per_route_sensitivity_unknown_linked_route_pairs"
        )
    else:
        return

    answer_changed = detail.get("answer_changed")
    if answer_changed not in {True, False}:
        raise AssertionError(
            "A PER route-sensitivity actor pair has no answer_changed value"
        )
    counts["per_route_sensitivity_pre_wrong_actor_pairs"] += 1
    counts[source_field] += 1
    if answer_changed:
        counts["per_route_sensitivity_answer_changed"] += 1
    else:
        counts["per_route_sensitivity_answer_unchanged"] += 1

    if detail.get("after_correct") is None:
        return
    outcome = str(detail.get("transition_outcome") or "")
    outcome_field = {
        "unchanged": "per_route_sensitivity_labeled_unchanged",
        "changed_still_wrong": (
            "per_route_sensitivity_labeled_changed_still_wrong"
        ),
        "repaired": "per_route_sensitivity_repaired",
    }.get(outcome)
    if outcome_field is None:
        raise AssertionError(
            "An after-correct-labeled PER route-sensitivity transition has "
            f"an impossible outcome: {outcome}"
        )
    counts["per_route_sensitivity_after_correct_labeled_pairs"] += 1
    counts[outcome_field] += 1
    if outcome in {"changed_still_wrong", "repaired"}:
        counts[
            "per_route_sensitivity_changed_after_correct_labeled"
        ] += 1


def _compact_matrix_per_route_sensitivity(
    matrix_summary_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Pool the PER-only route-observed sensitivity across matrix cells."""

    per_rows = [
        row for row in matrix_summary_rows if row["protocol"] == "per"
    ]
    counts = {
        field: sum(int(row[field]) for row in per_rows)
        for field in PER_ROUTE_SENSITIVITY_COUNT_FIELDS
    }
    total = counts["per_route_sensitivity_pre_wrong_actor_pairs"]
    changed = counts["per_route_sensitivity_answer_changed"]
    labeled = counts[
        "per_route_sensitivity_after_correct_labeled_pairs"
    ]
    changed_labeled = counts[
        "per_route_sensitivity_changed_after_correct_labeled"
    ]
    repaired = counts["per_route_sensitivity_repaired"]
    return {
        "scope": "matrix_4x2x5_per_only",
        "protocol": "per",
        "eligibility_bucket": "strict_pre_gate",
        "matrix_cells": len(per_rows),
        "cells_with_pre_wrong_actor_pairs": sum(
            int(row["per_route_sensitivity_pre_wrong_actor_pairs"]) > 0
            for row in per_rows
        ),
        **counts,
        "change_response_rate": _safe_rate(changed, total),
        "after_correct_label_coverage_over_uptake": _safe_rate(
            labeled, total
        ),
        "after_correct_label_coverage_over_changed": _safe_rate(
            changed_labeled, changed
        ),
        "repair_rate_over_after_correct_labeled_pairs": _safe_rate(
            repaired, labeled
        ),
        "aggregation": "pooled_count_micro_average",
        "sensitivity_definition": PER_ROUTE_SENSITIVITY_DEFINITION,
        "semantic_review_action_overridden": False,
        "critique_stance_or_usefulness_claim_allowed": False,
        "unknown_action_note": PENDING_UNKNOWN_ACTION_RECOVERY_NOTE,
    }


def _compact_matrix_revise_summary(
    matrix_summary_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Pool counts into four descriptive protocol-by-eligibility rows."""

    output: list[dict[str, Any]] = []
    for protocol in ("per", "broadcast"):
        protocol_rows = [
            row for row in matrix_summary_rows if row["protocol"] == protocol
        ]
        for prefix, bucket in (
            ("strict_", "strict_pre_gate"),
            ("gate_", "gate_confounded"),
        ):
            uptake_denominator_field = (
                f"{prefix}explicit_revise_pre_wrong_actor_pairs"
            )
            answer_unchanged_field = (
                f"{prefix}explicit_revise_answer_unchanged"
            )
            answer_changed_field = (
                f"{prefix}explicit_revise_answer_changed"
            )
            labeled_denominator_field = (
                f"{prefix}explicit_revise_after_correct_labeled_pairs"
            )
            labeled_unchanged_field = (
                f"{prefix}explicit_revise_labeled_unchanged"
            )
            labeled_changed_wrong_field = (
                f"{prefix}explicit_revise_labeled_changed_still_wrong"
            )
            repaired_field = f"{prefix}explicit_revise_repaired"
            changed_labeled_field = (
                f"{prefix}explicit_revise_changed_after_correct_labeled"
            )
            uptake_denominator = sum(
                int(row[uptake_denominator_field]) for row in protocol_rows
            )
            answer_unchanged = sum(
                int(row[answer_unchanged_field]) for row in protocol_rows
            )
            answer_changed = sum(
                int(row[answer_changed_field]) for row in protocol_rows
            )
            labeled_denominator = sum(
                int(row[labeled_denominator_field]) for row in protocol_rows
            )
            labeled_unchanged = sum(
                int(row[labeled_unchanged_field]) for row in protocol_rows
            )
            labeled_changed_wrong = sum(
                int(row[labeled_changed_wrong_field]) for row in protocol_rows
            )
            repaired = sum(
                int(row[repaired_field]) for row in protocol_rows
            )
            changed_labeled = sum(
                int(row[changed_labeled_field]) for row in protocol_rows
            )
            output.append(
                {
                    "scope": "matrix_4x2x5_per_broadcast",
                    "protocol": protocol,
                    "eligibility_bucket": bucket,
                    "matrix_cells": len(protocol_rows),
                    "cells_with_explicit_revise_pre_wrong_actor_pairs": sum(
                        int(row[uptake_denominator_field]) > 0
                        for row in protocol_rows
                    ),
                    "explicit_revise_pre_wrong_actor_pairs": (
                        uptake_denominator
                    ),
                    "answer_unchanged": answer_unchanged,
                    "answer_changed": answer_changed,
                    "change_response_rate": _safe_rate(
                        answer_changed, uptake_denominator
                    ),
                    "after_correct_labeled_pairs": labeled_denominator,
                    "after_correct_label_coverage_over_uptake": _safe_rate(
                        labeled_denominator, uptake_denominator
                    ),
                    "changed_after_correct_labeled": changed_labeled,
                    "after_correct_label_coverage_over_changed": _safe_rate(
                        changed_labeled, answer_changed
                    ),
                    "labeled_unchanged": labeled_unchanged,
                    "labeled_changed_still_wrong": labeled_changed_wrong,
                    "repaired": repaired,
                    "repair_rate_over_after_correct_labeled_pairs": _safe_rate(
                        repaired, labeled_denominator
                    ),
                    "aggregation": "pooled_count_micro_average",
                    "review_action_definition": REVISE_ACTION_DEFINITION,
                    "unknown_action_note": (
                        PENDING_UNKNOWN_ACTION_RECOVERY_NOTE
                    ),
                }
            )
    return output


def aggregate_release(
    release_root: Path,
    output_dir: Path,
    *,
    scope: str,
) -> dict[str, Any]:
    metadata = load_trajectory_metadata(release_root, scope=scope)
    if not metadata:
        raise ValueError("No PER/Broadcast trajectories matched the requested scope")

    summaries: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    status_sets: dict[tuple[str, str, str, str], dict[str, set[str]]] = defaultdict(
        lambda: {"release": set(), "provenance": set()}
    )
    for item in metadata.values():
        key = _group_key(item)
        if key not in summaries:
            summaries[key] = _new_summary(item)
        summaries[key]["trajectories"] += 1
        status_sets[key]["release"].add(str(item["release_status"]))
        status_sets[key]["provenance"].add(str(item["provenance_status"]))

    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / "full_release_symmetric_transitions.csv"
    detail_tmp = detail_path.with_suffix(".csv.tmp")
    denominator_counts: dict[
        tuple[str, str, str, str], Counter[str]
    ] = defaultdict(Counter)
    seen_trajectories: set[str] = set()
    detail_rows = 0

    with detail_tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DETAIL_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for trajectory_uid, events in iter_selected_trajectory_events(
            release_root / "data" / "messages", set(metadata)
        ):
            item = metadata[trajectory_uid]
            seen_trajectories.add(trajectory_uid)
            labels = build_frozen_answer_labels(
                events,
                final_answer=item.get("final_answer"),
                final_correct=_as_bool(item.get("final_correct")),
            )
            transitions = extract_symmetric_transitions(
                str(item["protocol_id"]),
                events,
                trajectory_uid=trajectory_uid,
            )
            key = _group_key(item)
            if transitions:
                summaries[key]["trajectories_with_review"] += 1
            for transition in transitions:
                labeled = attach_frozen_correctness(transition, labels)
                detail = _detail_row(item, labeled)
                _validate_detail_record(detail)
                writer.writerow(detail)
                detail_rows += 1
                _increment_summary(summaries[key], detail)
                _update_revise_conditioned_denominators(
                    denominator_counts[key], detail
                )
                _update_per_route_sensitivity_counts(
                    denominator_counts[key], detail
                )
                if (
                    detail.get("before_correct") is False
                    and detail.get("after_correct") is not None
                ):
                    denominator_counts[key]["labeled_pre_wrong_pairs"] += 1
                    prefixes = [""]
                    if bool(detail.get("strict_pre_gate_eligible")):
                        prefixes.append("strict_")
                    if (
                        detail.get("primary_eligibility_bucket")
                        == "gate_confounded"
                    ):
                        prefixes.append("gate_")
                    for prefix in prefixes:
                        if prefix:
                            denominator_counts[key][
                                f"{prefix}labeled_pre_wrong_pairs"
                            ] += 1
                        if detail.get("transition_outcome") == "repaired":
                            denominator_counts[key][
                                f"{prefix}repaired_over_labeled_pre_wrong"
                            ] += 1

    detail_tmp.replace(detail_path)
    missing = set(metadata) - seen_trajectories
    if missing:
        raise ValueError(
            f"{len(missing)} selected trajectories were missing messages; "
            f"examples={sorted(missing)[:5]}"
        )

    for key, row in summaries.items():
        row["release_statuses_json"] = json.dumps(
            sorted(status_sets[key]["release"]), ensure_ascii=False
        )
        row["provenance_statuses_json"] = json.dumps(
            sorted(status_sets[key]["provenance"]), ensure_ascii=False
        )
        _finalize_summary(row)
    _update_correctness_denominators(summaries, denominator_counts)
    for row in summaries.values():
        _validate_summary_row(row)

    all_summary_rows = [summaries[key] for key in sorted(summaries)]
    matrix_summary_rows = [
        row for row in all_summary_rows if bool(row["matrix_4x2x5"])
    ]
    matrix_revise_summary_rows = _compact_matrix_revise_summary(
        matrix_summary_rows
    )
    matrix_per_route_sensitivity = (
        _compact_matrix_per_route_sensitivity(matrix_summary_rows)
    )
    fields = _summary_fields()
    summary_path = output_dir / "full_release_transition_summary.csv"
    matrix_path = output_dir / "matrix_4x2x5_transition_summary.csv"
    for path, rows in (
        (summary_path, all_summary_rows),
        (matrix_path, matrix_summary_rows),
    ):
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    matrix_revise_path = (
        output_dir / "matrix_4x2x5_revise_conditioned_summary.csv"
    )
    with matrix_revise_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "scope",
                "protocol",
                "eligibility_bucket",
                "matrix_cells",
                "cells_with_explicit_revise_pre_wrong_actor_pairs",
                "explicit_revise_pre_wrong_actor_pairs",
                "answer_unchanged",
                "answer_changed",
                "change_response_rate",
                "after_correct_labeled_pairs",
                "after_correct_label_coverage_over_uptake",
                "changed_after_correct_labeled",
                "after_correct_label_coverage_over_changed",
                "labeled_unchanged",
                "labeled_changed_still_wrong",
                "repaired",
                "repair_rate_over_after_correct_labeled_pairs",
                "aggregation",
                "review_action_definition",
                "unknown_action_note",
            ],
        )
        writer.writeheader()
        writer.writerows(matrix_revise_summary_rows)

    matrix_per_route_path = (
        output_dir
        / "matrix_4x2x5_per_route_observed_sensitivity.csv"
    )
    with matrix_per_route_path.open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(matrix_per_route_sensitivity.keys()),
        )
        writer.writeheader()
        writer.writerow(matrix_per_route_sensitivity)

    comparison_gate = _comparison_gate(matrix_summary_rows)
    summary_json_path = output_dir / "full_release_transition_summary.json"
    payload = {
        "scope": scope,
        "trajectory_rows": len(metadata),
        "detail_transition_rows": detail_rows,
        "all_release_cells": all_summary_rows,
        "matrix_4x2x5_cells": matrix_summary_rows,
        "matrix_4x2x5_revise_conditioned_summary": (
            matrix_revise_summary_rows
        ),
        "matrix_4x2x5_per_route_observed_sensitivity": (
            matrix_per_route_sensitivity
        ),
        "matrix_protocol_comparison_gate": comparison_gate,
        "quality_control": {
            "detail_invariants_validated": True,
            "summary_partition_invariants_validated": True,
            "selected_trajectories_missing_messages": 0,
            "system_selected_answer_used_as_after": 0,
        },
        "correctness_definition": {
            "source": (
                "same-trajectory frozen evaluator submission PASS/FAIL and "
                "trajectory final_correct only"
            ),
            "legacy_transition_labels_used": False,
            "gold_answer_exact_match_used": False,
            "unchanged": "normalized actor answers are identical",
            "changed_still_wrong": (
                "answers differ; both have frozen labels and both are wrong"
            ),
            "repaired": (
                "answers differ; frozen before label is wrong and after is correct"
            ),
            "harmed": (
                "answers differ; frozen before label is correct and after is wrong"
            ),
        },
        "review_action_conditioning": {
            "definition": REVISE_ACTION_DEFINITION,
            "uptake_denominator": (
                "within strict_pre_gate or gate_confounded separately: "
                "explicit review_action=revise, before_correct=false, and a "
                "parsed eligible actor pair; after_correct is not required"
            ),
            "change_response_rate": (
                "answer_changed / explicit-revise pre-wrong actor pairs"
            ),
            "repair_complete_case_denominator": (
                "explicit-revise pre-wrong actor pairs with a frozen "
                "after_correct label"
            ),
            "repair_rate": (
                "repaired / after-correct-labeled explicit-revise pre-wrong "
                "actor pairs"
            ),
            "label_coverage_over_uptake": (
                "after-correct-labeled pairs / explicit-revise pre-wrong "
                "actor pairs"
            ),
            "label_coverage_over_changed": (
                "changed pairs with frozen after_correct / all changed pairs "
                "in the explicit-revise pre-wrong uptake denominator"
            ),
            "unknown_action_note": PENDING_UNKNOWN_ACTION_RECOVERY_NOTE,
            "causal_interpretation_allowed": False,
        },
        "per_route_observed_sensitivity": {
            "definition": PER_ROUTE_SENSITIVITY_DEFINITION,
            "semantic_review_action_overridden": False,
            "critique_stance_or_usefulness_claim_allowed": False,
            "result": matrix_per_route_sensitivity,
        },
    }
    summary_json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    release_manifest = release_root / "registry" / "release_manifest.json"
    embedded = json.loads(release_manifest.read_text(encoding="utf-8"))
    manifest_path = output_dir / "full_release_transition_manifest.json"
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": scope,
        "source_hf_repository": embedded.get("source_hf_repository"),
        "embedded_source_revision": embedded.get("source_hf_revision"),
        "release_manifest_sha256": _sha256(release_manifest),
        "extractor_sha256": _sha256(HERE / "symmetric_transition_extractor.py"),
        "aggregator_sha256": _sha256(Path(__file__).resolve()),
        "trajectory_rows": len(metadata),
        "selected_trajectory_rows_seen": len(seen_trajectories),
        "detail_transition_rows": detail_rows,
        "summary_cells": len(all_summary_rows),
        "matrix_summary_cells": len(matrix_summary_rows),
        "legacy_transition_labels_used": False,
        "gold_answer_exact_match_used": False,
        "figures_generated": False,
        "detail_invariants_validated": True,
        "summary_partition_invariants_validated": True,
        "review_action_revise_definition": REVISE_ACTION_DEFINITION,
        "explicit_revise_uptake_denominator": (
            "review_action=revise, before_correct=false, and a parsed eligible "
            "actor pair; after_correct is not required"
        ),
        "explicit_revise_repair_complete_case_denominator": (
            "the uptake denominator restricted to a frozen after_correct label"
        ),
        "pending_unknown_action_recovery_note": (
            PENDING_UNKNOWN_ACTION_RECOVERY_NOTE
        ),
        "per_route_sensitivity_definition": (
            PER_ROUTE_SENSITIVITY_DEFINITION
        ),
        "matrix_4x2x5_revise_conditioned_summary": (
            matrix_revise_summary_rows
        ),
        "matrix_4x2x5_per_route_observed_sensitivity": (
            matrix_per_route_sensitivity
        ),
        "outputs": {
            detail_path.name: _sha256(detail_path),
            summary_path.name: _sha256(summary_path),
            matrix_path.name: _sha256(matrix_path),
            matrix_revise_path.name: _sha256(matrix_revise_path),
            matrix_per_route_path.name: _sha256(matrix_per_route_path),
            summary_json_path.name: _sha256(summary_json_path),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "manifest": manifest,
        "summary": payload,
        "paths": {
            "detail": str(detail_path),
            "summary_csv": str(summary_path),
            "matrix_csv": str(matrix_path),
            "matrix_revise_csv": str(matrix_revise_path),
            "matrix_per_route_sensitivity_csv": str(
                matrix_per_route_path
            ),
            "summary_json": str(summary_json_path),
            "manifest": str(manifest_path),
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, default=DEFAULT_RELEASE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--scope",
        choices=["all-per-broadcast", "matrix-4x2x5"],
        default="all-per-broadcast",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = aggregate_release(
        args.release_root,
        args.output_dir,
        scope=args.scope,
    )
    manifest = result["manifest"]
    print(
        "Processed "
        f"{manifest['trajectory_rows']} trajectories into "
        f"{manifest['detail_transition_rows']} review opportunities across "
        f"{manifest['summary_cells']} cells."
    )
    for label, path in result["paths"].items():
        print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
