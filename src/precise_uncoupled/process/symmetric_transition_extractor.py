#!/usr/bin/env python3
"""Extract rebuttal-safe, symmetric review-to-actor transitions.

The estimand is deliberately narrow for both supported protocols:

    actor-produced candidate immediately before a review
        -> next actor-produced candidate after that review

The post-review candidate must be produced by the protocol's answer-producing
actor (PER Executor or a Broadcast poll/discussion source matched to the next
review anchor). A reviewer-proposed correction, system candidate-memory record,
protocol-selected candidate update, submission, or evaluator message is never
used as ``answer_after``.

Two eligibility levels are exposed because Broadcast can select a reviewer
proposal in protocol state before a later actor poll/discussion produces a new
answer:

``actor_response_observed``
    A genuine next actor-produced candidate exists.

``strict_pre_gate_eligible``
    The genuine actor response exists, both answers parse, candidate provenance
    aligns, and no system candidate selection or submission intervenes.

The first quantity can document whether an actor had another opportunity.  The
second is the strict pre-system-selection descriptive comparison. Neither
identifies a causal effect of review content or of the approval mechanism.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SUPPORTED_PROTOCOLS = {"per", "broadcast"}
SPECIAL_SENDERS = {"system", "evaluator", "reviewer", "planner"}

REVIEW_POSITION_RE = re.compile(
    r"\bReview\s+Position\s*:\s*([A-Za-z_]+)", re.IGNORECASE
)
SOURCE_RE = re.compile(r"\bSource\s*:\s*([^\n]+)", re.IGNORECASE)
EXPLICIT_ANSWER_RE = re.compile(
    r"(?:final\s+answer|answer)\s*(?:is|:)\s*"
    r"([^\n]{1,160})",
    re.IGNORECASE,
)
PROPOSED_CORRECTION_RE = re.compile(
    r"\bProposed\s+Correction\b[^:]*:\s*(.*)",
    re.IGNORECASE | re.DOTALL,
)


def _text(value: Any) -> str:
    return str(value or "")


def _lower(value: Any) -> str:
    return _text(value).strip().lower()


def _event_type(event: Mapping[str, Any]) -> str:
    return _lower(event.get("event_type") or event.get("type"))


def _stage(event: Mapping[str, Any]) -> str:
    return _lower(event.get("stage"))


def _sender(event: Mapping[str, Any]) -> str:
    return _text(event.get("sender")).strip()


def _role(event: Mapping[str, Any]) -> str:
    return _lower(event.get("role"))


def _sequence(event: Mapping[str, Any], fallback: int) -> int:
    raw = event.get("sequence_index")
    try:
        return int(raw)
    except (TypeError, ValueError):
        return fallback


def _event_uid(event: Mapping[str, Any], fallback: int) -> str:
    value = _text(event.get("event_uid") or event.get("message_uid")).strip()
    return value or f"sequence:{_sequence(event, fallback)}"


def _sanitize_boxed_spelling(text: Any) -> str:
    # Some normalized traces contain a literal backspace from ``\boxed``.
    return _text(text).replace("\x08oxed{", r"\boxed{")


def extract_all_boxed(text: Any) -> list[str]:
    """Return balanced ``\\boxed{...}`` payloads in textual order."""

    raw = _sanitize_boxed_spelling(text)
    marker = r"\boxed{"
    answers: list[str] = []
    start = 0
    while True:
        marker_at = raw.find(marker, start)
        if marker_at < 0:
            break
        cursor = marker_at + len(marker)
        depth = 1
        payload: list[str] = []
        while cursor < len(raw) and depth:
            char = raw[cursor]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    break
            payload.append(char)
            cursor += 1
        if depth == 0:
            answer = "".join(payload).strip()
            if answer:
                answers.append(answer)
            start = cursor + 1
        else:
            break
    return answers


def extract_explicit_answer(text: Any) -> str:
    """Extract a conservative final-answer string from actor-authored text."""

    raw = _sanitize_boxed_spelling(text).strip()
    boxed = extract_all_boxed(raw)
    if boxed:
        answer = boxed[-1]
        while nested := extract_all_boxed(answer):
            next_answer = nested[-1]
            if next_answer == answer:
                break
            answer = next_answer
        return "" if _is_placeholder_answer(answer) else answer

    matches = list(EXPLICIT_ANSWER_RE.finditer(raw))
    if not matches:
        return ""
    answer = matches[-1].group(1).strip().rstrip(".")
    # Reject long prose rather than silently turning it into an answer.
    if len(answer) > 120 or len(answer.split()) > 12:
        return ""
    return "" if _is_placeholder_answer(answer) else answer


def _is_placeholder_answer(value: Any) -> bool:
    key = re.sub(r"\s+", " ", _text(value)).strip().lower()
    return key in {
        "",
        ".",
        "..",
        "...",
        "…",
        "none",
        "[none]",
        "null",
        "n/a",
        "[no candidate answer]",
        "[no extracted final answer]",
        "no solution yet",
        "no advice yet",
    }


def normalize_answer(value: Any) -> str:
    """A comparison key, not a correctness evaluator."""

    text = _sanitize_boxed_spelling(value).strip()
    boxed = extract_all_boxed(text)
    if boxed:
        text = boxed[-1]
    text = re.sub(r"\\(?:mathrm|text)\{([^{}]*)\}", r"\1", text)
    text = text.replace("\\,", "").replace("\\ ", "")
    text = re.sub(r"\s+", "", text).strip("$.").lower()
    letters = re.sub(r"[^a-z]", "", text)
    if letters and len(letters) <= 8 and len(letters) == len(text):
        return "".join(sorted(letters))
    return text


def _content_excerpt(value: Any, limit: int = 240) -> str:
    return re.sub(r"\s+", " ", _text(value)).strip()[:limit]


def _extra_json(event: Mapping[str, Any]) -> dict[str, Any]:
    raw = event.get("extra_json")
    if isinstance(raw, dict):
        return raw
    if not _text(raw).strip():
        return {}
    try:
        value = json.loads(_text(raw))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _explicit_field_answer(value: Any) -> str:
    raw = _sanitize_boxed_spelling(value).strip()
    if _is_placeholder_answer(raw):
        return ""
    answer = extract_explicit_answer(raw)
    if answer:
        return answer
    # Do not fall back to the raw wrapper when an explicitly marked answer
    # contains only a placeholder such as ``\boxed{...}`` or
    # ``Candidate Answer: ...``.
    if r"\boxed{" in raw or EXPLICIT_ANSWER_RE.search(raw):
        return ""
    if len(raw) <= 120 and len(raw.split()) <= 12:
        return raw
    return ""


def candidate_answer_from_event(event: Mapping[str, Any]) -> str:
    """Parse only an answer explicitly authored in this actor event."""

    stage = _stage(event)
    if stage.startswith("poll_"):
        answer = _explicit_field_answer(_extra_json(event).get("candidate_answer"))
        if answer:
            return answer
        candidates = re.findall(
            r"^\s*candidate\s*=\s*(.+?)\s*$",
            _sanitize_boxed_spelling(event.get("content")),
            re.IGNORECASE | re.MULTILINE,
        )
        if candidates:
            return _explicit_field_answer(candidates[-1])
    return extract_explicit_answer(event.get("content"))


def is_actor_turn(protocol: str, event: Mapping[str, Any]) -> bool:
    """True only for answer-producing actor opportunities."""

    protocol = _lower(protocol)
    if protocol not in SUPPORTED_PROTOCOLS:
        raise ValueError(f"Unsupported protocol: {protocol!r}")

    stage = _stage(event)
    sender = _lower(event.get("sender"))
    if "candidate_memory" in stage or not _text(event.get("content")).strip():
        return False

    if protocol == "per":
        return (
            _event_type(event) == "message"
            and (
                stage.startswith("executor_")
                or stage == "executor"
                or stage.startswith("executor_post_eval")
            )
            and sender == "executor"
        )

    discussion = (
        _event_type(event) == "message"
        and (stage.startswith("discussion_") or stage == "discussion")
    )
    poll = _event_type(event) == "summary" and stage.startswith("poll_")
    return (
        (discussion or poll)
        and sender not in SPECIAL_SENDERS
        and _role(event) not in {"system", "evaluator"}
    )


def is_actor_candidate(protocol: str, event: Mapping[str, Any]) -> bool:
    """True only when an actor turn contains a parsed candidate answer."""

    return is_actor_turn(protocol, event) and bool(candidate_answer_from_event(event))


def is_review_message(protocol: str, event: Mapping[str, Any]) -> bool:
    protocol = _lower(protocol)
    stage = _stage(event)
    sender = _lower(event.get("sender"))
    if _event_type(event) != "message" or "candidate_memory" in stage:
        return False
    if protocol == "per":
        return stage.startswith("reviewer_") and sender == "reviewer"
    if protocol == "broadcast":
        return stage.startswith("approval_") and bool(
            REVIEW_POSITION_RE.search(_text(event.get("content")))
        )
    raise ValueError(f"Unsupported protocol: {protocol!r}")


def is_broadcast_review_anchor(event: Mapping[str, Any]) -> bool:
    stage = _stage(event)
    return (
        _event_type(event) == "message"
        and stage.startswith("candidate_review_")
        and "candidate_memory" not in stage
        and _lower(event.get("sender")) == "system"
    )


def system_candidate_update(event: Mapping[str, Any]) -> bool:
    stage = _stage(event)
    return (
        "candidate_update" in stage
        and "candidate_memory" not in stage
        and _lower(event.get("sender")) == "system"
    )


def submission_gate_kind(event: Mapping[str, Any]) -> str:
    """Return a hard post-review boundary, or an empty string."""

    stage = _stage(event)
    if "submission" in stage:
        return "submission"
    return ""


def _review_action(protocol: str, contents: Sequence[str]) -> str:
    if protocol == "broadcast":
        positions = [
            match.group(1).strip().lower()
            for content in contents
            for match in REVIEW_POSITION_RE.finditer(content)
        ]
        if not positions:
            return "unknown"
        return (
            "agree"
            if all(position == "approve" for position in positions)
            else "revise"
        )

    lowered = "\n".join(contents).lower()
    if "[agree]" in lowered or "[submit]" in lowered:
        return "agree"
    if any(
        marker in lowered
        for marker in (
            "[route",
            "fix instruction:",
            "diagnosis:",
            "needs_revision",
            "propose_correction",
        )
    ):
        return "revise"
    # A PER Reviewer response is still a review event when legacy prompt
    # formatting omits the structured action marker.
    return "unknown"


def _reviewed_answer(contents: Sequence[str]) -> str:
    for content in contents:
        marker = re.search(
            r"Candidate\s+(?:Answer|under\s+(?:group\s+)?review)\s*:\s*",
            _sanitize_boxed_spelling(content),
            re.IGNORECASE,
        )
        if not marker:
            continue
        suffix = _sanitize_boxed_spelling(content)[marker.end() :]
        boxed = extract_all_boxed(suffix)
        if boxed:
            return boxed[0]
    return ""


def _reviewer_proposals(contents: Sequence[str]) -> list[str]:
    proposals: list[str] = []
    for content in contents:
        match = PROPOSED_CORRECTION_RE.search(_sanitize_boxed_spelling(content))
        if not match:
            continue
        answer = extract_explicit_answer(match.group(1))
        if answer and normalize_answer(answer) not in {
            normalize_answer(item) for item in proposals
        }:
            proposals.append(answer)
    return proposals


def _anchor_source(content: Any) -> str:
    match = SOURCE_RE.search(_text(content))
    if not match:
        return ""
    return match.group(1).strip()


def _updated_candidate_answer(event: Mapping[str, Any]) -> str:
    content = _sanitize_boxed_spelling(event.get("content"))
    boxed = extract_all_boxed(content)
    if "candidate updated from" in content.lower() and len(boxed) >= 2:
        return _explicit_field_answer(boxed[-1])
    marker = re.search(
        r"Current\s+Candidate\s+For\s+Further\s+Review\s*:\s*",
        content,
        re.IGNORECASE,
    )
    if marker:
        answer = extract_explicit_answer(content[marker.end() :])
        if answer:
            return answer
    return _explicit_field_answer(boxed[-1]) if boxed else ""


@dataclass(frozen=True)
class CandidateRef:
    event_uid: str
    sequence_index: int
    sender: str
    role: str
    stage: str
    answer: str
    content_excerpt: str


@dataclass(frozen=True)
class ReviewMemoryProvenance:
    """Raw provenance for the event immediately after a PER review.

    This records candidate-memory bookkeeping only.  In particular,
    ``review_route`` is not treated as a semantic ``revise`` label because the
    frozen traces also emit it after some text-explicit ``[Agree]`` reviews.
    """

    event_uid: str = ""
    sequence_index: int = -1
    action: str = ""
    source: str = ""
    authority: str = ""
    stage: str = ""
    previous_event_uid: str = ""
    immediate: bool = False
    event_type_valid: bool = False
    system_authored: bool = False
    stage_matches_review: bool = False
    previous_event_uid_matches_review: bool = False
    source_matches_reviewer: bool = False
    authority_matches_reviewer: bool = False
    provenance_valid: bool = False
    link_status: str = "not_applicable"


def _candidate_ref(event: Mapping[str, Any], fallback: int) -> CandidateRef:
    return CandidateRef(
        event_uid=_event_uid(event, fallback),
        sequence_index=_sequence(event, fallback),
        sender=_sender(event),
        role=_text(event.get("role")).strip(),
        stage=_text(event.get("stage")).strip(),
        answer=candidate_answer_from_event(event),
        content_excerpt=_content_excerpt(event.get("content")),
    )


def _empty_candidate_ref() -> CandidateRef:
    return CandidateRef("", -1, "", "", "", "", "")


def _empty_review_memory(status: str = "not_applicable") -> ReviewMemoryProvenance:
    return ReviewMemoryProvenance(link_status=status)


def _candidate_memory_payload(
    event: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    """Return a candidate-memory payload and a fail-closed parse status."""

    raw = event.get("extra_json")
    if isinstance(raw, dict):
        extra = raw
    elif not _text(raw).strip():
        return {}, "missing_extra_json"
    else:
        try:
            parsed = json.loads(_text(raw))
        except json.JSONDecodeError:
            return {}, "malformed_extra_json"
        if not isinstance(parsed, dict):
            return {}, "extra_json_not_object"
        extra = parsed

    payload = extra.get("candidate_memory_event")
    if not isinstance(payload, dict):
        return {}, "missing_candidate_memory_event"
    return payload, ""


def _per_review_memory_provenance(
    events: Sequence[Mapping[str, Any]],
    review_index: int,
) -> ReviewMemoryProvenance:
    """Inspect only the immediate next event without changing review action."""

    if review_index + 1 >= len(events):
        return _empty_review_memory("no_immediate_next_event")

    review = events[review_index]
    memory = events[review_index + 1]
    payload, parse_status = _candidate_memory_payload(memory)
    review_uid = _event_uid(review, review_index)
    memory_previous_uid = _text(memory.get("previous_event_uid")).strip()
    review_stage = _text(review.get("stage")).strip()
    embedded_stage = _text(payload.get("stage")).strip()
    review_sender = _sender(review)
    source = _text(payload.get("source")).strip()
    authority = _text(payload.get("authority")).strip()
    action = _text(payload.get("action")).strip()

    immediate = (
        _sequence(memory, review_index + 1)
        == _sequence(review, review_index) + 1
    )
    event_type_valid = _event_type(memory) == "summary"
    system_authored = (
        _lower(memory.get("sender")) == "system"
        and _role(memory) == "system"
    )
    stage_matches_review = bool(
        review_stage
        and _text(memory.get("stage")).strip()
        == f"{review_stage}_candidate_memory"
        and embedded_stage == review_stage
    )
    previous_event_uid_matches_review = bool(
        review_uid
        and memory_previous_uid
        and memory_previous_uid == review_uid
    )
    source_matches_reviewer = bool(
        review_sender
        and source
        and source.casefold() == review_sender.casefold()
    )
    authority_matches_reviewer = (
        authority == "internal_reviewer_suggestion"
    )

    failures: list[str] = []
    if parse_status:
        failures.append(parse_status)
    if not immediate:
        failures.append("not_immediate_sequence")
    if not event_type_valid:
        failures.append("event_type_not_summary")
    if not system_authored:
        failures.append("not_system_authored")
    if not stage_matches_review:
        failures.append("stage_mismatch")
    if not previous_event_uid_matches_review:
        failures.append("previous_event_uid_mismatch")
    if not source_matches_reviewer:
        failures.append("source_mismatch")
    if not authority_matches_reviewer:
        failures.append("authority_mismatch")
    if not action:
        failures.append("missing_action")
    provenance_valid = not failures

    return ReviewMemoryProvenance(
        event_uid=_event_uid(memory, review_index + 1),
        sequence_index=_sequence(memory, review_index + 1),
        action=action,
        source=source,
        authority=authority,
        stage=embedded_stage,
        previous_event_uid=memory_previous_uid,
        immediate=immediate,
        event_type_valid=event_type_valid,
        system_authored=system_authored,
        stage_matches_review=stage_matches_review,
        previous_event_uid_matches_review=(
            previous_event_uid_matches_review
        ),
        source_matches_reviewer=source_matches_reviewer,
        authority_matches_reviewer=authority_matches_reviewer,
        provenance_valid=provenance_valid,
        link_status="linked" if provenance_valid else "|".join(failures),
    )


def _candidate_alignment(
    before: CandidateRef, reviewed_answer: str, link_method: str
) -> str:
    if not before.event_uid:
        return "missing_actor_before"
    if before.answer and reviewed_answer:
        return (
            "match"
            if normalize_answer(before.answer) == normalize_answer(reviewed_answer)
            else "mismatch"
        )
    if link_method == "immediate_previous_executor":
        return "structural_match"
    return "unverifiable"


def _find_previous_actor(
    protocol: str,
    events: Sequence[Mapping[str, Any]],
    before_index: int,
    *,
    preferred_sender: str = "",
) -> tuple[int, CandidateRef]:
    fallback: tuple[int, CandidateRef] | None = None
    preferred_key = preferred_sender.strip().lower()
    for index in range(before_index - 1, -1, -1):
        event = events[index]
        if submission_gate_kind(event):
            break
        if not is_actor_turn(protocol, event):
            continue
        ref = _candidate_ref(event, index)
        if fallback is None:
            fallback = (index, ref)
        if preferred_key and ref.sender.strip().lower() == preferred_key:
            return index, ref
        if not preferred_key:
            return index, ref
    return fallback or (-1, _empty_candidate_ref())


def _find_broadcast_actor_link(
    events: Sequence[Mapping[str, Any]],
    anchor_index: int,
    *,
    lower_bound: int = 0,
) -> tuple[int, CandidateRef]:
    """Link a system review anchor back to its actor-authored source event.

    A link is accepted only when both the anchor's named source and reviewed
    answer match an actor-authored poll/discussion event. The system anchor is
    therefore corroboration, never the candidate source.
    """

    anchor = events[anchor_index]
    source = _anchor_source(anchor.get("content")).strip().lower()
    reviewed_answer = _reviewed_answer([_text(anchor.get("content"))])
    if not source or not reviewed_answer:
        return -1, _empty_candidate_ref()
    reviewed_key = normalize_answer(reviewed_answer)
    for index in range(anchor_index - 1, lower_bound - 1, -1):
        event = events[index]
        if submission_gate_kind(event) or is_broadcast_review_anchor(event):
            break
        if not is_actor_candidate("broadcast", event):
            continue
        ref = _candidate_ref(event, index)
        if (
            ref.sender.strip().lower() == source
            and normalize_answer(ref.answer) == reviewed_key
        ):
            return index, ref
    return -1, _empty_candidate_ref()


def _trajectory_uid(
    events: Sequence[Mapping[str, Any]], explicit: str | None
) -> str:
    if explicit:
        return explicit
    for event in events:
        value = _text(event.get("trajectory_uid")).strip()
        if value:
            return value
    return ""


def _base_record(
    *,
    protocol: str,
    trajectory_uid: str,
    review_number: int,
    review_events: Sequence[Mapping[str, Any]],
    review_indices: Sequence[int],
    before: CandidateRef,
    after: CandidateRef,
    reviewed_answer: str,
    before_link_method: str,
    after_link_method: str,
    action: str,
    status: str,
    exclusion_reason: str,
    actor_turn_observed: bool,
    boundary_event: Mapping[str, Any] | None,
    boundary_index: int,
    updates: Sequence[Mapping[str, Any]],
    review_memory: ReviewMemoryProvenance,
) -> dict[str, Any]:
    contents = [_text(event.get("content")) for event in review_events]
    alignment = _candidate_alignment(
        before, reviewed_answer, before_link_method
    )
    actor_response_observed = bool(after.event_uid and after.answer)
    actor_pair_provenance_valid = bool(before.event_uid and after.event_uid)
    answers_parse = bool(before.answer and after.answer)
    intervening_updates = len(updates)
    strict_pre_gate_eligible = bool(
        actor_pair_provenance_valid
        and answers_parse
        and alignment in {"match", "structural_match"}
        and intervening_updates == 0
        and not exclusion_reason
    )
    selected_answers = [
        answer for update in updates if (answer := _updated_candidate_answer(update))
    ]
    first_review = review_events[0]
    last_review = review_events[-1]
    boundary = boundary_event or {}
    record = {
        "trajectory_uid": trajectory_uid,
        "protocol": protocol,
        "review_id": f"{trajectory_uid or 'trace'}.symmetric_review.{review_number}",
        "review_sequence_start": _sequence(first_review, review_indices[0]),
        "review_sequence_end": _sequence(last_review, review_indices[-1]),
        "review_event_uids_json": json.dumps(
            [
                _event_uid(event, index)
                for event, index in zip(review_events, review_indices)
            ],
            ensure_ascii=False,
        ),
        "review_stages_json": json.dumps(
            [_text(event.get("stage")) for event in review_events],
            ensure_ascii=False,
        ),
        "reviewer_senders_json": json.dumps(
            [_sender(event) for event in review_events], ensure_ascii=False
        ),
        "review_action": action,
        "review_memory_event_uid": review_memory.event_uid,
        "review_memory_sequence_index": review_memory.sequence_index,
        "review_memory_action": review_memory.action,
        "review_memory_source": review_memory.source,
        "review_memory_authority": review_memory.authority,
        "review_memory_stage": review_memory.stage,
        "review_memory_previous_event_uid": (
            review_memory.previous_event_uid
        ),
        "review_memory_immediate": review_memory.immediate,
        "review_memory_event_type_valid": review_memory.event_type_valid,
        "review_memory_system_authored": review_memory.system_authored,
        "review_memory_stage_matches_review": (
            review_memory.stage_matches_review
        ),
        "review_memory_previous_event_uid_matches_review": (
            review_memory.previous_event_uid_matches_review
        ),
        "review_memory_source_matches_reviewer": (
            review_memory.source_matches_reviewer
        ),
        "review_memory_authority_matches_reviewer": (
            review_memory.authority_matches_reviewer
        ),
        "review_memory_provenance_valid": review_memory.provenance_valid,
        "review_memory_link_status": review_memory.link_status,
        "review_route_observed": bool(
            review_memory.provenance_valid
            and review_memory.action == "review_route"
        ),
        "reviewed_answer": reviewed_answer,
        "review_feedback_excerpt": _content_excerpt("\n".join(contents)),
        "reviewer_proposed_answers_json": json.dumps(
            _reviewer_proposals(contents), ensure_ascii=False
        ),
        "before_event_uid": before.event_uid,
        "before_sequence_index": before.sequence_index,
        "before_sender": before.sender,
        "before_role": before.role,
        "before_stage": before.stage,
        "answer_before": before.answer,
        "before_content_excerpt": before.content_excerpt,
        "after_event_uid": after.event_uid,
        "after_sequence_index": after.sequence_index,
        "after_sender": after.sender,
        "after_role": after.role,
        "after_stage": after.stage,
        "answer_after": after.answer,
        "after_content_excerpt": after.content_excerpt,
        "before_candidate_link_method": before_link_method,
        "after_candidate_link_method": after_link_method,
        "candidate_alignment": alignment,
        "actor_turn_observed": actor_turn_observed,
        "actor_response_observed": actor_response_observed,
        "actor_pair_provenance_valid": actor_pair_provenance_valid,
        "answer_pair_parsed": answers_parse,
        "strict_pre_gate_eligible": strict_pre_gate_eligible,
        "intervening_system_candidate_updates": intervening_updates,
        "system_selected_answers_json": json.dumps(
            selected_answers, ensure_ascii=False
        ),
        "system_selected_answer_used_as_after": False,
        "boundary_event_uid": (
            _event_uid(boundary, boundary_index) if boundary_event else ""
        ),
        "boundary_sequence_index": (
            _sequence(boundary, boundary_index) if boundary_event else -1
        ),
        "boundary_stage": _text(boundary.get("stage")) if boundary_event else "",
        "status": status,
        "exclusion_reason": exclusion_reason,
    }
    _assert_safe_record(record)
    return record


def _assert_safe_record(record: Mapping[str, Any]) -> None:
    """Fail closed if a future edit could leak a non-actor into ``after``."""

    if not record.get("after_event_uid"):
        return
    sender = _lower(record.get("after_sender"))
    stage = _lower(record.get("after_stage"))
    protocol = _lower(record.get("protocol"))
    if sender in SPECIAL_SENDERS:
        raise AssertionError(f"Unsafe answer_after sender: {sender}")
    if protocol == "per" and not stage.startswith("executor"):
        raise AssertionError(f"Unsafe PER answer_after stage: {stage}")
    if protocol == "broadcast" and not (
        stage.startswith("discussion") or stage.startswith("poll")
    ):
        raise AssertionError(f"Unsafe Broadcast answer_after stage: {stage}")
    if record.get("system_selected_answer_used_as_after"):
        raise AssertionError("System-selected candidate cannot be answer_after")


def _extract_per(
    events: Sequence[Mapping[str, Any]], trajectory_uid: str
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    review_number = 0
    for index, event in enumerate(events):
        if not is_review_message("per", event):
            continue
        review_number += 1
        _, before = _find_previous_actor("per", events, index)
        after = _empty_candidate_ref()
        boundary_event: Mapping[str, Any] | None = None
        boundary_index = -1
        status = "no_actor_response_observed"
        exclusion_reason = "no_next_executor_candidate"
        actor_turn_observed = False

        for next_index in range(index + 1, len(events)):
            candidate = events[next_index]
            gate_kind = submission_gate_kind(candidate)
            if gate_kind:
                boundary_event = candidate
                boundary_index = next_index
                status = "no_actor_response_before_submission"
                exclusion_reason = "submission_precedes_next_executor_candidate"
                break
            if is_review_message("per", candidate):
                boundary_event = candidate
                boundary_index = next_index
                status = "no_actor_response_before_next_review"
                exclusion_reason = "additional_review_precedes_executor_response"
                break
            if is_actor_turn("per", candidate):
                after = _candidate_ref(candidate, next_index)
                actor_turn_observed = True
                if after.answer:
                    status = "actor_response_before_gate"
                    exclusion_reason = ""
                else:
                    status = "actor_response_unparsed_before_gate"
                    exclusion_reason = "next_executor_answer_unparsed"
                break

        if not before.event_uid:
            status = "missing_actor_candidate_before"
            exclusion_reason = "no_executor_candidate_since_previous_submission"

        contents = [_text(event.get("content"))]
        reviewed = _reviewed_answer(contents)
        review_memory = _per_review_memory_provenance(events, index)
        records.append(
            _base_record(
                protocol="per",
                trajectory_uid=trajectory_uid,
                review_number=review_number,
                review_events=[event],
                review_indices=[index],
                before=before,
                after=after,
                reviewed_answer=reviewed,
                before_link_method="immediate_previous_executor",
                after_link_method=(
                    "immediate_next_executor" if after.event_uid else ""
                ),
                action=_review_action("per", contents),
                status=status,
                exclusion_reason=exclusion_reason,
                actor_turn_observed=actor_turn_observed,
                boundary_event=boundary_event,
                boundary_index=boundary_index,
                updates=[],
                review_memory=review_memory,
            )
        )
    return records


def _extract_broadcast(
    events: Sequence[Mapping[str, Any]], trajectory_uid: str
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    review_number = 0
    index = 0
    while index < len(events):
        anchor = events[index]
        if not is_broadcast_review_anchor(anchor):
            index += 1
            continue

        review_indices: list[int] = []
        updates: list[Mapping[str, Any]] = []
        boundary_event: Mapping[str, Any] | None = None
        boundary_index = -1
        cursor = index + 1
        while cursor < len(events):
            candidate = events[cursor]
            if submission_gate_kind(candidate):
                boundary_event = candidate
                boundary_index = cursor
                break
            if is_broadcast_review_anchor(candidate):
                boundary_event = candidate
                boundary_index = cursor
                break
            if is_review_message("broadcast", candidate):
                review_indices.append(cursor)
            if system_candidate_update(candidate):
                # One logical protocol update is represented by a summary and a
                # public system message with the same stage. Count it once.
                if not any(
                    _stage(existing) == _stage(candidate) for existing in updates
                ):
                    updates.append(candidate)
            cursor += 1

        if not review_indices:
            index = max(cursor, index + 1)
            continue

        review_number += 1
        review_events = [events[item] for item in review_indices]
        _, before = _find_broadcast_actor_link(events, index)
        after = _empty_candidate_ref()
        after_link_method = ""
        actor_turn_observed = False
        status = "no_actor_response_observed"
        exclusion_reason = "no_linked_later_actor_candidate"

        response_interval_start = review_indices[-1] + 1
        response_interval_end = (
            boundary_index if boundary_index >= 0 else len(events)
        )
        actor_turn_observed = any(
            is_actor_turn("broadcast", events[item])
            for item in range(response_interval_start, response_interval_end)
        )

        if boundary_event is not None and is_broadcast_review_anchor(
            boundary_event
        ):
            _, after = _find_broadcast_actor_link(
                events,
                boundary_index,
                lower_bound=response_interval_start,
            )
            if after.event_uid:
                after_link_method = "next_anchor_source_answer_match_actor_event"
                if updates:
                    status = "actor_response_after_protocol_selection"
                    exclusion_reason = (
                        "system_selected_reviewer_proposal_before_actor_response"
                    )
                else:
                    status = "actor_response_before_gate"
                    exclusion_reason = ""
            elif actor_turn_observed:
                status = "actor_response_unlinked_to_next_review"
                exclusion_reason = (
                    "next_review_anchor_not_linked_to_actor_source_answer"
                )
            else:
                status = "no_actor_response_before_next_review"
                exclusion_reason = "next_group_review_precedes_actor_response"
        elif boundary_event is not None and submission_gate_kind(boundary_event):
            status = "no_actor_response_before_submission"
            exclusion_reason = "consensus_or_submission_precedes_actor_response"
        elif actor_turn_observed:
            status = "actor_response_unlinked_no_next_review_anchor"
            exclusion_reason = "actor_turn_has_no_subsequent_review_anchor"

        if not before.event_uid:
            status = "missing_actor_candidate_before"
            exclusion_reason = (
                "review_anchor_not_linked_to_actor_source_answer"
            )

        anchor_reviewed = _reviewed_answer([_text(anchor.get("content"))])
        review_marked = _reviewed_answer(
            [_text(event.get("content")) for event in review_events]
        )
        reviewed = anchor_reviewed or review_marked
        records.append(
            _base_record(
                protocol="broadcast",
                trajectory_uid=trajectory_uid,
                review_number=review_number,
                review_events=review_events,
                review_indices=review_indices,
                before=before,
                after=after,
                reviewed_answer=reviewed,
                before_link_method=(
                    "anchor_source_answer_match_actor_event"
                    if before.event_uid
                    else ""
                ),
                after_link_method=after_link_method,
                action=_review_action(
                    "broadcast",
                    [_text(event.get("content")) for event in review_events],
                ),
                status=status,
                exclusion_reason=exclusion_reason,
                actor_turn_observed=actor_turn_observed,
                boundary_event=boundary_event,
                boundary_index=boundary_index,
                updates=updates,
                review_memory=_empty_review_memory(
                    "not_applicable_broadcast_group_review"
                ),
            )
        )
        index = max(cursor, index + 1)
    return records


def extract_symmetric_transitions(
    protocol: str,
    events: Iterable[Mapping[str, Any]],
    *,
    trajectory_uid: str | None = None,
) -> list[dict[str, Any]]:
    """Extract symmetric transition records from one ordered trajectory."""

    protocol = _lower(protocol)
    if protocol not in SUPPORTED_PROTOCOLS:
        raise ValueError(f"Unsupported protocol: {protocol!r}")
    ordered = [
        event
        for _, event in sorted(
            enumerate(events),
            key=lambda pair: (_sequence(pair[1], pair[0]), pair[0]),
        )
    ]
    trace_uid = _trajectory_uid(ordered, trajectory_uid)
    if protocol == "per":
        return _extract_per(ordered, trace_uid)
    return _extract_broadcast(ordered, trace_uid)


def write_records_csv(records: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        raise ValueError("Refusing to write an empty transition CSV")
    fields = list(records[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            events.append(value)
    return events


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=sorted(SUPPORTED_PROTOCOLS), required=True)
    parser.add_argument("--events-jsonl", type=Path, required=True)
    parser.add_argument("--trajectory-uid", default="")
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    records = extract_symmetric_transitions(
        args.protocol,
        _read_jsonl(args.events_jsonl),
        trajectory_uid=args.trajectory_uid or None,
    )
    write_records_csv(records, args.output_csv)
    print(f"Wrote {args.output_csv} ({len(records)} review opportunities)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
