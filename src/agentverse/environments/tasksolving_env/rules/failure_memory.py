from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from agentverse.evaluation.common import extract_boxed


AUTHORITY_GUIDE = """Feedback Authority Guide:
- Peer or Reviewer revisions are internal suggestions. They may be useful or wrong; verify them against the task evidence and answer requirements.
- A system/protocol candidate update means the protocol selected the current candidate for further review or submission. It is not a correctness signal.
- Evaluator feedback is the external verifier signal based on evaluator-only reference material. Treat it as higher priority than peer or Reviewer suggestions, but remember it is a non-leaking hint, not the answer."""


def with_authority_guide(text: str) -> str:
    cleaned = str(text or "").strip()
    if AUTHORITY_GUIDE in cleaned:
        return cleaned
    return "\n\n".join(part for part in [AUTHORITY_GUIDE, cleaned] if part).strip()


def _compact_text(text: str, max_chars: int = 320) -> str:
    cleaned = str(text or "").strip()
    if not cleaned:
        return ""
    cleaned = re.sub(
        r"Final Answer For Evaluator:\s*.*?(?=\n\[(?:Route:Planner|Route:Executor|Agree)\]|\Z)",
        "",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    )
    cleaned = re.sub(
        r"\[(?:Route:Planner|Route:Executor|Route:Evaluator|Route:Evaluate|Agree)\]",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3].rstrip() + "..."


@dataclass
class FailedAttemptRecord:
    attempt_id: int
    submitted_answer: str
    evaluator_signal: str
    evaluator_hint: str
    repair_summary: str = ""
    source_stage: str = "evaluation"


class FailedAttemptMemory:
    """Per-question structured memory for failed evaluator submissions."""

    def __init__(self, limit: int = 5):
        self.limit = max(1, int(limit or 5))
        self.records: List[FailedAttemptRecord] = []

    def reset(self) -> None:
        self.records = []

    def add(
        self,
        *,
        submitted_answer: str,
        evaluator_hint: str,
        evaluator_signal: str = "FAIL",
        repair_summary: str = "",
        source_stage: str = "evaluation",
    ) -> FailedAttemptRecord:
        record = FailedAttemptRecord(
            attempt_id=len(self.records) + 1,
            submitted_answer=_compact_text(submitted_answer, max_chars=180)
            or "[no extracted answer]",
            evaluator_signal=str(evaluator_signal or "FAIL").strip() or "FAIL",
            evaluator_hint=_compact_text(evaluator_hint, max_chars=240)
            or "[no evaluator hint]",
            repair_summary=_compact_text(repair_summary, max_chars=220),
            source_stage=str(source_stage or "evaluation").strip() or "evaluation",
        )
        self.records.append(record)
        if len(self.records) > self.limit:
            self.records = self.records[-self.limit :]
            for idx, kept in enumerate(self.records, start=1):
                kept.attempt_id = idx
        return record

    def update_latest_repair_summary(self, repair_summary: str) -> None:
        if not self.records:
            return
        summary = _compact_text(repair_summary, max_chars=220)
        if summary:
            self.records[-1].repair_summary = summary

    def render(self) -> str:
        if not self.records:
            return (
                "Failed Attempt Memory For This Question:\n"
                "[No failed evaluator attempts yet.]"
            )

        lines = [
            "Failed Attempt Memory For This Question:",
            "Use this concise ledger to avoid repeating rejected answers unless the mismatch has been fixed.",
            "Evaluator feedback is the external verifier signal; it is not the answer.",
        ]
        for record in self.records:
            details = [
                f"signal={record.evaluator_signal}",
                f"hint={record.evaluator_hint}",
            ]
            if record.repair_summary:
                details.append(f"fix={record.repair_summary}")
            lines.extend(
                [
                    "",
                    (
                        f"- A{record.attempt_id} ({record.source_stage}): "
                        f"answer={record.submitted_answer}"
                    ),
                    "  " + " | ".join(details),
                    "  Do not resubmit this answer unless you can explain what changed.",
                ]
            )
        return "\n".join(lines).strip()

    def render_for_evaluator_hint(self, max_records: int = 3) -> str:
        if not self.records:
            return "[No previous failed evaluator attempts for this question.]"

        lines = ["Previous answers summary for evaluator hint generation:"]
        for record in self.records[-max(1, int(max_records or 1)) :]:
            lines.extend(
                [
                    f"- Attempt {record.attempt_id}: answer={record.submitted_answer}",
                    f"  signal={record.evaluator_signal}",
                    f"  hint={_compact_text(record.evaluator_hint, max_chars=160)}",
                ]
            )
            if record.repair_summary:
                lines.append(
                    f"  repair={_compact_text(record.repair_summary, max_chars=160)}"
                )
        return "\n".join(lines).strip()


def _normalize_candidate_answer(candidate_answer: str, fallback_text: str = "") -> str:
    boxed = extract_boxed(candidate_answer or "") or extract_boxed(fallback_text or "")
    if boxed:
        return f"\\boxed{{{boxed}}}"

    cleaned = _compact_text(candidate_answer, max_chars=240)
    if cleaned:
        return cleaned
    return ""


@dataclass
class CandidateRevisionRecord:
    event_id: int
    stage: str
    source: str
    action: str
    candidate_answer: str
    rationale_or_review: str = ""
    parent_event_id: Optional[int] = None
    evaluator_signal: str = ""

    def authority_label(self) -> str:
        action = str(self.action or "").strip().lower()
        source = str(self.source or "").strip().lower()
        stage = str(self.stage or "").strip().lower()
        if action == "evaluate" or "evaluator" in source or "evaluation" in stage:
            return "external_evaluator_signal"
        if source == "system" and action == "select_revision":
            return "protocol_candidate_selection_not_correctness"
        if source == "system":
            return "system_protocol_event_not_correctness"
        if "reviewer" in source or action in {"review_submit", "repair_instruction"}:
            return "internal_reviewer_suggestion"
        if action in {"revise", "approve", "reject", "propose_final", "select_for_review"}:
            return "peer_or_agent_suggestion"
        if action in {"propose"}:
            return "internal_candidate_proposal"
        return "internal_observation"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "stage": self.stage,
            "source": self.source,
            "action": self.action,
            "authority": self.authority_label(),
            "candidate_answer": self.candidate_answer,
            "rationale_or_review": self.rationale_or_review,
            "parent_event_id": self.parent_event_id,
            "evaluator_signal": self.evaluator_signal,
        }


class CandidateRevisionMemory:
    """Per-question structured memory for candidate answers and revisions."""

    def __init__(self, limit: int = 8):
        self.limit = max(1, int(limit or 8))
        self.records: List[CandidateRevisionRecord] = []

    def reset(self) -> None:
        self.records = []

    def add(
        self,
        *,
        stage: str,
        source: str,
        action: str,
        candidate_answer: str = "",
        rationale_or_review: str = "",
        parent_event_id: Optional[int] = None,
        evaluator_signal: str = "",
    ) -> Optional[CandidateRevisionRecord]:
        normalized_candidate = _normalize_candidate_answer(
            candidate_answer,
            fallback_text=rationale_or_review,
        )
        compact_rationale = _compact_text(rationale_or_review, max_chars=220)

        if not normalized_candidate and not compact_rationale:
            return None

        record = CandidateRevisionRecord(
            event_id=len(self.records) + 1,
            stage=str(stage or "unknown").strip() or "unknown",
            source=str(source or "unknown").strip() or "unknown",
            action=str(action or "observe").strip() or "observe",
            candidate_answer=normalized_candidate or "[no candidate answer]",
            rationale_or_review=compact_rationale,
            parent_event_id=parent_event_id,
            evaluator_signal=str(evaluator_signal or "").strip(),
        )
        self.records.append(record)
        if len(self.records) > self.limit:
            self.records = self.records[-self.limit :]
            old_to_new: Dict[int, int] = {}
            for idx, kept in enumerate(self.records, start=1):
                old_to_new[kept.event_id] = idx
                kept.event_id = idx
            for kept in self.records:
                if kept.parent_event_id in old_to_new:
                    kept.parent_event_id = old_to_new[kept.parent_event_id]
                elif kept.parent_event_id is not None:
                    kept.parent_event_id = None
        return record

    def render(self) -> str:
        if not self.records:
            return (
                "Candidate And Revision Memory For This Question:\n"
                "[No candidate or revision events yet.]"
            )

        lines = [
            "Candidate And Revision Memory For This Question:",
            "Use this concise ledger to track proposed answers, reviews, revisions, and submitted candidates.",
            "Peer and Reviewer revisions are suggestions to verify; evaluator feedback is higher-priority but still non-leaking.",
        ]
        for record in self.records:
            header = f"- C{record.event_id} {record.action} by {record.source}"
            if record.parent_event_id is not None:
                header += f" revising C{record.parent_event_id}"
            lines.extend(["", header, f"  candidate={record.candidate_answer}"])
            if record.evaluator_signal:
                lines.append(f"  evaluator_signal={record.evaluator_signal}")
            if record.rationale_or_review:
                lines.append(
                    f"  note={_compact_text(record.rationale_or_review, max_chars=180)}"
                )
        return "\n".join(lines).strip()

    def render_for_evaluator_hint(self, max_records: int = 5) -> str:
        if not self.records:
            return "[No previous candidate/revision trajectory for this question.]"

        lines = ["Previous reasoning summary for evaluator hint generation:"]
        for record in self.records[-max(1, int(max_records or 1)) :]:
            summary = (
                f"- C{record.event_id}: {record.action} by {record.source}; "
                f"candidate={record.candidate_answer}"
            )
            lines.append(summary)
            if record.rationale_or_review:
                lines.append(
                    f"  note={_compact_text(record.rationale_or_review, max_chars=160)}"
                )
        return "\n".join(lines).strip()
