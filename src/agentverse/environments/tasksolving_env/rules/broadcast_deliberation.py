from __future__ import annotations

import asyncio
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

from agentverse.evaluation.common import extract_boxed
from agentverse.environments.tasksolving_env.rules.critique_intervention_policy import (
    CritiqueInterventionDecision,
    RuntimeCritiqueInterventionPolicy,
)
from agentverse.environments.tasksolving_env.rules.evaluator import (
    BaseEvaluator,
    evaluator_registry,
)
from agentverse.environments.tasksolving_env.rules.failure_memory import (
    CandidateRevisionMemory,
    FailedAttemptMemory,
    with_authority_guide,
)
from agentverse.message import Message
from agentverse.utils import AGENT_TYPES

def _normalize_boxed(answer: str) -> str:
    answer = str(answer or "").strip()
    if not answer:
        return ""
    boxed = extract_boxed(answer)
    if boxed:
        return f"\\boxed{{{boxed}}}"
    return f"\\boxed{{{answer}}}"


def _submitted_final_answer_for_trace(answer: str) -> str:
    boxed = extract_boxed(answer or "")
    return f"\\boxed{{{boxed}}}" if boxed else "[No extracted final answer]"


def _coerce_evaluator_score(score: Any) -> bool:
    if isinstance(score, bool):
        return score
    if isinstance(score, int):
        return score == 1 or score >= 8
    if isinstance(score, (list, tuple)):
        return bool(score) and all(_coerce_evaluator_score(item) for item in score)
    return bool(score)


_INVALID_PUBLIC_TEXT = {
    "",
    "[empty]",
    "[none]",
    "none",
    "n/a",
    "na",
    "no comment",
    "no comments",
    "no feedback",
    "no substantive feedback",
    "nothing to add",
    "skip",
    "pass",
}


def _build_broadcast_augmented_message(
    content: str,
    decision: "CritiqueInterventionDecision",
    augment_scores: list,
) -> str:
    """Prepend predicted-score metadata to a public broadcast message.

    Message-level augmentation for the "wrong but useful" study: every public
    message is passed through unchanged, plus estimated correctness / trajectory
    value so peers can weigh how reliable/useful the (possibly partial,
    speculative, or locally-wrong) message is. Never filters.
    """
    parts = []
    show_traj = ("trajectory" in augment_scores) or ("integration" in augment_scores)
    if "correctness" in augment_scores:
        parts.append(
            f"Predicted correctness: {decision.credibility_prob:.2f} "
            "(estimated probability that this message is locally correct or directionally valid)"
        )
    if show_traj:
        parts.append(
            f"Predicted trajectory value: {decision.repair_prob:.2f} "
            "(estimated probability that incorporating this message improves the group's "
            "downstream reasoning trajectory or final answer, regardless of local correctness)"
        )
    if not parts:
        return str(content or "").strip()
    annotation = "\n".join(parts)
    return (
        "[The following metadata is an ESTIMATED, fallible property of the peer message "
        "below.\n"
        f"{annotation}\n"
        "These estimates may be imperfect. Use them as additional evidence to weigh the "
        "message; do not blindly obey them. You still decide.]\n\n"
        "Peer message:\n"
        f"{str(content or '').strip()}"
    ).strip()


def _is_meaningful_public_text(text: str) -> bool:
    raw = str(text or "").strip()
    if not raw:
        return False
    normalized = re.sub(r"\s+", " ", raw).strip().lower()
    if normalized in _INVALID_PUBLIC_TEXT:
        return False
    if normalized in {"[]", "{}", '""', "''"}:
        return False
    # A boxed answer alone is a valid public contribution.
    if extract_boxed(raw):
        return True
    # Very short acknowledgements are usually parser fallbacks, not useful speech.
    if len(normalized) < 4:
        return False
    return True


def _format_evaluator_feedback(mode: str, passed: bool, advice: str) -> str:
    normalized_mode = str(mode or "unknown").strip() or "unknown"
    signal = "PASS" if passed else "FAIL"
    clean_advice = str(advice or "").strip()
    lines = [
        f"Evaluation signal: {signal}",
        f"Verifier mode: {normalized_mode}",
        f"Score: {passed}",
    ]
    if clean_advice:
        lines.append(f"Hint: {clean_advice}")
    else:
        lines.append(f"Hint: Verifier: {signal} ({normalized_mode}).")
    return "\n".join(lines)


def _format_evaluator_verdict_only(mode: str, passed: bool) -> str:
    normalized_mode = str(mode or "unknown").strip() or "unknown"
    signal = "PASS" if passed else "FAIL"
    return "\n".join(
        [
            f"Evaluation signal: {signal}",
            f"Verifier mode: {normalized_mode}",
            f"Score: {passed}",
        ]
    )


def _format_evaluator_hint_only(advice: str) -> str:
    clean_advice = str(advice or "").strip()
    return f"Evaluation hint: {clean_advice}" if clean_advice else "Evaluation hint: [none]"


def _format_group_feedback(
    mode: str,
    passed: bool,
    advice: str,
    feedback_mode: str,
) -> str:
    normalized_mode = str(mode or "unknown").strip() or "unknown"
    signal = "PASS" if passed else "FAIL"
    normalized_feedback_mode = str(feedback_mode or "hint").strip().lower() or "hint"

    if normalized_feedback_mode == "hint":
        return _format_evaluator_feedback(mode, passed, advice)

    if passed:
        plain_advice = f"Verifier: PASS ({normalized_mode})."
    elif normalized_mode in {"exact", "numeric-verifier", "omni-rule"}:
        plain_advice = (
            f"Verifier: FAIL ({normalized_mode}). "
            "Re-check the reasoning and the final boxed answer."
        )
    else:
        plain_advice = (
            f"Verifier: FAIL ({normalized_mode}). "
            "Re-check the reasoning and the final answer."
        )

    lines = [
        f"Evaluation signal: {signal}",
        f"Verifier mode: {normalized_mode}",
        f"Score: {passed}",
        f"Advice: {plain_advice}",
    ]
    return "\n".join(lines)


def _append_log(
    logs: List[Dict[str, Any]],
    event: Dict[str, Any],
    live_log_sink: Any = None,
) -> None:
    logs.append(event)
    if live_log_sink is not None:
        try:
            live_log_sink(event)
        except Exception:
            pass


class BroadcastDeliberationRule:
    evaluator: BaseEvaluator

    def __init__(self, evaluator_config: Dict[str, Any] | None = None, **kwargs):
        self.discussion_rounds: int = int(kwargs.pop("discussion_rounds", 4))
        self.approval_rounds: int = int(kwargs.pop("approval_rounds", 2))
        self.speak_threshold: int = int(kwargs.pop("speak_threshold", 60))
        self.memory_mode: str = str(kwargs.pop("memory_mode", "summary_plus_recent"))
        self.recent_history_window: int = int(kwargs.pop("recent_history_window", 4))
        self.approval_policy: str = str(kwargs.pop("approval_policy", "unanimous")).strip().lower()
        self.candidate_selection_policy: str = str(
            kwargs.pop("candidate_selection_policy", "majority_revision")
        ).strip().lower()
        self.discussion_instruction: str = str(
            kwargs.pop(
                "discussion_instruction",
                (
                    "Contribute one focused message to the peer math discussion. "
                    "Avoid repeating points already made. "
                    "If you believe the group is ready to consider a concrete final answer, "
                    "remember that the answer may be numeric, symbolic, formulaic, set-valued, or conditional, and "
                    "include a short section exactly in this form:\n"
                    "Candidate Answer:\n"
                    "\\boxed{...}"
                ),
            )
        )
        feedback_mode = str(
            kwargs.pop("reviewer_feedback_mode", kwargs.pop("evaluator_feedback_mode", "hint"))
        ).strip().lower()
        if feedback_mode not in {"plain", "hint"}:
            feedback_mode = "hint"
        self.reviewer_feedback_mode: str = feedback_mode
        self.failed_attempt_memory_limit: int = int(
            kwargs.pop("failed_attempt_memory_limit", 3)
        )
        self.failed_attempt_memory = FailedAttemptMemory(self.failed_attempt_memory_limit)
        self.candidate_revision_memory_limit: int = int(
            kwargs.pop("candidate_revision_memory_limit", 5)
        )
        self.candidate_revision_memory = CandidateRevisionMemory(
            self.candidate_revision_memory_limit
        )

        # Message-level augmentation policy (mode="augment"): attach predicted
        # correctness + trajectory value to every public broadcast message.
        # Never filters/reroutes; the message always passes through.
        self.critique_intervention_policy = RuntimeCritiqueInterventionPolicy.from_config(
            kwargs.pop("critique_intervention_policy", None)
        )

        evaluator_config = dict(evaluator_config or {"type": "omni-verifier"})
        evaluator_type = str(evaluator_config.pop("type", "omni-verifier")).strip().lower()
        self.evaluator_type = evaluator_type
        self.evaluator = evaluator_registry.build(evaluator_type, **evaluator_config)

    def _augment_public_message(self, message: "Message | None") -> "Message | None":
        """Prepend predicted correctness / trajectory-value metadata to a public
        broadcast message, if augment mode is enabled. Returns a NEW Message with
        annotated content; the original protocol behavior is otherwise unchanged.
        """
        policy = getattr(self, "critique_intervention_policy", None)
        if message is None or policy is None or getattr(policy, "mode", "disabled") != "augment":
            return message
        content = getattr(message, "content", "") or ""
        if not content.strip():
            return message
        decision = policy.decide(
            route="executor",  # message-level: treat as an actionable peer signal
            review_text=content,
            candidate_text="",
            reviewer_role=getattr(message, "sender", "Agent"),
            review_turn=0,
            after_evaluator_fail=False,
            stage="broadcast_public_message",
        )
        if not getattr(decision, "enabled", False) or decision.action != "augment":
            return message
        annotated = _build_broadcast_augmented_message(
            content, decision, getattr(policy, "augment_scores", [])
        )
        new_msg = Message(content=annotated, sender=getattr(message, "sender", ""))
        return new_msg

    def reset(self) -> None:
        self.evaluator.reset()
        self.failed_attempt_memory.reset()
        self.candidate_revision_memory.reset()

    def _failed_attempt_memory_size(self) -> int:
        return len(self.failed_attempt_memory.records)

    def _candidate_revision_memory_size(self) -> int:
        return len(self.candidate_revision_memory.records)

    def _deliberator_debug_kwargs(self, stage: str) -> Dict[str, Any]:
        return {
            "debug_prompt_stage": stage,
            "debug_failed_attempt_memory_size": self._failed_attempt_memory_size(),
            "debug_candidate_revision_memory_size": self._candidate_revision_memory_size(),
            "debug_recent_history_window": self.recent_history_window,
            "debug_memory_mode": self.memory_mode,
        }

    def _set_evaluator_debug_context(self, agent: Any, stage: str) -> None:
        if agent is None:
            return
        setattr(agent, "current_debug_prompt_stage", stage)
        setattr(
            agent,
            "current_failed_attempt_memory_size",
            self._failed_attempt_memory_size(),
        )
        setattr(
            agent,
            "current_candidate_revision_memory_size",
            self._candidate_revision_memory_size(),
        )

    def _question_memory_context(self) -> str:
        parts: List[str] = []
        if self.candidate_revision_memory.records:
            parts.append(self.candidate_revision_memory.render())
        if self.failed_attempt_memory.records:
            parts.append(self.failed_attempt_memory.render())
        return "\n\n".join(parts).strip()

    def _advice_with_question_memory(self, advice: str) -> str:
        clean_advice = str(advice or "").strip()
        memory_context = self._question_memory_context()
        if not memory_context:
            return with_authority_guide(clean_advice)
        if memory_context in clean_advice:
            return with_authority_guide(clean_advice)
        return with_authority_guide(
            "\n\n".join(part for part in [clean_advice, memory_context] if part).strip()
        )

    def _record_candidate_event(
        self,
        logs: List[Dict[str, Any]],
        *,
        round_id: int,
        stage: str,
        source: str,
        action: str,
        candidate_text: str = "",
        rationale_or_review: str = "",
        evaluator_signal: str = "",
        live_log_sink: Any = None,
    ) -> None:
        record = self.candidate_revision_memory.add(
            stage=stage,
            source=source,
            action=action,
            candidate_answer=candidate_text,
            rationale_or_review=rationale_or_review,
            evaluator_signal=evaluator_signal,
        )
        if record is None:
            return
        _append_log(
            logs,
            {
                "type": "summary",
                "round": round_id,
                "stage": f"{stage}_candidate_memory",
                "sender": "system",
                "content": (
                    f"C{record.event_id} {record.action} "
                    f"{record.candidate_answer}"
                ),
                "candidate_memory_event": record.to_dict(),
                "trace_visible": False,
            },
            live_log_sink,
        )

    def _append_live_log(self, live_log_path: str, text: str) -> None:
        path = str(live_log_path or "").strip()
        if not path:
            return
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(text.rstrip() + "\n")

    def _public_memory_snapshot(self, agent: Any) -> str:
        if hasattr(agent, "_public_context"):
            try:
                snapshot = agent._public_context(self.memory_mode, self.recent_history_window)
                return str(snapshot or "").strip() or "[empty]"
            except Exception:
                return "[unavailable]"
        return "[unavailable]"

    def _private_memory_snapshot(self, agent: Any) -> str:
        if hasattr(agent, "_private_context"):
            try:
                snapshot = agent._private_context()
                return str(snapshot or "").strip() or "[empty]"
            except Exception:
                return "[unavailable]"
        return "[unavailable]"

    def _own_speak_history_snapshot(self, agent: Any) -> str:
        if hasattr(agent, "_own_speak_history_context"):
            try:
                snapshot = agent._own_speak_history_context()
                return str(snapshot or "").strip() or "[empty]"
            except Exception:
                return "[unavailable]"
        return "[unavailable]"

    def _record_agent_speaking_turn(self, agent: Any, turn_label: str, content: str) -> None:
        if hasattr(agent, "record_speaking_turn"):
            try:
                agent.record_speaking_turn(turn_label, content)
            except Exception:
                pass

    def _review_position_label(self, approved: bool, revised_answer: str) -> str:
        if approved:
            return "approve"
        if str(revised_answer or "").strip():
            return "propose_correction"
        return "needs_revision"

    def _log_live_outer_round_start(
        self,
        *,
        live_log_path: str,
        round_id: int,
        task_description: str,
        advice: str,
        current_candidate: str,
    ) -> None:
        self._append_live_log(
            live_log_path,
            "\n".join(
                [
                    "",
                    "=" * 80,
                    f"OUTER ROUND {round_id} START",
                    f"Current Candidate Answer: {current_candidate or '[None]'}",
                    f"Current Advice: {advice or '[None]'}",
                    "Problem:",
                    task_description or "[empty]",
                    "=" * 80,
                ]
            ),
        )

    def _log_live_turn_state(
        self,
        *,
        live_log_path: str,
        round_id: int,
        turn_idx: int,
        current_candidate: str,
        advice: str,
        deliberators: List[Any],
    ) -> None:
        lines = [
            "",
            "-" * 80,
            f"OUTER ROUND {round_id} DISCUSSION TURN {turn_idx} PRE-POLL",
            f"Current Candidate Answer: {current_candidate or '[None]'}",
            f"Current Advice: {advice or '[None]'}",
        ]
        for agent in deliberators:
            lines.extend(
                [
                    "",
                    f"Agent: {agent.name}",
                    "Public Memory Snapshot:",
                    self._public_memory_snapshot(agent),
                    "Private Deferred Notes:",
                    self._private_memory_snapshot(agent),
                    "Own Speaking History:",
                    self._own_speak_history_snapshot(agent),
                ]
            )
        self._append_live_log(live_log_path, "\n".join(lines))

    def _log_live_poll_results(
        self,
        *,
        live_log_path: str,
        round_id: int,
        turn_idx: int,
        deliberators: List[Any],
        polls: List[Dict[str, Any]],
        selected_speaker: str,
        selected_score: int,
    ) -> None:
        lines = [
            "",
            f"OUTER ROUND {round_id} DISCUSSION TURN {turn_idx} POLL RESULTS",
            f"Selected Speaker: {selected_speaker}",
            f"Selected Speaker Confidence: {selected_score}",
        ]
        for agent, poll in zip(deliberators, polls):
            lines.extend(
                [
                    "",
                    f"Agent: {agent.name}",
                    f"Confidence Score: {int(poll.get('score', 1))}",
                    f"Parse Status: {'parsed_json' if poll.get('parse_succeeded') else 'fallback'}",
                    f"Reason To Speak: {str(poll.get('reason', '') or '').strip() or '[empty]'}",
                    "Intended Contribution:",
                    str(poll.get("intent", "") or "").strip() or "[empty]",
                    f"Candidate Answer: {str(poll.get('candidate_answer', '') or '').strip() or '[None]'}",
                ]
            )
            if not poll.get("parse_succeeded"):
                lines.extend(
                    [
                        "Raw Poll Response:",
                        str(poll.get("raw_text", "") or "").strip() or "[empty]",
                    ]
                )
        self._append_live_log(live_log_path, "\n".join(lines))

    def _log_live_speaker_message(
        self,
        *,
        live_log_path: str,
        round_id: int,
        turn_idx: int,
        speaker_name: str,
        content: str,
    ) -> None:
        self._append_live_log(
            live_log_path,
            "\n".join(
                [
                    "",
                    f"OUTER ROUND {round_id} DISCUSSION TURN {turn_idx} SPOKEN MESSAGE",
                    f"Speaker: {speaker_name}",
                    "Message:",
                    content or "[empty]",
                ]
            ),
        )

    def _log_live_skipped_speaker_message(
        self,
        *,
        live_log_path: str,
        round_id: int,
        turn_idx: int,
        speaker_name: str,
    ) -> None:
        self._append_live_log(
            live_log_path,
            "\n".join(
                [
                    "",
                    f"OUTER ROUND {round_id} DISCUSSION TURN {turn_idx} SPOKEN MESSAGE SKIPPED",
                    f"Speaker: {speaker_name}",
                    "Reason: empty or non-substantive public message was not added to shared memory.",
                ]
            ),
        )

    def _log_live_deferred_notes(
        self,
        *,
        live_log_path: str,
        round_id: int,
        turn_idx: int,
        deferred_notes: List[Tuple[str, str]],
    ) -> None:
        if not deferred_notes:
            return
        lines = [
            "",
            f"OUTER ROUND {round_id} DISCUSSION TURN {turn_idx} DEFERRED NOTES",
        ]
        for agent_name, note in deferred_notes:
            lines.extend(
                [
                    "",
                    f"Agent: {agent_name}",
                    note,
                ]
            )
        self._append_live_log(live_log_path, "\n".join(lines))

    def _log_live_candidate_review(
        self,
        *,
        live_log_path: str,
        round_id: int,
        turn_idx: int,
        candidate_answer: str,
        candidate_source: str,
    ) -> None:
        self._append_live_log(
            live_log_path,
            "\n".join(
                [
                    "",
                    f"OUTER ROUND {round_id} DISCUSSION TURN {turn_idx} CANDIDATE REVIEW START",
                    f"Candidate Source: {candidate_source or 'unknown'}",
                    f"Candidate Answer: {candidate_answer or '[None]'}",
                ]
            ),
        )

    def _log_live_approval_round(
        self,
        *,
        live_log_path: str,
        round_id: int,
        approval_round: int,
        current_candidate: str,
        deliberators: List[Any],
        responses: List[Dict[str, Any]],
    ) -> None:
        lines = [
            "",
            f"OUTER ROUND {round_id} APPROVAL ROUND {approval_round}",
            f"Candidate Under Review: {current_candidate or '[None]'}",
        ]
        for agent, response in zip(deliberators, responses):
            revised_answer = str(response.get("revised_answer", "") or "").strip()
            feedback = str(response.get("feedback", "") or "").strip()
            review_position = self._review_position_label(
                bool(response.get("approve")),
                revised_answer,
            )
            valid_review = (
                bool(response.get("approve"))
                or bool(_normalize_boxed(revised_answer))
                or _is_meaningful_public_text(feedback)
            )
            if not valid_review:
                lines.extend(
                    [
                        "",
                        f"Agent: {agent.name}",
                        "Review skipped: empty or non-substantive approval review was not added to shared memory.",
                    ]
                )
                continue
            lines.extend(
                [
                    "",
                    f"Agent: {agent.name}",
                    f"Review Position: {review_position}",
                    f"Parse Status: {'parsed_json' if response.get('parse_succeeded') else 'fallback'}",
                    "Peer Review:",
                    feedback or "[no substantive feedback]",
                    f"Proposed Correction (peer suggestion): {revised_answer or '[None]'}",
                    "Public Memory Snapshot:",
                    self._public_memory_snapshot(agent),
                    "Private Deferred Notes:",
                    self._private_memory_snapshot(agent),
                    "Own Speaking History:",
                    self._own_speak_history_snapshot(agent),
                ]
            )
            if not response.get("parse_succeeded"):
                lines.extend(
                    [
                        "Raw Approval Response:",
                        str(response.get("raw_text", "") or "").strip() or "[empty]",
                    ]
                )
        self._append_live_log(live_log_path, "\n".join(lines))

    def _log_live_final_proposals(
        self,
        *,
        live_log_path: str,
        round_id: int,
        deliberators: List[Any],
        proposals: List[Dict[str, Any]],
        selected_answer: str,
        selected_source: str,
    ) -> None:
        lines = [
            "",
            f"OUTER ROUND {round_id} FINAL PROPOSALS",
            f"Selected Final Proposal Source: {selected_source or '[None]'}",
            f"Selected Final Proposal Answer: {selected_answer or '[None]'}",
        ]
        for agent, proposal in zip(deliberators, proposals):
            answer = str(proposal.get("final_answer", "") or "").strip()
            rationale = str(proposal.get("rationale", "") or "").strip()
            if not (answer or _is_meaningful_public_text(rationale)):
                continue
            lines.extend(
                [
                    "",
                    f"Agent: {agent.name}",
                    f"Confidence: {int(proposal.get('confidence', 1))}",
                    f"Final Answer: {answer or '[None]'}",
                    "Rationale:",
                    rationale,
                ]
            )
        self._append_live_log(live_log_path, "\n".join(lines))

    def _log_live_evaluation(
        self,
        *,
        live_log_path: str,
        round_id: int,
        final_output: str,
        evaluation_feedback: str,
        passed: bool,
    ) -> None:
        self._append_live_log(
            live_log_path,
            "\n".join(
                [
                    "",
                    f"OUTER ROUND {round_id} EVALUATION",
                    f"Passed: {passed}",
                    "Final Output Submitted To Evaluator:",
                    final_output or "[empty]",
                    "Evaluator Feedback:",
                    evaluation_feedback or "[empty]",
                ]
            ),
        )

    def _broadcast_public_message(self, deliberators: List[Any], message: Message | None) -> None:
        if message is None:
            return
        for agent in deliberators:
            agent.add_message_to_memory([message])

    def _approval_reached(self, approvals: List[bool]) -> bool:
        if not approvals:
            return False
        if self.approval_policy == "majority":
            return sum(1 for item in approvals if item) > (len(approvals) // 2)
        return all(approvals)

    def _select_poll_candidate(
        self,
        deliberators: List[Any],
        polls: List[Dict[str, Any]],
    ) -> Tuple[str, str]:
        best_candidate = ""
        best_source = ""
        best_score = -1

        for agent, poll in zip(deliberators, polls):
            candidate = _normalize_boxed(poll.get("candidate_answer", ""))
            if not candidate:
                continue
            score = int(poll.get("score", 1))
            if score > best_score:
                best_candidate = candidate
                best_source = agent.name
                best_score = score

        return best_candidate, best_source

    def _choose_candidate(self, current_candidate: str, revisions: List[str]) -> str:
        clean_revisions = [_normalize_boxed(item) for item in revisions if _normalize_boxed(item)]
        if not clean_revisions:
            return current_candidate
        if self.candidate_selection_policy == "latest_revision":
            return clean_revisions[-1]
        if self.candidate_selection_policy == "first_revision":
            return clean_revisions[0]
        counts = Counter(clean_revisions)
        best_answer, _ = counts.most_common(1)[0]
        return best_answer

    def _format_final_output(
        self,
        candidate_answer: str,
        *,
        consensus_reached: bool,
        candidate_source: str,
    ) -> str:
        boxed = _normalize_boxed(candidate_answer)
        if not boxed:
            boxed = "\\boxed{}"
        consensus_label = "consensus_reached" if consensus_reached else "consensus_not_reached"
        source_label = candidate_source or "unknown"
        return (
            "Broadcast-Deliberation Final Output\n"
            f"Consensus Status: {consensus_label}\n"
            f"Candidate Source: {source_label}\n"
            "Final Answer For Evaluator:\n"
            f"{boxed}"
        )

    def _candidate_review_message(self, candidate_answer: str, candidate_source: str) -> Message:
        boxed = _normalize_boxed(candidate_answer) or "\\boxed{}"
        return Message(
            content=(
                "Candidate answer under group review:\n"
                f"Source: {candidate_source or 'unknown'}\n"
                f"Candidate Answer:\n{boxed}"
            ),
            sender="system",
            receiver={"all"},
        )

    def _candidate_update_message(
        self,
        previous_candidate: str,
        updated_candidate: str,
        approval_round: int,
    ) -> Message:
        return Message(
            content=(
                "Protocol candidate update:\n"
                "Source Type: system protocol state (not evaluator feedback; not proof of correctness)\n"
                f"Reason: peer revision suggestions were proposed in approval round {approval_round}, "
                "and the configured candidate-selection policy selected a current candidate for further review.\n"
                f"Previous Candidate: {previous_candidate or '[None]'}\n"
                f"Current Candidate For Further Review: {updated_candidate or '[None]'}\n"
                "Instruction: Treat this as the candidate to scrutinize next. Do not assume a peer revision is correct just because it became the current candidate."
            ),
            sender="system",
            receiver={"all"},
        )

    async def _approval_loop(
        self,
        *,
        task_description: str,
        deliberators: List[Any],
        advice: str,
        candidate_answer: str,
        round_id: int,
        live_log_path: str = "",
        live_log_sink: Any = None,
    ) -> Tuple[bool, str, List[Dict[str, Any]]]:
        logs: List[Dict[str, Any]] = []
        current_candidate = _normalize_boxed(candidate_answer)
        consensus = False

        for approval_round in range(self.approval_rounds):
            responses = await asyncio.gather(
                *[
                    agent.aapprove_candidate(
                        task_description=task_description,
                        advice=self._advice_with_question_memory(advice),
                        candidate_answer=current_candidate,
                        memory_mode=self.memory_mode,
                        recent_history_window=self.recent_history_window,
                        approval_round_idx=approval_round,
                        failed_attempt_ledger_size=self._failed_attempt_memory_size(),
                        revision_ledger_size=self._candidate_revision_memory_size(),
                    )
                    for agent in deliberators
                ]
            )

            self._log_live_approval_round(
                live_log_path=live_log_path,
                round_id=round_id,
                approval_round=approval_round,
                current_candidate=current_candidate,
                deliberators=deliberators,
                responses=responses,
            )

            approvals: List[bool] = []
            revisions: List[str] = []

            for agent, response in zip(deliberators, responses):
                approvals.append(bool(response.get("approve")))
                revised = _normalize_boxed(response.get("revised_answer", ""))
                if revised:
                    revisions.append(revised)
                feedback = str(response.get("feedback", "") or "").strip()

                review_position = self._review_position_label(
                    bool(response.get("approve")),
                    revised,
                )
                valid_review = (
                    bool(response.get("approve"))
                    or bool(revised)
                    or _is_meaningful_public_text(feedback)
                )
                if not valid_review:
                    _append_log(
                        logs,
                        {
                            "type": "summary",
                            "round": round_id,
                            "stage": f"approval_{approval_round}_skipped",
                            "sender": agent.name,
                            "content": "Skipped empty or non-substantive approval review.",
                        },
                        live_log_sink,
                    )
                    continue

                if revised:
                    self._record_candidate_event(
                        logs,
                        round_id=round_id,
                        stage=f"approval_{approval_round}",
                        source=agent.name,
                        action="revise",
                        candidate_text=revised,
                        rationale_or_review=feedback,
                        live_log_sink=live_log_sink,
                    )
                elif bool(response.get("approve")):
                    self._record_candidate_event(
                        logs,
                        round_id=round_id,
                        stage=f"approval_{approval_round}",
                        source=agent.name,
                        action="approve",
                        candidate_text=current_candidate,
                        rationale_or_review=feedback,
                        live_log_sink=live_log_sink,
                    )
                else:
                    self._record_candidate_event(
                        logs,
                        round_id=round_id,
                        stage=f"approval_{approval_round}",
                        source=agent.name,
                        action="reject",
                        candidate_text=current_candidate,
                        rationale_or_review=feedback,
                        live_log_sink=live_log_sink,
                    )

                content = (
                    "Source Type: peer review/suggestion (not evaluator feedback; not system proof)\n"
                    f"Candidate under review: {current_candidate or '[None]'}\n"
                    f"Review Position: {review_position}\n"
                    f"Peer Review: {feedback or '[no substantive feedback]'}"
                )
                if revised:
                    content += (
                        "\nProposed Correction (peer suggestion; verify independently): "
                        f"{revised}"
                    )

                feedback_message = Message(
                    content=content,
                    sender=agent.name,
                    sender_agent=agent,
                    receiver=agent.get_receiver(),
                )
                self._record_agent_speaking_turn(
                    agent,
                    f"Outer round {round_id}, approval round {approval_round}",
                    content,
                )
                _append_log(
                    logs,
                    {
                        "type": "message",
                        "round": round_id,
                        "stage": f"approval_{approval_round}",
                        "sender": agent.name,
                        "content": content,
                    },
                    live_log_sink,
                )
                self._broadcast_public_message(deliberators, feedback_message)

            if self._approval_reached(approvals):
                consensus = True
                break

            updated_candidate = self._choose_candidate(current_candidate, revisions)
            if updated_candidate and updated_candidate != current_candidate:
                _append_log(
                    logs,
                    {
                        "type": "summary",
                        "round": round_id,
                        "stage": f"approval_{approval_round}_candidate_update",
                        "sender": "system",
                        "content": (
                            f"Candidate updated from {current_candidate or '[None]'} "
                            f"to {updated_candidate}"
                        ),
                    },
                    live_log_sink,
                )
                candidate_update_message = self._candidate_update_message(
                    current_candidate,
                    updated_candidate,
                    approval_round,
                )
                self._broadcast_public_message(deliberators, candidate_update_message)
                _append_log(
                    logs,
                    {
                        "type": "message",
                        "round": round_id,
                        "stage": f"approval_{approval_round}_candidate_update",
                        "sender": "system",
                        "content": candidate_update_message.content,
                    },
                    live_log_sink,
                )
                current_candidate = updated_candidate
                self._record_candidate_event(
                    logs,
                    round_id=round_id,
                    stage=f"approval_{approval_round}_candidate_update",
                    source="system",
                    action="select_revision",
                    candidate_text=updated_candidate,
                    rationale_or_review=(
                        "Protocol selected this candidate from peer revisions for further review; this is not a correctness signal."
                    ),
                    live_log_sink=live_log_sink,
                )

        return consensus, current_candidate, logs

    async def _final_proposal_round(
        self,
        *,
        task_description: str,
        deliberators: List[Any],
        advice: str,
        round_id: int,
        live_log_path: str = "",
        live_log_sink: Any = None,
    ) -> Tuple[str, str, List[Dict[str, Any]]]:
        logs: List[Dict[str, Any]] = []
        proposals = await asyncio.gather(
            *[
                agent.apropose_final_answer(
                    task_description=task_description,
                    advice=self._advice_with_question_memory(advice),
                    memory_mode=self.memory_mode,
                    recent_history_window=self.recent_history_window,
                    failed_attempt_ledger_size=self._failed_attempt_memory_size(),
                    revision_ledger_size=self._candidate_revision_memory_size(),
                )
                for agent in deliberators
            ]
        )

        best_answer = ""
        best_source = ""
        best_confidence = -1

        for agent, proposal in zip(deliberators, proposals):
            answer = _normalize_boxed(proposal.get("final_answer", ""))
            confidence = int(proposal.get("confidence", 0))
            rationale = str(proposal.get("rationale", "") or "").strip()
            valid_rationale = _is_meaningful_public_text(rationale)
            if answer or valid_rationale:
                self._record_agent_speaking_turn(
                    agent,
                    f"Outer round {round_id}, final proposal round",
                    (
                        f"Confidence: {confidence}\n"
                        f"Final Answer: {answer or '[None]'}\n"
                        f"Rationale: {rationale if valid_rationale else '[empty]'}"
                    ),
                )
                _append_log(
                    logs,
                    {
                        "type": "summary",
                        "round": round_id,
                        "stage": "final_proposal",
                        "sender": agent.name,
                        "content": (
                            f"confidence={confidence}\n"
                            f"answer={answer or '[None]'}\n"
                            f"rationale={rationale if valid_rationale else ''}"
                        ),
                    },
                    live_log_sink,
                )
                self._record_candidate_event(
                    logs,
                    round_id=round_id,
                    stage="final_proposal",
                    source=agent.name,
                    action="propose_final",
                    candidate_text=answer,
                    rationale_or_review=rationale,
                    live_log_sink=live_log_sink,
                )
            if answer and confidence > best_confidence:
                best_answer = answer
                best_source = agent.name
                best_confidence = confidence

        if best_answer:
            final_message = Message(
                content=f"Final proposal selected:\n{best_answer}",
                sender=best_source,
                sender_agent=next((a for a in deliberators if a.name == best_source), None),
                receiver={"all"},
            )
            _append_log(
                logs,
                {
                    "type": "message",
                    "round": round_id,
                    "stage": "final_proposal_selected",
                    "sender": best_source,
                    "content": final_message.content,
                },
                live_log_sink,
            )
            self._broadcast_public_message(deliberators, final_message)
            self._record_candidate_event(
                logs,
                round_id=round_id,
                stage="final_proposal_selected",
                source=best_source,
                action="submit",
                candidate_text=best_answer,
                rationale_or_review="Final proposal selected for evaluator submission.",
                live_log_sink=live_log_sink,
            )

        self._log_live_final_proposals(
            live_log_path=live_log_path,
            round_id=round_id,
            deliberators=deliberators,
            proposals=proposals,
            selected_answer=best_answer,
            selected_source=best_source,
        )

        return best_answer, best_source, logs

    async def astep(
        self,
        task_description: str,
        agents: Dict[Any, Any],
        advice: str = "No advice yet.",
        previous_plan: str = "No solution yet.",
        ground_truth: str = "",
        reference_solution: str = "",
        round_id: int = 0,
        live_log_path: str = "",
        live_log_sink: Any = None,
        **kwargs,
    ) -> Tuple[str, str, str, List[Dict[str, Any]], bool]:
        logs: List[Dict[str, Any]] = []
        deliberators = list(agents.get("deliberators", []) or [])
        evaluator_agent = agents.get(AGENT_TYPES.EVALUATION, None)

        if not deliberators:
            raise ValueError("broadcast-deliberation requires at least one `agent_type: deliberator`.")

        current_candidate = _normalize_boxed(previous_plan)
        if current_candidate in {"\\boxed{No solution yet.}", "\\boxed{No advice yet.}"}:
            current_candidate = ""
        candidate_source = "previous_attempt" if current_candidate else ""
        consensus_reached = False

        self._log_live_outer_round_start(
            live_log_path=live_log_path,
            round_id=round_id,
            task_description=task_description,
            advice=advice,
            current_candidate=current_candidate,
        )

        discussion_instruction = self.discussion_instruction

        for turn_idx in range(self.discussion_rounds):
            self._log_live_turn_state(
                live_log_path=live_log_path,
                round_id=round_id,
                turn_idx=turn_idx,
                current_candidate=current_candidate,
                advice=advice,
                deliberators=deliberators,
            )
            polls = await asyncio.gather(
                *[
                    agent.abroadcast_poll(
                        task_description=task_description,
                        advice=self._advice_with_question_memory(advice),
                        candidate_answer=current_candidate,
                        memory_mode=self.memory_mode,
                        recent_history_window=self.recent_history_window,
                        turn_idx=turn_idx,
                        outer_round_idx=round_id,
                        failed_attempt_ledger_size=self._failed_attempt_memory_size(),
                        revision_ledger_size=self._candidate_revision_memory_size(),
                    )
                    for agent in deliberators
                ]
            )

            best_idx = 0
            best_score = -1
            poll_candidate, poll_candidate_source = self._select_poll_candidate(
                deliberators,
                polls,
            )
            deferred_notes: List[Tuple[str, str]] = []

            for idx, (agent, poll) in enumerate(zip(deliberators, polls)):
                score = int(poll.get("score", 1))
                poll_reason = str(poll.get("reason", "") or "").strip()
                poll_intent = str(poll.get("intent", "") or "").strip()
                poll_candidate_answer = str(poll.get("candidate_answer", "") or "")
                poll_trace_visible = (
                    _is_meaningful_public_text(poll_reason)
                    or _is_meaningful_public_text(poll_intent)
                    or bool(_normalize_boxed(poll_candidate_answer))
                )
                _append_log(
                    logs,
                    {
                        "type": "summary",
                        "round": round_id,
                        "stage": f"poll_{turn_idx}",
                        "discussion_turn": turn_idx,
                        "sender": agent.name,
                        "score": score,
                        "reason": poll_reason,
                        "intent": poll_intent,
                        "candidate_answer": poll_candidate_answer,
                        "parse_succeeded": bool(poll.get("parse_succeeded")),
                        "trace_visible": poll_trace_visible,
                        "content": (
                            f"score={score}\n"
                            f"reason={poll_reason}\n"
                            f"intent={poll_intent}\n"
                            f"candidate={poll_candidate_answer or '[None]'}"
                        ),
                    },
                    live_log_sink,
                )
                if score > best_score:
                    best_idx = idx
                    best_score = score

            for idx, (agent, poll) in enumerate(zip(deliberators, polls)):
                if idx == best_idx or int(poll.get("score", 0)) < self.speak_threshold:
                    continue
                reason = str(poll.get("reason", "") or "").strip()
                intent = str(poll.get("intent", "") or "").strip()
                if not (
                    _is_meaningful_public_text(reason)
                    or _is_meaningful_public_text(intent)
                    or _normalize_boxed(poll.get("candidate_answer", ""))
                ):
                    continue
                note = (
                    f"Deferred note from discussion turn {turn_idx}.\n"
                    f"Reason to speak: {reason}\n"
                    f"Intended contribution: {intent}"
                )
                agent.add_private_note(note)
                deferred_notes.append((agent.name, note))
                _append_log(
                    logs,
                    {
                        "type": "summary",
                        "round": round_id,
                        "stage": f"deferred_{turn_idx}",
                        "sender": agent.name,
                        "content": note,
                    },
                    live_log_sink,
                )

            speaker = deliberators[best_idx]
            selected_poll = polls[best_idx]
            self._log_live_poll_results(
                live_log_path=live_log_path,
                round_id=round_id,
                turn_idx=turn_idx,
                deliberators=deliberators,
                polls=polls,
                selected_speaker=speaker.name,
                selected_score=best_score,
            )
            self._log_live_deferred_notes(
                live_log_path=live_log_path,
                round_id=round_id,
                turn_idx=turn_idx,
                deferred_notes=deferred_notes,
            )
            public_message = await speaker.abroadcast_speak(
                task_description=task_description,
                advice=self._advice_with_question_memory(advice),
                candidate_answer=current_candidate,
                phase="discussion",
                phase_instruction=discussion_instruction,
                current_turn_label=f"Outer round {round_id}, discussion turn {turn_idx}",
                **self._deliberator_debug_kwargs(f"discussion_turn_{turn_idx}_speak"),
            )
            public_content = getattr(public_message, "content", "") or ""
            valid_public_speech = _is_meaningful_public_text(public_content)
            if valid_public_speech:
                self._record_agent_speaking_turn(
                    speaker,
                    f"Outer round {round_id}, discussion turn {turn_idx}",
                    public_content,
                )

                _append_log(
                    logs,
                    {
                        "type": "message",
                        "round": round_id,
                        "stage": f"discussion_{turn_idx}",
                        "sender": speaker.name,
                        "content": public_content,
                    },
                    live_log_sink,
                )
                # Message-level augmentation: peers receive the message + its
                # predicted correctness / trajectory value (augment mode only;
                # no-op otherwise). The original message always passes through.
                broadcast_message = self._augment_public_message(public_message)
                self._broadcast_public_message(deliberators, broadcast_message)
                self._log_live_speaker_message(
                    live_log_path=live_log_path,
                    round_id=round_id,
                    turn_idx=turn_idx,
                    speaker_name=speaker.name,
                    content=public_content,
                )
            else:
                _append_log(
                    logs,
                    {
                        "type": "summary",
                        "round": round_id,
                        "stage": f"discussion_{turn_idx}_skipped",
                        "sender": speaker.name,
                        "content": "Skipped empty or non-substantive public message.",
                    },
                    live_log_sink,
                )
                self._log_live_skipped_speaker_message(
                    live_log_path=live_log_path,
                    round_id=round_id,
                    turn_idx=turn_idx,
                    speaker_name=speaker.name,
                )

            spoken_candidate = (
                _normalize_boxed(extract_boxed(public_content))
                if valid_public_speech
                else ""
            )
            selected_poll_candidate = _normalize_boxed(selected_poll.get("candidate_answer", ""))
            proposed_candidate = spoken_candidate or selected_poll_candidate or poll_candidate
            proposed_candidate_source = (
                speaker.name
                if (spoken_candidate or selected_poll_candidate)
                else poll_candidate_source
            )

            if proposed_candidate:
                current_candidate = proposed_candidate
                candidate_source = proposed_candidate_source
                self._record_candidate_event(
                    logs,
                    round_id=round_id,
                    stage=f"candidate_review_{turn_idx}",
                    source=candidate_source or "unknown",
                    action="select_for_review",
                    candidate_text=current_candidate,
                    rationale_or_review=(
                        f"Candidate promoted during discussion turn {turn_idx}."
                    ),
                    live_log_sink=live_log_sink,
                )
                self._log_live_candidate_review(
                    live_log_path=live_log_path,
                    round_id=round_id,
                    turn_idx=turn_idx,
                    candidate_answer=current_candidate,
                    candidate_source=candidate_source,
                )
                candidate_review_message = self._candidate_review_message(
                    current_candidate,
                    candidate_source,
                )
                self._broadcast_public_message(deliberators, candidate_review_message)
                _append_log(
                    logs,
                    {
                        "type": "message",
                        "round": round_id,
                        "stage": f"candidate_review_{turn_idx}",
                        "sender": "system",
                        "content": candidate_review_message.content,
                    },
                    live_log_sink,
                )
                consensus_reached, current_candidate, approval_logs = await self._approval_loop(
                    task_description=task_description,
                    deliberators=deliberators,
                    advice=advice,
                    candidate_answer=current_candidate,
                    round_id=round_id,
                    live_log_path=live_log_path,
                    live_log_sink=live_log_sink,
                )
                logs.extend(approval_logs)
                if consensus_reached:
                    break

        if not current_candidate:
            current_candidate, candidate_source, proposal_logs = await self._final_proposal_round(
                task_description=task_description,
                deliberators=deliberators,
                advice=advice,
                round_id=round_id,
                live_log_path=live_log_path,
                live_log_sink=live_log_sink,
            )
            logs.extend(proposal_logs)

        final_output = self._format_final_output(
            current_candidate,
            consensus_reached=consensus_reached,
            candidate_source=candidate_source,
        )
        _append_log(
            logs,
            {
                "type": "message",
                "round": round_id,
                "stage": "evaluation_submission",
                "sender": "system",
                "content": (
                    "Actual extracted final answer sent to evaluator: "
                    f"{_submitted_final_answer_for_trace(final_output)}"
                ),
            },
            live_log_sink,
        )

        self._set_evaluator_debug_context(
            evaluator_agent,
            "evaluation_verdict",
        )
        evaluation = await self.evaluator.astep(
            agent=evaluator_agent,
            solution=[],
            result=[],
            task_description=task_description,
            all_role_description=[getattr(agent, "role_description", "") for agent in deliberators],
            reviewer_output=final_output,
            submitted_candidate=final_output,
            ground_truth=str(ground_truth),
            reference_solution=str(reference_solution or ""),
            debug_stage_prefix="evaluation",
            previous_answer_summary=self.failed_attempt_memory.render_for_evaluator_hint(),
            previous_reasoning_summary=self.candidate_revision_memory.render_for_evaluator_hint(),
            current_reasoning_summary=self.candidate_revision_memory.render_for_evaluator_hint(
                max_records=5
            ),
        )
        eval_result = dict(evaluation.content) if isinstance(evaluation.content, dict) else {}
        passed = _coerce_evaluator_score(evaluation.score)
        eval_result.setdefault("mode", self.evaluator_type)
        eval_result.setdefault("passed", passed)
        eval_result.setdefault("signal", "PASS" if passed else "FAIL")
        eval_result.setdefault("final_answer", extract_boxed(final_output))
        eval_advice = str(evaluation.advice or "")
        direct_protocol_fail = bool(eval_result.get("direct_protocol_fail"))
        verdict_advice = str(eval_result.get("verdict_advice") or eval_advice or "").strip()
        hint_advice = str(eval_result.get("hint_advice") or "").strip()
        if direct_protocol_fail:
            eval_sender = "system"
        else:
            eval_sender = getattr(evaluation, "sender", "") or getattr(
                evaluator_agent,
                "name",
                "Evaluator",
            )
        evaluation_feedback = _format_evaluator_feedback(
            eval_result["mode"],
            passed,
            hint_advice or verdict_advice or eval_advice,
        )
        group_feedback = _format_group_feedback(
            eval_result["mode"],
            passed,
            hint_advice or verdict_advice or eval_advice,
            self.reviewer_feedback_mode,
        )
        evaluation_boxed = _normalize_boxed(eval_result.get("final_answer", "")) or "\\boxed{}"
        self._record_candidate_event(
            logs,
            round_id=round_id,
            stage="evaluation",
            source=eval_sender,
            action="evaluate",
            candidate_text=evaluation_boxed,
            rationale_or_review=group_feedback,
            evaluator_signal=eval_result.get("signal", "PASS" if passed else "FAIL"),
            live_log_sink=live_log_sink,
        )
        failed_attempt_context = ""
        if not passed:
            submitted_answer = (
                _normalize_boxed(eval_result.get("final_answer", ""))
                or _normalize_boxed(final_output)
                or final_output
            )
            self.failed_attempt_memory.add(
                submitted_answer=submitted_answer,
                evaluator_signal=str(eval_result.get("signal", "FAIL") or "FAIL"),
                evaluator_hint=group_feedback,
                repair_summary=(
                    "Next broadcast round should use the evaluator hint to revise the shared candidate."
                ),
                source_stage="broadcast_evaluation",
            )
            failed_attempt_context = self.failed_attempt_memory.render()

        eval_log_content = _format_evaluator_verdict_only(
            eval_result["mode"],
            passed,
        )
        if eval_result.get("judge_device"):
            eval_log_content += (
                f"\nJudge student final answer: {eval_result.get('judge_student_final_answer')}"
                f"\nJudge equivalence judgement: {eval_result.get('judge_equivalence_judgement')}"
                f"\nJudge device: {eval_result.get('judge_device')}"
                f"\nJudge dtype: {eval_result.get('judge_dtype')}"
            )
        _append_log(
            logs,
                {
                    "type": "message",
                    "round": round_id,
                    "stage": "evaluation",
                    "sender": eval_sender,
                    "content": eval_log_content,
                    "force_numbered_system_trace": bool(direct_protocol_fail),
                },
                live_log_sink,
            )
        if not passed and hint_advice:
            _append_log(
                logs,
                {
                    "type": "message",
                    "round": round_id,
                    "stage": "evaluation_hint",
                    "sender": eval_sender,
                    "content": _format_evaluator_hint_only(hint_advice),
                    "force_numbered_system_trace": bool(direct_protocol_fail),
                },
                live_log_sink,
            )
        _append_log(
            logs,
            {
                "type": "meta",
                "round": round_id,
                "stage": "evaluation_result",
                "sender": eval_sender,
                "content": "PASS" if passed else "FAIL",
                "mode": eval_result["mode"],
                "signal": eval_result.get("signal", "PASS" if passed else "FAIL"),
                "correctness": 1 if passed else 0,
                "final_answer": eval_result.get("final_answer", ""),
                "advice": group_feedback,
                "verdict_advice": verdict_advice,
                "hint_advice": hint_advice,
                "raw_evaluator_advice": evaluation_feedback,
                "failed_attempt_memory": failed_attempt_context,
                "reviewer_feedback_mode": self.reviewer_feedback_mode,
                "reference_solution_used": bool(eval_result.get("reference_solution_used")),
                "judge_student_final_answer": eval_result.get("judge_student_final_answer", ""),
                "judge_equivalence_judgement": eval_result.get("judge_equivalence_judgement", ""),
                "judge_device": eval_result.get("judge_device"),
                "judge_dtype": eval_result.get("judge_dtype"),
                "rule_prediction": eval_result.get("rule_prediction", ""),
            },
            live_log_sink,
        )

        evaluation_message = Message(
            content=(
                "Outer-loop evaluator feedback:\n"
                f"{group_feedback}\n"
                f"{self.candidate_revision_memory.render() if self.candidate_revision_memory.records else ''}\n"
                f"{failed_attempt_context}\n"
                f"Final MAS answer reviewed:\n{evaluation_boxed}"
            ),
            sender=eval_sender,
            sender_agent=evaluator_agent,
            receiver={"all"},
        )
        self._broadcast_public_message(deliberators, evaluation_message)
        self._log_live_evaluation(
            live_log_path=live_log_path,
            round_id=round_id,
            final_output=final_output,
            evaluation_feedback=group_feedback,
            passed=passed,
        )

        _append_log(
            logs,
            {
                "type": "meta",
                "round": round_id,
                "stage": "system",
                "sender": "system",
                "content": "Good score! Accept!" if passed else "Bad score! Reject!",
            },
            live_log_sink,
        )

        system_message = Message(
            content="Good score! Accept!" if passed else "Bad score! Reject!",
            sender="system",
            receiver={"all"},
        )
        self._broadcast_public_message(deliberators, system_message)

        if not passed:
            final_advice = (
                f"{group_feedback}\n\n"
                f"{self.candidate_revision_memory.render() if self.candidate_revision_memory.records else ''}\n\n"
                f"{failed_attempt_context}\n"
                f"[Candidate]\n{final_output}\n[EndCandidate]"
            ).strip()
        else:
            final_advice = group_feedback

        return final_output, final_advice, final_output, logs, passed
