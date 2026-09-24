from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


# RELEASE PATCH (NeurIPS26_Precise-but-Uncoupled): upstream these tests sat in
# process/tests/ so parents[1] WAS the process directory. In this release they
# live in tests/ at the repository root, so the path is computed explicitly.
# The assertions below are unchanged.
PROCESS_DIR = Path(__file__).resolve().parents[1] / "src" / "precise_uncoupled" / "process"
sys.path.insert(0, str(PROCESS_DIR))

from symmetric_transition_extractor import (  # noqa: E402
    candidate_answer_from_event,
    extract_all_boxed,
    extract_explicit_answer,
    extract_symmetric_transitions,
)


def event(
    sequence: int,
    stage: str,
    sender: str,
    content: str,
    *,
    event_type: str = "message",
    role: str = "assistant",
    extra_json: str | dict | None = None,
    previous_event_uid: str = "",
) -> dict:
    result = {
        "event_uid": f"e{sequence}",
        "trajectory_uid": "trace-1",
        "sequence_index": sequence,
        "event_type": event_type,
        "stage": stage,
        "sender": sender,
        "role": role,
        "content": content,
    }
    if extra_json is not None:
        result["extra_json"] = extra_json
    if previous_event_uid:
        result["previous_event_uid"] = previous_event_uid
    return result


class BoxedExtractionTests(unittest.TestCase):
    def test_nested_boxed_and_literal_backspace_are_supported(self) -> None:
        self.assertEqual(extract_all_boxed(r"x \boxed{\frac{1}{2}}"), [r"\frac{1}{2}"])
        self.assertEqual(extract_all_boxed("x \x08oxed{B}"), ["B"])
        self.assertEqual(extract_explicit_answer(r"\boxed{\boxed{B}}"), "B")
        self.assertEqual(extract_explicit_answer("Candidate Answer: ..."), "")
        placeholder_poll = event(
            0,
            "poll_0",
            "Mira",
            "score=0\ncandidate=\\boxed{...}",
            event_type="summary",
        )
        placeholder_poll["extra_json"] = json.dumps(
            {"candidate_answer": r"\boxed{...}"}
        )
        self.assertEqual(candidate_answer_from_event(placeholder_poll), "")


class PerTransitionTests(unittest.TestCase):
    def _review_memory(
        self,
        sequence: int,
        *,
        review_sequence: int,
        action: str = "review_route",
        source: str = "Reviewer",
        authority: str = "internal_reviewer_suggestion",
        review_stage: str = "reviewer_0",
        event_stage: str | None = None,
        previous_event_uid: str | None = None,
        extra_json: str | dict | None = None,
    ) -> dict:
        payload = (
            {
                "candidate_memory_event": {
                    "action": action,
                    "authority": authority,
                    "candidate_answer": r"\boxed{A}",
                    "source": source,
                    "stage": review_stage,
                }
            }
            if extra_json is None
            else extra_json
        )
        return event(
            sequence,
            event_stage or f"{review_stage}_candidate_memory",
            "system",
            r"C5 review \boxed{A}",
            event_type="summary",
            role="system",
            extra_json=payload,
            previous_event_uid=(
                f"e{review_sequence}"
                if previous_event_uid is None
                else previous_event_uid
            ),
        )

    def test_source_linked_review_route_never_overrides_explicit_agree(
        self,
    ) -> None:
        events = [
            event(0, "executor_0", "Executor", r"\boxed{A}"),
            event(1, "reviewer_0", "Reviewer", r"\boxed{A} [Agree]"),
            self._review_memory(2, review_sequence=1),
            event(3, "executor_1", "Executor", r"\boxed{B}"),
        ]
        [record] = extract_symmetric_transitions("per", events)
        self.assertEqual(record["review_action"], "agree")
        self.assertEqual(record["review_memory_action"], "review_route")
        self.assertTrue(record["review_memory_provenance_valid"])
        self.assertTrue(
            record["review_memory_previous_event_uid_matches_review"]
        )
        self.assertTrue(record["review_route_observed"])

    def test_source_linked_review_route_leaves_unknown_action_unknown(
        self,
    ) -> None:
        events = [
            event(0, "executor_0", "Executor", r"\boxed{A}"),
            event(
                1,
                "reviewer_0",
                "Reviewer",
                "I checked the reasoning and return this review.",
            ),
            self._review_memory(2, review_sequence=1),
            event(3, "executor_1", "Executor", r"\boxed{B}"),
        ]
        [record] = extract_symmetric_transitions("per", events)
        self.assertEqual(record["review_action"], "unknown")
        self.assertTrue(record["review_route_observed"])
        self.assertEqual(record["review_memory_link_status"], "linked")

    def test_review_memory_provenance_fails_closed_for_bad_links(
        self,
    ) -> None:
        cases = {
            "malformed": self._review_memory(
                2,
                review_sequence=1,
                extra_json="{",
            ),
            "wrong_previous_uid": self._review_memory(
                2,
                review_sequence=1,
                previous_event_uid="wrong-event",
            ),
            "wrong_source": self._review_memory(
                2,
                review_sequence=1,
                source="Planner",
            ),
            "wrong_authority": self._review_memory(
                2,
                review_sequence=1,
                authority="peer_or_agent_suggestion",
            ),
            "wrong_stage": self._review_memory(
                2,
                review_sequence=1,
                event_stage="other_candidate_memory",
            ),
            "nonconsecutive_sequence": self._review_memory(
                3,
                review_sequence=1,
            ),
        }
        for name, memory in cases.items():
            with self.subTest(name=name):
                events = [
                    event(0, "executor_0", "Executor", r"\boxed{A}"),
                    event(
                        1,
                        "reviewer_0",
                        "Reviewer",
                        "I checked the reasoning.",
                    ),
                    memory,
                    event(4, "executor_1", "Executor", r"\boxed{B}"),
                ]
                [record] = extract_symmetric_transitions("per", events)
                self.assertEqual(record["review_action"], "unknown")
                self.assertFalse(record["review_memory_provenance_valid"])
                self.assertFalse(record["review_route_observed"])
                self.assertNotEqual(
                    record["review_memory_link_status"], "linked"
                )

    def test_reviewer_proposal_is_never_used_as_after(self) -> None:
        events = [
            event(0, "executor_0", "Executor", r"Initial \boxed{A}"),
            event(
                1,
                "reviewer_0",
                "Reviewer",
                r"[ROUTE: EXECUTOR] Proposed correction: \boxed{B}",
            ),
            event(
                2,
                "reviewer_0_candidate_memory",
                "system",
                r"C5 review \boxed{B}",
                event_type="summary",
                role="system",
            ),
            event(3, "executor_1", "Executor", r"I reconsidered it: \boxed{C}"),
        ]
        [record] = extract_symmetric_transitions("per", events)
        self.assertEqual(record["answer_before"], "A")
        self.assertEqual(record["answer_after"], "C")
        self.assertEqual(json.loads(record["reviewer_proposed_answers_json"]), ["B"])
        self.assertEqual(record["after_sender"], "Executor")
        self.assertTrue(record["actor_response_observed"])
        self.assertTrue(record["strict_pre_gate_eligible"])
        self.assertFalse(record["system_selected_answer_used_as_after"])

    def test_submission_blocks_later_executor(self) -> None:
        events = [
            event(0, "executor_0", "Executor", r"\boxed{A}"),
            event(1, "reviewer_0", "Reviewer", "Please revise."),
            event(
                2,
                "evaluation_submission",
                "system",
                r"Submitted \boxed{B}",
                role="system",
            ),
            event(3, "executor_post_eval_fail", "Executor", r"\boxed{C}"),
        ]
        [record] = extract_symmetric_transitions("per", events)
        self.assertEqual(record["status"], "no_actor_response_before_submission")
        self.assertEqual(record["answer_after"], "")
        self.assertFalse(record["actor_response_observed"])
        self.assertFalse(record["strict_pre_gate_eligible"])

    def test_unparsed_executor_turn_is_not_an_observed_candidate_response(self) -> None:
        events = [
            event(0, "executor_0", "Executor", r"\boxed{A}"),
            event(1, "reviewer_0", "Reviewer", "Please finish the calculation."),
            event(2, "executor_1", "Executor", "I am still calculating the value."),
        ]
        [record] = extract_symmetric_transitions("per", events)
        self.assertTrue(record["actor_turn_observed"])
        self.assertFalse(record["actor_response_observed"])
        self.assertEqual(record["status"], "actor_response_unparsed_before_gate")
        self.assertFalse(record["strict_pre_gate_eligible"])

    def test_evaluator_and_system_candidates_are_ignored(self) -> None:
        events = [
            event(0, "executor_0", "Executor", r"\boxed{A}"),
            event(1, "reviewer_0", "Reviewer", r"Try \boxed{B}."),
            event(2, "evaluation_hint", "Evaluator", r"Hint \boxed{D}", role="evaluator"),
            event(
                3,
                "executor_1_candidate_memory",
                "system",
                r"C5 propose \boxed{E}",
                event_type="summary",
                role="system",
            ),
            event(4, "executor_1", "Executor", r"\boxed{C}"),
        ]
        [record] = extract_symmetric_transitions("per", events)
        self.assertEqual(record["answer_after"], "C")
        self.assertEqual(record["after_event_uid"], "e4")


class BroadcastTransitionTests(unittest.TestCase):
    def _anchor(
        self,
        sequence: int = 1,
        answer: str = "A",
        *,
        source: str = "Mira",
        turn: int = 0,
    ) -> dict:
        return event(
            sequence,
            f"candidate_review_{turn}",
            "system",
            "Candidate answer under group review:\n"
            f"Source: {source}\n"
            f"Candidate Answer: \\\\boxed{{{answer}}}",
            role="system",
        )

    def _review(self, sequence: int, sender: str, position: str, proposal: str) -> dict:
        return event(
            sequence,
            "approval_0",
            sender,
            "Candidate under review: \\boxed{A}\n"
            f"Review Position: {position}\n"
            "Peer Review: check it\n"
            f"Proposed Correction (peer suggestion; verify independently): \\boxed{{{proposal}}}",
        )

    def test_system_selected_revision_is_not_after(self) -> None:
        events = [
            event(0, "discussion_0", "Mira", r"Public candidate \boxed{A}"),
            self._anchor(),
            self._review(2, "Rowan", "propose_correction", "B"),
            self._review(3, "Talia", "propose_correction", "C"),
            event(
                4,
                "approval_0_candidate_update",
                "system",
                r"Candidate updated from \boxed{A} to \boxed{B}",
                role="system",
            ),
            event(
                5,
                "evaluation_submission",
                "system",
                r"Submitted \boxed{B}",
                role="system",
            ),
        ]
        [record] = extract_symmetric_transitions("broadcast", events)
        self.assertEqual(record["answer_before"], "A")
        self.assertEqual(record["answer_after"], "")
        self.assertEqual(json.loads(record["system_selected_answers_json"]), ["B"])
        self.assertFalse(record["system_selected_answer_used_as_after"])
        self.assertEqual(record["status"], "no_actor_response_before_submission")
        self.assertFalse(record["strict_pre_gate_eligible"])

    def test_later_discussion_is_actor_response_but_not_gate_free_after_update(self) -> None:
        events = [
            event(0, "discussion_0", "Mira", r"\boxed{A}"),
            self._anchor(),
            self._review(2, "Rowan", "propose_correction", "B"),
            event(
                3,
                "approval_0_candidate_update",
                "system",
                r"Candidate updated from \boxed{A} to \boxed{B}",
                role="system",
            ),
            event(4, "discussion_1", "Talia", r"I re-derived it: \boxed{C}"),
            self._anchor(5, "C", source="Talia", turn=1),
        ]
        [record] = extract_symmetric_transitions("broadcast", events)
        self.assertEqual(record["answer_after"], "C")
        self.assertEqual(record["after_sender"], "Talia")
        self.assertTrue(record["actor_response_observed"])
        self.assertEqual(record["status"], "actor_response_after_protocol_selection")
        self.assertFalse(record["strict_pre_gate_eligible"])

    def test_later_discussion_without_update_is_strictly_eligible(self) -> None:
        events = [
            event(0, "discussion_0", "Mira", r"\boxed{A}"),
            self._anchor(),
            self._review(2, "Rowan", "needs_revision", "B"),
            event(3, "discussion_1", "Talia", r"Independent revision \boxed{C}"),
            self._anchor(4, "C", source="Talia", turn=1),
        ]
        [record] = extract_symmetric_transitions("broadcast", events)
        self.assertEqual(record["answer_after"], "C")
        self.assertEqual(record["status"], "actor_response_before_gate")
        self.assertTrue(record["strict_pre_gate_eligible"])

    def test_system_or_reviewer_messages_cannot_satisfy_actor_response(self) -> None:
        events = [
            event(0, "discussion_0", "Mira", r"\boxed{A}"),
            self._anchor(),
            self._review(2, "Rowan", "propose_correction", "B"),
            event(
                3,
                "approval_0_candidate_update",
                "system",
                r"Candidate updated from \boxed{A} to \boxed{B}",
                role="system",
            ),
            event(
                4,
                "evaluation_submission",
                "system",
                r"\boxed{B}",
                role="system",
            ),
            event(5, "discussion_1", "Talia", r"\boxed{C}"),
        ]
        [record] = extract_symmetric_transitions("broadcast", events)
        self.assertFalse(record["actor_response_observed"])
        self.assertEqual(record["answer_after"], "")

    def test_candidate_alignment_mismatch_fails_closed_without_actor_link(self) -> None:
        events = [
            event(0, "discussion_0", "Mira", r"\boxed{Z}"),
            self._anchor(answer="A"),
            self._review(2, "Rowan", "needs_revision", "B"),
            event(3, "discussion_1", "Talia", r"\boxed{C}"),
            self._anchor(4, "C", source="Talia", turn=1),
        ]
        [record] = extract_symmetric_transitions("broadcast", events)
        self.assertEqual(record["candidate_alignment"], "missing_actor_before")
        self.assertEqual(record["status"], "missing_actor_candidate_before")
        self.assertEqual(record["answer_before"], "")
        self.assertFalse(record["strict_pre_gate_eligible"])

    def test_poll_origin_is_linked_by_next_anchor_source_and_answer(self) -> None:
        events = [
            event(0, "discussion_0", "Mira", r"\boxed{A}"),
            self._anchor(),
            self._review(2, "Talia", "needs_revision", "B"),
            event(
                3,
                "poll_1",
                "Mira",
                "score=90\ncandidate=\\boxed{B}",
                event_type="summary",
            ),
            event(
                4,
                "poll_1",
                "Rowan",
                "score=95\ncandidate=\\boxed{C}",
                event_type="summary",
            ),
            event(
                5,
                "poll_1",
                "Talia",
                "score=70\ncandidate=\\boxed{D}",
                event_type="summary",
            ),
            event(
                6,
                "discussion_1",
                "Mira",
                "I am comparing the alternatives; no final answer here.",
            ),
            self._anchor(7, "C", source="Rowan", turn=1),
        ]
        [record] = extract_symmetric_transitions("broadcast", events)
        self.assertEqual(record["answer_after"], "C")
        self.assertEqual(record["after_sender"], "Rowan")
        self.assertEqual(record["after_stage"], "poll_1")
        self.assertEqual(
            record["after_candidate_link_method"],
            "next_anchor_source_answer_match_actor_event",
        )
        self.assertTrue(record["actor_response_observed"])
        self.assertTrue(record["strict_pre_gate_eligible"])


if __name__ == "__main__":
    unittest.main()
