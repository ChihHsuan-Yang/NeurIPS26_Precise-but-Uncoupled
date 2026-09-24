from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path


# RELEASE PATCH (NeurIPS26_Precise-but-Uncoupled): upstream these tests sat in
# process/tests/ so parents[1] WAS the process directory. In this release they
# live in tests/ at the repository root, so the path is computed explicitly.
# The assertions below are unchanged.
PROCESS_DIR = Path(__file__).resolve().parents[1] / "src" / "precise_uncoupled" / "process"
sys.path.insert(0, str(PROCESS_DIR))

from aggregate_release_transitions import (  # noqa: E402
    PER_ROUTE_SENSITIVITY_COUNT_FIELDS,
    PER_ROUTE_SENSITIVITY_DEFINITION,
    REVISE_ACTION_DEFINITION,
    _compact_matrix_per_route_sensitivity,
    _compact_matrix_revise_summary,
    attach_frozen_correctness,
    build_frozen_answer_labels,
    classify_transition_outcome,
    primary_eligibility_bucket,
    _parse_pass_fail,
    _update_correctness_denominators,
    _update_per_route_sensitivity_counts,
    _update_revise_conditioned_denominators,
)
from symmetric_transition_extractor import extract_symmetric_transitions  # noqa: E402


def event(
    sequence: int,
    stage: str,
    sender: str,
    content: str,
    *,
    event_type: str = "message",
    role: str = "assistant",
    extra_json: str = "{}",
) -> dict:
    return {
        "event_uid": f"e{sequence}",
        "trajectory_uid": "trace-aggregate",
        "sequence_index": sequence,
        "event_type": event_type,
        "stage": stage,
        "sender": sender,
        "role": role,
        "content": content,
        "extra_json": extra_json,
    }


class FrozenLabelTests(unittest.TestCase):
    def test_evaluation_signal_is_parsed_without_keyword_ambiguity(self) -> None:
        self.assertIs(
            _parse_pass_fail(
                "Evaluation signal: FAIL\nAdvice: a future answer may PASS."
            ),
            False,
        )
        self.assertIsNone(_parse_pass_fail("The candidate may PASS or FAIL."))

    def test_runtime_evaluator_labels_create_repair_classification(self) -> None:
        events = [
            event(
                0,
                "evaluation_submission",
                "system",
                r"Actual extracted final answer sent to evaluator: \boxed{A}",
                role="system",
            ),
            event(1, "evaluation", "Evaluator", "Evaluation signal: FAIL", role="evaluator"),
            event(2, "executor_0", "Executor", r"\boxed{A}"),
            event(3, "reviewer_0", "Reviewer", "Diagnosis: revise."),
            event(4, "executor_1", "Executor", r"\boxed{B}"),
            event(
                5,
                "evaluation_submission",
                "system",
                r"Actual extracted final answer sent to evaluator: \boxed{B}",
                role="system",
            ),
            event(6, "evaluation", "Evaluator", "Evaluation signal: PASS", role="evaluator"),
        ]
        labels = build_frozen_answer_labels(
            events, final_answer="B", final_correct=True
        )
        [transition] = extract_symmetric_transitions("per", events)
        labeled = attach_frozen_correctness(transition, labels)
        self.assertIs(labeled["before_correct"], False)
        self.assertIs(labeled["after_correct"], True)
        self.assertTrue(labeled["correctness_pair_labeled"])
        self.assertEqual(labeled["transition_outcome"], "repaired")
        self.assertEqual(labeled["primary_eligibility_bucket"], "strict_pre_gate")

    def test_conflicting_frozen_labels_fail_closed(self) -> None:
        events = [
            event(
                0,
                "evaluation_submission",
                "system",
                r"\boxed{A}",
                role="system",
            ),
            event(1, "evaluation", "Evaluator", "PASS", role="evaluator"),
            event(
                2,
                "evaluation_submission",
                "system",
                r"\boxed{A}",
                role="system",
            ),
            event(3, "evaluation", "Evaluator", "FAIL", role="evaluator"),
        ]
        labels = build_frozen_answer_labels(events)
        item = labels["a"]
        self.assertIsNone(item.value)
        self.assertTrue(item.conflict)


class ClassificationTests(unittest.TestCase):
    def test_required_transition_outcomes(self) -> None:
        self.assertEqual(
            classify_transition_outcome("A", "A", False, False),
            (False, "unchanged"),
        )
        self.assertEqual(
            classify_transition_outcome("A", "B", False, False),
            (True, "changed_still_wrong"),
        )
        self.assertEqual(
            classify_transition_outcome("A", "B", False, True),
            (True, "repaired"),
        )

    def test_gate_confounded_bucket_precedes_strict(self) -> None:
        record = {
            "status": "actor_response_after_protocol_selection",
            "actor_turn_observed": True,
            "answer_pair_parsed": True,
            "intervening_system_candidate_updates": 1,
            "strict_pre_gate_eligible": False,
        }
        self.assertEqual(primary_eligibility_bucket(record), "gate_confounded")

    def test_parser_incomplete_and_no_after_are_separate(self) -> None:
        parser = {
            "status": "actor_response_unparsed_before_gate",
            "actor_turn_observed": True,
            "answer_pair_parsed": False,
            "intervening_system_candidate_updates": 0,
            "strict_pre_gate_eligible": False,
        }
        no_after = {
            "status": "no_actor_response_before_submission",
            "actor_turn_observed": False,
            "answer_pair_parsed": False,
            "intervening_system_candidate_updates": 0,
            "strict_pre_gate_eligible": False,
        }
        self.assertEqual(primary_eligibility_bucket(parser), "parser_incomplete")
        self.assertEqual(primary_eligibility_bucket(no_after), "no_after")

    def test_strict_and_gate_repair_denominators_remain_separate(self) -> None:
        key = ("dataset", "slice", "actor", "per")
        summaries = {key: {}}
        counts = {
            key: Counter(
                {
                    "labeled_pre_wrong_pairs": 10,
                    "repaired_over_labeled_pre_wrong": 4,
                    "strict_labeled_pre_wrong_pairs": 3,
                    "strict_repaired_over_labeled_pre_wrong": 2,
                    "gate_labeled_pre_wrong_pairs": 6,
                    "gate_repaired_over_labeled_pre_wrong": 1,
                }
            )
        }
        _update_correctness_denominators(summaries, counts)
        row = summaries[key]
        self.assertEqual(row["labeled_pre_wrong_pairs"], 10)
        self.assertEqual(row["strict_labeled_pre_wrong_pairs"], 3)
        self.assertEqual(row["gate_labeled_pre_wrong_pairs"], 6)
        self.assertEqual(row["repair_rate_labeled_pre_wrong"], 0.4)
        self.assertEqual(row["strict_repair_rate_labeled_pre_wrong"], 2 / 3)
        self.assertEqual(row["gate_repair_rate_labeled_pre_wrong"], 1 / 6)

    def test_revise_conditioned_breakdown_is_bucketed_and_pre_wrong_only(
        self,
    ) -> None:
        counts: Counter[str] = Counter()
        strict = {
            "review_action": "revise",
            "strict_pre_gate_eligible": True,
            "primary_eligibility_bucket": "strict_pre_gate",
            "before_correct": False,
            "after_correct": False,
        }
        for outcome in ("unchanged", "changed_still_wrong", "repaired"):
            row = dict(strict)
            row["transition_outcome"] = outcome
            row["answer_changed"] = outcome != "unchanged"
            if outcome == "repaired":
                row["after_correct"] = True
            _update_revise_conditioned_denominators(counts, row)
        _update_revise_conditioned_denominators(
            counts,
            {
                **strict,
                "after_correct": None,
                "answer_changed": True,
                "transition_outcome": "changed_correctness_unlabeled",
            },
        )

        gate = {
            "review_action": "revise",
            "strict_pre_gate_eligible": False,
            "primary_eligibility_bucket": "gate_confounded",
            "before_correct": False,
            "after_correct": True,
            "answer_changed": True,
            "transition_outcome": "repaired",
        }
        _update_revise_conditioned_denominators(counts, gate)

        ignored = [
            {
                **strict,
                "review_action": "agree",
                "answer_changed": False,
                "transition_outcome": "unchanged",
            },
            {
                **strict,
                "review_action": "unknown",
                "answer_changed": False,
                "transition_outcome": "unchanged",
            },
            {
                **strict,
                "before_correct": True,
                "answer_changed": False,
                "transition_outcome": "unchanged",
            },
        ]
        for row in ignored:
            _update_revise_conditioned_denominators(counts, row)

        key = ("dataset", "slice", "actor", "per")
        summaries = {key: {}}
        _update_correctness_denominators(summaries, {key: counts})
        summary = summaries[key]
        self.assertEqual(
            summary["strict_explicit_revise_pre_wrong_actor_pairs"], 4
        )
        self.assertEqual(
            summary["strict_explicit_revise_answer_unchanged"], 1
        )
        self.assertEqual(
            summary["strict_explicit_revise_answer_changed"], 3
        )
        self.assertEqual(
            summary["strict_explicit_revise_after_correct_labeled_pairs"], 3
        )
        self.assertEqual(
            summary["strict_explicit_revise_labeled_unchanged"], 1
        )
        self.assertEqual(
            summary["strict_explicit_revise_labeled_changed_still_wrong"], 1
        )
        self.assertEqual(
            summary["strict_explicit_revise_repaired"], 1
        )
        self.assertEqual(
            summary["strict_explicit_revise_changed_after_correct_labeled"],
            2,
        )
        self.assertEqual(
            summary["strict_explicit_revise_change_response_rate"], 3 / 4
        )
        self.assertEqual(
            summary[
                "strict_explicit_revise_after_correct_label_coverage_over_uptake"
            ],
            3 / 4,
        )
        self.assertEqual(
            summary[
                "strict_explicit_revise_after_correct_label_coverage_over_changed"
            ],
            2 / 3,
        )
        self.assertEqual(
            summary[
                "strict_explicit_revise_repair_rate_over_after_correct_labeled_pairs"
            ],
            1 / 3,
        )
        self.assertEqual(
            summary["gate_explicit_revise_pre_wrong_actor_pairs"], 1
        )
        self.assertEqual(
            summary["gate_explicit_revise_change_response_rate"], 1.0
        )
        self.assertEqual(
            summary[
                "gate_explicit_revise_repair_rate_over_after_correct_labeled_pairs"
            ],
            1.0,
        )

    def test_compact_matrix_revise_summary_pools_counts_not_cell_rates(
        self,
    ) -> None:
        rows = [
            {
                "protocol": "per",
                "strict_explicit_revise_pre_wrong_actor_pairs": 3,
                "strict_explicit_revise_answer_unchanged": 1,
                "strict_explicit_revise_answer_changed": 2,
                "strict_explicit_revise_after_correct_labeled_pairs": 2,
                "strict_explicit_revise_labeled_unchanged": 1,
                "strict_explicit_revise_labeled_changed_still_wrong": 0,
                "strict_explicit_revise_repaired": 1,
                "strict_explicit_revise_changed_after_correct_labeled": 1,
                "gate_explicit_revise_pre_wrong_actor_pairs": 0,
                "gate_explicit_revise_answer_unchanged": 0,
                "gate_explicit_revise_answer_changed": 0,
                "gate_explicit_revise_after_correct_labeled_pairs": 0,
                "gate_explicit_revise_labeled_unchanged": 0,
                "gate_explicit_revise_labeled_changed_still_wrong": 0,
                "gate_explicit_revise_repaired": 0,
                "gate_explicit_revise_changed_after_correct_labeled": 0,
            },
            {
                "protocol": "per",
                "strict_explicit_revise_pre_wrong_actor_pairs": 4,
                "strict_explicit_revise_answer_unchanged": 1,
                "strict_explicit_revise_answer_changed": 3,
                "strict_explicit_revise_after_correct_labeled_pairs": 3,
                "strict_explicit_revise_labeled_unchanged": 1,
                "strict_explicit_revise_labeled_changed_still_wrong": 1,
                "strict_explicit_revise_repaired": 1,
                "strict_explicit_revise_changed_after_correct_labeled": 2,
                "gate_explicit_revise_pre_wrong_actor_pairs": 1,
                "gate_explicit_revise_answer_unchanged": 1,
                "gate_explicit_revise_answer_changed": 0,
                "gate_explicit_revise_after_correct_labeled_pairs": 1,
                "gate_explicit_revise_labeled_unchanged": 1,
                "gate_explicit_revise_labeled_changed_still_wrong": 0,
                "gate_explicit_revise_repaired": 0,
                "gate_explicit_revise_changed_after_correct_labeled": 0,
            },
            {
                "protocol": "broadcast",
                "strict_explicit_revise_pre_wrong_actor_pairs": 0,
                "strict_explicit_revise_answer_unchanged": 0,
                "strict_explicit_revise_answer_changed": 0,
                "strict_explicit_revise_after_correct_labeled_pairs": 0,
                "strict_explicit_revise_labeled_unchanged": 0,
                "strict_explicit_revise_labeled_changed_still_wrong": 0,
                "strict_explicit_revise_repaired": 0,
                "strict_explicit_revise_changed_after_correct_labeled": 0,
                "gate_explicit_revise_pre_wrong_actor_pairs": 3,
                "gate_explicit_revise_answer_unchanged": 0,
                "gate_explicit_revise_answer_changed": 3,
                "gate_explicit_revise_after_correct_labeled_pairs": 2,
                "gate_explicit_revise_labeled_unchanged": 0,
                "gate_explicit_revise_labeled_changed_still_wrong": 1,
                "gate_explicit_revise_repaired": 1,
                "gate_explicit_revise_changed_after_correct_labeled": 2,
            },
        ]
        compact = _compact_matrix_revise_summary(rows)
        per_strict = compact[0]
        self.assertEqual(per_strict["matrix_cells"], 2)
        self.assertEqual(
            per_strict["explicit_revise_pre_wrong_actor_pairs"], 7
        )
        self.assertEqual(per_strict["answer_changed"], 5)
        self.assertEqual(per_strict["change_response_rate"], 5 / 7)
        self.assertEqual(per_strict["after_correct_labeled_pairs"], 5)
        self.assertEqual(
            per_strict["after_correct_label_coverage_over_uptake"], 5 / 7
        )
        self.assertEqual(
            per_strict["after_correct_label_coverage_over_changed"], 3 / 5
        )
        self.assertEqual(
            per_strict["repair_rate_over_after_correct_labeled_pairs"], 2 / 5
        )
        broadcast_gate = compact[3]
        self.assertEqual(
            broadcast_gate["explicit_revise_pre_wrong_actor_pairs"], 3
        )
        self.assertEqual(broadcast_gate["change_response_rate"], 1.0)
        self.assertEqual(
            broadcast_gate["repair_rate_over_after_correct_labeled_pairs"],
            1 / 2,
        )
        self.assertEqual(
            broadcast_gate["review_action_definition"],
            REVISE_ACTION_DEFINITION,
        )
        self.assertIn("reviewer-issued", REVISE_ACTION_DEFINITION)
        self.assertIn("not an independently evaluator-verified", REVISE_ACTION_DEFINITION)

    def test_per_route_sensitivity_includes_only_safe_source_strata(
        self,
    ) -> None:
        counts: Counter[str] = Counter()
        strict_pre_wrong = {
            "protocol": "per",
            "strict_pre_gate_eligible": True,
            "primary_eligibility_bucket": "strict_pre_gate",
            "before_correct": False,
        }
        included = [
            {
                **strict_pre_wrong,
                "review_action": "revise",
                "review_route_observed": False,
                "answer_changed": False,
                "after_correct": False,
                "transition_outcome": "unchanged",
            },
            {
                **strict_pre_wrong,
                "review_action": "unknown",
                "review_route_observed": True,
                "answer_changed": True,
                "after_correct": True,
                "transition_outcome": "repaired",
            },
            {
                **strict_pre_wrong,
                "review_action": "revise",
                "review_route_observed": True,
                "answer_changed": True,
                "after_correct": None,
                "transition_outcome": "changed_correctness_unlabeled",
            },
        ]
        for row in included:
            _update_per_route_sensitivity_counts(counts, row)

        ignored = [
            {
                **strict_pre_wrong,
                "review_action": "agree",
                "review_route_observed": True,
                "answer_changed": True,
                "after_correct": True,
                "transition_outcome": "repaired",
            },
            {
                **strict_pre_wrong,
                "review_action": "unknown",
                "review_route_observed": False,
                "answer_changed": True,
                "after_correct": True,
                "transition_outcome": "repaired",
            },
            {
                **strict_pre_wrong,
                "protocol": "broadcast",
                "review_action": "unknown",
                "review_route_observed": True,
                "answer_changed": True,
                "after_correct": True,
                "transition_outcome": "repaired",
            },
            {
                **strict_pre_wrong,
                "strict_pre_gate_eligible": False,
                "review_action": "unknown",
                "review_route_observed": True,
                "answer_changed": True,
                "after_correct": True,
                "transition_outcome": "repaired",
            },
        ]
        for row in ignored:
            _update_per_route_sensitivity_counts(counts, row)

        key = ("dataset", "slice", "actor", "per")
        summaries = {key: {}}
        _update_correctness_denominators(summaries, {key: counts})
        summary = summaries[key]
        self.assertEqual(
            summary["per_route_sensitivity_pre_wrong_actor_pairs"], 3
        )
        self.assertEqual(
            summary["per_route_sensitivity_explicit_revise_pairs"], 2
        )
        self.assertEqual(
            summary["per_route_sensitivity_unknown_linked_route_pairs"], 1
        )
        self.assertEqual(
            summary["per_route_sensitivity_answer_unchanged"], 1
        )
        self.assertEqual(
            summary["per_route_sensitivity_answer_changed"], 2
        )
        self.assertEqual(
            summary["per_route_sensitivity_after_correct_labeled_pairs"], 2
        )
        self.assertEqual(
            summary["per_route_sensitivity_labeled_unchanged"], 1
        )
        self.assertEqual(
            summary["per_route_sensitivity_labeled_changed_still_wrong"], 0
        )
        self.assertEqual(summary["per_route_sensitivity_repaired"], 1)
        self.assertEqual(
            summary[
                "per_route_sensitivity_changed_after_correct_labeled"
            ],
            1,
        )
        self.assertEqual(
            summary["per_route_sensitivity_change_response_rate"], 2 / 3
        )
        self.assertEqual(
            summary[
                "per_route_sensitivity_after_correct_label_coverage_over_uptake"
            ],
            2 / 3,
        )
        self.assertEqual(
            summary[
                "per_route_sensitivity_after_correct_label_coverage_over_changed"
            ],
            1 / 2,
        )
        self.assertEqual(
            summary[
                "per_route_sensitivity_repair_rate_over_after_correct_labeled_pairs"
            ],
            1 / 2,
        )

    def test_compact_per_route_sensitivity_pools_per_counts_only(
        self,
    ) -> None:
        def row(protocol: str, **updates: int) -> dict:
            result = {
                "protocol": protocol,
                **{field: 0 for field in PER_ROUTE_SENSITIVITY_COUNT_FIELDS},
            }
            result.update(updates)
            return result

        rows = [
            row(
                "per",
                per_route_sensitivity_pre_wrong_actor_pairs=3,
                per_route_sensitivity_explicit_revise_pairs=2,
                per_route_sensitivity_unknown_linked_route_pairs=1,
                per_route_sensitivity_answer_unchanged=1,
                per_route_sensitivity_answer_changed=2,
                per_route_sensitivity_after_correct_labeled_pairs=2,
                per_route_sensitivity_labeled_unchanged=1,
                per_route_sensitivity_labeled_changed_still_wrong=0,
                per_route_sensitivity_repaired=1,
                per_route_sensitivity_changed_after_correct_labeled=1,
            ),
            row(
                "per",
                per_route_sensitivity_pre_wrong_actor_pairs=2,
                per_route_sensitivity_explicit_revise_pairs=1,
                per_route_sensitivity_unknown_linked_route_pairs=1,
                per_route_sensitivity_answer_unchanged=0,
                per_route_sensitivity_answer_changed=2,
                per_route_sensitivity_after_correct_labeled_pairs=1,
                per_route_sensitivity_labeled_unchanged=0,
                per_route_sensitivity_labeled_changed_still_wrong=1,
                per_route_sensitivity_repaired=0,
                per_route_sensitivity_changed_after_correct_labeled=1,
            ),
            row(
                "broadcast",
                per_route_sensitivity_pre_wrong_actor_pairs=99,
                per_route_sensitivity_answer_changed=99,
            ),
        ]
        compact = _compact_matrix_per_route_sensitivity(rows)
        self.assertEqual(compact["matrix_cells"], 2)
        self.assertEqual(compact["cells_with_pre_wrong_actor_pairs"], 2)
        self.assertEqual(
            compact["per_route_sensitivity_pre_wrong_actor_pairs"], 5
        )
        self.assertEqual(
            compact["per_route_sensitivity_explicit_revise_pairs"], 3
        )
        self.assertEqual(
            compact["per_route_sensitivity_unknown_linked_route_pairs"], 2
        )
        self.assertEqual(compact["change_response_rate"], 4 / 5)
        self.assertEqual(
            compact["after_correct_label_coverage_over_uptake"], 3 / 5
        )
        self.assertEqual(
            compact["after_correct_label_coverage_over_changed"], 2 / 4
        )
        self.assertEqual(
            compact["repair_rate_over_after_correct_labeled_pairs"], 1 / 3
        )
        self.assertEqual(
            compact["sensitivity_definition"],
            PER_ROUTE_SENSITIVITY_DEFINITION,
        )
        self.assertIs(compact["semantic_review_action_overridden"], False)
        self.assertIs(
            compact["critique_stance_or_usefulness_claim_allowed"], False
        )


if __name__ == "__main__":
    unittest.main()
