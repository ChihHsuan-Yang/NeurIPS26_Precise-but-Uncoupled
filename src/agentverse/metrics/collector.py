"""Core metrics extraction from structured logs and agent token counters.

Supports two collection modes:
1. Structured logs (tasksolving): call extract_from_logs() after agentverse.run()
2. Incremental (simulation): record_message() is called by trace_state for each message
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from agentverse.evaluation.common import extract_boxed
from agentverse.metrics.models import (
    AgentMetrics,
    ConfidencePollRecord,
    ExampleMetrics,
    RoundMetrics,
    TokenUsage,
)

# Patterns for reviewer routing detection
_AGREE_RE = re.compile(r"\[Agree\]", re.IGNORECASE)
_ROUTE_RE = re.compile(r"\[Route:\s*\w+\]", re.IGNORECASE)
_SELF_REVISE_RE = re.compile(r"\[(?:Self-Revise|Revise)\]", re.IGNORECASE)
_SUBMIT_RE = re.compile(r"\[Submit\]", re.IGNORECASE)
_SCORE_RE = re.compile(r"\bScore:\s*(True|False)\b", re.IGNORECASE)
_EVAL_SIGNAL_RE = re.compile(r"\bEvaluation signal:\s*(PASS|FAIL)\b", re.IGNORECASE)
_VERIFIER_RE = re.compile(r"\bVerifier:\s*(PASS|FAIL)\b", re.IGNORECASE)


class MetricsCollector:
    """Collects metrics for a single example."""

    def __init__(self, example_idx: int, input_text: str, label: str):
        self.example_idx = example_idx
        self.input_text = input_text
        self.label = label

        # Per-agent message counts
        self._agent_messages: dict[str, int] = defaultdict(int)
        # Per-round → per-agent message counts
        self._round_agent_messages: dict[int, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        # Per-round final evaluation result within one outer/system try.
        self._round_eval: dict[int, bool | None] = {}
        self._evaluation_attempts: list[bool] = []
        # Reviewer behavior
        self._reviewer_agrees = 0
        self._reviewer_reroutes = 0
        # Correctness
        self._correct: int | None = None
        self._first_round_correct = False
        # Predicted answer
        self._predicted_answer = ""
        # Rounds seen
        self._rounds_seen: set[int] = set()
        # Total messages
        self._total_messages = 0
        # Agent token data (populated by snapshot_agent_tokens)
        self._agent_tokens: dict[str, TokenUsage] = {}
        self._agent_models: dict[str, str] = {}
        self._agent_invocations: dict[str, int] = {}
        # Confidence polls (broadcast-deliberation)
        self._confidence_scores: list[ConfidencePollRecord] = []

    # ------------------------------------------------------------------
    # Mode 1: Parse structured logs (tasksolving environments)
    # ------------------------------------------------------------------

    def extract_from_logs(self, logs: list[dict[str, Any]]) -> None:
        """Parse the structured log dicts returned by environment.step().

        Works for task-basic, task-per, broadcast-deliberation, and
        single-agent-reflect style log formats.
        """
        if not logs:
            return

        last_executor_content = ""

        for entry in logs:
            if not isinstance(entry, dict):
                continue

            entry_type = entry.get("type", "")
            round_id = entry.get("round", 0)
            stage = entry.get("stage", "")
            sender = entry.get("sender", "")
            content = entry.get("content", "")

            self._rounds_seen.add(round_id)

            # Count actual agent utterances, including broadcast confidence polls.
            is_poll = str(stage).startswith("poll_")
            if (entry_type == "message" or is_poll) and sender and sender != "system":
                self._agent_messages[sender] += 1
                self._round_agent_messages[round_id][sender] += 1
                self._total_messages += 1

            # Track last executor output for predicted answer extraction
            if entry_type == "message" and stage.startswith("executor"):
                last_executor_content = content
            # For basic env, the execution stage is "execution"
            if entry_type == "message" and stage == "execution" and sender != "system":
                last_executor_content = content

            # Detect evaluation results
            if entry_type == "message" and stage.startswith("evaluation"):
                passed = _parse_evaluation_passed(content)
                if passed is None:
                    continue
                self._evaluation_attempts.append(passed)
                self._round_eval[round_id] = passed
            if str(stage).startswith("poll_"):
                discussion_turn = entry.get("discussion_turn")
                if discussion_turn is None:
                    try:
                        discussion_turn = int(str(stage).split("_", 1)[1])
                    except Exception:
                        discussion_turn = 0
                score = entry.get("score")
                if score is None:
                    score = _extract_prefixed_int(content, "score", default=1)
                try:
                    score = int(score)
                except Exception:
                    score = 1
                self._confidence_scores.append(
                    ConfidencePollRecord(
                        outer_round_id=round_id,
                        discussion_turn=int(discussion_turn),
                        agent_name=str(sender),
                        score=max(1, min(score, 100)),
                        reason=str(entry.get("reason", "") or ""),
                        intent=str(entry.get("intent", "") or ""),
                        candidate_answer=str(entry.get("candidate_answer", "") or ""),
                    )
                )

            if stage == "evaluation_result":
                final_answer = str(entry.get("final_answer", "") or "").strip()
                if final_answer:
                    self._predicted_answer = final_answer

            # Detect system accept/reject (final correctness)
            if stage == "system" and sender == "system":
                if "Good score! Accept!" in content:
                    self._correct = 1
                elif "Bad score! Reject!" in content:
                    self._correct = 0

            # Detect reviewer behavior patterns
            if entry_type == "message" and (
                "reviewer" in stage.lower() or "critic" in sender.lower()
                or "self_review" in stage.lower()
            ):
                if _AGREE_RE.search(content):
                    self._reviewer_agrees += 1
                if _ROUTE_RE.search(content) or _SELF_REVISE_RE.search(content):
                    self._reviewer_reroutes += 1

        # Extract predicted answer from last executor output
        if not self._predicted_answer:
            self._predicted_answer = _extract_boxed(last_executor_content) or ""

    # ------------------------------------------------------------------
    # Mode 2: Incremental recording (simulation environments)
    # ------------------------------------------------------------------

    def record_message(self, sender: str, turn_idx: int) -> None:
        """Record a single message (called by trace_state for simulations)."""
        if not sender:
            return
        self._agent_messages[sender] += 1
        self._rounds_seen.add(turn_idx)
        self._round_agent_messages[turn_idx][sender] += 1
        self._total_messages += 1

    # ------------------------------------------------------------------
    # Token snapshot from agent LLM objects
    # ------------------------------------------------------------------

    def snapshot_agent_tokens(self, environment: Any) -> None:
        """Read cumulative token counts from each agent's LLM.

        Since TaskSolving.from_task() creates fresh agents per example,
        the counters represent exactly this example's usage.
        """
        iter_fn = getattr(environment, "_iter_agent_objects", None)
        if iter_fn is None:
            return

        for agent in iter_fn():
            name = getattr(agent, "name", None) or agent.__class__.__name__
            llm = getattr(agent, "llm", None)
            if llm is None:
                continue
            memory = getattr(agent, "memory", None)

            prompt = getattr(llm, "total_prompt_tokens", 0)
            completion = getattr(llm, "total_completion_tokens", 0)
            if memory is not None:
                prompt += int(getattr(memory, "summary_prompt_tokens", 0) or 0)
                completion += int(getattr(memory, "summary_completion_tokens", 0) or 0)
            self._agent_tokens[name] = TokenUsage(
                prompt_tokens=prompt,
                completion_tokens=completion,
                total_tokens=prompt + completion,
            )
            self._agent_models[name] = getattr(llm, "model", "")
            self._agent_invocations[name] = int(
                getattr(llm, "total_request_count", 0) or 0
            ) + int(getattr(memory, "summary_request_count", 0) or 0)

    # ------------------------------------------------------------------
    # Finalize
    # ------------------------------------------------------------------

    def finalize(self, wall_time: float = 0.0) -> ExampleMetrics:
        """Produce the final ExampleMetrics."""
        # Build per-agent metrics
        all_agent_names = set(self._agent_messages.keys()) | set(
            self._agent_tokens.keys()
        )
        agent_metrics: dict[str, AgentMetrics] = {}
        total_tokens = TokenUsage()

        for name in sorted(all_agent_names):
            tokens = self._agent_tokens.get(name, TokenUsage())
            total_tokens += tokens
            agent_metrics[name] = AgentMetrics(
                agent_name=name,
                model=self._agent_models.get(name, ""),
                tokens=tokens,
                message_count=self._agent_messages.get(name, 0),
                invocation_count=self._agent_invocations.get(
                    name, self._agent_messages.get(name, 0)
                ),
            )

        total_model_calls = sum(
            metrics.invocation_count for metrics in agent_metrics.values()
        )

        # Build per-round metrics
        round_metrics: list[RoundMetrics] = []
        for rid in sorted(self._rounds_seen):
            round_metrics.append(
                RoundMetrics(
                    round_id=rid,
                    evaluation_passed=self._round_eval.get(rid),
                    agent_messages=dict(self._round_agent_messages.get(rid, {})),
                )
            )

        system_try_results = [
            self._round_eval.get(rid) is True for rid in sorted(self._rounds_seen)
        ]
        first_system_try_correct = system_try_results[0] if system_try_results else False
        first_success_system_try = _first_true_index(system_try_results)
        first_success_attempt = first_success_system_try

        # If correctness was never explicitly set (e.g. simulation), default to -1
        correct = self._correct if self._correct is not None else -1

        return ExampleMetrics(
            example_idx=self.example_idx,
            input_text=self.input_text,
            label=self.label,
            predicted_answer=self._predicted_answer,
            correct=correct,
            total_tokens=total_tokens,
            total_messages=self._total_messages,
            total_rounds=len(self._rounds_seen),
            system_try_count=len(self._rounds_seen),
            first_system_try_correct=first_system_try_correct,
            first_success_system_try=first_success_system_try,
            system_try_results=system_try_results,
            evaluator_attempt_count=len(self._evaluation_attempts),
            first_success_attempt=first_success_attempt,
            evaluation_attempt_results=list(self._evaluation_attempts),
            total_model_calls=total_model_calls,
            agent_metrics=agent_metrics,
            round_metrics=round_metrics,
            confidence_poll_count=len(self._confidence_scores),
            confidence_scores=self._confidence_scores,
            wall_time_seconds=round(wall_time, 3),
            first_round_correct=first_system_try_correct,
            reviewer_agree_count=self._reviewer_agrees,
            reviewer_reroute_count=self._reviewer_reroutes,
        )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _extract_boxed(text: str) -> str:
    """Extract the last \\boxed{...} answer from text."""
    return extract_boxed(text or "")


def _extract_prefixed_int(text: str, field: str, default: int = 0) -> int:
    match = re.search(rf"{re.escape(field)}\s*=\s*(-?\d+)", text or "")
    if not match:
        return default
    try:
        return int(match.group(1))
    except Exception:
        return default


def _parse_evaluation_passed(content: str) -> bool | None:
    text = str(content or "")
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


def _first_true_index(values: list[bool]) -> int | None:
    for idx, value in enumerate(values, start=1):
        if value is True:
            return idx
    return None
