"""Aggregate per-example metrics into a run-level summary."""

from __future__ import annotations

from agentverse.metrics.models import (
    AgentMetrics,
    ExampleMetrics,
    RunSummary,
    TokenUsage,
)


class RunAggregator:
    """Accumulates ExampleMetrics and computes RunSummary."""

    def __init__(self, run_id: str, task: str, model: str = ""):
        self.run_id = run_id
        self.task = task
        self.model = model
        self._examples: list[ExampleMetrics] = []

    def add_example(self, metrics: ExampleMetrics) -> None:
        self._examples.append(metrics)

    def summarize(self) -> RunSummary:
        n = len(self._examples)
        if n == 0:
            return RunSummary(run_id=self.run_id, task=self.task, model=self.model)

        # Correctness (only count examples where correct is 0 or 1, skip -1)
        scored = [e for e in self._examples if e.correct >= 0]
        accuracy = sum(e.correct for e in scored) / len(scored) if scored else 0.0
        first_round_acc = (
            sum(1 for e in scored if _first_system_try_correct(e)) / len(scored)
            if scored
            else 0.0
        )

        # Totals and averages
        total_tokens = TokenUsage()
        total_rounds = 0
        total_messages = 0
        total_model_calls = 0
        total_evaluator_attempts = 0
        total_confidence_polls = 0
        confidence_score_sum = 0

        for e in self._examples:
            total_tokens += e.total_tokens
            total_rounds += e.total_rounds
            total_messages += e.total_messages
            total_model_calls += e.total_model_calls
            total_evaluator_attempts += e.evaluator_attempt_count
            total_confidence_polls += e.confidence_poll_count
            confidence_score_sum += sum(item.score for item in e.confidence_scores)

        avg_tokens = TokenUsage(
            prompt_tokens=total_tokens.prompt_tokens // n,
            completion_tokens=total_tokens.completion_tokens // n,
            total_tokens=total_tokens.total_tokens // n,
        )

        # Per-agent aggregation
        per_agent: dict[str, AgentMetrics] = {}
        for e in self._examples:
            for name, am in e.agent_metrics.items():
                if name not in per_agent:
                    per_agent[name] = AgentMetrics(
                        agent_name=name,
                        model=am.model,
                    )
                per_agent[name].tokens += am.tokens
                per_agent[name].message_count += am.message_count
                per_agent[name].invocation_count += am.invocation_count

        success_attempts = [
            _first_success_system_try(e)
            for e in scored
            if _first_success_system_try(e) is not None
        ]
        max_attempt = max(
            [_system_try_count(e) for e in scored] + success_attempts + [1]
        )
        pass_at_attempt = {
            str(k): round(
                sum(
                    1
                    for e in scored
                    if _first_success_system_try(e) is not None
                    and _first_success_system_try(e) <= k
                )
                / len(scored),
                4,
            )
            for k in range(1, max_attempt + 1)
        } if scored else {}
        solved_by_attempt_count: dict[str, int] = {}
        for attempt in success_attempts:
            key = str(attempt)
            solved_by_attempt_count[key] = solved_by_attempt_count.get(key, 0) + 1

        return RunSummary(
            run_id=self.run_id,
            task=self.task,
            model=self.model,
            total_examples=n,
            accuracy=accuracy,
            first_round_accuracy=first_round_acc,
            total_tokens=total_tokens,
            avg_rounds=total_rounds / n,
            avg_messages=total_messages / n,
            avg_tokens=avg_tokens,
            total_model_calls=total_model_calls,
            avg_model_calls=total_model_calls / n,
            avg_evaluator_attempts=total_evaluator_attempts / n,
            avg_attempts_to_success=(
                sum(success_attempts) / len(success_attempts)
                if success_attempts
                else 0.0
            ),
            pass_at_attempt=pass_at_attempt,
            solved_by_attempt_count=solved_by_attempt_count,
            total_confidence_polls=total_confidence_polls,
            avg_confidence_score=(
                confidence_score_sum / total_confidence_polls if total_confidence_polls else 0.0
            ),
            per_agent_summary=per_agent,
        )


def _system_try_count(metrics: ExampleMetrics) -> int:
    if int(getattr(metrics, "system_try_count", 0) or 0) > 0:
        return int(metrics.system_try_count)
    if getattr(metrics, "round_metrics", None):
        return len(metrics.round_metrics)
    return int(getattr(metrics, "total_rounds", 0) or 0)


def _first_system_try_correct(metrics: ExampleMetrics) -> bool:
    if hasattr(metrics, "first_system_try_correct"):
        return bool(getattr(metrics, "first_system_try_correct"))
    return bool(getattr(metrics, "first_round_correct", False))


def _first_success_system_try(metrics: ExampleMetrics) -> int | None:
    value = getattr(metrics, "first_success_system_try", None)
    if value is not None:
        return int(value)
    return getattr(metrics, "first_success_attempt", None)
