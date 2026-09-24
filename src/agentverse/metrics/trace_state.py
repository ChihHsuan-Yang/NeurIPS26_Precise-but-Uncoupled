"""Global trace state with metrics collector support.

Duplicates agentverse/tracing/trace_state.py and extends log_selected_messages()
to also feed messages into the active MetricsCollector (for simulation environments
that don't return structured logs).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from agentverse.metrics.trace_logger import MetricsTraceLogger

if TYPE_CHECKING:
    from agentverse.message import Message
    from agentverse.metrics.collector import MetricsCollector


@dataclass
class TraceState:
    trace: MetricsTraceLogger
    name_to_A: dict[str, str]
    collector: MetricsCollector | None = None


_STATE: TraceState | None = None


def enable_trace(trace: MetricsTraceLogger, name_to_A: dict[str, str]) -> None:
    global _STATE
    _STATE = TraceState(trace=trace, name_to_A=name_to_A)


def disable_trace() -> None:
    global _STATE
    _STATE = None


def is_enabled() -> bool:
    return _STATE is not None


def set_collector(collector: MetricsCollector) -> None:
    """Attach a metrics collector to the global trace state."""
    if _STATE is not None:
        _STATE.collector = collector


def clear_collector() -> None:
    """Detach the active metrics collector."""
    if _STATE is not None:
        _STATE.collector = None


def log_selected_messages(turn_idx: int, selected_messages: list[Message]) -> None:
    """Log accepted messages to trace file and feed into metrics collector.

    This is the hook used by simulation environments (which don't return
    structured logs). Tasksolving environments use MetricsCollector.extract_from_logs()
    instead, so the collector hook here is optional.
    """
    if _STATE is None:
        return

    for m in selected_messages:
        if m is None:
            continue
        sender = getattr(m, "sender", "")
        content = (getattr(m, "content", "") or "").strip()
        if not sender or not content:
            continue

        # Write to plain-text trace
        A = _STATE.name_to_A.get(sender)
        if A:
            _STATE.trace.log_turn(agent_number=A, text=content, turn_idx=turn_idx)

        # Feed into metrics collector (incremental mode for simulations)
        if _STATE.collector is not None:
            _STATE.collector.record_message(sender, turn_idx)
