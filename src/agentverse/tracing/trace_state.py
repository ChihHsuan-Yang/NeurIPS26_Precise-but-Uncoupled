from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Any, List

# Import your logger class
from agentverse.tracing.trace_logger import SimulationTraceLogger
from agentverse.message import Message


@dataclass
class TraceState:
    trace: SimulationTraceLogger
    name_to_A: Dict[str, str]


_STATE: Optional[TraceState] = None


def enable_trace(trace: SimulationTraceLogger, name_to_A: Dict[str, str]) -> None:
    global _STATE
    _STATE = TraceState(trace=trace, name_to_A=name_to_A)


def disable_trace() -> None:
    global _STATE
    _STATE = None


def is_enabled() -> bool:
    return _STATE is not None


def log_selected_messages(turn_idx: int, selected_messages: List[Message]) -> None:
    """
    Log exactly the messages that the environment accepted this step.
    This is the clean "turn-based" trace, no stdout scraping involved.
    """

    if _STATE is None:
        print(1)
        return
    for m in selected_messages:
        if m is None:
            print(2)
            continue
        sender = getattr(m, "sender", "")
        content = (getattr(m, "content", "") or "").strip()

        if not sender or not content:
            print(3)
            continue
        A = _STATE.name_to_A.get(sender)
        if not A:
            print(4)
            _STATE.trace.log_turn(agent_number=int(sender)+1, text=content, turn_idx=turn_idx)

            continue
        print(5)
        _STATE.trace.log_turn(agent_number= A, text=content, turn_idx=turn_idx)

