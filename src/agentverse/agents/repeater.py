"""Repeater wrapper for agents in Shapley value experiments.

A RepeaterAgent wraps an existing agent and short-circuits astep()
to return the last received message (echo mode) without making any LLM call.
The pipeline sees a normal agent interface and operates unchanged.
"""
from __future__ import annotations

from typing import Any, List

from agentverse.message import (
    CriticMessage,
    ExecutorMessage,
    Message,
    SolverMessage,
)


class RepeaterAgent:
    """Wraps a real agent, echoing the last memory message instead of calling the LLM.

    Echo mode: returns the most recent message from memory (zero new information).
    For Reviewer (critic), no routing tag is appended — _parse_route() defaults
    to "executor", letting the inner loop run naturally.

    Attributes:
        _inner: The original agent being wrapped.
        _agent_type: One of 'solver', 'executor', 'critic'.
    """

    def __init__(self, inner: Any, agent_type: str):
        self._inner = inner
        self._agent_type = agent_type

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the wrapped agent."""
        return getattr(self._inner, name)

    async def astep(self, **kwargs) -> Message:
        """Return an echo of the last memory message without calling the LLM."""
        echo_content = self._get_echo_content(kwargs)

        if self._agent_type == "critic":
            return CriticMessage(
                content=echo_content,
                sender=self._inner.name,
                sender_agent=self._inner,
                is_agree=None,
            )
        elif self._agent_type == "solver":
            return SolverMessage(
                content=echo_content,
                sender=self._inner.name,
                sender_agent=self._inner,
                receiver=self._inner.get_receiver(),
            )
        elif self._agent_type == "executor":
            return ExecutorMessage(
                content=echo_content,
                sender=self._inner.name,
                sender_agent=self._inner,
            )
        else:
            raise ValueError(f"Unknown agent_type for repeater: {self._agent_type}")

    def _get_echo_content(self, kwargs: dict) -> str:
        """Get the last message from memory to echo. Falls back to task_description."""
        mem = getattr(self._inner, "memory", None)
        if mem is not None:
            msgs = getattr(mem, "messages", [])
            if msgs:
                content = getattr(msgs[-1], "content", "")
                if content:
                    return str(content)
        # First call or empty memory: echo task_description or former_solution
        return (
            kwargs.get("task_description", "")
            or kwargs.get("former_solution", "")
            or ""
        )

    def add_message_to_memory(self, messages: List[Message]) -> None:
        """Delegate to inner agent so chat history stays coherent for other agents."""
        self._inner.add_message_to_memory(messages)

    def reset(self) -> None:
        """Reset the inner agent."""
        self._inner.reset()
