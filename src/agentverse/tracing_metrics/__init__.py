"""Backward-compatible shim for the renamed metrics package.

New code should import from ``agentverse.metrics``.
"""

from agentverse.metrics import *  # noqa: F401,F403
from agentverse.metrics import trace_state
