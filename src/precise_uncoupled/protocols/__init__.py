"""The four protocol runtimes (Track B) -- facade over `agentverse.environments`.

Each paper protocol is one registered `env_type`. Importing this module registers
all four with the AgentVerse environment registry, which is what
`agentverse_command.benchmark` builds from.

  config dir                         env_type                        class
  configs/paper/baseline_llm         task-single-llm-benchmark       SingleLLMBenchmarkEnvironment
  configs/paper/single_agent_hint_llm task-single-agent-reflect      SingleAgentReflectEnvironment
  configs/paper/per_hint_llm         task-per                        PerEnvironment
  configs/paper/broadcast_hint_llm   task-broadcast-deliberation     BroadcastDeliberationEnvironment

TRACK B CAVEAT -- READ BEFORE COMPARING NUMBERS.
These runtimes are vendored at AgentVerse HEAD b4a2db6, which POST-DATES the
submitted run's base be9c47c9. Re-running them reproduces the experiment's
DESIGN, not the submitted run's exact outputs. The analysis modules are
byte-identical to be9c47c9; the protocol modules are not. See
docs/PROVENANCE.md "Track B is not byte-exact" and docs/LIMITATIONS.md.

There is also NO SEED: the paper ran at temperature 0, but a hosted
OpenAI-compatible endpoint exposes no seed and its runtime is not frozen. Two
independent executions of the same config differ. Measured envelope (P1A, N=195):
PER FinalPass range 0.51 pp, Broadcast FinalPass range 2.05 pp.
"""

from agentverse.environments.tasksolving_env.single_llm_benchmark import (  # noqa: F401
    SingleLLMBenchmarkEnvironment,
)
from agentverse.environments.tasksolving_env.single_agent_reflect import (  # noqa: F401
    SingleAgentReflectEnvironment,
)
from agentverse.environments.tasksolving_env.per import PerEnvironment  # noqa: F401
from agentverse.environments.tasksolving_env.broadcast_deliberation import (  # noqa: F401
    BroadcastDeliberationEnvironment,
)

#: config directory -> registered env_type, for docs and validation.
PAPER_PROTOCOLS = {
    "baseline_llm": "task-single-llm-benchmark",
    "single_agent_hint_llm": "task-single-agent-reflect",
    "per_hint_llm": "task-per",
    "broadcast_hint_llm": "task-broadcast-deliberation",
    "per_6_inner_rounds_hint_llm": "task-per",
    "per_ack_required_hint_llm": "task-per",
    "per_embedded_advice_hint_llm": "task-per",
}

__all__ = [
    "PAPER_PROTOCOLS",
    "BroadcastDeliberationEnvironment",
    "PerEnvironment",
    "SingleAgentReflectEnvironment",
    "SingleLLMBenchmarkEnvironment",
]
