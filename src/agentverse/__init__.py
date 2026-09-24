"""Top-level AgentVerse package with lazy exports.

RELEASE NOTE (NeurIPS26_Precise-but-Uncoupled): the AgentVerse *simulation*
half is not part of this paper. Its lazy exports (order/describer/selector/
updater/visibility registries and ``Simulation``) were removed here; the
task-solving exports below are unchanged. See docs/PROVENANCE.md.

Historically this module imported most of the runtime stack eagerly, which made
lightweight tooling such as offline trace analysis pull in optional dependencies
like model backends. The package-level API is preserved here, but the actual
imports only happen when an attribute is first accessed.
"""

from __future__ import annotations

from importlib import import_module


_LAZY_EXPORTS = {
    "output_parser_registry": ("agentverse.output_parser", "output_parser_registry"),
    "env_registry": ("agentverse.environments", "env_registry"),
    "decision_maker_registry": (
        "agentverse.environments.tasksolving_env.rules.decision_maker",
        "decision_maker_registry",
    ),
    "evaluator_registry": (
        "agentverse.environments.tasksolving_env.rules.evaluator",
        "evaluator_registry",
    ),
    "executor_registry": (
        "agentverse.environments.tasksolving_env.rules.executor",
        "executor_registry",
    ),
    "role_assigner_registry": (
        "agentverse.environments.tasksolving_env.rules.role_assigner",
        "role_assigner_registry",
    ),
    "TaskSolving": ("agentverse.tasksolving", "TaskSolving"),
    "prepare_task_config": ("agentverse.initialization", "prepare_task_config"),
    "load_agent": ("agentverse.initialization", "load_agent"),
    "load_environment": ("agentverse.initialization", "load_environment"),
    "load_llm": ("agentverse.initialization", "load_llm"),
    "load_memory": ("agentverse.initialization", "load_memory"),
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _LAZY_EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
