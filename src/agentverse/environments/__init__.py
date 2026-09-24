# AgentVerse/agentverse/environments/__init__.py
from typing import Dict
from agentverse.registry import Registry


env_registry = Registry(name="EnvironmentRegistry")


from .base import BaseEnvironment, BaseRule

# from .basic import PipelineEnvironment
# RELEASE NOTE: the AgentVerse simulation-half environments (basic, reflection,
# pokemon, broadcast, prisoner_dilemma, sde_team, sde_team_given_tests) are not
# used by this paper and were removed from this release. See docs/PROVENANCE.md.

from .tasksolving_env.basic import BasicEnvironment
from .tasksolving_env.broadcast_deliberation import BroadcastDeliberationEnvironment  # noqa: F401
from .tasksolving_env.per import PerEnvironment  # noqa: F401
# RELEASE NOTE: DHD (task-diverse-hypothesis) is a separate, post-paper project; removed.
