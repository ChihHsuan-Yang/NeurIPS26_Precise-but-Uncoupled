# from .agent import Agent
from agentverse.registry import Registry

agent_registry = Registry(name="AgentRegistry")


from .base import BaseAgent
# RELEASE NOTE: simulation-half agents (conversation/broadcast/tool/reflection/
# prisoner_dilemma) removed; this paper uses only tasksolving agents.

from agentverse.agents.tasksolving_agent.role_assigner import RoleAssignerAgent
from agentverse.agents.tasksolving_agent.critic import CriticAgent
from agentverse.agents.tasksolving_agent.deliberator import DeliberationAgent
from agentverse.agents.tasksolving_agent.evaluator import EvaluatorAgent
from agentverse.agents.tasksolving_agent.solver import SolverAgent
from agentverse.agents.tasksolving_agent.manager import ManagerAgent
from agentverse.agents.tasksolving_agent.executor import ExecutorAgent
from agentverse.agents.repeater import RepeaterAgent
