from __future__ import annotations

from enum import Enum
from typing import Dict, List, Union

from agentverse.agents.base import BaseAgent
from agentverse.environments import BaseEnvironment
from agentverse.environments.tasksolving_env.rules.single_agent_reflect import (
    SingleAgentReflectRule,
    prepare_single_agent_reflect_rule_kwargs,
)

from .. import env_registry as EnvironmentRegistry


@EnvironmentRegistry.register("task-single-agent-reflect")
class SingleAgentReflectEnvironment(BaseEnvironment):
    class Config:
        arbitrary_types_allowed = True

    rule: SingleAgentReflectRule
    agents: Dict[Enum, Union[BaseAgent, List[BaseAgent]]] = None

    task_description: str = ""
    ground_truth: str = ""
    reference_solution: str = ""

    cnt_turn: int = 0
    max_rounds: int = 1
    success: bool = False

    def __init__(self, **kwargs):
        max_rounds = int(kwargs.pop("max_rounds", 1))
        rule_config = kwargs.pop("rule", {}) or {}
        rule = SingleAgentReflectRule(
            **prepare_single_agent_reflect_rule_kwargs(rule_config)
        )
        super().__init__(rule=rule, **kwargs)
        object.__setattr__(self, "max_rounds", max_rounds)

    async def step(
        self,
        advice: str = "No advice yet.",
        previous_plan: str = "No solution yet.",
    ):
        result, advice, previous_plan, logs, success = await self.rule.astep(
            task_description=self.task_description,
            agents=self.agents,
            advice=advice,
            previous_plan=previous_plan,
            ground_truth=self.ground_truth,
            reference_solution=self.reference_solution,
            round_id=self.cnt_turn,
            live_log_sink=self.live_log_sink,
        )
        self.cnt_turn += 1
        self.success = bool(success)
        return result, advice, previous_plan, logs, self.success

    def is_done(self) -> bool:
        return self.success or (self.cnt_turn >= self.max_rounds)

    def set_task_description(self, task_description: str = ""):
        self.task_description = task_description

    def set_ground_truth(self, ground_truth: str = ""):
        self.ground_truth = str(ground_truth)

    def set_reference_solution(self, reference_solution: str = ""):
        self.reference_solution = str(reference_solution or "")

    def reset(self) -> None:
        self.cnt_turn = 0
        self.success = False
        self.rule.reset()
        for agent in self._iter_agent_objects():
            agent.reset()
