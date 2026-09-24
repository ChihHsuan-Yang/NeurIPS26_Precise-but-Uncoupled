# AgentVerse/agentverse/environments/tasksolving_env/per.py
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Union

from agentverse.environments import BaseEnvironment
from agentverse.agents.base import BaseAgent
from agentverse.environments.tasksolving_env.rules.per import (
    PerRule,
    prepare_per_rule_kwargs,
)

from .. import env_registry as EnvironmentRegistry


@EnvironmentRegistry.register("task-per")
class PerEnvironment(BaseEnvironment):
    """
    PER Environment:
      Planner / Executor / Reviewer (LLMs)
      + up to `max_rounds` outer rounds
      + each outer round may include more than one evaluator submission
    """

    # --- Pydantic fix: allow arbitrary non-pydantic types like PerRule ---
    class Config:
        arbitrary_types_allowed = True

    rule: PerRule
    agents: Dict[Enum, Union[BaseAgent, List[BaseAgent]]] = None

    task_description: str = ""
    ground_truth: str = ""
    reference_solution: str = ""

    cnt_turn: int = 0
    max_rounds: int = 1  # number of outer rounds
    success: bool = False

    def __init__(self, **kwargs):
        # IMPORTANT: do NOT set self.xxx before super().__init__ (pydantic not ready yet)

        # max_rounds can be set from yaml: environment.max_rounds
        max_rounds = int(kwargs.pop("max_rounds", 1))

        rule_config = kwargs.pop("rule", {}) or {}
        rule = PerRule(**prepare_per_rule_kwargs(rule_config))

        # BaseEnvironment is a pydantic model; this call must happen before setting attrs
        super().__init__(rule=rule, **kwargs)

        # After pydantic init, safe to assign
        object.__setattr__(self, "max_rounds", max_rounds)

    async def step(
        self,
        advice: str = "No advice yet.",
        previous_plan: str = "No solution yet.",
    ):
        # Delegate main loop logic to PerRule (you'll implement PerRule.astep)
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
        # done when success OR the protocol has used max_rounds outer rounds
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
