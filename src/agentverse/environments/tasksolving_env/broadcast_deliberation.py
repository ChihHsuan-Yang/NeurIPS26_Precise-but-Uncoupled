from __future__ import annotations

import os
from enum import Enum
from typing import Any, Dict, List, Union

from agentverse.agents.base import BaseAgent
from agentverse.environments import BaseEnvironment
from agentverse.environments.tasksolving_env.rules.broadcast_deliberation import (
    BroadcastDeliberationRule,
)

from .. import env_registry as EnvironmentRegistry


@EnvironmentRegistry.register("task-broadcast-deliberation")
class BroadcastDeliberationEnvironment(BaseEnvironment):
    class Config:
        arbitrary_types_allowed = True

    rule: BroadcastDeliberationRule
    agents: Dict[Enum | str, Union[BaseAgent, List[BaseAgent]]] = None

    task_description: str = ""
    ground_truth: str = ""
    reference_solution: str = ""

    cnt_turn: int = 0
    max_rounds: int = 1
    success: bool = False
    result_output_path: str = ""
    live_output_path: str = ""

    def __init__(self, **kwargs):
        max_rounds = int(kwargs.pop("max_rounds", 1))
        result_output_path = str(kwargs.get("result_output_path", "") or "")
        rule_config = kwargs.pop("rule", {}) or {}
        evaluator_config = dict(rule_config.pop("evaluator", {}) or {})
        rule = BroadcastDeliberationRule(
            evaluator_config=evaluator_config,
            **rule_config,
        )
        super().__init__(rule=rule, **kwargs)
        object.__setattr__(self, "max_rounds", max_rounds)
        object.__setattr__(self, "result_output_path", result_output_path)
        object.__setattr__(self, "live_output_path", self._derive_live_output_path())

    def _derive_live_output_path(self) -> str:
        result_output_path = str(self.result_output_path or "").strip()
        if not result_output_path:
            return ""
        result_dir = os.path.dirname(os.path.dirname(result_output_path))
        base_name = os.path.splitext(os.path.basename(result_output_path))[0]
        return os.path.join(
            result_dir,
            "live_records",
            f"{base_name}.broadcast_live.txt",
        )

    def _initialize_live_log(self) -> None:
        live_output_path = str(self.live_output_path or "").strip()
        if not live_output_path:
            return
        os.makedirs(os.path.dirname(live_output_path), exist_ok=True)
        with open(live_output_path, "w", encoding="utf-8") as f:
            f.write("Broadcast-Deliberation Live Log\n")
            f.write(f"Task Description:\n{self.task_description or '[empty]'}\n\n")
            f.write(f"Ground Truth:\n{self.ground_truth or '[empty]'}\n\n")
            f.write(f"Final Result File:\n{self.result_output_path or '[unset]'}\n")
            f.write("=" * 80 + "\n")

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
            live_log_path=self.live_output_path,
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
        object.__setattr__(self, "live_output_path", self._derive_live_output_path())
        self._initialize_live_log()
        for agent in self._iter_agent_objects():
            agent.reset()
