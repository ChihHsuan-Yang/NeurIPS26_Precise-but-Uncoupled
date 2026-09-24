# AgentVerse/agentverse/environments/tasksolving_env/basic.py
import asyncio
from enum import Enum
from typing import Any, Dict, List, Tuple, Union

from colorama import Fore

from agentverse.environments import BaseEnvironment
from agentverse.agents.base import BaseAgent
from agentverse.logging import logger
from agentverse.message import Message, SolverMessage, ExecutorMessage
from agentverse.utils import AGENT_TYPES  # <-- ADD

from .. import env_registry as EnvironmentRegistry
from agentverse.environments.tasksolving_env.rules import TasksolvingRule


@EnvironmentRegistry.register("task-basic")
class BasicEnvironment(BaseEnvironment):
    rule: TasksolvingRule
    agents: Dict[Enum, Union[BaseAgent, List[BaseAgent]]] = None

    task_description: str
    ground_truth: str = ""
    reference_solution: str = ""

    cnt_turn: int = 0
    max_turns: int = 10
    success: bool = False

    def __init__(self, **kwargs):
        rule_config = kwargs.pop("rule", {})
        role_assigner_config = rule_config.pop(
            "role_assigner", {"type": "role_description"}
        )
        decision_maker_config = rule_config.pop("decision_maker", {"type": "vertical"})
        executor_config = rule_config.pop("executor", {"type": "none"})
        evaluator_config = rule_config.pop("evaluator", {"type": "basic"})
        rule = TasksolvingRule(
            role_assigner_config=role_assigner_config,
            decision_maker_config=decision_maker_config,
            executor_config=executor_config,
            evaluator_config=evaluator_config,
        )
        super().__init__(rule=rule, **kwargs)

    def _safe_agent_name(self, agent_or_agents) -> str:
        """Return a stable agent name for trace mapping."""
        try:
            if isinstance(agent_or_agents, list):
                # shouldn’t happen for singletons, but just in case
                return agent_or_agents[0].name if agent_or_agents else "unknown"
            return agent_or_agents.name
        except Exception:
            return "unknown"

    def _log_message_objects(
        self,
        logs: List[Dict[str, Any]],
        messages: List[Message],
        stage: str,
    ) -> None:
        """
        Record each message as its own event.
        IMPORTANT: use sender_agent.name (stable) rather than role_description (unstable).
        """
        if not messages:
            self._append_log_event(
                logs,
                {
                    "type": "message",
                    "round": self.cnt_turn,
                    "stage": stage,
                    "sender": "system",
                    "content": "[Silence]",
                },
            )
            return

        for m in messages:
            sender_agent = getattr(m, "sender_agent", None)
            if sender_agent is not None and getattr(sender_agent, "name", None):
                sender = sender_agent.name  # <-- critical fix
            else:
                sender = getattr(m, "sender", "unknown")

            content = getattr(m, "content", "")
            self._append_log_event(
                logs,
                {
                    "type": "message",
                    "round": self.cnt_turn,
                    "stage": stage,
                    "sender": sender,
                    "content": content,
                },
            )

    async def step(
        self, advice: str = "No advice yet.", previous_plan: str = "No solution yet."
    ) -> List[Message]:
        logs: List[Dict[str, Any]] = []
        logger.info(f"Loop Round {self.cnt_turn}")

        # ================== EXPERT RECRUITMENT ==================
        assigned_agents = await self.rule.role_assign(
            self.task_description, self.agents, self.cnt_turn, advice
        )
        description = "\n".join([a.role_description for a in assigned_agents])

        # Always include sender for summary logs
        role_assigner_name = self._safe_agent_name(self.agents[AGENT_TYPES.ROLE_ASSIGNMENT])
        self._append_log_event(
            logs,
            {
                "type": "summary",
                "round": self.cnt_turn,
                "stage": "role_assign",
                "sender": role_assigner_name,  # <-- critical fix
                "content": f"Role Assignment:\n{description}",
            },
        )
        logger.info("", f"Role Assignment:\n{description}", Fore.CYAN)
        # ================== EXPERT RECRUITMENT ==================

        # ================== DECISION MAKING ==================
        plan: List[SolverMessage] = await self.rule.decision_making(
            self.task_description, self.agents, previous_plan, advice
        )
        self._log_message_objects(logs, plan, stage="decision_making")

        flatten_plan = "\n".join([p.content for p in plan])
        planner_name = self._safe_agent_name(self.agents[AGENT_TYPES.SOLVER])
        self._append_log_event(
            logs,
            {
                "type": "summary",
                "round": self.cnt_turn,
                "stage": "decision_making",
                "sender": planner_name,  # <-- stable sender
                "content": f"Decision Plan:\n{flatten_plan}",
            },
        )
        logger.info("", f"Decision Plan:\n{flatten_plan}", Fore.YELLOW)
        # ================== DECISION MAKING ==================

        # ================== EXECUTION ==================
        result: List[ExecutorMessage] = await self.rule.execute(
            self.task_description, self.agents, plan
        )
        self._log_message_objects(logs, result, stage="execution")

        #flatten_result = "\n".join([r.content for r in result])
        flatten_result = "\n".join([r.content for r in result]).strip()
        if not flatten_result:
            flatten_result = "[No execution output]"
        executor_name = self._safe_agent_name(self.agents[AGENT_TYPES.EXECUTION])
        self._append_log_event(
            logs,
            {
                "type": "summary",
                "round": self.cnt_turn,
                "stage": "execution",
                "sender": executor_name,  # <-- stable sender
                "content": f"Execution Result:\n{flatten_result}",
            },
        )
        logger.info("", "Execution Result:", Fore.GREEN)
        logger.info("", flatten_result, Fore.GREEN)
        # ================== EXECUTION ==================

        # ================== EVALUATION ==================
        score, advice = await self.rule.evaluate(
            self.task_description,
            self.agents,
            plan,
            result,
            ground_truth=self.ground_truth,
            reference_solution=self.reference_solution,
        )

        evaluator_name = self._safe_agent_name(self.agents[AGENT_TYPES.EVALUATION])
        self._append_log_event(
            logs,
            {
                "type": "message",
                "round": self.cnt_turn,
                "stage": "evaluation",
                "sender": evaluator_name,  # <-- stable sender
                "content": f"Evaluation result: Score: {score}\nAdvice: {advice}",
            },
        )
        logger.info("", f"Evaluation result:\nScore: {score}\nAdvice: {advice}", Fore.YELLOW)

        if score is not None and (
            (isinstance(score, bool) and score is True)
            or (isinstance(score, (list, tuple)) and all([s >= 8 for s in score]))
            or (isinstance(score, int) and score == 1)
        ):
            self._append_log_event(
                logs,
                {
                    "type": "message",
                    "round": self.cnt_turn,
                    "stage": "system",
                    "sender": "system",
                    "content": "Good score! Accept!",
                },
            )
            self.success = True
        else:
            self._append_log_event(
                logs,
                {
                    "type": "message",
                    "round": self.cnt_turn,
                    "stage": "system",
                    "sender": "system",
                    "content": "Bad score! Reject!",
                },
            )

        self.cnt_turn += 1
        return flatten_result, advice, flatten_plan, logs, self.success

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
