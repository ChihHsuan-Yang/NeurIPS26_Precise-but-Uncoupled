# agentverse/tasksolving.py
import asyncio
import os
import copy
from typing import Any, Dict, Optional

import logging

from agentverse.environments.tasksolving_env.basic import BasicEnvironment
from agentverse.initialization import (
    load_agent,
    load_environment,
    prepare_task_config,
    prepare_task_config_from_path,
)
from agentverse.utils import AGENT_TYPES


openai_logger = logging.getLogger("openai")
openai_logger.setLevel(logging.WARNING)


class TaskSolving:
    environment: BasicEnvironment
    task: str = ""
    logs: list = []

    def __init__(
        self,
        environment: BasicEnvironment,
        task: str = "",
        result_output_path: Optional[str] = None,
    ):
        self.environment = environment
        self.task = task
        self.result_output_path = result_output_path

    @classmethod
    def from_task_with_repeaters(
        cls,
        task: str,
        tasks_dir: str,
        repeater_configs: dict[str, str],
        model_overrides: dict[str, dict] | None = None,
    ) -> "TaskSolving":
        """Build a TaskSolving with some agents replaced by echo repeaters.

        Args:
            task: Task name (e.g. "tasksolving/mgsm/alcf_per").
            tasks_dir: Root directory containing task configs.
            repeater_configs: Mapping of agent_name -> agent_type for agents
                to wrap as repeaters (e.g. {"Planner": "solver"}).
            model_overrides: Optional mapping of agent_name -> dict with keys
                like "model", "temperature", "max_tokens" to override on
                active (non-repeater) agents.
        """
        from agentverse.agents.repeater import RepeaterAgent

        instance = cls.from_task(task, tasks_dir)
        agents = instance.environment.agents

        def _apply_model_override(agent):
            if not model_overrides or agent.name not in model_overrides:
                return
            overrides = model_overrides[agent.name]
            if "model" in overrides:
                agent.llm.args.model = overrides["model"]
            if "temperature" in overrides:
                agent.llm.args.temperature = overrides["temperature"]
            if "max_tokens" in overrides:
                agent.llm.args.max_tokens = overrides["max_tokens"]

        for agent_type_enum, agent_obj in list(agents.items()):
            if isinstance(agent_obj, list):
                for a in agent_obj:
                    if a.name not in repeater_configs:
                        _apply_model_override(a)
                agents[agent_type_enum] = [
                    RepeaterAgent(a, repeater_configs[a.name])
                    if a.name in repeater_configs
                    else a
                    for a in agent_obj
                ]
            elif agent_obj is not None:
                if agent_obj.name in repeater_configs:
                    agents[agent_type_enum] = RepeaterAgent(
                        agent_obj, repeater_configs[agent_obj.name]
                    )
                else:
                    _apply_model_override(agent_obj)

        return instance

    @classmethod
    def from_task(
        cls,
        task: str,
        tasks_dir: str,
        env_overrides: Optional[Dict[str, Any]] = None,
        result_output_path: Optional[str] = None,
    ):
        """Build an AgentVerse from a task name.
        The task name should correspond to a directory in `tasks` directory.
        Then this method will load the configuration from the yaml file in that directory.
        """
        # Prepare the config of the task
        task_config = prepare_task_config(task, tasks_dir)

        # Build the environment
        env_config = copy.deepcopy(task_config["environment"])
        env_type = task_config.get("environment", {}).get("env_type", "") or task_config.get("environment", {}).get("type", "")

        # Build agents for all pipeline (task)
        agents = {}
        for i, agent_config in enumerate(task_config["agents"]):
            if agent_config.get("agent_type", "") == "deliberator":
                agents.setdefault("deliberators", []).append(load_agent(agent_config))
                continue

            if agent_config.get("agent_type", "") == "hypothesizer":
                agents.setdefault("hypothesizers", []).append(load_agent(agent_config))
                continue

            if agent_config.get("agent_type", "") == "integrator":
                agents["integrator"] = load_agent(agent_config)
                continue
            '''
            if agent_config.get("agent_type", "") == "critic":
                agent = load_agent(agent_config)
                agents[AGENT_TYPES.CRITIC] = [
                    copy.deepcopy(agent)
                    for _ in range(task_config.get("cnt_agents", 1) - 1)
                ]
                '''

            if agent_config.get("agent_type", "") == "critic":
                agent = load_agent(agent_config)

                # For PER we want exactly one reviewer (A3), not a critic group
                if env_type == "task-per":
                    agents[AGENT_TYPES.CRITIC] = agent
                else:
                    n = task_config.get("cnt_agents", 1)
                    agents[AGENT_TYPES.CRITIC] = [agent] + [copy.deepcopy(agent) for _ in range(n - 1)]

            else:
                agent_type = AGENT_TYPES.from_string(agent_config.get("agent_type", ""))
                agents[agent_type] = load_agent(agent_config)

        env_config["agents"] = agents

        env_config["task_description"] = task_config.get("task_description", "")
        env_config["max_rounds"] = task_config.get("max_rounds", 3)
        if result_output_path is not None:
            env_config["result_output_path"] = result_output_path

        if env_overrides:
            rule_overrides = env_overrides.get("rule", {}) or {}
            if rule_overrides:
                existing_rule = env_config.get("rule", {}) or {}
                evaluator_overrides = rule_overrides.get("evaluator", {}) or {}
                if evaluator_overrides:
                    merged_evaluator = dict(existing_rule.get("evaluator", {}) or {})
                    merged_evaluator.update(evaluator_overrides)
                    existing_rule["evaluator"] = merged_evaluator
                    rule_overrides = dict(rule_overrides)
                    rule_overrides.pop("evaluator", None)
                existing_rule.update(rule_overrides)
                env_config["rule"] = existing_rule

            for key, value in env_overrides.items():
                if key == "rule":
                    continue
                env_config[key] = value

        environment: BasicEnvironment = load_environment(env_config)

        return cls(
            environment=environment,
            task=task,
            result_output_path=result_output_path,
        )

    @classmethod
    def from_config_path(
        cls,
        config_path: str,
        *,
        task_name: str = "",
        env_overrides: Optional[Dict[str, Any]] = None,
        result_output_path: Optional[str] = None,
    ):
        """Build a TaskSolving pipeline directly from a config.yaml path."""

        task_config = prepare_task_config_from_path(
            config_path,
            default_output_parser_name=task_name or "dummy",
        )

        env_config = copy.deepcopy(task_config["environment"])
        env_type = task_config.get("environment", {}).get("env_type", "") or task_config.get(
            "environment", {}
        ).get("type", "")

        agents = {}
        for i, agent_config in enumerate(task_config["agents"]):
            if agent_config.get("agent_type", "") == "deliberator":
                agents.setdefault("deliberators", []).append(load_agent(agent_config))
                continue

            if agent_config.get("agent_type", "") == "hypothesizer":
                agents.setdefault("hypothesizers", []).append(load_agent(agent_config))
                continue

            if agent_config.get("agent_type", "") == "integrator":
                agents["integrator"] = load_agent(agent_config)
                continue

            if agent_config.get("agent_type", "") == "critic":
                agent = load_agent(agent_config)
                if env_type == "task-per":
                    agents[AGENT_TYPES.CRITIC] = agent
                else:
                    n = task_config.get("cnt_agents", 1)
                    agents[AGENT_TYPES.CRITIC] = [agent] + [
                        copy.deepcopy(agent) for _ in range(n - 1)
                    ]
            else:
                agent_type = AGENT_TYPES.from_string(agent_config.get("agent_type", ""))
                agents[agent_type] = load_agent(agent_config)

        env_config["agents"] = agents
        env_config["task_description"] = task_config.get("task_description", "")
        env_config["max_rounds"] = task_config.get("max_rounds", 3)
        if result_output_path is not None:
            env_config["result_output_path"] = result_output_path

        if env_overrides:
            rule_overrides = env_overrides.get("rule", {}) or {}
            if rule_overrides:
                existing_rule = env_config.get("rule", {}) or {}
                evaluator_overrides = rule_overrides.get("evaluator", {}) or {}
                if evaluator_overrides:
                    merged_evaluator = dict(existing_rule.get("evaluator", {}) or {})
                    merged_evaluator.update(evaluator_overrides)
                    existing_rule["evaluator"] = merged_evaluator
                    rule_overrides = dict(rule_overrides)
                    rule_overrides.pop("evaluator", None)
                existing_rule.update(rule_overrides)
                env_config["rule"] = existing_rule

            for key, value in env_overrides.items():
                if key == "rule":
                    continue
                env_config[key] = value

        environment: BasicEnvironment = load_environment(env_config)
        return cls(
            environment=environment,
            task=task_name or config_path,
            result_output_path=result_output_path,
        )

    def run(self):
        """Run the environment from scratch until it is done."""
        self.reset()
        self.logs = []
        advice = "No advice yet."
        previous_plan = "No solution yet."
        while not self.environment.is_done():
            result, advice, previous_plan, logs, success = asyncio.run(
                self.environment.step(advice, previous_plan)
            )
            self.logs += logs
        self.environment.report_metrics()
        self.save_result(previous_plan, result, self.environment.get_spend())
        return previous_plan, result, self.logs

    def singleagent_thinking(self, preliminary_solution, advice) -> str:
        preliminary_solution = self.environment.solve(
            former_solution=preliminary_solution,
            critic_opinions=[(self.environment.evaluator, advice)],
        )
        return preliminary_solution

    def reset(self):
        self.environment.reset()

    def prepare_example(
        self,
        *,
        task_description: str,
        ground_truth: str,
        reference_solution: str = "",
        result_output_path: Optional[str] = None,
    ) -> None:
        if result_output_path is not None:
            self.result_output_path = result_output_path
            if hasattr(self.environment, "result_output_path"):
                object.__setattr__(
                    self.environment,
                    "result_output_path",
                    str(result_output_path or ""),
                )
            if hasattr(self.environment, "_derive_live_output_path"):
                try:
                    object.__setattr__(
                        self.environment,
                        "live_output_path",
                        self.environment._derive_live_output_path(),
                    )
                except Exception:
                    pass

        # Isolate per-example resource accounting. The benchmark loop reuses one
        # TaskSolving across examples; LLM token/request counters only accumulate
        # (+=) and are never otherwise reset, so without this the per-example
        # metrics snapshot records cumulative within-worker values. See
        # rebuttal P0-E accounting audit.
        if hasattr(self.environment, "reset_token_counters"):
            self.environment.reset_token_counters()

        self.environment.set_task_description(task_description)
        self.environment.set_ground_truth(ground_truth)
        if hasattr(self.environment, "set_reference_solution"):
            self.environment.set_reference_solution(reference_solution)

    def save_result(self, plan: str, result: str, spend: float):
        """Save the result to the result file"""
        result_file_path = self.result_output_path or ("./results/" + self.task + ".txt")
        os.makedirs(os.path.dirname(result_file_path), exist_ok=True)
        with open(result_file_path, "w") as f:
            f.write("[Final Plan]\n" + plan + "\n\n")
            f.write("[Result]\n" + result)
            f.write(f"[Spent]\n${spend}")
