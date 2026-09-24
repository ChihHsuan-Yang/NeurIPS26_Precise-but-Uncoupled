from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, TYPE_CHECKING

import yaml
from agentverse.logging import logger

# RELEASE NOTE (NeurIPS26_Precise-but-Uncoupled): upstream AgentVerse optionally
# imported BMTools here for its tool-using SIMULATION agents. BMTools was present
# in the source repository only as a DANGLING git submodule reference (a gitlink
# with no .gitmodules), so it could not be checked out, and no protocol in this
# paper uses tools. The import and its startup warning were removed; load_tools()
# below now raises a clear error if a config ever asks for a tool.

from agentverse.llms import llm_registry


def load_llm(llm_config: Dict):
    llm_type = llm_config.pop("llm_type", "text-davinci-003")

    return llm_registry.build(llm_type, **llm_config)


def load_tools(tool_config: List[Dict]):
    if len(tool_config) == 0:
        return []
    raise NotImplementedError(
        "Tool-using agents are not part of the 'Precise but Uncoupled' release. "
        "Upstream AgentVerse loaded them through BMTools, which this release does "
        "not vendor (it was a dangling submodule reference). No paper protocol "
        "uses tools; remove `tools` from your config."
    )


from agentverse.agents import agent_registry
from agentverse.environments import BaseEnvironment, env_registry
from agentverse.memory import memory_registry
from agentverse.memory_manipulator import memory_manipulator_registry

from agentverse.output_parser import output_parser_registry

if TYPE_CHECKING:
    from agentverse.agents import BaseAgent


def load_memory(memory_config: Dict):
    memory_type = memory_config.pop("memory_type", "chat_history")
    return memory_registry.build(memory_type, **memory_config)


def load_memory_manipulator(memory_manipulator_config: Dict):
    memory_manipulator_type = memory_manipulator_config.pop(
        "memory_manipulator_type", "basic"
    )
    return memory_manipulator_registry.build(
        memory_manipulator_type, **memory_manipulator_config
    )


def load_environment(env_config: Dict) -> BaseEnvironment:
    env_type = env_config.pop("env_type", "basic")
    return env_registry.build(env_type, **env_config)


def load_agent(agent_config: Dict) -> BaseAgent:
    agent_type = agent_config.pop("agent_type", "conversation")
    agent = agent_registry.build(agent_type, **agent_config)
    return agent


def prepare_task_config(task, tasks_dir):
    """Read the yaml config of the given task in `tasks` directory."""
    all_task_dir = tasks_dir
    task_path = os.path.join(all_task_dir, task)
    config_path = os.path.join(task_path, "config.yaml")
    if not os.path.exists(task_path):
        all_tasks = []
        for task in os.listdir(all_task_dir):
            if (
                os.path.isdir(os.path.join(all_task_dir, task))
                and task != "__pycache__"
            ):
                all_tasks.append(task)
                for subtask in os.listdir(os.path.join(all_task_dir, task)):
                    if (
                        os.path.isdir(os.path.join(all_task_dir, task, subtask))
                        and subtask != "__pycache__"
                    ):
                        all_tasks.append(f"{task}/{subtask}")
        raise ValueError(f"Task {task} not found. Available tasks: {all_tasks}")
    if not os.path.exists(config_path):
        raise ValueError(
            "You should include the config.yaml file in the task directory"
        )
    return prepare_task_config_from_path(config_path, default_output_parser_name=task)


def prepare_task_config_from_path(
    config_path: str,
    *,
    default_output_parser_name: str = "dummy",
):
    """Read a task config from an explicit config.yaml path."""

    task_config = yaml.safe_load(open(config_path))
    task_config = _apply_prompt_preset(task_config, config_path=Path(config_path))
    return _prepare_loaded_task_config(
        task_config,
        default_output_parser_name=default_output_parser_name,
    )


def _resolve_prompt_preset_path(raw_path: str, *, config_path: Path) -> Path:
    candidate = Path(str(raw_path)).expanduser()
    if candidate.is_absolute():
        return candidate

    local = (config_path.parent / candidate).resolve()
    if local.exists():
        return local

    repo_root = Path(__file__).resolve().parents[1]
    return (repo_root / candidate).resolve()


def _flatten_prompt_mapping(payload: Dict[str, Any]) -> Dict[str, str]:
    prompts = payload.get("prompts", payload)
    if not isinstance(prompts, dict):
        raise ValueError("Prompt preset must contain a mapping or a top-level `prompts` mapping.")

    flattened: Dict[str, str] = {}

    def visit(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                next_prefix = f"{prefix}.{key}" if prefix else str(key)
                visit(next_prefix, child)
            return
        flattened[prefix] = "" if value is None else str(value)

    visit("", prompts)
    return flattened


def _load_prompt_preset(task_config: Dict[str, Any], *, config_path: Path) -> Dict[str, str]:
    preset_spec = task_config.get("prompt_preset") or task_config.get("prompt_preset_path")
    if not preset_spec:
        return {}

    if isinstance(preset_spec, dict):
        raw_path = preset_spec.get("path")
    else:
        raw_path = preset_spec
    if not raw_path:
        raise ValueError("`prompt_preset` must be a path string or a mapping with `path`.")

    preset_path = _resolve_prompt_preset_path(str(raw_path), config_path=config_path)
    if not preset_path.exists():
        raise FileNotFoundError(f"Prompt preset does not exist: {preset_path}")
    preset_payload = yaml.safe_load(preset_path.read_text(encoding="utf-8")) or {}
    if not isinstance(preset_payload, dict):
        raise ValueError(f"Prompt preset must be a mapping: {preset_path}")
    return _flatten_prompt_mapping(preset_payload)


def _resolve_prompt_refs(value: Any, prompts: Dict[str, str]) -> Any:
    if isinstance(value, dict) and set(value.keys()) == {"prompt_ref"}:
        ref = str(value["prompt_ref"])
        if ref not in prompts:
            raise KeyError(f"Prompt reference {ref!r} was not found in prompt preset.")
        return prompts[ref]
    if isinstance(value, str) and value.startswith("prompt_ref:"):
        ref = value.split(":", 1)[1].strip()
        if ref not in prompts:
            raise KeyError(f"Prompt reference {ref!r} was not found in prompt preset.")
        return prompts[ref]
    if isinstance(value, dict):
        return {key: _resolve_prompt_refs(child, prompts) for key, child in value.items()}
    if isinstance(value, list):
        return [_resolve_prompt_refs(child, prompts) for child in value]
    return value


def _apply_prompt_preset(task_config: Dict[str, Any], *, config_path: Path) -> Dict[str, Any]:
    if not isinstance(task_config, dict):
        return task_config
    prompts = _load_prompt_preset(task_config, config_path=config_path)
    if not prompts:
        return task_config
    return _resolve_prompt_refs(task_config, prompts)


def _prepare_loaded_task_config(
    task_config: Dict,
    *,
    default_output_parser_name: str,
):
    if not isinstance(task_config, dict):
        raise ValueError("Task config must be a mapping.")

    for i, agent_configs in enumerate(task_config["agents"]):
        memory_cfg = agent_configs.get("memory", {})
        # Pass the agent's LLM model name to memory for summarization calls
        llm_model = (agent_configs.get("llm") or {}).get("model")
        if llm_model and memory_cfg.get("has_summary") and "summary_model" not in memory_cfg:
            memory_cfg["summary_model"] = llm_model
        agent_configs["memory"] = load_memory(memory_cfg)
        if agent_configs.get("tool_memory", None) is not None:
            agent_configs["tool_memory"] = load_memory(agent_configs["tool_memory"])
        llm = load_llm(agent_configs.get("llm", "text-davinci-003"))
        agent_configs["llm"] = llm

        memory_manipulator = load_memory_manipulator(
            agent_configs.get("memory_manipulator", {})
        )
        agent_configs["memory_manipulator"] = memory_manipulator

        agent_configs["tools"] = load_tools(agent_configs.get("tools", []))

        # Build the output parser
        output_parser_config = agent_configs.get("output_parser", {"type": "dummy"})
        if output_parser_config.get("type", None) == "role_assigner":
            output_parser_config["cnt_critic_agents"] = task_config.get(
                "cnt_critic_agents", 0
            )
        output_parser_name = output_parser_config.pop("type", default_output_parser_name)
        agent_configs["output_parser"] = output_parser_registry.build(
            output_parser_name, **output_parser_config
        )

    return task_config
