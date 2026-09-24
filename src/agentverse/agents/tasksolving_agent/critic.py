# AgentVerse/agentverse/agents/tasksolving_agent/critic.py
from __future__ import annotations

import json
import re
from colorama import Fore
from agentverse.logging import get_logger
import bdb
from string import Template
from typing import TYPE_CHECKING, List, Union

from agentverse.message import Message
from agentverse.evaluation.common import extract_boxed

from agentverse.agents import agent_registry
from agentverse.agents.base import BaseAgent, format_fatal_auth_error, is_fatal_auth_error
from agentverse.utils import AgentCriticism
from agentverse.message import CriticMessage

logger = get_logger()


def _extract_boxed_answer(text: str) -> str:
    return extract_boxed(text or "")


def _build_empty_review_fallback(preliminary_solution: str, advice: str) -> str:
    boxed = _extract_boxed_answer(preliminary_solution or "")
    boxed_line = f"\\boxed{{{boxed}}}" if boxed else "\\boxed{}"

    diagnosis_lines = [
        "- Reviewer generation returned empty or invalid content.",
        "- A safe reviewer verdict could not be recovered from the model response.",
    ]
    fix_lines = [
        "- Re-issue a concise review and restate exactly one boxed final answer.",
        "- If the current solution looks correct, preserve the boxed answer; otherwise revise it before resubmitting.",
    ]

    if "FAIL" in (advice or "") or "Score: False" in (advice or ""):
        fix_lines.append("- Address the evaluator failure before submitting another final answer.")

    return (
        "Diagnosis:\n"
        + "\n".join(diagnosis_lines)
        + "\nFix Instruction:\n"
        + "\n".join(fix_lines)
        + "\nFinal Answer For Evaluator:\n"
        + boxed_line
        + "\n[Route:Executor]"
    )


@agent_registry.register("critic")
class CriticAgent(BaseAgent):
    max_history: int = 3
    tools: List[dict] = []
    tool_names: List[str] = []
    tool_descriptions: str = ""

    def __init__(self, *args, **kwargs):
        tool_config_file = kwargs.pop("tool_config", "")
        tools = []
        tool_names = []
        tool_descriptions = ""
        if tool_config_file != "":
            try:
                with open(tool_config_file, "r") as f:
                    tools_dict = json.load(f)
                tools = tools_dict["tools_json"]
                tool_names = [t["name"] for t in tools]
                tool_descriptions = "\n".join(
                    [f"- {t['name']}: " + t["description"] for t in tools]
                )
                kwargs.update({"tools": tools})
                kwargs.update({"tool_names": tool_names})
                kwargs.update({"tool_descriptions": tool_descriptions})
            except Exception as e:
                logger.error(e)
                logger.warn("Failed to load tool config file.")
        super().__init__(
            *args,
            **kwargs,
        )

    def step(self, env_description: str = "") -> CriticMessage:
        pass

    async def astep(
        self,
        preliminary_solution: str,
        advice: str = "No advice yet.",
        task_description: str = "",
        all_roles: str = "",
        **kwargs,
    ) -> CriticMessage:
        """Asynchronous version of step"""
        logger.debug("", self.name, Fore.MAGENTA)
        prepend_prompt, append_prompt, prompt_token = self.get_all_prompts(
            preliminary_solution=preliminary_solution,
            advice=advice,
            task_description=task_description,
            role_description=self.role_description,
            agent_name=self.name,
            all_roles=all_roles,
            # tool_names=self.tool_names,
            tool_descriptions=self.tool_descriptions,
        )

        max_send_token = self.history_token_budget(prompt_token)

        history = await self.memory.to_messages(
            self.name,
            start_index=self.history_start_index(),
            max_send_token=max_send_token,
            model=self.llm.args.model,
        )
        self.log_prompt_budget(
            stage=str(kwargs.get("debug_prompt_stage", "reviewer")),
            prompt_token=prompt_token,
            history=history,
            extra_debug={
                "failed_attempt_ledger_size": int(
                    kwargs.get("debug_failed_attempt_memory_size", 0) or 0
                ),
                "revision_ledger_size": int(
                    kwargs.get("debug_candidate_revision_memory_size", 0) or 0
                ),
                "private_note_count": 0,
                "speak_history_count": 0,
            },
        )
        parsed_response: Union[AgentCriticism, None] = None
        last_raw_response: str = ""

        for i in range(self.max_retry):
            try:
                response = await self.llm.agenerate_response(
                    prepend_prompt, history, append_prompt
                )
                if isinstance(response, str):
                    last_raw_response = response
                else:
                    last_raw_response = (getattr(response, "content", "") or "").strip()
                parsed_response = self.output_parser.parse(response)
                break
            except (KeyboardInterrupt, bdb.BdbQuit):
                raise
            except Exception as e:
                if is_fatal_auth_error(e):
                    logger.error(format_fatal_auth_error(e))
                    raise RuntimeError(format_fatal_auth_error(e)) from e
                logger.error(e)
                logger.warn("Retrying...")
                continue

        if parsed_response is None:
            logger.error(f"{self.name} failed to generate valid response.")

        # IMPORTANT: if parser returns empty criticism (often when [Agree]),
        # keep the raw model output so trace/logger doesn't drop the reviewer turn.
        content = ""
        is_agree = False
        if parsed_response is not None:
            is_agree = bool(parsed_response.is_agree)
            content = (parsed_response.criticism or "").strip()

        if not content:
            content = (last_raw_response or "").strip()
        if not content:
            content = _build_empty_review_fallback(preliminary_solution, advice)

        message = CriticMessage(
            content=content,
            criticism=content,
            sender=self.name,
            sender_agent=self,
            is_agree=is_agree,
        )
        return message

    def _fill_prompt_template(
        self, preliminary_solution: str, advice: str, task_description: str
    ) -> str:
        """Fill the placeholders in the prompt template

        In the conversation agent, three placeholders are supported:
        - ${role_description}
        - ${task_description}
        - ${preliminary_solution}
        - ${advice}
        """
        input_arguments = {
            "role_description": self.role_description,
            "task_description": task_description,
            "preliminary_solution": preliminary_solution,
            "advice": advice,
        }
        return Template(self.prompt_template).safe_substitute(input_arguments)

    def add_message_to_memory(self, messages: List[Message]) -> None:
        self.memory.add_message(messages)

    def reset(self) -> None:
        """Reset the agent"""
        self.memory.reset()
        # TODO: reset receiver
