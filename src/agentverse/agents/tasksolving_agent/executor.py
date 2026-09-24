# AgentVerse/agentverse/agents/tasksolving_agent/executor.py
from __future__ import annotations

from agentverse.logging import get_logger
from colorama import Fore
import bdb
from string import Template
from typing import TYPE_CHECKING, List, Any

from agentverse.message import ExecutorMessage, Message, SolverMessage
from agentverse.utils import AgentFinish, AgentAction

from agentverse.agents import agent_registry
from agentverse.agents.base import BaseAgent, format_fatal_auth_error, is_fatal_auth_error
import requests

logger = get_logger()


@agent_registry.register("executor")
class ExecutorAgent(BaseAgent):
    max_history: int = 5

    def step(
        self, task_description: str, solution: str, tools: List[dict] = [], **kwargs
    ) -> ExecutorMessage:
        pass

    async def astep(
        self, task_description: str, solution: str, tools: List[dict] = [], **kwargs
    ) -> ExecutorMessage:
        logger.debug("", self.name, Fore.MAGENTA)
        prepend_prompt, append_prompt, prompt_token = self.get_all_prompts(
            task_description=task_description,
            solution=solution,
            agent_name=self.name,
            **kwargs,
        )

        max_send_token = self.history_token_budget(prompt_token)

        disable_chat_history = bool(kwargs.get("disable_chat_history", False))
        history = []
        if not disable_chat_history:
            history = await self.memory.to_messages(
                self.name,
                start_index=self.history_start_index(),
                max_send_token=max_send_token,
                model=self.llm.args.model,
            )
        self.log_prompt_budget(
            stage=str(kwargs.get("debug_prompt_stage", "executor")),
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
                "chat_history_disabled": disable_chat_history,
            },
        )
        parsed_response = None
        for i in range(self.max_retry):
            try:
                response = await self.llm.agenerate_response(
                    prepend_prompt, history, append_prompt, tools
                )
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
            parsed_response = AgentAction(tool="", tool_input="", log="")
        if isinstance(parsed_response, AgentFinish):
            message = ExecutorMessage(
                content=parsed_response.return_values["output"],
                sender=self.name,
                sender_agent=self,
            )
        elif isinstance(parsed_response, AgentAction):
            message = ExecutorMessage(
                content=parsed_response.log,
                sender=self.name,
                sender_agent=self,
                tool_name=parsed_response.tool,
                tool_input=parsed_response.tool_input,
            )
        else:
            raise ValueError(
                f"Error response type: {type(parsed_response)}. Only support \
                    AgentFinish and AgentAction. Modify your output parser."
            )
        return message

    def add_message_to_memory(self, messages: List[Message]) -> None:
        self.memory.add_message(messages)

    def reset(self) -> None:
        """Reset the agent"""
        self.memory.reset()
        # TODO: reset receiver
