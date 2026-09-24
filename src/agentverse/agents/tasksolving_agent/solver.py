# AgentVerse/agentverse/agents/tasksolving_agent/solver.py
from __future__ import annotations

import asyncio
from colorama import Fore

from agentverse.logging import get_logger
import bdb
from string import Template
from typing import TYPE_CHECKING, List, Tuple

# from agentverse.environments import PipelineEnvironment
from agentverse.message import SolverMessage, Message, CriticMessage

from agentverse.agents import agent_registry
from agentverse.agents.base import BaseAgent, format_fatal_auth_error, is_fatal_auth_error
from agentverse.utils import AgentCriticism

logger = get_logger()


@agent_registry.register("solver")
class SolverAgent(BaseAgent):
    max_history: int = 5

    def step(
        self, former_solution: str, advice: str, task_description: str = "", **kwargs
    ) -> SolverMessage:
        pass

    async def astep(
        self, former_solution: str, advice: str, task_description: str = "", **kwargs
    ) -> SolverMessage:
        """Asynchronous version of step"""
        logger.debug("", self.name, Fore.MAGENTA)
        # prompt = self._fill_prompt_template(
        #     former_solution, critic_opinions, advice, task_description
        # )
        prepend_prompt, append_prompt, prompt_token = self.get_all_prompts(
            former_solution=former_solution,
            task_description=task_description,
            advice=advice,
            role_description=self.role_description,
            **kwargs,
        )

        max_send_token = self.history_token_budget(prompt_token)

        history = await self.memory.to_messages(
            self.name,
            start_index=self.history_start_index(),
            max_send_token=max_send_token,
            model=self.llm.args.model,
        )
        self.log_prompt_budget(
            stage=str(kwargs.get("debug_prompt_stage", kwargs.get("phase", "solver"))),
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
        parsed_response = None
        for i in range(self.max_retry):
            try:
                response = await self.llm.agenerate_response(
                    prepend_prompt, history, append_prompt
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

        message = SolverMessage(
            content=""
            if parsed_response is None
            else parsed_response.return_values["output"],
            sender=self.name,
            sender_agent=self,          # <-- ADD THIS (important)
            receiver=self.get_receiver(),
        )
        return message

    def _fill_prompt_template(
        self,
        former_solution: str,
        critic_opinions: List[AgentCriticism],
        advice: str,
        task_description: str,
    ) -> str:
        """Fill the placeholders in the prompt template

        In the role_assigner agent, three placeholders are supported:
        - ${task_description}
        - ${former_solution}
        - ${critic_messages}
        - ${advice}
        """
        input_arguments = {
            "task_description": task_description,
            "former_solution": former_solution,
            "critic_opinions": "\n".join(
                [
                    f"{critic.sender_agent.role_description} said: {critic.criticism}"
                    for critic in critic_opinions
                ]
            ),
            "advice": advice,
        }
        # if discussion_mode:
        #     template = Template(self.prompt_template[1])
        # else:
        template = Template(self.prompt_template)
        return template.safe_substitute(input_arguments)

    def add_message_to_memory(self, messages: List[Message]) -> None:
        self.memory.add_message(messages)

    def reset(self) -> None:
        """Reset the agent"""
        self.memory.reset()
        # TODO: reset receiver
