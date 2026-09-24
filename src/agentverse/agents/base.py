import json
import logging
from abc import abstractmethod
from typing import List, NamedTuple, Optional, Set, Union
from string import Template

from pydantic import BaseModel, Field
from agentverse.llms import BaseLLM

from agentverse.logging import logger
from agentverse.llms.utils import count_message_tokens, count_string_tokens
from agentverse.memory import BaseMemory, ChatHistoryMemory
from agentverse.message import Message
from agentverse.output_parser import OutputParser
from agentverse.memory_manipulator import BaseMemoryManipulator


def is_fatal_auth_error(error: Exception) -> bool:
    raw = str(error or "").lower()
    markers = (
        "token introspection",
        "token is either not active or invalid",
        "authenticationerror",
        "error code: 401",
        "status code: 401",
        "permission denied from internal policies",
        "high-assurance timeout",
    )
    return any(marker in raw for marker in markers)


def format_fatal_auth_error(error: Exception) -> str:
    return (
        "ALCF authentication/session failure detected. "
        "This should not be retried as a normal model error. "
        "Please re-authenticate with the ALCF endpoint, then rerun the command. "
        f"Original error: {error}"
    )


class BaseAgent(BaseModel):
    name: str
    llm: BaseLLM
    output_parser: OutputParser
    prepend_prompt_template: str = Field(default="")
    append_prompt_template: str = Field(default="")
    first_attempt_prepend_prompt_template: str = Field(default="")
    first_attempt_append_prompt_template: str = Field(default="")
    prompt_template: str = Field(default="")
    role_description: str = Field(default="")
    memory: BaseMemory = Field(default_factory=ChatHistoryMemory)
    memory_manipulator: BaseMemoryManipulator = Field(
        default_factory=BaseMemoryManipulator
    )
    max_retry: int = Field(default=3)
    receiver: Set[str] = Field(default=set({"all"}))
    async_mode: bool = Field(default=True)

    @abstractmethod
    def step(self, env_description: str = "") -> Message:
        """Get one step response"""
        pass

    @abstractmethod
    def astep(self, env_description: str = "") -> Message:
        """Asynchronous version of step"""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the agent"""
        pass

    @abstractmethod
    def add_message_to_memory(self, messages: List[Message]) -> None:
        """Add a message to the memory"""
        pass

    def get_spend(self) -> float:
        return self.llm.get_spend()

    def get_spend_formatted(self) -> str:
        two_trailing = f"${self.get_spend():.2f}"
        if two_trailing == "$0.00":
            return f"${self.get_spend():.6f}"
        return two_trailing

    def get_all_prompts(
        self,
        use_first_attempt_prompt: bool = False,
        prepend_prompt_template_override: Optional[str] = None,
        append_prompt_template_override: Optional[str] = None,
        **kwargs,
    ):
        """Render one request without mutating the agent's prompt templates.

        Request-scoped overrides are needed by concurrent protocols that reuse an
        agent object for multiple prompt modes. Mutating the model fields around an
        ``await`` lets another coroutine observe the temporary templates.
        """
        prepend_template = (
            self.prepend_prompt_template
            if prepend_prompt_template_override is None
            else prepend_prompt_template_override
        )
        append_template = (
            self.append_prompt_template
            if append_prompt_template_override is None
            else append_prompt_template_override
        )
        if use_first_attempt_prompt:
            prepend_template = (
                self.first_attempt_prepend_prompt_template
                or prepend_template
            )
            append_template = (
                self.first_attempt_append_prompt_template
                or append_template
            )

        prepend_prompt = Template(prepend_template).safe_substitute(
            **kwargs
        )
        append_prompt = Template(append_template).safe_substitute(**kwargs)

        # TODO: self.llm.args.model is not generalizable
        num_prepend_prompt_token = count_string_tokens(
            prepend_prompt, self.llm.args.model
        )
        num_append_prompt_token = count_string_tokens(
            append_prompt, self.llm.args.model
        )

        # Safeguard: some backends / models may not support token counting,
        # which returns None. Treat those as 0 to avoid crashes.
        if num_prepend_prompt_token is None:
            num_prepend_prompt_token = 0
        if num_append_prompt_token is None:
            num_append_prompt_token = 0



        return (
            prepend_prompt,
            append_prompt,
            num_prepend_prompt_token + num_append_prompt_token,
        )

    def get_receiver(self) -> Set[str]:
        return self.receiver

    def history_start_index(self) -> int:
        max_history = getattr(self, "max_history", None)
        if max_history is None:
            return 0
        try:
            value = int(max_history)
        except (TypeError, ValueError):
            return 0
        return 0 if value <= 0 else -value

    def history_token_budget(self, prompt_token: int) -> int:
        llm_args = getattr(getattr(self, "llm", None), "args", None)
        model_name = str(getattr(llm_args, "model", "") or "")
        client_args = getattr(getattr(self, "llm", None), "client_args", {}) or {}
        model_limit = int(
            self.llm.send_token_limit(
                model_name,
                base_url=client_args.get("base_url"),
            )
        )
        requested_max_tokens = int(getattr(llm_args, "max_tokens", 0) or 0)
        if model_name in {
            "gpt-oss-120b",
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "gpt-oss-120b-131072",
        }:
            safety_margin = 2048
        else:
            safety_margin = 256
        return max(0, model_limit - int(prompt_token or 0) - requested_max_tokens - safety_margin)

    def log_prompt_budget(
        self,
        *,
        stage: str,
        prompt_token: int,
        history: List[dict],
        extra_debug: dict | None = None,
    ) -> None:
        model = getattr(getattr(self, "llm", None), "args", None)
        model_name = getattr(model, "model", "")
        try:
            history_tokens = count_message_tokens(history, model_name) if history else 0
        except Exception:
            history_tokens = -1

        summary_text = str(getattr(self.memory, "summary", "") or "")
        try:
            summary_tokens = (
                count_string_tokens(summary_text, model_name) if summary_text else 0
            )
        except Exception:
            summary_tokens = -1

        recent_history_count = 0
        for message in history or []:
            if not isinstance(message, dict):
                continue
            content = str(message.get("content", "") or "")
            role = str(message.get("role", "") or "")
            if role == "system" and content.startswith(
                "This reminds you of these events from your past:"
            ):
                continue
            recent_history_count += 1

        payload = {
            "agent": self.name,
            "stage": str(stage or "unknown"),
            "model": model_name,
            "estimated_prompt_tokens": (
                prompt_token + history_tokens if history_tokens >= 0 else prompt_token
            ),
            "prompt_template_tokens": prompt_token,
            "history_tokens": history_tokens,
            "summary_tokens": summary_tokens,
            "summary_length_tokens": summary_tokens,
            "summary_chars": len(summary_text),
            "summary_length_chars": len(summary_text),
            "recent_history_count": recent_history_count,
            "memory_has_summary": bool(getattr(self.memory, "has_summary", False)),
            "summary_keep_last_n_messages": int(
                getattr(self.memory, "summary_keep_last_n_messages", 0) or 0
            ),
            "summary_keep_recent_token_budget": int(
                getattr(self.memory, "summary_keep_recent_token_budget", 0) or 0
            ),
            "summary_update_every_n_messages": int(
                getattr(self.memory, "summary_update_every_n_messages", 0) or 0
            ),
        }
        if extra_debug:
            payload.update(extra_debug)
        logger.info(f"[PROMPT BUDGET] {json.dumps(payload, ensure_ascii=True)}")

    def set_receiver(self, receiver: Union[Set[str], str]) -> None:
        if isinstance(receiver, str):
            self.receiver = set({receiver})
        elif isinstance(receiver, set):
            self.receiver = receiver
        else:
            raise ValueError(
                "input argument `receiver` must be a string or a set of string"
            )

    def add_receiver(self, receiver: Union[Set[str], str]) -> None:
        if isinstance(receiver, str):
            self.receiver.add(receiver)
        elif isinstance(receiver, set):
            self.receiver = self.receiver.union(receiver)
        else:
            raise ValueError(
                "input argument `receiver` must be a string or a set of string"
            )

    def remove_receiver(self, receiver: Union[Set[str], str]) -> None:
        if isinstance(receiver, str):
            try:
                self.receiver.remove(receiver)
            except KeyError as e:
                logger.warn(f"Receiver {receiver} not found.")
        elif isinstance(receiver, set):
            self.receiver = self.receiver.difference(receiver)
        else:
            raise ValueError(
                "input argument `receiver` must be a string or a set of string"
            )
