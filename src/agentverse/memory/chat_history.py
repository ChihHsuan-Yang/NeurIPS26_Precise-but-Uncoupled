# AgentVerse/agentverse/memory/chat_history.py
import json
import logging
import os
from openai import OpenAI
import copy
from typing import List, Optional, Tuple, Dict
import asyncio

from agentverse.message import Message, ExecutorMessage
from . import memory_registry
from .base import BaseMemory
from agentverse.llms.utils import count_message_tokens, count_string_tokens
from agentverse.llms import OpenAIChat
from agentverse.llms.openai import DEFAULT_CLIENT as openai_client
from pydantic import Field, PrivateAttr

@memory_registry.register("chat_history")
class ChatHistoryMemory(BaseMemory):
    messages: List[Message] = Field(default=[])
    has_summary: bool = False
    max_summary_tlength: int = 500
    summary_update_every_n_messages: int = 5
    summary_keep_last_n_messages: int = 3
    summary_keep_recent_token_budget: int = 200
    summary_preserve_latest_final_answer_message: bool = True
    last_trimmed_index: int = 0
    summary: str = ""
    summary_model: str = "gpt-3.5-turbo"
    summary_request_count: int = 0
    summary_prompt_tokens: int = 0
    summary_completion_tokens: int = 0
    _summary_job: Optional[asyncio.Task] = PrivateAttr(default=None)
    _summary_updates_disabled: bool = PrivateAttr(default=False)
    SUMMARIZATION_PROMPT: str = '''Your task is to create a concise running summary of actions and information results in the provided text, focusing on key and potentially important information to remember.

You will receive the current summary and your latest actions. Combine them, adding relevant key information from the latest development in 1st person past tense and keeping the summary concise.

Summary So Far:
"""
{summary}
"""

Latest Development:
"""
{new_events}
"""
'''

    @staticmethod
    def _coerce_summary_text(value: object) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            parts: List[str] = []
            for item in value:
                if isinstance(item, dict):
                    for key in ("text", "content", "value", "output_text"):
                        text = ChatHistoryMemory._coerce_summary_text(item.get(key))
                        if text:
                            parts.append(text)
                            break
                    else:
                        parts.append(str(item))
                else:
                    text = ChatHistoryMemory._coerce_summary_text(item)
                    if text:
                        parts.append(text)
            return "\n".join(part for part in parts if part).strip()
        if isinstance(value, dict):
            for key in ("text", "content", "value", "output_text"):
                text = ChatHistoryMemory._coerce_summary_text(value.get(key))
                if text:
                    return text
            try:
                return json.dumps(value, ensure_ascii=False, sort_keys=True)
            except Exception:
                return str(value)
        return str(value)


    def add_message(self, messages: List[Message]) -> None:
        for message in messages:
            self.messages.append(message)

        # Only summarize when summary mode is actually enabled.
        if self.has_summary:
            self._schedule_summary_update()

    def _summary_knobs(self) -> Tuple[int, int, int]:
        update_every = max(1, int(self.summary_update_every_n_messages or 1))
        keep_last_n = max(0, int(self.summary_keep_last_n_messages or 0))
        keep_recent_tokens = max(0, int(self.summary_keep_recent_token_budget or 0))
        return update_every, keep_last_n, keep_recent_tokens

    def _token_count_model(self) -> str:
        return str(self.summary_model or "gpt-3.5-turbo")

    def _message_token_length(self, message: Message, model: str) -> int:
        try:
            return int(
                count_message_tokens(
                    {
                        "role": "assistant",
                        "content": (
                            f"[{getattr(message, 'sender', '')}]: "
                            f"{getattr(message, 'content', '')}"
                        ).strip()
                    },
                    model,
                )
                or 0
            )
        except Exception:
            return 0

    def _message_has_final_answer_marker(self, message: Message) -> bool:
        content = str(getattr(message, "content", "") or "")
        lowered = content.lower()
        markers = (
            "\\boxed{",
            "final answer:",
            "final answer for evaluator:",
            "submitted answer:",
            "actual extracted final answer sent to evaluator:",
        )
        return any(marker in lowered for marker in markers)

    def _recent_tail_start_index(self, model: str) -> int:
        _, keep_last_n, keep_recent_tokens = self._summary_knobs()
        if not self.messages:
            return 0
        if keep_recent_tokens <= 0:
            return max(0, len(self.messages) - keep_last_n)

        running_tokens = 0
        start_index = len(self.messages)
        for idx in range(len(self.messages) - 1, -1, -1):
            token_count = self._message_token_length(self.messages[idx], model)
            if start_index == len(self.messages) or running_tokens + token_count <= keep_recent_tokens:
                running_tokens += token_count
                start_index = idx
                continue
            break
        return max(0, start_index)

    def _latest_final_answer_index(self) -> Optional[int]:
        if not bool(self.summary_preserve_latest_final_answer_message):
            return None
        for idx in range(len(self.messages) - 1, -1, -1):
            if self._message_has_final_answer_marker(self.messages[idx]):
                return idx
        return None

    def _summary_cutoff_index(self, model: Optional[str] = None) -> int:
        token_model = str(model or self._token_count_model())
        cutoff_index = self._recent_tail_start_index(token_model)
        latest_final_answer_index = self._latest_final_answer_index()
        if latest_final_answer_index is not None:
            cutoff_index = min(cutoff_index, latest_final_answer_index)
        return max(0, cutoff_index)

    def _messages_to_summary_events(self, messages: List[Message]) -> List[dict]:
        events: List[dict] = []
        for message in messages:
            sender = str(getattr(message, "sender", "") or "").strip() or "assistant"
            content = str(getattr(message, "content", "") or "").strip()
            tool_name = str(getattr(message, "tool_name", "") or "").strip()
            if not content and not tool_name:
                continue
            if getattr(message, "sender", "") == "function":
                events.append(
                    {
                        "role": "function",
                        "content": content,
                        "name": tool_name or "function",
                    }
                )
                continue
            if tool_name:
                content = (
                    f"[{sender}]: {content}\n"
                    f"Function call: {tool_name}"
                ).strip()
            else:
                content = f"[{sender}]: {content}".strip()
            events.append({"role": "assistant", "content": content})
        return events

    def _pending_summary_batch(
        self,
        *,
        force: bool = False,
    ) -> Tuple[List[Message], int]:
        update_every, _, keep_recent_tokens = self._summary_knobs()
        cutoff_index = self._summary_cutoff_index()
        if cutoff_index <= self.last_trimmed_index:
            return [], cutoff_index
        pending = self.messages[self.last_trimmed_index:cutoff_index]
        if keep_recent_tokens > 0:
            return pending, cutoff_index
        if not force and len(pending) < update_every:
            return [], cutoff_index
        return pending, cutoff_index

    def _schedule_summary_update(self) -> None:
        if not self.has_summary:
            return
        if self._summary_updates_disabled:
            return
        if self._summary_job is not None and not self._summary_job.done():
            return
        pending, cutoff_index = self._pending_summary_batch(force=False)
        if not pending:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._summary_job = loop.create_task(
            self._summary_task(
                events=self._messages_to_summary_events(pending),
                cutoff_index=cutoff_index,
            )
        )

    async def _summary_task(self, events: List[dict], cutoff_index: int):
        try:
            await self.update_running_summary(
                new_events=events,
                model=self.summary_model,
            )
            self.last_trimmed_index = max(self.last_trimmed_index, cutoff_index)
        except Exception as e:
            self._summary_updates_disabled = True
            print(
                "Summary update failed; disabling further summary updates for this worker:",
                e,
            )
        finally:
            self._summary_job = None
            self._schedule_summary_update()

    async def _ensure_summary_current(self, *, model: str) -> None:
        if not self.has_summary:
            return
        if self._summary_updates_disabled:
            return
        if self._summary_job is not None and not self._summary_job.done():
            try:
                await self._summary_job
            except Exception:
                pass
        pending, cutoff_index = self._pending_summary_batch(force=True)
        if not pending:
            return
        await self._summary_task(
            events=self._messages_to_summary_events(pending),
            cutoff_index=cutoff_index,
        )

    def to_string(self, add_sender_prefix: bool = False) -> str:
        if add_sender_prefix:
            return "\n".join(
                [
                    f"[{message.sender}]: {message.content}"
                    if message.sender != ""
                    else message.content
                    for message in self.messages
                ]
            )
        else:
            return "\n".join([message.content for message in self.messages])


    #added this to ensure that message-passing isn't extremely large for later API calls. 
    def to_string_last_n(self, n: int, add_sender_prefix: bool = True) -> str:
        return "\n".join([
            f"[{message.sender}]: {message.content}"
            for message in self.messages[-n:]
        ])

    def to_string_summary_last_n(self, n: int = 3) -> str:
        summary_block = ""

        if self.summary != "":
            summary_block = (
                "Summary of earlier conversation:\n"
                f"{self.summary}\n\n"
            )

        recent_start = self._summary_cutoff_index()
        recent_messages = self.messages[recent_start:]
        if not recent_messages and n > 0:
            recent_messages = self.messages[-n:]

        last_messages = "\n".join(
            [
                f"[{message.sender}]: {message.content}"
                for message in recent_messages
            ]
        )

        return summary_block + "Recent conversation:\n" + last_messages

    async def to_messages(
        self,
        my_name: str = "",
        start_index: int = 0,
        max_summary_length: int = 0,
        max_send_token: int = 0,
        model: str = "gpt-3.5-turbo",
    ) -> List[dict]:
        if self.has_summary:
            await self._ensure_summary_current(model=model)
        messages = []

        if self.has_summary:
            start_index = self.last_trimmed_index

        for message in self.messages[start_index:]:
            if message.sender == my_name:
                if isinstance(message, ExecutorMessage):
                    if message.tool_name != "":
                        messages.append(
                            {
                                "role": "assistant",
                                "content": f"[{message.sender}]: {message.content}"
                                if message.content != ""
                                else "",
                                "function_call": {
                                    "name": message.tool_name,
                                    "arguments": json.dumps(message.tool_input),
                                },
                            }
                        )
                        continue
                messages.append(
                    {
                        "role": "assistant",
                        "content": f"[{message.sender}]: {message.content}",
                    }
                )
                continue
            if message.sender == "function":
                messages.append(
                    {
                        "role": "function",
                        "content": message.content,
                        "name": message.tool_name,
                    }
                )
                continue
            messages.append(
                {
                    "role": "assistant",
                    "content": f"[{message.sender}]: {message.content}",
                }
            )

        if self.has_summary:
            if max_summary_length == 0:
                max_summary_length = self.max_summary_tlength
            prompt = []
            history_budget = max_send_token
            if self.summary != "":
                history_budget = max(0, history_budget - max_summary_length)
            pinned_indexes: List[int] = []
            if bool(self.summary_preserve_latest_final_answer_message):
                source_messages = self.messages[start_index:]
                for idx, message in enumerate(source_messages):
                    if self._message_has_final_answer_marker(message):
                        pinned_indexes = [idx]
            add_history_upto_token_limit(
                prompt, messages, history_budget, model, pinned_indexes=pinned_indexes
            )
            if self.summary != "":
                prompt.insert(0, self.summary_message())
            messages = prompt
        return messages

    def reset(self) -> None:
        self.messages = []
        self.last_trimmed_index = 0
        self.summary = ""
        self.summary_request_count = 0
        self.summary_prompt_tokens = 0
        self.summary_completion_tokens = 0
        self._summary_job = None
        self._summary_updates_disabled = False

    async def trim_messages(
        self, current_message_chain: List[Dict], model: str, history: List[Dict]
    ) -> Tuple[Dict, List[Dict]]:
        new_messages_not_in_chain = [
            msg for msg in history if msg not in current_message_chain
        ]

        if not new_messages_not_in_chain:
            return self.summary_message(), []

        new_summary_message = await self.update_running_summary(
            new_events=new_messages_not_in_chain, model=model
        )

        last_message = new_messages_not_in_chain[-1]
        self.last_trimmed_index += history.index(last_message)

        return new_summary_message, new_messages_not_in_chain

    async def update_running_summary(
        self,
        new_events: List[Dict],
        model: str = "gpt-3.5-turbo",
        max_summary_length: Optional[int] = None,
    ) -> dict:
        if not new_events:
            return self.summary_message()
        if max_summary_length is None:
            max_summary_length = self.max_summary_tlength

        new_events = copy.deepcopy(new_events)

        # Replace "assistant" with "you". This produces much better first person past tense results.
        for event in new_events:
            if event["role"].lower() == "assistant":
                event["role"] = "you"

            elif event["role"].lower() == "system":
                event["role"] = "your computer"

            # Delete all user messages
            elif event["role"] == "user":
                new_events.remove(event)

        prompt_template_length = len(
            self.SUMMARIZATION_PROMPT.format(summary="", new_events="")
        )
        max_input_tokens = OpenAIChat.send_token_limit(model) - max_summary_length
        summary_tlength = count_string_tokens(
            self._coerce_summary_text(self.summary), model
        )
        batch: List[Dict] = []
        batch_tlength = 0

        for event in new_events:
            event_tlength = count_message_tokens(event, model)

            if (
                batch_tlength + event_tlength
                > max_input_tokens - prompt_template_length - summary_tlength
            ):
                await self._update_summary_with_batch(batch, model, max_summary_length)
                summary_tlength = count_string_tokens(
                    self._coerce_summary_text(self.summary), model
                )
                batch = [event]
                batch_tlength = event_tlength
            else:
                batch.append(event)
                batch_tlength += event_tlength

        if batch:
            await self._update_summary_with_batch(batch, model, max_summary_length)

        return self.summary_message()

    async def _update_summary_with_batch(
        self, new_events_batch: List[dict], model: str, max_summary_length: int
    ) -> None:
        prompt = self.SUMMARIZATION_PROMPT.format(
            summary=self.summary, new_events=new_events_batch
        )

        self.summary_request_count += 1
        response = openai_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            max_tokens=max_summary_length,
            temperature=0.5,
        )
        usage = getattr(response, "usage", None)
        if usage is not None:
            self.summary_prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
            self.summary_completion_tokens += int(
                getattr(usage, "completion_tokens", 0) or 0
            )

        raw_summary = getattr(response.choices[0].message, "content", "")
        self.summary = self._coerce_summary_text(raw_summary)
    
    
    def summary_message(self) -> dict:
        return {
            "role": "system",
            "content": f"This reminds you of these events from your past: \n{self.summary}",
        }


def add_history_upto_token_limit(
    prompt: List[dict],
    history: List[dict],
    t_limit: int,
    model: str,
    pinned_indexes: Optional[List[int]] = None,
) -> List[Message]:
    pinned_index_set = set(pinned_indexes or [])
    current_prompt_length = 0
    trimmed_messages: List[Dict] = []
    selected_messages: Dict[int, Dict] = {}
    for idx in sorted(pinned_index_set):
        if 0 <= idx < len(history):
            message = history[idx]
            token_to_add = count_message_tokens(message, model)
            if current_prompt_length == 0 or current_prompt_length + token_to_add <= t_limit:
                selected_messages[idx] = message
                current_prompt_length += token_to_add
            else:
                trimmed_messages.append(message)
    limit_reached = False
    for idx in range(len(history) - 1, -1, -1):
        message = history[idx]
        if idx in pinned_index_set:
            continue
        token_to_add = count_message_tokens(message, model)
        if current_prompt_length + token_to_add > t_limit:
            limit_reached = True

        if not limit_reached:
            selected_messages[idx] = message
            current_prompt_length += token_to_add
        else:
            trimmed_messages.insert(0, message)
    for idx in sorted(selected_messages):
        prompt.append(selected_messages[idx])
    return trimmed_messages
