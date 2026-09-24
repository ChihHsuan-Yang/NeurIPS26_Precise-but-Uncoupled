from __future__ import annotations

import bdb
import json
import re
from typing import Any, Dict, List

from colorama import Fore
from pydantic import Field

from agentverse.agents import agent_registry
from agentverse.agents.base import BaseAgent, format_fatal_auth_error, is_fatal_auth_error
from agentverse.evaluation.common import extract_boxed
from agentverse.llms.utils import count_string_tokens
from agentverse.llms.utils.jsonrepair import JsonRepair
from agentverse.logging import get_logger
from agentverse.message import Message

logger = get_logger()


def _parse_json_dict(candidate: str) -> Dict[str, Any]:
    cleaned = str(candidate or "").strip()
    if not cleaned:
        return {}

    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        pass

    try:
        repaired = JsonRepair(cleaned).repair()
        parsed = json.loads(repaired)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _iter_balanced_object_candidates(text: str) -> List[str]:
    raw = str(text or "")
    candidates: List[str] = []
    n = len(raw)

    for start_idx, char in enumerate(raw):
        if char != "{":
            continue

        depth = 0
        in_string = False
        escape = False
        string_char = ""

        for end_idx in range(start_idx, n):
            current = raw[end_idx]

            if in_string:
                if escape:
                    escape = False
                elif current == "\\":
                    escape = True
                elif current == string_char:
                    in_string = False
                continue

            if current in {'"', "'"}:
                in_string = True
                string_char = current
                continue

            if current == "{":
                depth += 1
            elif current == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(raw[start_idx : end_idx + 1].strip())
                    break

    return candidates


def _extract_json_object(
    text: str,
    expected_keys: List[str] | None = None,
) -> Dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        return {}

    expected = set(expected_keys or [])
    candidates: List[str] = [raw]

    fenced_blocks = re.findall(
        r"```(?:json|JSON)?\s*(.*?)```",
        raw,
        re.DOTALL,
    )
    candidates.extend(block.strip() for block in fenced_blocks if block.strip())
    candidates.extend(_iter_balanced_object_candidates(raw))

    seen: set[str] = set()
    for candidate in candidates:
        candidate = str(candidate or "").strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)

        parsed = _parse_json_dict(candidate)
        if not parsed:
            continue
        if expected and not any(key in parsed for key in expected):
            continue
        return parsed

    return {}

def _normalize_boxed(answer: str) -> str:
    answer = str(answer or "").strip()
    if not answer:
        return ""
    boxed = extract_boxed(answer)
    if boxed:
        return f"\\boxed{{{boxed}}}"
    return f"\\boxed{{{answer}}}"


def _extract_labeled_boxed(text: str, labels: List[str]) -> str:
    raw = str(text or "")
    for label in labels:
        pattern = rf"(?is){re.escape(label)}\s*[:=]\s*(.*)"
        match = re.search(pattern, raw)
        if match:
            boxed = extract_boxed(match.group(1))
            if boxed:
                return f"\\boxed{{{boxed}}}"
    return ""


def _extract_int_field(text: str, field_name: str, default: int = 1) -> int:
    pattern = rf'(?i)(?:"{re.escape(field_name)}"|{re.escape(field_name)})\s*[:=]\s*(\d{{1,3}})'
    match = re.search(pattern, text or "")
    if not match:
        return default
    try:
        return int(match.group(1))
    except Exception:
        return default


def _extract_bool_field(text: str, field_name: str, default: bool = False) -> bool:
    pattern = rf'(?i)(?:"{re.escape(field_name)}"|{re.escape(field_name)})\s*[:=]\s*(true|false)'
    match = re.search(pattern, text or "")
    if not match:
        return default
    return match.group(1).lower() == "true"


def _coerce_bool_value(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0", ""}:
            return False
    return default


@agent_registry.register("deliberator")
class DeliberationAgent(BaseAgent):
    max_history: int = 5
    max_private_notes: int = 8
    max_speak_history: int = 8
    max_public_context_tokens: int = 2400
    max_private_context_tokens: int = 600
    max_speak_history_context_tokens: int = 600
    max_advice_context_tokens: int = 2400
    max_candidate_context_tokens: int = 300
    max_manual_prompt_tokens: int = 12000
    group_description: str = "math broadcast-deliberation group"
    answer_format_guidance: str = (
        "Omni-MATH answers are not always single numbers. A valid final answer may be "
        "a symbolic expression, formula, set, interval, condition, equality, inequality, "
        "or other compact mathematical form."
    )
    intent_language_guidance: str = "using mathematically precise language"
    review_focus_guidance: str = (
        "Reason carefully about whether it is mathematically complete and equivalent "
        "to the intended solution."
    )
    review_feedback_guidance: str = (
        "a concise mathematical review explaining why you accept it or what is missing/wrong"
    )
    final_answer_guidance: str = (
        "your best boxed final answer like \\boxed{...}, preserving exact mathematical form"
    )
    private_notes: List[str] = Field(default_factory=list)
    speak_history: List[str] = Field(default_factory=list)
    current_turn_label: str = ""

    def step(self, env_description: str = "", **kwargs) -> Message:
        raise NotImplementedError("Use `astep` or the broadcast deliberation helpers.")

    async def _agenerate_text(
        self,
        prepend_prompt: str,
        history: List[dict] | None = None,
        append_prompt: str = "",
    ) -> str:
        history = history or []
        for _ in range(self.max_retry):
            try:
                logger.debug(prepend_prompt, f"{self.name} Prompt", Fore.CYAN)
                response = await self.llm.agenerate_response(
                    prepend_prompt,
                    history,
                    append_prompt,
                )
                return (getattr(response, "content", "") or "").strip()
            except (KeyboardInterrupt, bdb.BdbQuit):
                raise
            except Exception as exc:
                if is_fatal_auth_error(exc):
                    logger.error(format_fatal_auth_error(exc))
                    raise RuntimeError(format_fatal_auth_error(exc)) from exc
                logger.error(exc)
                logger.warn("Retrying...")
        logger.error(f"{self.name} failed to generate valid response.")
        return ""

    def _public_context(self, memory_mode: str, recent_history_window: int) -> str:
        mem = self.memory
        if memory_mode == "full_history":
            if hasattr(mem, "to_string"):
                return mem.to_string(add_sender_prefix=True)
        if memory_mode == "recent_only":
            if hasattr(mem, "to_string_last_n"):
                return mem.to_string_last_n(recent_history_window, add_sender_prefix=True)
        if memory_mode == "summary_plus_recent":
            if hasattr(mem, "to_string_summary_last_n"):
                return mem.to_string_summary_last_n(recent_history_window)
        if hasattr(mem, "to_string_summary_last_n"):
            return mem.to_string_summary_last_n(max(1, min(recent_history_window, 2)))
        if hasattr(mem, "to_string"):
            return mem.to_string(add_sender_prefix=True)
        return ""

    def _estimate_text_tokens(self, text: str) -> int:
        try:
            return int(count_string_tokens(text or "", self.llm.args.model) or 0)
        except Exception:
            return 0

    def _trim_text_for_prompt(
        self,
        text: str,
        *,
        token_budget: int,
        keep: str = "tail",
        marker: str = "\n...[truncated for prompt budget]...\n",
    ) -> str:
        raw = str(text or "").strip()
        if not raw or token_budget <= 0:
            return raw

        current_tokens = self._estimate_text_tokens(raw)
        if current_tokens <= token_budget:
            return raw

        marker = str(marker or "\n...[truncated]...\n")
        low = 32
        high = max(32, len(raw))
        best = raw[-32:] if keep == "tail" else raw[:32]

        def build(chars: int) -> str:
            chars = max(32, int(chars))
            if keep == "tail":
                return marker.lstrip("\n") + raw[-chars:]
            if keep == "head":
                return raw[:chars] + marker.rstrip("\n")
            head_chars = max(16, int(chars * 0.45))
            tail_chars = max(16, chars - head_chars)
            return raw[:head_chars].rstrip() + marker + raw[-tail_chars:].lstrip()

        while low <= high:
            mid = (low + high) // 2
            candidate = build(mid)
            candidate_tokens = self._estimate_text_tokens(candidate)
            if candidate_tokens <= token_budget:
                best = candidate
                low = mid + 1
            else:
                high = mid - 1

        return best

    def _fit_manual_prompt_to_budget(
        self,
        *,
        sections: Dict[str, str],
        build_prompt,
        total_budget: int,
        keep_modes: Dict[str, str] | None = None,
    ) -> tuple[str, Dict[str, str], Dict[str, int]]:
        keep_modes = keep_modes or {}
        if total_budget <= 0:
            prompt = build_prompt(sections)
            token_counts = {
                key: self._estimate_text_tokens(value)
                for key, value in sections.items()
            }
            token_counts["prompt_tokens"] = self._estimate_text_tokens(prompt)
            return prompt, sections, token_counts

        original_sections = {
            key: str(value or "")
            for key, value in sections.items()
        }
        current_sections = dict(original_sections)

        min_budgets = {
            "public_context": 300,
            "advice": 400,
            "own_speak_history": 150,
            "private_context": 100,
            "candidate_answer": 80,
        }
        reduction_order = [
            "public_context",
            "advice",
            "own_speak_history",
            "private_context",
            "candidate_answer",
        ]

        prompt = build_prompt(current_sections)
        prompt_tokens = self._estimate_text_tokens(prompt)
        if prompt_tokens <= total_budget:
            token_counts = {
                key: self._estimate_text_tokens(value)
                for key, value in current_sections.items()
            }
            token_counts["prompt_tokens"] = prompt_tokens
            return prompt, current_sections, token_counts

        stall_rounds = 0
        while prompt_tokens > total_budget and stall_rounds < 2:
            changed = False
            overflow = max(128, prompt_tokens - total_budget)
            for name in reduction_order:
                raw = original_sections.get(name, "")
                if not raw:
                    continue
                current_text = current_sections.get(name, "")
                current_budget = self._estimate_text_tokens(current_text)
                min_budget = min_budgets.get(name, 64)
                if current_budget <= min_budget:
                    continue
                reduction = max(128, overflow // 2, int(current_budget * 0.2))
                new_budget = max(min_budget, current_budget - reduction)
                new_text = self._trim_text_for_prompt(
                    raw,
                    token_budget=new_budget,
                    keep=keep_modes.get(name, "tail"),
                )
                if new_text == current_text:
                    continue
                current_sections[name] = new_text
                changed = True
                prompt = build_prompt(current_sections)
                prompt_tokens = self._estimate_text_tokens(prompt)
                if prompt_tokens <= total_budget:
                    break
            if changed:
                stall_rounds = 0
            else:
                stall_rounds += 1

        if prompt_tokens > total_budget:
            for name in reduction_order:
                raw = original_sections.get(name, "")
                if not raw:
                    continue
                current_sections[name] = self._trim_text_for_prompt(
                    raw,
                    token_budget=min_budgets.get(name, 64),
                    keep=keep_modes.get(name, "tail"),
                )
            prompt = build_prompt(current_sections)
            prompt_tokens = self._estimate_text_tokens(prompt)

        token_counts = {
            key: self._estimate_text_tokens(value)
            for key, value in current_sections.items()
        }
        token_counts["prompt_tokens"] = prompt_tokens
        return prompt, current_sections, token_counts

    def _private_context(self) -> str:
        if not self.private_notes:
            return "No deferred private notes."
        return "\n".join(self.private_notes[-self.max_private_notes :])

    def _own_speak_history_context(self) -> str:
        if not self.speak_history:
            return "No prior speaking history in this run."
        return "\n\n".join(self.speak_history[-self.max_speak_history :])

    def _recent_public_message_count(self, recent_history_window: int) -> int:
        message_count = len(getattr(self.memory, "messages", []) or [])
        return min(max(0, int(recent_history_window or 0)), message_count)

    def _prompt_debug_context(
        self,
        *,
        recent_history_window: int,
        failed_attempt_ledger_size: int = 0,
        revision_ledger_size: int = 0,
        memory_mode: str = "",
        advice_tokens: int = 0,
        public_context_tokens: int = 0,
        private_context_tokens: int = 0,
        own_speak_history_tokens: int = 0,
    ) -> Dict[str, Any]:
        return {
            "memory_mode": str(memory_mode or ""),
            "recent_history_count": self._recent_public_message_count(
                recent_history_window
            ),
            "failed_attempt_ledger_size": int(failed_attempt_ledger_size or 0),
            "revision_ledger_size": int(revision_ledger_size or 0),
            "private_note_count": len(self.private_notes),
            "speak_history_count": len(self.speak_history),
            "advice_tokens": int(advice_tokens or 0),
            "public_context_tokens": int(public_context_tokens or 0),
            "private_context_tokens": int(private_context_tokens or 0),
            "own_speak_history_tokens": int(own_speak_history_tokens or 0),
        }

    def _log_manual_prompt_budget(
        self,
        *,
        stage: str,
        prompt: str,
        recent_history_window: int,
        failed_attempt_ledger_size: int = 0,
        revision_ledger_size: int = 0,
        memory_mode: str = "",
        advice_tokens: int = 0,
        public_context_tokens: int = 0,
        private_context_tokens: int = 0,
        own_speak_history_tokens: int = 0,
    ) -> None:
        try:
            prompt_tokens = count_string_tokens(prompt, self.llm.args.model) or 0
        except Exception:
            prompt_tokens = 0
        self.log_prompt_budget(
            stage=stage,
            prompt_token=prompt_tokens,
            history=[],
            extra_debug=self._prompt_debug_context(
                recent_history_window=recent_history_window,
                failed_attempt_ledger_size=failed_attempt_ledger_size,
                revision_ledger_size=revision_ledger_size,
                memory_mode=memory_mode,
                advice_tokens=advice_tokens,
                public_context_tokens=public_context_tokens,
                private_context_tokens=private_context_tokens,
                own_speak_history_tokens=own_speak_history_tokens,
            ),
        )

    def add_message_to_memory(self, messages: List[Message]) -> None:
        self.memory.add_message(messages)

    def set_current_turn_label(self, turn_label: str) -> None:
        self.current_turn_label = str(turn_label or "").strip()

    def record_speaking_turn(self, turn_label: str, content: str) -> None:
        label = str(turn_label or self.current_turn_label or "[unknown turn]").strip()
        text = str(content or "").strip()
        if not text:
            return
        self.speak_history.append(f"{label}\n{text}")
        if len(self.speak_history) > self.max_speak_history:
            self.speak_history = self.speak_history[-self.max_speak_history :]

    def add_private_note(self, note: str) -> None:
        cleaned = str(note or "").strip()
        if not cleaned:
            return
        self.private_notes.append(cleaned)
        if len(self.private_notes) > self.max_private_notes:
            self.private_notes = self.private_notes[-self.max_private_notes :]

    def reset(self) -> None:
        self.memory.reset()
        self.private_notes = []
        self.speak_history = []
        self.current_turn_label = ""

    async def astep(
        self,
        env_description: str = "",
        task_description: str = "",
        advice: str = "No advice yet.",
        candidate_answer: str = "",
        phase: str = "discussion",
        phase_instruction: str = "",
        current_turn_label: str = "",
        **kwargs,
    ) -> Message:
        turn_label = str(current_turn_label or self.current_turn_label or "").strip()
        trimmed_advice = self._trim_text_for_prompt(
            advice,
            token_budget=self.max_advice_context_tokens,
            keep="head_tail",
        )
        trimmed_candidate_answer = self._trim_text_for_prompt(
            candidate_answer,
            token_budget=self.max_candidate_context_tokens,
            keep="head_tail",
        )
        trimmed_private_context = self._trim_text_for_prompt(
            self._private_context(),
            token_budget=self.max_private_context_tokens,
            keep="tail",
        )
        trimmed_own_speak_history = self._trim_text_for_prompt(
            self._own_speak_history_context(),
            token_budget=self.max_speak_history_context_tokens,
            keep="tail",
        )
        prepend_prompt, append_prompt, prompt_token = self.get_all_prompts(
            env_description=env_description,
            task_description=task_description,
            advice=trimmed_advice,
            candidate_answer=trimmed_candidate_answer,
            phase=phase,
            phase_instruction=phase_instruction,
            private_memory=trimmed_private_context,
            current_turn_label=turn_label,
            own_speak_history=trimmed_own_speak_history,
            role_description=self.role_description,
            agent_name=self.name,
            **kwargs,
        )

        max_send_token = self.history_token_budget(prompt_token)
        history = await self.memory.to_messages(
            self.name,
            start_index=-self.max_history,
            max_send_token=max_send_token,
            model=self.llm.args.model,
        )
        self.log_prompt_budget(
            stage=str(kwargs.get("debug_prompt_stage", phase or "discussion")),
            prompt_token=prompt_token,
            history=history,
            extra_debug=self._prompt_debug_context(
                recent_history_window=int(kwargs.get("debug_recent_history_window", 0) or 0),
                failed_attempt_ledger_size=int(
                    kwargs.get("debug_failed_attempt_memory_size", 0) or 0
                ),
                revision_ledger_size=int(
                    kwargs.get("debug_candidate_revision_memory_size", 0) or 0
                ),
                memory_mode=str(kwargs.get("debug_memory_mode", "") or ""),
                advice_tokens=self._estimate_text_tokens(trimmed_advice),
                public_context_tokens=0,
                private_context_tokens=self._estimate_text_tokens(trimmed_private_context),
                own_speak_history_tokens=self._estimate_text_tokens(trimmed_own_speak_history),
            ),
        )
        content = await self._agenerate_text(prepend_prompt, history, append_prompt)
        return Message(
            content=content,
            sender=self.name,
            sender_agent=self,
            receiver=self.get_receiver(),
        )

    async def abroadcast_poll(
        self,
        *,
        task_description: str,
        advice: str,
        candidate_answer: str,
        memory_mode: str,
        recent_history_window: int,
        turn_idx: int,
        outer_round_idx: int,
        failed_attempt_ledger_size: int = 0,
        revision_ledger_size: int = 0,
    ) -> Dict[str, Any]:
        turn_label = (
            f"Outer round {outer_round_idx}, discussion turn {turn_idx}"
        )
        self.set_current_turn_label(turn_label)
        trimmed_advice = self._trim_text_for_prompt(
            advice,
            token_budget=self.max_advice_context_tokens,
            keep="head_tail",
        )
        trimmed_candidate_answer = self._trim_text_for_prompt(
            candidate_answer,
            token_budget=self.max_candidate_context_tokens,
            keep="head_tail",
        )
        trimmed_public_context = self._trim_text_for_prompt(
            self._public_context(memory_mode, recent_history_window),
            token_budget=self.max_public_context_tokens,
            keep="tail",
        )
        trimmed_private_context = self._trim_text_for_prompt(
            self._private_context(),
            token_budget=self.max_private_context_tokens,
            keep="tail",
        )
        trimmed_own_speak_history = self._trim_text_for_prompt(
            self._own_speak_history_context(),
            token_budget=self.max_speak_history_context_tokens,
            keep="tail",
        )
        def build_prompt(parts: Dict[str, str]) -> str:
            return f"""You are {self.name}, one peer in a {self.group_description}.

Problem:
{task_description}

Outer-loop feedback:
{parts["advice"]}

Current MAS candidate answer:
{parts["candidate_answer"] or "[None yet]"}

Your reasoning style:
{self.role_description}

Public discussion context:
{parts["public_context"]}

Your deferred private notes:
{parts["private_context"]}

Your own speaking history:
{parts["own_speak_history"]}

Current turn label:
{turn_label}

Current outer round: {outer_round_idx}
Current discussion turn: {turn_idx}

This is the mandatory turn-allocation poll. Your numeric confidence score is used to decide who gets to speak next.
You must provide an integer confidence score from 1 to 100.
If you do not provide a valid score in the required JSON format, your poll may be treated as score 1.
{self.answer_format_guidance}

Return a JSON object with exactly these keys:
- "score": integer from 1 to 100 for how strongly you want to speak now
- "reason": short reason for the score
- "intent": what you would say if selected, {self.intent_language_guidance}
- "candidate_answer": your current best final answer in boxed form, following the answer-format guidance above, or empty string if you do not want to propose one now
- "has_candidate_answer": true or false

Return JSON only. Do not add any extra prose before or after the JSON object.
Be conservative with very high scores. Use high scores only when you think speaking now would materially help."""
        prompt, fitted_sections, token_counts = self._fit_manual_prompt_to_budget(
            sections={
                "advice": trimmed_advice,
                "candidate_answer": trimmed_candidate_answer,
                "public_context": trimmed_public_context,
                "private_context": trimmed_private_context,
                "own_speak_history": trimmed_own_speak_history,
            },
            build_prompt=build_prompt,
            total_budget=self.max_manual_prompt_tokens,
            keep_modes={
                "advice": "head_tail",
                "candidate_answer": "head_tail",
                "public_context": "tail",
                "private_context": "tail",
                "own_speak_history": "tail",
            },
        )

        self._log_manual_prompt_budget(
            stage=f"discussion_turn_{turn_idx}_poll",
            prompt=prompt,
            recent_history_window=recent_history_window,
            failed_attempt_ledger_size=failed_attempt_ledger_size,
            revision_ledger_size=revision_ledger_size,
            memory_mode=memory_mode,
            advice_tokens=token_counts.get("advice", 0),
            public_context_tokens=token_counts.get("public_context", 0),
            private_context_tokens=token_counts.get("private_context", 0),
            own_speak_history_tokens=token_counts.get("own_speak_history", 0),
        )

        text = await self._agenerate_text(prompt)
        data = _extract_json_object(
            text,
            expected_keys=[
                "score",
                "reason",
                "intent",
                "candidate_answer",
                "has_candidate_answer",
            ],
        )

        score = data.get("score", _extract_int_field(text, "score", default=1))
        try:
            score = int(score)
        except Exception:
            score = 1
        score = max(1, min(score, 100))

        intent = str(data.get("intent", "") or "").strip()
        reason = str(data.get("reason", "") or "").strip()
        candidate = _normalize_boxed(str(data.get("candidate_answer", "") or "").strip())
        has_candidate = _coerce_bool_value(
            data.get("has_candidate_answer"),
            default=False,
        ) or bool(candidate)

        if not intent and not data:
            intent = text.strip()

        if not candidate:
            candidate = _extract_labeled_boxed(
                intent or text,
                labels=["candidate_answer", "candidate answer", "final_answer", "final answer"],
            ) or _normalize_boxed(extract_boxed(intent or text))
        has_candidate = has_candidate or bool(candidate)

        return {
            "score": score,
            "reason": reason,
            "intent": intent,
            "candidate_answer": candidate,
            "has_candidate_answer": bool(candidate) or has_candidate,
            "raw_text": text,
            "parse_succeeded": bool(data),
        }

    async def abroadcast_speak(
        self,
        *,
        task_description: str,
        advice: str,
        candidate_answer: str,
        phase: str = "discussion",
        phase_instruction: str = "",
        current_turn_label: str = "",
        **kwargs,
    ) -> Message:
        return await self.astep(
            task_description=task_description,
            advice=advice,
            candidate_answer=candidate_answer,
            phase=phase,
            phase_instruction=phase_instruction,
            current_turn_label=current_turn_label,
            **kwargs,
        )

    async def aapprove_candidate(
        self,
        *,
        task_description: str,
        advice: str,
        candidate_answer: str,
        memory_mode: str,
        recent_history_window: int,
        approval_round_idx: int,
        failed_attempt_ledger_size: int = 0,
        revision_ledger_size: int = 0,
    ) -> Dict[str, Any]:
        turn_label = f"Approval round {approval_round_idx}"
        self.set_current_turn_label(turn_label)
        trimmed_advice = self._trim_text_for_prompt(
            advice,
            token_budget=self.max_advice_context_tokens,
            keep="head_tail",
        )
        trimmed_candidate_answer = self._trim_text_for_prompt(
            candidate_answer,
            token_budget=self.max_candidate_context_tokens,
            keep="head_tail",
        )
        trimmed_public_context = self._trim_text_for_prompt(
            self._public_context(memory_mode, recent_history_window),
            token_budget=self.max_public_context_tokens,
            keep="tail",
        )
        trimmed_private_context = self._trim_text_for_prompt(
            self._private_context(),
            token_budget=self.max_private_context_tokens,
            keep="tail",
        )
        trimmed_own_speak_history = self._trim_text_for_prompt(
            self._own_speak_history_context(),
            token_budget=self.max_speak_history_context_tokens,
            keep="tail",
        )
        def build_prompt(parts: Dict[str, str]) -> str:
            return f"""You are {self.name}, one peer in a {self.group_description}.

Problem:
{task_description}

Current candidate answer:
{parts["candidate_answer"] or "[None]"}

Outer-loop feedback:
{parts["advice"]}

Your reasoning style:
{self.role_description}

Public discussion context:
{parts["public_context"]}

Your deferred private notes:
{parts["private_context"]}

Your own speaking history:
{parts["own_speak_history"]}

Current turn label:
{turn_label}

Current approval round: {approval_round_idx}

{self.answer_format_guidance}
{self.review_focus_guidance}
If you think the candidate is incomplete, imprecise, or wrong, try to correct it.
Your review will be shared with all peers in this review turn.

Return a JSON object with exactly these keys:
- "approve": true if you accept the candidate answer as the current MAS answer, otherwise false
- "feedback": {self.review_feedback_guidance}
- "revised_answer": a boxed answer like \\boxed{{...}} if you think the candidate should be replaced or completed, otherwise empty string

All agents, including the one who originally proposed the candidate answer, must participate in this approval discussion.
"""
        prompt, fitted_sections, token_counts = self._fit_manual_prompt_to_budget(
            sections={
                "advice": trimmed_advice,
                "candidate_answer": trimmed_candidate_answer,
                "public_context": trimmed_public_context,
                "private_context": trimmed_private_context,
                "own_speak_history": trimmed_own_speak_history,
            },
            build_prompt=build_prompt,
            total_budget=self.max_manual_prompt_tokens,
            keep_modes={
                "advice": "head_tail",
                "candidate_answer": "head_tail",
                "public_context": "tail",
                "private_context": "tail",
                "own_speak_history": "tail",
            },
        )

        self._log_manual_prompt_budget(
            stage=f"approval_round_{approval_round_idx}_review",
            prompt=prompt,
            recent_history_window=recent_history_window,
            failed_attempt_ledger_size=failed_attempt_ledger_size,
            revision_ledger_size=revision_ledger_size,
            memory_mode=memory_mode,
            advice_tokens=token_counts.get("advice", 0),
            public_context_tokens=token_counts.get("public_context", 0),
            private_context_tokens=token_counts.get("private_context", 0),
            own_speak_history_tokens=token_counts.get("own_speak_history", 0),
        )

        text = await self._agenerate_text(prompt)
        data = _extract_json_object(
            text,
            expected_keys=[
                "approve",
                "feedback",
                "revised_answer",
            ],
        )
        revised = _normalize_boxed(str(data.get("revised_answer", "") or "").strip())
        approve = _coerce_bool_value(data.get("approve"), default=False)
        if not data:
            approve = _extract_bool_field(text, "approve", default=False)
        feedback = str(data.get("feedback", "") or "").strip()
        if not feedback:
            feedback = text
        if not revised:
            revised = _extract_labeled_boxed(
                text,
                labels=["revised_answer", "revised answer", "suggested revision"],
            )
        return {
            "approve": approve,
            "feedback": feedback,
            "revised_answer": revised,
            "raw_text": text,
            "parse_succeeded": bool(data),
        }

    async def apropose_final_answer(
        self,
        *,
        task_description: str,
        advice: str,
        memory_mode: str,
        recent_history_window: int,
        failed_attempt_ledger_size: int = 0,
        revision_ledger_size: int = 0,
    ) -> Dict[str, Any]:
        turn_label = "Final proposal round"
        self.set_current_turn_label(turn_label)
        trimmed_advice = self._trim_text_for_prompt(
            advice,
            token_budget=self.max_advice_context_tokens,
            keep="head_tail",
        )
        trimmed_public_context = self._trim_text_for_prompt(
            self._public_context(memory_mode, recent_history_window),
            token_budget=self.max_public_context_tokens,
            keep="tail",
        )
        trimmed_private_context = self._trim_text_for_prompt(
            self._private_context(),
            token_budget=self.max_private_context_tokens,
            keep="tail",
        )
        trimmed_own_speak_history = self._trim_text_for_prompt(
            self._own_speak_history_context(),
            token_budget=self.max_speak_history_context_tokens,
            keep="tail",
        )
        def build_prompt(parts: Dict[str, str]) -> str:
            return f"""You are {self.name}, one peer in a {self.group_description}.

Problem:
{task_description}

Outer-loop feedback:
{parts["advice"]}

Your reasoning style:
{self.role_description}

Public discussion context:
{parts["public_context"]}

Your deferred private notes:
{parts["private_context"]}

Your own speaking history:
{parts["own_speak_history"]}

Current turn label:
{turn_label}

{self.answer_format_guidance}

Return a JSON object with exactly these keys:
- "confidence": integer from 1 to 100 for how much you trust your proposed final answer
- "final_answer": {self.final_answer_guidance}
- "rationale": a short justification
"""
        prompt, fitted_sections, token_counts = self._fit_manual_prompt_to_budget(
            sections={
                "advice": trimmed_advice,
                "public_context": trimmed_public_context,
                "private_context": trimmed_private_context,
                "own_speak_history": trimmed_own_speak_history,
            },
            build_prompt=build_prompt,
            total_budget=self.max_manual_prompt_tokens,
            keep_modes={
                "advice": "head_tail",
                "public_context": "tail",
                "private_context": "tail",
                "own_speak_history": "tail",
            },
        )

        self._log_manual_prompt_budget(
            stage="final_proposal",
            prompt=prompt,
            recent_history_window=recent_history_window,
            failed_attempt_ledger_size=failed_attempt_ledger_size,
            revision_ledger_size=revision_ledger_size,
            memory_mode=memory_mode,
            advice_tokens=token_counts.get("advice", 0),
            public_context_tokens=token_counts.get("public_context", 0),
            private_context_tokens=token_counts.get("private_context", 0),
            own_speak_history_tokens=token_counts.get("own_speak_history", 0),
        )

        text = await self._agenerate_text(prompt)
        data = _extract_json_object(
            text,
            expected_keys=[
                "confidence",
                "score",
                "final_answer",
                "rationale",
            ],
        )
        confidence = data.get(
            "confidence",
            data.get("score", _extract_int_field(text, "confidence", default=1)),
        )
        try:
            confidence = int(confidence)
        except Exception:
            confidence = 1
        confidence = max(1, min(confidence, 100))
        final_answer = _normalize_boxed(str(data.get("final_answer", "") or "").strip())
        rationale = str(data.get("rationale", "") or "").strip()
        if not final_answer:
            final_answer = _extract_labeled_boxed(
                text,
                labels=["final_answer", "final answer", "candidate_answer", "candidate answer"],
            ) or _normalize_boxed(extract_boxed(text))
        return {
            "confidence": confidence,
            "final_answer": final_answer,
            "rationale": rationale or text,
            "raw_text": text,
            "parse_succeeded": bool(data),
        }
