from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple
from agentverse.evaluation.common import extract_boxed
from agentverse.llms.utils import count_string_tokens
from agentverse.environments.tasksolving_env.rules.evaluator import (
    BaseEvaluator,
    evaluator_registry,
)
from agentverse.environments.tasksolving_env.rules.critique_intervention_policy import (
    CritiqueInterventionDecision,
    RuntimeCritiqueInterventionPolicy,
)
from agentverse.environments.tasksolving_env.rules.failure_memory import (
    CandidateRevisionMemory,
    FailedAttemptMemory,
    with_authority_guide,
)
from agentverse.message import Message
from agentverse.utils import AGENT_TYPES


def _submitted_final_answer_for_trace(text: str) -> str:
    boxed = extract_boxed(text or "")
    return f"\\boxed{{{boxed}}}" if boxed else "[No extracted final answer]"


def _memory_size(mem: Any) -> int:
    """Return number of stored Message objects, or -1 if unknown."""
    if mem is None:
        return -1
    msgs = getattr(mem, "messages", None)
    if isinstance(msgs, list):
        return len(msgs)
    try:
        return len(mem)
    except Exception:
        return -1


def _memory_tail(mem: Any, k: int = 2) -> List[Tuple[str, str]]:
    """Return last k messages as (sender, content_preview)."""
    if mem is None:
        return []
    msgs = getattr(mem, "messages", None)
    if not isinstance(msgs, list) or len(msgs) == 0:
        return []
    tail = msgs[-k:]
    out: List[Tuple[str, str]] = []
    for m in tail:
        sender = str(getattr(m, "sender", "") or "")
        content = str(getattr(m, "content", "") or "")
        content = " ".join(content.split())
        if len(content) > 120:
            content = content[:120] + "…"
        out.append((sender, content))
    return out


def _broadcast_to_agents(agent_list: List[Any], msg: Message | None) -> None:
    """
    Put the same Message into every agent's memory so they share a chat history.
    Also prints debug info to verify memory is actually growing.
    """
    if msg is None:
        return

    for a in agent_list:
        name = getattr(a, "name", type(a).__name__)
        try:
            mem = getattr(a, "memory", None)
            before = _memory_size(mem)

            if not hasattr(a, "add_message_to_memory"):
                raise AttributeError("missing add_message_to_memory()")

            a.add_message_to_memory([msg])

            mem = getattr(a, "memory", None)
            after = _memory_size(mem)
            tail = _memory_tail(mem, k=2)

            print(f"[MEM] {name} size={before}->{after} last={tail}")
        except Exception as exc:
            print(f"[MEM] {name} memory_update_failed: {exc}")


def _configure_reasoning_history(agent: Any, recent_limit: int = 3) -> None:
    if agent is None:
        return
    memory = getattr(agent, "memory", None)
    if bool(getattr(memory, "has_summary", False)):
        return
    if not hasattr(agent, "max_history"):
        return
    try:
        configured = int(getattr(agent, "max_history", recent_limit) or recent_limit)
    except Exception:
        configured = recent_limit
    try:
        agent.max_history = max(1, min(configured, int(recent_limit or 1)))
    except Exception:
        pass


def _enable_full_question_history(agent: Any) -> None:
    """Backward-compatible helper kept for other rule imports.

    Single-shot baseline imports this name from the PER utilities module.
    Preserve the historical behavior there: allow full same-question history.
    """
    if agent is None:
        return
    if hasattr(agent, "max_history"):
        try:
            agent.max_history = 0
        except Exception:
            pass


def _compact_candidate_handoff(text: str, max_chars: int = 240) -> str:
    boxed = extract_boxed(text or "")
    if boxed:
        return f"\\boxed{{{boxed}}}"
    cleaned = " ".join(str(text or "").split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3].rstrip() + "..."


def _estimate_text_tokens(text: str, model: str) -> int:
    try:
        return int(count_string_tokens(text or "", model) or 0)
    except Exception:
        return 0


def _trim_text_for_prompt(
    text: str,
    *,
    model: str,
    token_budget: int,
    keep: str = "head_tail",
    marker: str = "\n...[truncated for prompt budget]...\n",
) -> str:
    raw = str(text or "").strip()
    if not raw or token_budget <= 0:
        return raw

    current_tokens = _estimate_text_tokens(raw, model)
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
        candidate_tokens = _estimate_text_tokens(candidate, model)
        if candidate_tokens <= token_budget:
            best = candidate
            low = mid + 1
        else:
            high = mid - 1

    return best


def _parse_route(review_text: str) -> str:
    """
    Returns one of: "planner", "executor", "agree", "evaluate"
    Default: "executor" (safer)
    """
    t = (review_text or "").strip()
    if not t:
        return "executor"

    if "[Route:Evaluator]" in t or "[Route:Evaluate]" in t or "[Submit]" in t:
        return "evaluate"
    if "[Agree]" in t:
        return "agree"

    match = re.search(r"\[Route:(Planner|Executor|Evaluator|Evaluate)\]", t, re.IGNORECASE)
    if not match:
        return "executor"
    return match.group(1).lower()


def _extract_candidate_from_advice(advice: str) -> str:
    """
    If we embed the last candidate answer in advice, extract it for reviewer context.
    Format:
      [Candidate]
      ...
      [EndCandidate]
    """
    if not advice:
        return ""
    match = re.search(r"\[Candidate\](.*?)\[EndCandidate\]", advice, re.DOTALL)
    return match.group(1).strip() if match else ""


def _coerce_evaluator_score(score: Any) -> bool:
    if isinstance(score, bool):
        return score
    if isinstance(score, int):
        return score == 1 or score >= 8
    if isinstance(score, (list, tuple)):
        return bool(score) and all(_coerce_evaluator_score(item) for item in score)
    return bool(score)


def prepare_per_rule_kwargs(rule_config: Dict[str, Any] | None) -> Dict[str, Any]:
    normalized = dict(rule_config or {})

    evaluator_config = dict(normalized.pop("evaluator", {}) or {})
    legacy_type = normalized.pop("verifier_mode", None)
    if legacy_type is not None:
        evaluator_config["type"] = legacy_type
    evaluator_config.setdefault("type", "numeric-verifier")

    legacy_field_map = {
        "verifier_data_name": "data_name",
        "numeric_tolerance": "numeric_tolerance",
        "omni_judge_model_path": "omni_judge_model_path",
        "omni_judge_max_new_tokens": "omni_judge_max_new_tokens",
        "omni_judge_device": "omni_judge_device",
        "omni_judge_dtype": "omni_judge_dtype",
    }
    for legacy_key, evaluator_key in legacy_field_map.items():
        if legacy_key in normalized:
            evaluator_config[evaluator_key] = normalized.pop(legacy_key)

    normalized["evaluator_config"] = evaluator_config
    return normalized


def _default_fail_route(eval_advice: str, review_text: str = "") -> str:
    combined = f"{eval_advice}\n{review_text}".lower()
    executor_markers = [
        "boxed answer",
        "boxed",
        "format",
        "latex",
        "markup",
        "unit",
        "token",
        "raw numeric",
        "presentation",
        "output",
        "missing",
    ]
    if any(marker in combined for marker in executor_markers):
        return "[Route:Executor]"
    return "[Route:Planner]"


def _force_post_fail_route(review_text: str, eval_advice: str) -> str:
    route = _default_fail_route(eval_advice, review_text)
    text = (review_text or "").strip()
    if not text:
        return (
            "Diagnosis:\n"
            "- The evaluator returned FAIL.\n"
            "Fix Instruction:\n"
            "- Revise the current attempt based on evaluator feedback.\n"
            f"{route}"
        )

    text = re.sub(
        r"\[(?:Agree|Route:Planner|Route:Executor|Route:Evaluator|Route:Evaluate)\]\s*$",
        "",
        text,
        flags=re.IGNORECASE,
    ).rstrip()
    return f"{text}\n{route}"


def _format_evaluator_feedback(mode: str, passed: bool, advice: str) -> str:
    normalized_mode = str(mode or "unknown").strip() or "unknown"
    signal = "PASS" if passed else "FAIL"
    clean_advice = str(advice or "").strip()
    lines = [
        f"Evaluation signal: {signal}",
        f"Verifier mode: {normalized_mode}",
        f"Score: {passed}",
    ]
    if clean_advice:
        lines.append(f"Hint: {clean_advice}")
    else:
        lines.append(f"Hint: Verifier: {signal} ({normalized_mode}).")
    return "\n".join(lines)


def _format_evaluator_verdict_only(mode: str, passed: bool) -> str:
    normalized_mode = str(mode or "unknown").strip() or "unknown"
    signal = "PASS" if passed else "FAIL"
    return "\n".join(
        [
            f"Evaluation signal: {signal}",
            f"Verifier mode: {normalized_mode}",
            f"Score: {passed}",
        ]
    )


def _format_evaluator_hint_only(advice: str) -> str:
    clean_advice = str(advice or "").strip()
    return f"Evaluation hint: {clean_advice}" if clean_advice else "Evaluation hint: [none]"


def _has_substantive_feedback(text: str) -> bool:
    clean = str(text or "").strip()
    return bool(clean) and clean.lower() != "no advice yet."


def _strip_terminal_route_tag(text: str) -> str:
    return re.sub(
        r"\n?\[(?:Agree|Route:Planner|Route:Executor|Route:Evaluator|Route:Evaluate)\]\s*$",
        "",
        str(text or "").strip(),
        flags=re.IGNORECASE,
    ).rstrip()


def _build_action_interface_instruction(
    *,
    actor_label: str,
    expects_final_answer: bool,
    feedback_location: str,
) -> str:
    revise_target = "your revised solution" if expects_final_answer else "your revised plan"
    final_answer_line = (
        "- In `Revise`, give a concise corrected solution and keep the LAST line as EXACTLY one boxed final answer."
        if expects_final_answer
        else "- In `Revise`, provide only a compact revised plan and do NOT provide the final answer."
    )
    patch_behavior_line = (
        "- Treat the critique as a patch request to the current draft; do not restart from scratch unless the critique says the whole approach is invalid."
    )
    concision_line = (
        "- Keep `Acknowledge`, `Decision`, and `Action` to one short sentence or bullet each."
    )
    final_answer_guard = (
        "- Do not add any section after the boxed final answer. If you are uncertain, still give your best corrected boxed answer."
        if expects_final_answer
        else "- Keep the revised plan short (at most 3 brief bullets or 3 short paragraphs)."
    )
    return "\n".join(
        [
            "ACTION INTERFACE REQUIREMENT (mandatory for this turn):",
            f"- You are the {actor_label} responding to reviewer/system critique carried in {feedback_location}.",
            "- Before revising, explicitly process the critique instead of bypassing it.",
            patch_behavior_line,
            "- Use EXACTLY these top-level headers in this order:",
            "  Acknowledge:",
            "  Decision:",
            "  Action:",
            "  Revise:",
            concision_line,
            "- In `Acknowledge`, state the concrete issue the reviewer/system identified.",
            "- In `Decision`, choose exactly one of: accept / partially accept / reject.",
            "- In `Action`, state the concrete change you will make next.",
            f"- In `Revise`, provide {revise_target}.",
            final_answer_line,
            final_answer_guard,
        ]
    )


def _build_system_feedback_message(label: str, feedback: str) -> Message | None:
    clean = _strip_terminal_route_tag(feedback)
    if not clean:
        return None
    return Message(
        sender="system",
        content=f"{label}\n{clean}".strip(),
    )


def _build_cip_challenge_feedback(
    review_text: str,
    decision: CritiqueInterventionDecision,
) -> str:
    return (
        "Critique Intervention Policy decision: CHALLENGE.\n"
        "The reviewer critique may be useful, but the policy is not confident "
        "enough to make it binding. Re-check the critique against the problem "
        "and current candidate before revising. If you revise, make a concrete "
        "answer-level change rather than restating the same candidate.\n"
        f"Policy scores: repair_prob={decision.repair_prob:.4f}, "
        f"harm_prob={decision.harm_prob:.4f}.\n\n"
        "Original reviewer feedback:\n"
        f"{str(review_text or '').strip()}"
    ).strip()


def _build_cip_augmented_feedback(
    review_text: str,
    decision: CritiqueInterventionDecision,
    augment_scores: list[str],
) -> str:
    """Prepend predicted-score metadata to the reviewer message.

    This does NOT filter or re-route the critique. The full reviewer message is
    always passed; the solver additionally sees the requested predicted score(s)
    so it can weigh how reliable / useful the critique is likely to be.
    """
    parts: list[str] = []
    # Accept both the new term ("trajectory") and legacy ("integration").
    show_traj = ("trajectory" in augment_scores) or ("integration" in augment_scores)
    if "correctness" in augment_scores:
        parts.append(
            f"Predicted correctness: {decision.credibility_prob:.2f} "
            "(estimated probability that the signal is locally correct or directionally valid)"
        )
    if show_traj:
        # Wording deliberately avoids implying "revise"/"incorporate": the score
        # is about whether *carefully considering* the signal helps, not an
        # instruction to change the answer.
        parts.append(
            f"Predicted trajectory value: {decision.repair_prob:.2f} "
            "(estimated probability that carefully considering this signal will improve the "
            "downstream reasoning trajectory or final answer, regardless of whether the "
            "signal is locally correct)"
        )
    if not parts:
        return str(review_text or "").strip()
    annotation = "\n".join(parts)
    return (
        "[The following metadata is an ESTIMATED, fallible property of the reasoning "
        "signal below.\n"
        f"{annotation}\n"
        "These estimates may be imperfect. Use them as evidence, not as instructions. "
        "Do not change your answer unless you identify a concrete reasoning reason.]\n\n"
        "Reviewer feedback:\n"
        f"{str(review_text or '').strip()}"
    ).strip()


def _build_placebo_feedback(review_text: str) -> str:
    """Length/structure-matched baseline: same wrapper shape as the augmented
    conditions but with NO score information. Isolates the score content from the
    mere presence of extra framing text.
    """
    return (
        "[No signal-level estimates are available for this reasoning signal.\n"
        "This note is informational only. Use the reviewer feedback as usual.\n"
        "Do not change your answer unless you identify a concrete reasoning reason.]\n\n"
        "Reviewer feedback:\n"
        f"{str(review_text or '').strip()}"
    ).strip()


def _build_cip_block_feedback(
    review_text: str,
    decision: CritiqueInterventionDecision,
) -> str:
    return (
        "Critique Intervention Policy decision: DO NOT APPLY THIS CRITIQUE YET.\n"
        "The policy estimates elevated harm risk for acting on the reviewer "
        "critique. Preserve the current candidate unless the evaluator or a "
        "fresh plan identifies a concrete answer-level error.\n"
        f"Policy scores: repair_prob={decision.repair_prob:.4f}, "
        f"harm_prob={decision.harm_prob:.4f}.\n\n"
        "Reviewer feedback being held back:\n"
        f"{str(review_text or '').strip()}"
    ).strip()


def _format_reviewer_feedback(
    mode: str,
    passed: bool,
    advice: str,
    reviewer_feedback_mode: str,
) -> str:
    normalized_mode = str(mode or "unknown").strip() or "unknown"
    signal = "PASS" if passed else "FAIL"
    feedback_mode = str(reviewer_feedback_mode or "plain").strip().lower() or "plain"

    if feedback_mode == "hint":
        return _format_evaluator_feedback(mode, passed, advice)

    if passed:
        plain_advice = f"Verifier: PASS ({normalized_mode})."
    elif normalized_mode in {"exact", "numeric-verifier", "omni-rule"}:
        plain_advice = (
            f"Verifier: FAIL ({normalized_mode}). "
            "Re-check the reasoning and the final Reviewer's boxed answer."
        )
    else:
        plain_advice = (
            f"Verifier: FAIL ({normalized_mode}). "
            "Re-check the reasoning and the final reviewer answer."
        )

    lines = [
        f"Evaluation signal: {signal}",
        f"Verifier mode: {normalized_mode}",
        f"Score: {passed}",
        f"Advice: {plain_advice}",
    ]
    return "\n".join(lines)


def _extract_reviewer_submission_candidate(review_text: str) -> str:
    raw = str(review_text or "").strip()
    if not raw:
        return ""
    match = re.search(
        r"Final Answer For Evaluator:\s*(.*?)(?=\n\[(?:Route:Planner|Route:Executor|Route:Evaluator|Route:Evaluate|Agree)\]\s*$|\Z)",
        raw,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return ""
    block = match.group(1).strip()
    if not block:
        return ""
    boxed = extract_boxed(block)
    if boxed:
        return f"\\boxed{{{boxed}}}"
    return block


def _build_candidate_trace_context(
    *,
    plan_text: str,
    exec_text: str,
    review_text: str = "",
    candidate_text: str = "",
    source_label: str = "",
) -> str:
    sections: List[str] = []
    if str(plan_text or "").strip():
        sections.append(f"Planner plan:\n{str(plan_text).strip()}")
    if str(exec_text or "").strip():
        sections.append(f"Executor output:\n{str(exec_text).strip()}")
    if str(review_text or "").strip():
        sections.append(f"Reviewer message:\n{str(review_text).strip()}")
    if str(candidate_text or "").strip():
        label = str(source_label or "Current submitted candidate").strip()
        sections.append(f"{label}:\n{str(candidate_text).strip()}")
    return "\n\n".join(sections).strip()


def _append_log(
    logs: List[Dict[str, Any]],
    event: Dict[str, Any],
    live_log_sink: Any = None,
) -> None:
    logs.append(event)
    if live_log_sink is not None:
        try:
            live_log_sink(event)
        except Exception:
            pass


class PerRule:
    """
    Planner -> Executor -> Reviewer with an inner-loop evaluator that is built
    from the shared evaluator registry.

    Reviewer routing still controls the next inner step, but PASS / FAIL comes
    from the configured evaluator instead of being hardcoded in this class.
    """

    evaluator: BaseEvaluator

    def __init__(self, evaluator_config: Dict[str, Any] | None = None, **kwargs):
        self.inner_rounds: int = int(kwargs.pop("inner_rounds", 2))
        self.stop_on_agree: bool = bool(kwargs.pop("stop_on_agree", True))
        self.allow_reviewer_override: bool = bool(kwargs.pop("allow_reviewer_override", False))
        self.require_explicit_critique_uptake: bool = bool(
            kwargs.pop("require_explicit_critique_uptake", False)
        )
        self.embed_action_feedback_in_history: bool = bool(
            kwargs.pop("embed_action_feedback_in_history", False)
        )
        feedback_mode = str(kwargs.pop("reviewer_feedback_mode", "plain")).strip().lower()
        if feedback_mode not in {"plain", "hint"}:
            feedback_mode = "plain"
        self.reviewer_feedback_mode: str = feedback_mode
        self.failed_attempt_memory_limit: int = int(
            kwargs.pop("failed_attempt_memory_limit", 3)
        )
        self.failed_attempt_memory = FailedAttemptMemory(self.failed_attempt_memory_limit)
        self.candidate_revision_memory_limit: int = int(
            kwargs.pop("candidate_revision_memory_limit", 5)
        )
        self.candidate_revision_memory = CandidateRevisionMemory(
            self.candidate_revision_memory_limit
        )
        self.prompt_advice_token_budget: int = int(
            kwargs.pop("prompt_advice_token_budget", 0) or 0
        )
        self.prompt_plan_token_budget: int = int(
            kwargs.pop("prompt_plan_token_budget", 0) or 0
        )
        self.prompt_solution_token_budget: int = int(
            kwargs.pop("prompt_solution_token_budget", 0) or 0
        )
        self.evaluator_reasoning_token_budget: int = int(
            kwargs.pop("evaluator_reasoning_token_budget", 0) or 0
        )
        self.evaluator_memory_token_budget: int = int(
            kwargs.pop("evaluator_memory_token_budget", 0) or 0
        )
        self.critique_intervention_policy = RuntimeCritiqueInterventionPolicy.from_config(
            kwargs.pop("critique_intervention_policy", None)
        )

        evaluator_config = dict(evaluator_config or {"type": "numeric-verifier"})
        evaluator_type = str(evaluator_config.pop("type", "numeric-verifier")).strip().lower()
        self.evaluator_type = evaluator_type
        self.evaluator = evaluator_registry.build(evaluator_type, **evaluator_config)

    def reset(self) -> None:
        self.evaluator.reset()
        self.failed_attempt_memory.reset()
        self.candidate_revision_memory.reset()

    def _failed_attempt_memory_size(self) -> int:
        return len(self.failed_attempt_memory.records)

    def _candidate_revision_memory_size(self) -> int:
        return len(self.candidate_revision_memory.records)

    def _agent_debug_kwargs(self, stage: str) -> Dict[str, Any]:
        return {
            "debug_prompt_stage": stage,
            "debug_failed_attempt_memory_size": self._failed_attempt_memory_size(),
            "debug_candidate_revision_memory_size": self._candidate_revision_memory_size(),
        }

    def _set_evaluator_debug_context(self, agent: Any, stage: str) -> None:
        if agent is None:
            return
        setattr(agent, "current_debug_prompt_stage", stage)
        setattr(
            agent,
            "current_failed_attempt_memory_size",
            self._failed_attempt_memory_size(),
        )
        setattr(
            agent,
            "current_candidate_revision_memory_size",
            self._candidate_revision_memory_size(),
        )

    def _question_memory_context(self) -> str:
        parts: List[str] = []
        if self.candidate_revision_memory.records:
            parts.append(self.candidate_revision_memory.render())
        if self.failed_attempt_memory.records:
            parts.append(self.failed_attempt_memory.render())
        return "\n\n".join(parts).strip()

    def _advice_with_question_memory(self, advice: str) -> str:
        clean_advice = str(advice or "").strip()
        memory_context = self._question_memory_context()
        if not memory_context:
            return with_authority_guide(clean_advice)
        if memory_context in clean_advice:
            return with_authority_guide(clean_advice)
        return with_authority_guide(
            "\n\n".join(part for part in [clean_advice, memory_context] if part).strip()
        )

    def _trim_for_agent_prompt(
        self,
        text: str,
        *,
        model: str,
        token_budget: int,
        keep: str = "head_tail",
    ) -> str:
        if token_budget <= 0:
            return str(text or "").strip()
        return _trim_text_for_prompt(
            text,
            model=model,
            token_budget=token_budget,
            keep=keep,
        )

    def _prepare_action_advice(
        self,
        raw_feedback: str,
        *,
        actor_label: str,
        expects_final_answer: bool,
        include_question_memory: bool,
        model: str,
    ) -> str:
        clean_feedback = str(raw_feedback or "").strip()

        if self.embed_action_feedback_in_history:
            base_advice = (
                self._advice_with_question_memory("")
                if include_question_memory
                else ""
            )
            feedback_location = "the recent shared chat history"
        else:
            base_advice = (
                self._advice_with_question_memory(clean_feedback)
                if include_question_memory
                else clean_feedback
            )
            feedback_location = "the reviewer/system feedback in this prompt"

        if not (
            self.require_explicit_critique_uptake and _has_substantive_feedback(clean_feedback)
        ):
            return self._trim_for_agent_prompt(
                base_advice,
                model=model,
                token_budget=self.prompt_advice_token_budget,
                keep="head_tail",
            )

        instruction = _build_action_interface_instruction(
            actor_label=actor_label,
            expects_final_answer=expects_final_answer,
            feedback_location=feedback_location,
        )
        return self._trim_for_agent_prompt(
            "\n\n".join(
                part for part in [instruction, base_advice] if str(part).strip()
            ).strip(),
            model=model,
            token_budget=self.prompt_advice_token_budget,
            keep="head_tail",
        )

    def _embed_feedback_into_shared_context(
        self,
        agent_list: List[Any],
        *,
        label: str,
        feedback: str,
    ) -> None:
        if not self.embed_action_feedback_in_history:
            return
        msg = _build_system_feedback_message(label, feedback)
        if msg is not None:
            _broadcast_to_agents(agent_list, msg)

    def _record_candidate_event(
        self,
        logs: List[Dict[str, Any]],
        *,
        round_id: int,
        stage: str,
        source: str,
        action: str,
        candidate_text: str = "",
        rationale_or_review: str = "",
        evaluator_signal: str = "",
        live_log_sink: Any = None,
    ) -> None:
        record = self.candidate_revision_memory.add(
            stage=stage,
            source=source,
            action=action,
            candidate_answer=candidate_text,
            rationale_or_review=rationale_or_review,
            evaluator_signal=evaluator_signal,
        )
        if record is None:
            return
        _append_log(
            logs,
            {
                "type": "summary",
                "round": round_id,
                "stage": f"{stage}_candidate_memory",
                "sender": "system",
                "content": (
                    f"C{record.event_id} {record.action} "
                    f"{record.candidate_answer}"
                ),
                "candidate_memory_event": record.to_dict(),
                "trace_visible": False,
            },
            live_log_sink,
        )

    def _apply_critique_intervention_policy(
        self,
        logs: List[Dict[str, Any]],
        *,
        round_id: int,
        stage: str,
        route: str,
        review_text: str,
        candidate_text: str,
        reviewer_role: str,
        review_turn: int,
        after_evaluator_fail: bool,
        live_log_sink: Any = None,
    ) -> Tuple[str, str, CritiqueInterventionDecision]:
        decision = self.critique_intervention_policy.decide(
            route=route,
            review_text=review_text,
            candidate_text=candidate_text,
            reviewer_role=reviewer_role,
            review_turn=review_turn,
            after_evaluator_fail=after_evaluator_fail,
            stage=stage,
        )
        if decision.enabled:
            _append_log(
                logs,
                {
                    "type": "meta",
                    "round": round_id,
                    "stage": f"{stage}_critique_intervention_policy",
                    "sender": "system",
                    "content": (
                        "CIP decision: "
                        f"{decision.action} "
                        f"(would_action={decision.would_action or decision.action}, "
                        f"credibility_prob={decision.credibility_prob:.4f}, "
                        f"repair_prob={decision.repair_prob:.4f}, "
                        f"harm_prob={decision.harm_prob:.4f}). "
                        f"{decision.reason}"
                    ),
                    "critique_intervention_policy": decision.to_log_dict(),
                },
                live_log_sink,
            )
        if not decision.enabled or decision.action == "accept":
            return route, review_text, decision
        if decision.action == "augment":
            # Always pass the message; only prepend score metadata. Keep route.
            augmented = _build_cip_augmented_feedback(
                review_text,
                decision,
                getattr(self.critique_intervention_policy, "augment_scores", []),
            )
            return route, augmented, decision
        if decision.action == "challenge":
            challenged_feedback = _build_cip_challenge_feedback(review_text, decision)
            return (
                self.critique_intervention_policy.challenge_route,
                challenged_feedback,
                decision,
            )
        if decision.action == "ignore":
            blocked_feedback = _build_cip_block_feedback(review_text, decision)
            return self.critique_intervention_policy.harm_route, blocked_feedback, decision
        return route, review_text, decision

    async def astep(
        self,
        task_description: str,
        agents: Dict[Any, Any],
        advice: str = "No advice yet.",
        previous_plan: str = "No solution yet.",
        ground_truth: str = "",
        reference_solution: str = "",
        round_id: int = 0,
        live_log_sink: Any = None,
        **kwargs,
    ) -> Tuple[str, str, str, List[Dict[str, Any]], bool]:
        logs: List[Dict[str, Any]] = []

        planner = agents.get(AGENT_TYPES.SOLVER, None)
        executor = agents.get(AGENT_TYPES.EXECUTION, None)
        reviewer_obj = agents.get(AGENT_TYPES.CRITIC, None)
        reviewer = reviewer_obj[0] if isinstance(reviewer_obj, list) and reviewer_obj else reviewer_obj
        evaluator_agent = agents.get(AGENT_TYPES.EVALUATION, None)

        if planner is None or executor is None or reviewer is None:
            raise ValueError(
                "PER requires agents: solver(Planner), executor(Executor), critic(Reviewer). "
                f"Got planner={planner is not None}, executor={executor is not None}, reviewer={reviewer is not None}"
            )

        shared_reasoning_agents = [planner, reviewer]
        if self.embed_action_feedback_in_history:
            shared_reasoning_agents.append(executor)
        for agent in shared_reasoning_agents:
            _configure_reasoning_history(agent)

        cur_previous_plan = previous_plan or ""
        last_plan_text = cur_previous_plan
        last_exec_text = ""
        last_review_text = ""
        current_candidate = ""
        current_candidate_trace = ""

        queued_planner_advice = ""
        queued_executor_advice = ""
        next_actor = "planner"

        if "[Route:Planner]" in (advice or ""):
            queued_planner_advice = advice
            self._embed_feedback_into_shared_context(
                shared_reasoning_agents,
                label="System feedback carried from the previous outer try:",
                feedback=advice,
            )
            if self.embed_action_feedback_in_history:
                queued_planner_advice = ""
            next_actor = "planner"
        elif "[Route:Executor]" in (advice or ""):
            queued_executor_advice = advice
            self._embed_feedback_into_shared_context(
                shared_reasoning_agents,
                label="System feedback carried from the previous outer try:",
                feedback=advice,
            )
            if self.embed_action_feedback_in_history:
                queued_executor_advice = ""
            next_actor = "executor"
        elif "Verifier: FAIL" in (advice or ""):
            failed_candidate = _extract_candidate_from_advice(advice)
            queued_planner_advice = advice
            self._embed_feedback_into_shared_context(
                shared_reasoning_agents,
                label="System feedback carried from the previous outer try:",
                feedback=advice,
            )
            if self.embed_action_feedback_in_history:
                queued_planner_advice = ""
            if failed_candidate and not last_exec_text:
                last_exec_text = failed_candidate
            if failed_candidate and not current_candidate:
                current_candidate = failed_candidate
                current_candidate_trace = (
                    "Carried-over submitted candidate from the previous outer round:\n"
                    f"{failed_candidate}"
                )
            next_actor = "planner"

        for k in range(self.inner_rounds):
            reviewer_advice = advice
            if next_actor == "planner":
                planner_raw_feedback = queued_planner_advice or advice
                planner_advice = self._prepare_action_advice(
                    planner_raw_feedback,
                    actor_label="Planner",
                    expects_final_answer=False,
                    include_question_memory=True,
                    model=str(planner.llm.args.model),
                )
                planner_previous_plan = self._trim_for_agent_prompt(
                    last_plan_text,
                    model=str(planner.llm.args.model),
                    token_budget=self.prompt_plan_token_budget,
                    keep="head_tail",
                )
                plan_msg = await planner.astep(
                    former_solution=planner_previous_plan,
                    previous_plan=planner_previous_plan,
                    advice=planner_advice,
                    task_description=task_description,
                    **self._agent_debug_kwargs(f"planner_inner_{k}"),
                )

                plan_text = getattr(plan_msg, "content", "") or ""
                _append_log(
                    logs,
                    {
                        "type": "message",
                        "round": round_id,
                        "stage": f"planner_{k}",
                        "sender": getattr(plan_msg, "sender", "Planner"),
                        "content": plan_text,
                    },
                    live_log_sink,
                )
                _broadcast_to_agents(shared_reasoning_agents, plan_msg)

                queued_planner_advice = ""
                last_plan_text = plan_text or last_plan_text
                reviewer_advice = planner_advice

                executor_advice = self._prepare_action_advice(
                    "",
                    actor_label="Executor",
                    expects_final_answer=True,
                    include_question_memory=False,
                    model=str(executor.llm.args.model),
                )
                executor_solution = self._trim_for_agent_prompt(
                    last_plan_text,
                    model=str(executor.llm.args.model),
                    token_budget=self.prompt_plan_token_budget,
                    keep="head_tail",
                )
                exec_msg = await executor.astep(
                    task_description=task_description,
                    solution=executor_solution,
                    advice=executor_advice,
                    tools=[],
                    disable_chat_history=not self.embed_action_feedback_in_history,
                    **self._agent_debug_kwargs(f"executor_inner_{k}"),
                )
                exec_text = getattr(exec_msg, "content", "") or "[No execution output]"
                _append_log(
                    logs,
                    {
                        "type": "message",
                        "round": round_id,
                        "stage": f"executor_{k}",
                        "sender": getattr(exec_msg, "sender", "Executor"),
                        "content": exec_text,
                    },
                    live_log_sink,
                )
                _broadcast_to_agents(shared_reasoning_agents, exec_msg)
                last_exec_text = exec_text
                if extract_boxed(exec_text) or (not current_candidate and exec_text.strip()):
                    current_candidate = exec_text.strip()
                    current_candidate_trace = _build_candidate_trace_context(
                        plan_text=last_plan_text,
                        exec_text=exec_text,
                        candidate_text=current_candidate,
                        source_label="Current candidate after Executor",
                    )
                self._record_candidate_event(
                    logs,
                    round_id=round_id,
                    stage=f"executor_{k}",
                    source=getattr(exec_msg, "sender", "Executor"),
                    action="propose",
                    candidate_text=exec_text,
                    rationale_or_review=exec_text,
                    live_log_sink=live_log_sink,
                )
            elif next_actor == "executor":
                exec_raw_feedback = str(queued_executor_advice or advice or "").strip()
                exec_advice = self._prepare_action_advice(
                    exec_raw_feedback,
                    actor_label="Executor",
                    expects_final_answer=True,
                    include_question_memory=False,
                    model=str(executor.llm.args.model),
                )
                executor_solution = self._trim_for_agent_prompt(
                    last_plan_text,
                    model=str(executor.llm.args.model),
                    token_budget=self.prompt_plan_token_budget,
                    keep="head_tail",
                )
                exec_msg = await executor.astep(
                    task_description=task_description,
                    solution=executor_solution,
                    advice=exec_advice,
                    tools=[],
                    disable_chat_history=not self.embed_action_feedback_in_history,
                    **self._agent_debug_kwargs(f"executor_inner_{k}"),
                )
                exec_text = getattr(exec_msg, "content", "") or "[No execution output]"
                _append_log(
                    logs,
                    {
                        "type": "message",
                        "round": round_id,
                        "stage": f"executor_{k}",
                        "sender": getattr(exec_msg, "sender", "Executor"),
                        "content": exec_text,
                    },
                    live_log_sink,
                )
                _broadcast_to_agents(shared_reasoning_agents, exec_msg)

                queued_executor_advice = ""
                last_exec_text = exec_text
                reviewer_advice = exec_advice
                if extract_boxed(exec_text) or (not current_candidate and exec_text.strip()):
                    current_candidate = exec_text.strip()
                    current_candidate_trace = _build_candidate_trace_context(
                        plan_text=last_plan_text,
                        exec_text=exec_text,
                        candidate_text=current_candidate,
                        source_label="Current candidate after Executor",
                    )
                self._record_candidate_event(
                    logs,
                    round_id=round_id,
                    stage=f"executor_{k}",
                    source=getattr(exec_msg, "sender", "Executor"),
                    action="propose",
                    candidate_text=exec_text,
                    rationale_or_review=exec_text,
                    live_log_sink=live_log_sink,
                )
            reviewer_solution = self._trim_for_agent_prompt(
                last_exec_text or "[No execution output]",
                model=str(reviewer.llm.args.model),
                token_budget=self.prompt_solution_token_budget,
                keep="head_tail",
            )
            reviewer_previous_plan = self._trim_for_agent_prompt(
                last_plan_text,
                model=str(reviewer.llm.args.model),
                token_budget=self.prompt_plan_token_budget,
                keep="head_tail",
            )
            reviewer_prompt_advice = self._trim_for_agent_prompt(
                self._advice_with_question_memory(reviewer_advice),
                model=str(reviewer.llm.args.model),
                token_budget=self.prompt_advice_token_budget,
                keep="head_tail",
            )
            review_msg = await reviewer.astep(
                preliminary_solution=reviewer_solution,
                previous_plan=reviewer_previous_plan,
                advice=reviewer_prompt_advice,
                task_description=task_description,
                all_roles="Planner / Executor / Reviewer",
                **self._agent_debug_kwargs(f"reviewer_inner_{k}"),
            )

            review_text = getattr(review_msg, "content", "") or ""
            last_review_text = review_text
            _append_log(
                logs,
                {
                    "type": "message",
                    "round": round_id,
                    "stage": f"reviewer_{k}",
                    "sender": getattr(review_msg, "sender", "Reviewer"),
                    "content": review_text,
                },
                live_log_sink,
            )
            _broadcast_to_agents(shared_reasoning_agents, review_msg)
            route = _parse_route(review_text)
            reviewer_candidate = ""
            if route in {"agree", "evaluate"}:
                reviewer_candidate = _extract_reviewer_submission_candidate(review_text)
                if reviewer_candidate:
                    current_candidate = reviewer_candidate
                    current_candidate_trace = _build_candidate_trace_context(
                        plan_text=last_plan_text,
                        exec_text=last_exec_text,
                        review_text=review_text,
                        candidate_text=current_candidate,
                        source_label="Current candidate submitted by Reviewer",
                    )
            self._record_candidate_event(
                logs,
                round_id=round_id,
                stage=f"reviewer_{k}",
                source=getattr(review_msg, "sender", "Reviewer"),
                action="review_submit" if reviewer_candidate else "review_route",
                    candidate_text=reviewer_candidate,
                    rationale_or_review=review_text,
                    live_log_sink=live_log_sink,
                )
            route, effective_review_text, _ = self._apply_critique_intervention_policy(
                logs,
                round_id=round_id,
                stage=f"reviewer_{k}",
                route=route,
                review_text=review_text,
                candidate_text=current_candidate or last_exec_text,
                reviewer_role=getattr(review_msg, "sender", "Reviewer"),
                review_turn=k,
                after_evaluator_fail=False,
                live_log_sink=live_log_sink,
            )
            last_review_text = effective_review_text
            if route in ("agree", "evaluate") and self.stop_on_agree:
                break

            if route == "planner":
                self._embed_feedback_into_shared_context(
                    shared_reasoning_agents,
                    label="Reviewer feedback for the next action:",
                    feedback=effective_review_text,
                )
                queued_planner_advice = effective_review_text
                if self.embed_action_feedback_in_history:
                    queued_planner_advice = ""
                next_actor = "planner"
            elif route == "executor":
                self._embed_feedback_into_shared_context(
                    shared_reasoning_agents,
                    label="Reviewer feedback for the next action:",
                    feedback=effective_review_text,
                )
                queued_executor_advice = effective_review_text
                if self.embed_action_feedback_in_history:
                    queued_executor_advice = ""
                next_actor = "executor"
            elif route in ("agree", "evaluate"):
                next_actor = "planner"
            else:
                self._embed_feedback_into_shared_context(
                    shared_reasoning_agents,
                    label="Reviewer feedback for the next action:",
                    feedback=effective_review_text,
                )
                queued_executor_advice = effective_review_text
                if self.embed_action_feedback_in_history:
                    queued_executor_advice = ""
                next_actor = "executor"

        async def evaluate_current_candidate(
            submitted_candidate: str,
            reasoning_trace: str,
            stage_name: str,
            stage_result_name: str,
        ) -> Tuple[bool, Dict[str, Any], str, str]:
            candidate_for_eval = str(submitted_candidate or "").strip()
            current_reasoning_trace = str(reasoning_trace or "").strip()
            evaluator_model = str(evaluator_agent.llm.args.model)
            previous_answer_summary = self._trim_for_agent_prompt(
                self.failed_attempt_memory.render_for_evaluator_hint(),
                model=evaluator_model,
                token_budget=self.evaluator_memory_token_budget,
                keep="head_tail",
            )
            previous_reasoning_summary = self._trim_for_agent_prompt(
                self.candidate_revision_memory.render_for_evaluator_hint(),
                model=evaluator_model,
                token_budget=self.evaluator_memory_token_budget,
                keep="head_tail",
            )
            current_reasoning_summary = self._trim_for_agent_prompt(
                current_reasoning_trace or candidate_for_eval,
                model=evaluator_model,
                token_budget=self.evaluator_reasoning_token_budget,
                keep="head_tail",
            )
            self._set_evaluator_debug_context(
                evaluator_agent,
                f"{stage_name}_verdict",
            )
            _append_log(
                logs,
                {
                    "type": "message",
                    "round": round_id,
                    "stage": f"{stage_name}_submission",
                    "sender": "system",
                    "content": (
                        "Actual extracted final answer sent to evaluator: "
                        f"{_submitted_final_answer_for_trace(candidate_for_eval)}"
                    ),
                },
                live_log_sink,
            )
            self._record_candidate_event(
                logs,
                round_id=round_id,
                stage=f"{stage_name}_submission",
                source="system",
                action="submit",
                candidate_text=candidate_for_eval,
                rationale_or_review="Submitted current candidate to evaluator.",
                live_log_sink=live_log_sink,
            )
            evaluation = await self.evaluator.astep(
                agent=evaluator_agent,
                solution=[],
                result=[],
                task_description=task_description,
                all_role_description=[
                    getattr(planner, "role_description", "Planner"),
                    getattr(executor, "role_description", "Executor"),
                    getattr(reviewer, "role_description", "Reviewer"),
                ],
                reviewer_output=candidate_for_eval,
                submitted_candidate=candidate_for_eval,
                ground_truth=str(ground_truth),
                reference_solution=str(reference_solution or ""),
                debug_stage_prefix=stage_name,
                previous_answer_summary=previous_answer_summary,
                previous_reasoning_summary=previous_reasoning_summary,
                current_reasoning_summary=current_reasoning_summary,
            )
            eval_result = (
                dict(evaluation.content) if isinstance(evaluation.content, dict) else {}
            )
            passed = _coerce_evaluator_score(evaluation.score)
            eval_result.setdefault("mode", self.evaluator_type)
            eval_result.setdefault("passed", passed)
            eval_result.setdefault("signal", "PASS" if passed else "FAIL")
            eval_result.setdefault("final_answer", "")
            eval_advice = str(evaluation.advice or "")
            direct_protocol_fail = bool(eval_result.get("direct_protocol_fail"))
            if direct_protocol_fail:
                eval_sender = "system"
            else:
                eval_sender = getattr(evaluation, "sender", "") or getattr(
                    evaluator_agent,
                    "name",
                    "Evaluator",
                )
            verdict_advice = str(
                eval_result.get("verdict_advice") or eval_advice or ""
            ).strip()
            hint_advice = str(eval_result.get("hint_advice") or "").strip()
            eval_feedback = (
                hint_advice
                or verdict_advice
                or str(evaluation.advice or "").strip()
            )

            verdict_log_content = _format_evaluator_verdict_only(
                eval_result["mode"],
                passed,
            )
            if eval_result.get("judge_device"):
                verdict_log_content += (
                    f"\nJudge student final answer: {eval_result.get('judge_student_final_answer')}"
                    f"\nJudge equivalence judgement: {eval_result.get('judge_equivalence_judgement')}"
                    f"\nJudge device: {eval_result.get('judge_device')}"
                    f"\nJudge dtype: {eval_result.get('judge_dtype')}"
                )

            _append_log(
                logs,
                {
                    "type": "message",
                    "round": round_id,
                    "stage": stage_name,
                    "sender": eval_sender,
                    "content": verdict_log_content,
                    "force_numbered_system_trace": bool(direct_protocol_fail),
                },
                live_log_sink,
            )
            if not passed and hint_advice:
                _append_log(
                    logs,
                    {
                        "type": "message",
                        "round": round_id,
                        "stage": f"{stage_name}_hint",
                        "sender": eval_sender,
                        "content": _format_evaluator_hint_only(hint_advice),
                        "force_numbered_system_trace": bool(direct_protocol_fail),
                    },
                    live_log_sink,
                )
            _append_log(
                logs,
                {
                    "type": "meta",
                    "round": round_id,
                    "stage": stage_result_name,
                    "sender": eval_sender,
                    "content": "PASS" if passed else "FAIL",
                    "mode": eval_result["mode"],
                    "signal": eval_result.get("signal", "PASS" if passed else "FAIL"),
                    "correctness": 1 if passed else 0,
                    "final_answer": eval_result.get("final_answer", ""),
                    "advice": eval_feedback,
                    "verdict_advice": verdict_advice,
                    "hint_advice": hint_advice,
                    "reviewer_feedback_mode": self.reviewer_feedback_mode,
                    "reference_solution_used": bool(eval_result.get("reference_solution_used")),
                    "judge_student_final_answer": eval_result.get("judge_student_final_answer", ""),
                    "judge_equivalence_judgement": eval_result.get("judge_equivalence_judgement", ""),
                    "judge_device": eval_result.get("judge_device"),
                    "judge_dtype": eval_result.get("judge_dtype"),
                    "rule_prediction": eval_result.get("rule_prediction", ""),
                },
                live_log_sink,
            )
            eval_final_answer = str(eval_result.get("final_answer", "") or "").strip()
            self._record_candidate_event(
                logs,
                round_id=round_id,
                stage=stage_name,
                source=eval_sender,
                action="evaluate",
                candidate_text=(
                    f"\\boxed{{{eval_final_answer}}}"
                    if eval_final_answer
                    else candidate_for_eval
                ),
                rationale_or_review=eval_feedback,
                evaluator_signal=eval_result.get("signal", "PASS" if passed else "FAIL"),
                live_log_sink=live_log_sink,
            )

            return passed, eval_result, eval_feedback, eval_sender

        reviewer_output = (last_review_text or "").strip()
        submission_candidate = (
            str(current_candidate or "").strip()
            or str(last_exec_text or "").strip()
        )
        submission_trace = current_candidate_trace or _build_candidate_trace_context(
            plan_text=last_plan_text,
            exec_text=last_exec_text,
            review_text=reviewer_output,
            candidate_text=submission_candidate,
            source_label="Current candidate before evaluation",
        )
        passed, eval_result, reviewer_eval_advice, eval_sender = await evaluate_current_candidate(
            submission_candidate,
            submission_trace,
            stage_name="evaluation",
            stage_result_name="evaluation_result",
        )

        final_success = passed
        post_review_content = ""
        next_round_guidance = ""
        latest_reviewer_output = reviewer_output
        latest_eval_feedback = reviewer_eval_advice
        if not passed:
            submitted_answer = (
                str(eval_result.get("final_answer", "") or "").strip()
                or submission_candidate
                or last_exec_text
            )
            self.failed_attempt_memory.add(
                submitted_answer=submitted_answer,
                evaluator_signal=str(eval_result.get("signal", "FAIL") or "FAIL"),
                evaluator_hint=reviewer_eval_advice,
                repair_summary="Pending Planner repair plan.",
                source_stage="initial_evaluation",
            )
            planner_fail_advice = self._advice_with_question_memory(
                reviewer_eval_advice
            )
            if self.embed_action_feedback_in_history:
                self._embed_feedback_into_shared_context(
                    shared_reasoning_agents,
                    label="System feedback after evaluator rejection:",
                    feedback=reviewer_eval_advice,
                )
                planner_fail_advice = self._prepare_action_advice(
                    "",
                    actor_label="Planner",
                    expects_final_answer=False,
                    include_question_memory=True,
                    model=str(planner.llm.args.model),
                )
            else:
                planner_fail_advice = self._prepare_action_advice(
                    reviewer_eval_advice,
                    actor_label="Planner",
                    expects_final_answer=False,
                    include_question_memory=True,
                    model=str(planner.llm.args.model),
                )
            planner_previous_plan = self._trim_for_agent_prompt(
                last_plan_text,
                model=str(planner.llm.args.model),
                token_budget=self.prompt_plan_token_budget,
                keep="head_tail",
            )
            repair_plan_msg = await planner.astep(
                former_solution=planner_previous_plan,
                previous_plan=planner_previous_plan,
                advice=planner_fail_advice,
                task_description=task_description,
                **self._agent_debug_kwargs("planner_post_eval_fail"),
            )
            post_review_content = getattr(repair_plan_msg, "content", "") or ""
            _append_log(
                logs,
                {
                    "type": "message",
                    "round": round_id,
                    "stage": "planner_post_eval_fail",
                    "sender": getattr(repair_plan_msg, "sender", "Planner"),
                    "content": post_review_content,
                },
                live_log_sink,
            )
            _broadcast_to_agents(shared_reasoning_agents, repair_plan_msg)
            self._record_candidate_event(
                logs,
                round_id=round_id,
                stage="planner_post_eval_fail",
                source=getattr(repair_plan_msg, "sender", "Planner"),
                action="repair_instruction",
                candidate_text=post_review_content,
                rationale_or_review=post_review_content,
                live_log_sink=live_log_sink,
            )
            self.failed_attempt_memory.update_latest_repair_summary(post_review_content)
            next_round_guidance = post_review_content.strip()
            last_plan_text = post_review_content or last_plan_text

            repair_instruction = (
                "Current internal Planner repair plan (not evaluator ground truth):\n"
                f"{post_review_content.strip() or reviewer_eval_advice}"
            ).strip()
            repair_exec_advice = (
                "\n\n".join(
                    part
                    for part in [
                        reviewer_eval_advice.strip(),
                        repair_instruction,
                    ]
                    if part
                ).strip()
            )
            repair_exec_advice = self._prepare_action_advice(
                repair_exec_advice,
                actor_label="Executor",
                expects_final_answer=True,
                include_question_memory=False,
                model=str(executor.llm.args.model),
            )
            repair_executor_solution = self._trim_for_agent_prompt(
                last_plan_text,
                model=str(executor.llm.args.model),
                token_budget=self.prompt_plan_token_budget,
                keep="head_tail",
            )

            repair_exec_msg = await executor.astep(
                task_description=task_description,
                solution=repair_executor_solution,
                advice=repair_exec_advice,
                tools=[],
                disable_chat_history=not self.embed_action_feedback_in_history,
                **self._agent_debug_kwargs("executor_post_eval_fail"),
            )
            repair_exec_text = (
                getattr(repair_exec_msg, "content", "") or "[No execution output]"
            )
            _append_log(
                logs,
                {
                    "type": "message",
                    "round": round_id,
                    "stage": "executor_post_eval_fail",
                    "sender": getattr(repair_exec_msg, "sender", "Executor"),
                    "content": repair_exec_text,
                },
                live_log_sink,
            )
            _broadcast_to_agents(shared_reasoning_agents, repair_exec_msg)
            last_exec_text = repair_exec_text
            if extract_boxed(repair_exec_text) or (not current_candidate and repair_exec_text.strip()):
                current_candidate = repair_exec_text.strip()
                current_candidate_trace = _build_candidate_trace_context(
                    plan_text=last_plan_text,
                    exec_text=repair_exec_text,
                    candidate_text=current_candidate,
                    source_label="Current candidate after repair Executor",
                )
            self._record_candidate_event(
                logs,
                round_id=round_id,
                stage="executor_post_eval_fail",
                source=getattr(repair_exec_msg, "sender", "Executor"),
                action="revise",
                candidate_text=repair_exec_text,
                rationale_or_review=repair_exec_text,
                live_log_sink=live_log_sink,
            )

            repair_review_advice = self._advice_with_question_memory(
                "\n\n".join(
                    part
                    for part in [
                        reviewer_eval_advice.strip(),
                        repair_instruction,
                    ]
                    if part
                ).strip()
            ) or reviewer_eval_advice
            repair_reviewer_solution = self._trim_for_agent_prompt(
                last_exec_text,
                model=str(reviewer.llm.args.model),
                token_budget=self.prompt_solution_token_budget,
                keep="head_tail",
            )
            repair_reviewer_previous_plan = self._trim_for_agent_prompt(
                last_plan_text,
                model=str(reviewer.llm.args.model),
                token_budget=self.prompt_plan_token_budget,
                keep="head_tail",
            )
            repair_reviewer_advice = self._trim_for_agent_prompt(
                repair_review_advice,
                model=str(reviewer.llm.args.model),
                token_budget=self.prompt_advice_token_budget,
                keep="head_tail",
            )
            repair_review_msg = await reviewer.astep(
                preliminary_solution=repair_reviewer_solution,
                previous_plan=repair_reviewer_previous_plan,
                advice=repair_reviewer_advice,
                task_description=task_description,
                all_roles="Planner / Executor / Reviewer",
                **self._agent_debug_kwargs("reviewer_post_eval_pipeline"),
            )
            repair_review_text = getattr(repair_review_msg, "content", "") or ""
            latest_reviewer_output = repair_review_text.strip() or latest_reviewer_output
            _append_log(
                logs,
                {
                    "type": "message",
                    "round": round_id,
                    "stage": "reviewer_post_eval_pipeline",
                    "sender": getattr(repair_review_msg, "sender", "Reviewer"),
                    "content": repair_review_text,
                },
                live_log_sink,
            )
            _broadcast_to_agents(shared_reasoning_agents, repair_review_msg)
            repair_route = _parse_route(repair_review_text)
            repair_reviewer_candidate = ""
            if repair_route in {"agree", "evaluate"}:
                repair_reviewer_candidate = _extract_reviewer_submission_candidate(
                    repair_review_text
                )
                if repair_reviewer_candidate:
                    current_candidate = repair_reviewer_candidate
                    current_candidate_trace = _build_candidate_trace_context(
                        plan_text=last_plan_text,
                        exec_text=last_exec_text,
                        review_text=repair_review_text,
                        candidate_text=current_candidate,
                        source_label="Current candidate submitted by Reviewer",
                    )
            self._record_candidate_event(
                logs,
                round_id=round_id,
                stage="reviewer_post_eval_pipeline",
                source=getattr(repair_review_msg, "sender", "Reviewer"),
                action="review_submit" if repair_reviewer_candidate else "review_route",
                candidate_text=repair_reviewer_candidate,
                rationale_or_review=repair_review_text,
                live_log_sink=live_log_sink,
            )

            repair_submission_candidate = (
                str(current_candidate or "").strip()
                or str(last_exec_text or "").strip()
                or latest_reviewer_output
            )
            repair_submission_trace = current_candidate_trace or _build_candidate_trace_context(
                plan_text=last_plan_text,
                exec_text=last_exec_text,
                review_text=latest_reviewer_output,
                candidate_text=repair_submission_candidate,
                source_label="Current candidate before repair evaluation",
            )
            repair_passed, repair_eval_result, repair_eval_feedback, _ = await evaluate_current_candidate(
                repair_submission_candidate,
                repair_submission_trace,
                stage_name="evaluation_post_pipeline_repair",
                stage_result_name="evaluation_post_pipeline_repair_result",
            )
            final_success = repair_passed
            latest_eval_feedback = repair_eval_feedback
            next_round_guidance = latest_reviewer_output
            if not repair_passed:
                repair_submitted_answer = (
                    str(repair_eval_result.get("final_answer", "") or "").strip()
                    or repair_submission_candidate
                )
                self.failed_attempt_memory.add(
                    submitted_answer=repair_submitted_answer,
                    evaluator_signal=str(repair_eval_result.get("signal", "FAIL") or "FAIL"),
                    evaluator_hint=repair_eval_feedback,
                    repair_summary=repair_review_text,
                    source_stage="repair_evaluation",
                )
                latest_eval_feedback = repair_eval_feedback
            if not repair_passed and _parse_route(latest_reviewer_output) not in {
                "planner",
                "executor",
            }:
                latest_reviewer_output = _force_post_fail_route(
                    latest_reviewer_output,
                    latest_eval_feedback,
                )
                next_round_guidance = latest_reviewer_output

        if (
            self.allow_reviewer_override
            and passed is False
            and "[Agree]" in (post_review_content or "")
        ):
            final_success = True

        _append_log(
            logs,
            {
                "type": "meta",
                "round": round_id,
                "stage": "system",
                "sender": "system",
                "content": "Good score! Accept!" if final_success else "Bad score! Reject!",
            },
            live_log_sink,
        )

        if final_success:
            final_advice = latest_eval_feedback
        else:
            candidate_handoff = _compact_candidate_handoff(
                current_candidate or latest_reviewer_output
            )
            final_advice_lines = [
                latest_eval_feedback,
                "[Candidate]",
                candidate_handoff,
                "[EndCandidate]",
            ]
            if next_round_guidance and next_round_guidance != latest_reviewer_output:
                final_advice_lines.append(next_round_guidance)
            final_advice = "\n".join(
                [line for line in final_advice_lines if str(line).strip()]
            ).strip()

        final_plan = last_plan_text or cur_previous_plan
        final_result = current_candidate or latest_reviewer_output or "[No candidate submitted]"
        return final_result, final_advice, final_plan, logs, final_success
