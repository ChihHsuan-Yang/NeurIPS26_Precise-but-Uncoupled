"""Evaluator-style candidate correctness labels for main-paper metrics."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any, Optional

from agentverse.evaluation.common import exact_match, extract_final_answer
from agentverse.metrics.candidate_label_cache import (
    CachedCandidateLabel,
    PersistentCandidateLabelCache,
)
from agentverse.metrics.repair import TraceExample, _is_evaluator_turn


@dataclass
class CandidateLabel:
    value: Optional[bool]
    source: str
    reason: str = ""


@dataclass
class EvaluatorAgentMessage:
    score: Optional[bool]
    advice: str = ""


class OfflineEvaluatorAgent:
    """Small offline runner for a saved `agent_type: evaluator` config."""

    def __init__(
        self,
        *,
        name: str,
        prepend_prompt_template: str,
        append_prompt_template: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ):
        self.name = name
        self.prepend_prompt_template = prepend_prompt_template
        self.append_prompt_template = append_prompt_template
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def reset(self) -> None:
        return None

    async def astep(
        self,
        *,
        solution: str,
        result: str,
        task_description: str,
        all_role_description: str,
    ) -> EvaluatorAgentMessage:
        try:
            from openai import AsyncOpenAI
        except Exception as exc:
            raise RuntimeError("The `openai` package is required for evaluator-agent replay.") from exc

        api_key = os.environ.get("OPENAI_API_KEY")
        base_url = os.environ.get("OPENAI_BASE_URL")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set. Source `use_alcf.sh` before running.")

        values = {
            "solution": solution,
            "result": result,
            "task_description": task_description,
            "all_role_description": all_role_description,
        }
        prepend = Template(self.prepend_prompt_template).safe_substitute(values)
        append = Template(self.append_prompt_template).safe_substitute(values)
        messages = []
        if prepend:
            messages.append({"role": "system", "content": prepend})
        if append:
            messages.append({"role": "user", "content": append})

        async with AsyncOpenAI(api_key=api_key, base_url=base_url) as client:
            response = await client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                messages=messages,
            )
        text = response.choices[0].message.content or ""
        score, advice = _parse_evaluator_agent_response(text)
        return EvaluatorAgentMessage(score=score, advice=advice or text)


class CandidateCorrectnessEvaluator:
    """Label candidate answers without a separate post-trace LLM judge.

    The main-paper path reuses runtime evaluator labels for submitted final
    answers, and otherwise compares extracted candidates to ground truth using
    the same evaluator family as the benchmark when that evaluator is available.
    """

    def __init__(
        self,
        *,
        evaluator_type: str = "exact",
        numeric_tolerance: str = "0",
        data_name: str = "omni-math",
        omni_judge_model_path: str | None = None,
        omni_judge_max_new_tokens: int = 300,
        omni_judge_device: str = "auto",
        omni_judge_dtype: str = "auto",
        evaluator_agent_config_path: str | Path | None = None,
        reuse_runtime_final_labels: bool = True,
        cache_candidate_labels: bool = True,
        candidate_label_cache_path: str | Path | None = None,
    ):
        self.evaluator_type = normalize_evaluator_type(evaluator_type)
        self.numeric_tolerance = str(numeric_tolerance or "0")
        self.data_name = str(data_name or "omni-math")
        self.omni_judge_model_path = omni_judge_model_path
        self.omni_judge_max_new_tokens = int(omni_judge_max_new_tokens or 300)
        self.omni_judge_device = str(omni_judge_device or "auto")
        self.omni_judge_dtype = str(omni_judge_dtype or "auto")
        self.evaluator_agent_config_path = (
            Path(evaluator_agent_config_path).expanduser()
            if evaluator_agent_config_path
            else None
        )
        self.reuse_runtime_final_labels = bool(reuse_runtime_final_labels)
        self.cache_candidate_labels = bool(cache_candidate_labels)
        self._cache: dict[tuple[str, str, str, str], CandidateLabel] = {}
        self._llm_agent = None
        self._role_descriptions: list[str] = []
        self._cache_signature = self._build_cache_signature()
        self._persistent_cache = (
            PersistentCandidateLabelCache(candidate_label_cache_path)
            if self.cache_candidate_labels and candidate_label_cache_path
            else None
        )
        self.cache_hits = 0
        self.evaluator_calls = 0
        self.persistent_cache_hits = 0

    def label_candidate(
        self,
        *,
        problem: str,
        gold_answer: str,
        candidate_answer: str,
        final_candidate: str = "",
        runtime_final: Optional[bool] = None,
    ) -> CandidateLabel:
        candidate = str(candidate_answer or "").strip()
        gold = str(gold_answer or "").strip()
        if not candidate or not gold:
            return CandidateLabel(None, "missing", "Missing candidate or gold answer.")

        if (
            self.reuse_runtime_final_labels
            and runtime_final is not None
            and candidate == str(final_candidate or "").strip()
        ):
            return CandidateLabel(
                bool(runtime_final),
                "runtime_evaluator",
                "Reused pass/fail recorded during task solving.",
            )

        cache_key = (self.evaluator_type, str(problem or ""), gold, candidate)
        cached = self._cache.get(cache_key)
        if cached is not None:
            self.cache_hits += 1
            return cached

        if self._persistent_cache is not None:
            persisted = self._persistent_cache.lookup(
                signature=self._cache_signature,
                problem=str(problem or ""),
                gold_answer=gold,
                candidate_answer=candidate,
            )
            if persisted is not None:
                self.persistent_cache_hits += 1
                restored = CandidateLabel(
                    persisted.value,
                    persisted.source,
                    persisted.reason,
                )
                self._cache[cache_key] = restored
                return restored

        self.evaluator_calls += 1
        result = self._label_with_evaluator(
            problem=str(problem or ""),
            gold_answer=gold,
            candidate_answer=candidate,
        )
        self._cache[cache_key] = result
        if self._persistent_cache is not None and result.value is not None:
            self._persistent_cache.store(
                signature=self._cache_signature,
                problem=str(problem or ""),
                gold_answer=gold,
                candidate_answer=candidate,
                label=CachedCandidateLabel(
                    value=bool(result.value),
                    source=result.source,
                    reason=result.reason,
                ),
            )
        return result

    def _label_with_evaluator(
        self,
        *,
        problem: str,
        gold_answer: str,
        candidate_answer: str,
    ) -> CandidateLabel:
        if self.evaluator_type in {"exact", "numeric-verifier"}:
            # exact/numeric-verifier correctness = extracted final answer equals reference.
            passed = exact_match(
                candidate_answer,
                gold_answer,
                tol=self.numeric_tolerance if self.evaluator_type == "numeric-verifier" else "0",
            )
            return CandidateLabel(
                passed,
                self.evaluator_type,
                "Compared extracted final answers with deterministic equivalence.",
            )

        if self.evaluator_type == "omni-rule":
            return self._label_with_omni_rule(problem, gold_answer, candidate_answer)

        if self.evaluator_type in {"omni-judge", "omni-verifier"}:
            return self._label_with_omni_judge(problem, gold_answer, candidate_answer)

        if self.evaluator_type in {"llm", "agent", "openai"}:
            return self._label_with_llm_agent(problem, gold_answer, candidate_answer)

        return CandidateLabel(
            None,
            f"unsupported:{self.evaluator_type}",
            "Evaluator type cannot be reconstructed offline without the benchmark agent.",
        )

    def _label_with_omni_rule(
        self,
        problem: str,
        gold_answer: str,
        candidate_answer: str,
    ) -> CandidateLabel:
        try:
            from agentverse.evaluation.omni_rule import get_cached_omni_rule_evaluator

            evaluator = get_cached_omni_rule_evaluator(self.data_name)
            row = evaluator.evaluate(
                [
                    {
                        "problem": problem,
                        "answer": str(gold_answer or ""),
                        "model_generation": _as_model_generation(candidate_answer),
                    }
                ]
            )[0]
        except Exception as exc:
            return CandidateLabel(None, "omni-rule-error", str(exc))

        return CandidateLabel(
            bool(row.get("correctness")),
            "omni-rule",
            "Compared candidate with Omni-MATH rule evaluator.",
        )

    def _label_with_omni_judge(
        self,
        problem: str,
        gold_answer: str,
        candidate_answer: str,
    ) -> CandidateLabel:
        try:
            from agentverse.evaluation.omni_judge import (
                DEFAULT_OMNI_JUDGE_MODEL_ID,
                get_cached_omni_judge_evaluator,
            )

            evaluator = get_cached_omni_judge_evaluator(
                model_path=self.omni_judge_model_path or DEFAULT_OMNI_JUDGE_MODEL_ID,
                max_new_tokens=self.omni_judge_max_new_tokens,
                device_preference=self.omni_judge_device,
                dtype_preference=self.omni_judge_dtype,
            )
            row = evaluator.evaluate(
                [
                    {
                        "problem": problem,
                        "answer": str(gold_answer or ""),
                        "model_generation": _as_model_generation(candidate_answer),
                    }
                ]
            )[0]
        except Exception as exc:
            return CandidateLabel(None, f"{self.evaluator_type}-error", str(exc))

        return CandidateLabel(
            bool(row.get("correctness")),
            self.evaluator_type,
            "Compared candidate with the benchmark Omni-Judge evaluator.",
        )

    def _label_with_llm_agent(
        self,
        problem: str,
        gold_answer: str,
        candidate_answer: str,
    ) -> CandidateLabel:
        agent = self._get_llm_evaluator_agent()
        if agent is None:
            return CandidateLabel(
                None,
                "llm-evaluator-agent-missing",
                "No run config with `agent_type: evaluator` was available.",
            )

        try:
            agent.reset()
            message = asyncio.run(
                agent.astep(
                    solution=_as_evaluator_submission(candidate_answer),
                    result=f"Ground truth:\n{gold_answer}",
                    task_description=problem,
                    all_role_description="\n".join(self._role_descriptions),
                )
            )
        except Exception as exc:
            return CandidateLabel(None, "llm-evaluator-agent-error", str(exc))

        return CandidateLabel(
            _coerce_score_to_bool(getattr(message, "score", None)),
            "evaluator_agent",
            str(getattr(message, "advice", "") or ""),
        )

    def _get_llm_evaluator_agent(self):
        if self._llm_agent is not None:
            return self._llm_agent
        if self.evaluator_agent_config_path is None:
            return None

        agent, role_descriptions = _load_evaluator_agent_from_run_config(
            self.evaluator_agent_config_path
        )
        self._llm_agent = agent
        self._role_descriptions = role_descriptions
        return self._llm_agent

    def _build_cache_signature(self) -> str:
        payload: dict[str, Any] = {
            "evaluator_type": self.evaluator_type,
            "numeric_tolerance": self.numeric_tolerance,
            "data_name": self.data_name,
            "omni_judge_model_path": self.omni_judge_model_path,
            "omni_judge_max_new_tokens": self.omni_judge_max_new_tokens,
            "omni_judge_device": self.omni_judge_device,
            "omni_judge_dtype": self.omni_judge_dtype,
        }
        if self.evaluator_type in {"llm", "agent", "openai"}:
            payload["evaluator_agent_config_hash"] = _config_file_hash(
                self.evaluator_agent_config_path
            )
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def infer_evaluator_type(examples: list[TraceExample]) -> str:
    """Infer benchmark evaluator type from saved evaluator feedback text."""
    for example in examples:
        evaluator_names = {
            name for name in example.agent_map.values() if "evaluator" in name.lower()
        }
        for turn in example.turns:
            if not _is_evaluator_turn(turn, evaluator_names):
                continue
            inferred = infer_evaluator_type_from_text(turn.content)
            if inferred:
                return inferred
    return "exact"


def infer_evaluator_type_from_text(text: str) -> str:
    raw = str(text or "")
    match = re.search(r"\bVerifier:\s*(?:PASS|FAIL)\s*\(([^)]+)\)", raw, re.IGNORECASE)
    if match:
        mode = match.group(1).split(";", 1)[0].strip().lower()
        normalized = normalize_evaluator_type(mode)
        if normalized != "same_as_benchmark":
            return normalized

    lowered = raw.lower()
    for marker in (
        "numeric-verifier",
        "omni-verifier",
        "omni-judge",
        "omni-rule",
        "exact",
    ):
        if marker in lowered:
            return marker
    return ""


def normalize_evaluator_type(value: str) -> str:
    normalized = str(value or "exact").strip().lower().replace("_", "-")
    aliases = {
        "same": "same_as_benchmark",
        "same-as-benchmark": "same_as_benchmark",
        "benchmark": "same_as_benchmark",
        "numeric": "numeric-verifier",
        "omni": "omni-verifier",
        "rule": "omni-rule",
    }
    return aliases.get(normalized, normalized)


def _config_file_hash(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    try:
        raw = path.read_text(encoding="utf-8")
    except Exception:
        return None
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _as_model_generation(candidate_answer: str) -> str:
    final_answer = extract_final_answer(candidate_answer)
    if "\\boxed" in str(candidate_answer):
        return str(candidate_answer)
    return f"\\boxed{{{final_answer}}}" if final_answer else str(candidate_answer or "")


def _as_evaluator_submission(candidate_answer: str) -> str:
    return "Final Answer For Evaluator:\n" + _as_model_generation(candidate_answer)


def _coerce_score_to_bool(score: Any) -> Optional[bool]:
    if isinstance(score, bool):
        return score
    if isinstance(score, int):
        if score in {0, 1}:
            return bool(score)
        return score >= 8
    if isinstance(score, (list, tuple)):
        values = [_coerce_score_to_bool(item) for item in score]
        values = [value for value in values if value is not None]
        if not values:
            return None
        return all(values)
    return None


def _load_evaluator_agent_from_run_config(config_path: Path):
    import yaml

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    agent_configs = payload.get("agents", []) or []
    evaluator_config = None
    role_descriptions: list[str] = []
    for item in agent_configs:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role_description", "") or item.get("name", "") or "").strip()
        if role:
            role_descriptions.append(role)
        if str(item.get("agent_type", "")).strip().lower() == "evaluator":
            evaluator_config = copy.deepcopy(item)

    if evaluator_config is None:
        raise ValueError(f"No `agent_type: evaluator` found in {config_path}")

    llm_cfg = copy.deepcopy(evaluator_config.get("llm", {}) or {})
    agent = OfflineEvaluatorAgent(
        name=str(evaluator_config.get("name", "Evaluator") or "Evaluator"),
        prepend_prompt_template=str(evaluator_config.get("prepend_prompt_template", "") or ""),
        append_prompt_template=str(evaluator_config.get("append_prompt_template", "") or ""),
        model=str(llm_cfg.get("model", "") or ""),
        temperature=float(llm_cfg.get("temperature", 0.0) or 0.0),
        max_tokens=int(llm_cfg.get("max_tokens", 1024) or 1024),
    )
    return agent, role_descriptions


def _parse_evaluator_agent_response(text: str) -> tuple[Optional[bool], str]:
    raw = str(text or "")
    correctness = re.search(r"\bCorrectness:\s*([01])\b", raw, re.IGNORECASE)
    if correctness:
        score = correctness.group(1) == "1"
    else:
        verdict = re.search(r"\bVerdict:\s*(PASS|FAIL)\b", raw, re.IGNORECASE)
        score = verdict.group(1).upper() == "PASS" if verdict else None

    response = re.search(r"\bResponse:\s*(.+)", raw, re.IGNORECASE | re.DOTALL)
    advice = response.group(1).strip() if response else raw.strip()
    return score, advice
