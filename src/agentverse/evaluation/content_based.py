"""Content-based helpers for offline trace analysis.

This module provides a small, optional post-evaluator that can label:
- whether a candidate answer matches the reference answer
- whether a repair path meaningfully uses system feedback

The default behavior is heuristic-only so trace analysis can run without API
credentials. When configured in ``llm`` or ``hybrid`` mode, the helper uses an
OpenAI-compatible chat endpoint, which is suitable for the repo's
``openai/gpt-oss-120b`` deployments.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from agentverse.evaluation.common import exact_match, extract_final_answer
from agentverse.evaluation.feedback_label_cache import (
    CachedFeedbackLabel,
    PersistentFeedbackLabelCache,
)

try:
    from openai import AzureOpenAI, OpenAI
except Exception:  # pragma: no cover - optional dependency
    AzureOpenAI = None
    OpenAI = None


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_STOPWORDS = {
    "a",
    "an",
    "and",
    "answer",
    "be",
    "candidate",
    "did",
    "does",
    "evaluator",
    "feedback",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "path",
    "repair",
    "review",
    "signal",
    "system",
    "that",
    "the",
    "this",
    "to",
    "use",
    "used",
    "uses",
    "using",
    "with",
}


@dataclass
class ContentJudgeResult:
    """One offline content judgement."""

    value: Optional[bool]
    source: str
    reason: str = ""


class TraceAnalysisLabeler:
    """Heuristic or LLM-assisted labeler for offline trace analysis."""

    def __init__(
        self,
        mode: str = "heuristic",
        model: str = "openai/gpt-oss-120b",
        temperature: float = 0.0,
        max_tokens: int = 400,
        cache_feedback_labels: bool = True,
        feedback_incorporation_cache_path: str | Path | None = None,
    ):
        normalized = str(mode or "heuristic").strip().lower()
        if normalized not in {"heuristic", "llm", "hybrid"}:
            normalized = "heuristic"

        self.mode = normalized
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.cache_feedback_labels = bool(cache_feedback_labels)
        self._client = self._build_client() if normalized in {"llm", "hybrid"} else None
        self._correctness_cache: dict[tuple[str, str, str], ContentJudgeResult] = {}
        self._feedback_incorporation_cache: dict[tuple[str, str], ContentJudgeResult] = {}
        self._feedback_cache_signature = self._build_feedback_cache_signature()
        self._persistent_feedback_cache = (
            PersistentFeedbackLabelCache(feedback_incorporation_cache_path)
            if self.cache_feedback_labels and feedback_incorporation_cache_path
            else None
        )
        self.llm_request_count = 0
        self.correctness_cache_hits = 0
        self.feedback_incorporation_cache_hits = 0
        self.persistent_feedback_cache_hits = 0

    @property
    def llm_available(self) -> bool:
        return self._client is not None

    def judge_correctness(
        self,
        *,
        problem: str,
        gold_answer: str,
        candidate_answer: str,
    ) -> ContentJudgeResult:
        candidate = str(candidate_answer or "").strip()
        gold = str(gold_answer or "").strip()
        if not candidate or not gold:
            return ContentJudgeResult(value=None, source="missing", reason="Missing candidate or gold.")

        cache_key = (str(problem or ""), gold, candidate)
        cached = self._correctness_cache.get(cache_key)
        if cached is not None:
            self.correctness_cache_hits += 1
            return cached

        if exact_match(candidate, gold):
            result = ContentJudgeResult(value=True, source="exact", reason="Exact-match success.")
            self._correctness_cache[cache_key] = result
            return result

        if self.mode == "heuristic":
            result = ContentJudgeResult(value=False, source="exact_mismatch", reason="Exact-match mismatch.")
            self._correctness_cache[cache_key] = result
            return result

        if self.mode in {"llm", "hybrid"} and self._client is not None:
            prompt = (
                "You are a strict math answer equivalence judge.\n"
                "Decide whether the candidate answer is equivalent to the gold answer.\n"
                "Return JSON only with keys: correct (boolean), reason (string).\n\n"
                f"Problem:\n{problem}\n\n"
                f"Gold answer:\n{gold}\n\n"
                f"Candidate answer:\n{candidate}\n"
            )
            parsed = self._ask_json(prompt)
            if parsed is not None and isinstance(parsed.get("correct"), bool):
                result = ContentJudgeResult(
                    value=bool(parsed["correct"]),
                    source="llm",
                    reason=str(parsed.get("reason", "") or ""),
                )
                self._correctness_cache[cache_key] = result
                return result

        result = ContentJudgeResult(value=False, source="exact_mismatch", reason="Exact-match mismatch.")
        self._correctness_cache[cache_key] = result
        return result

    def judge_feedback_uptake(
        self,
        *,
        evaluator_feedback: str,
        repair_path: str,
    ) -> ContentJudgeResult:
        """Backward-compatible alias for feedback incorporation labels."""
        return self.judge_feedback_incorporation(
            feedback_text=evaluator_feedback,
            response_path=repair_path,
            feedback_source="system",
        )

    def judge_feedback_incorporation(
        self,
        *,
        feedback_text: str,
        response_path: str,
        feedback_source: str = "feedback",
    ) -> ContentJudgeResult:
        feedback = str(feedback_text or "").strip()
        path = str(response_path or "").strip()
        if not feedback or not path:
            return ContentJudgeResult(value=None, source="missing", reason="Missing feedback or repair path.")

        cache_key = (feedback, path)
        cached = self._feedback_incorporation_cache.get(cache_key)
        if cached is not None:
            self.feedback_incorporation_cache_hits += 1
            return cached

        if self._persistent_feedback_cache is not None:
            persisted = self._persistent_feedback_cache.lookup(
                signature=self._feedback_cache_signature,
                feedback_source=feedback_source,
                feedback_text=feedback,
                response_path=path,
            )
            if persisted is not None:
                self.persistent_feedback_cache_hits += 1
                restored = ContentJudgeResult(
                    value=persisted.value,
                    source=persisted.source,
                    reason=persisted.reason,
                )
                self._feedback_incorporation_cache[cache_key] = restored
                return restored

        heuristic = self._heuristic_feedback_incorporation(feedback, path)
        if self.mode == "heuristic" or self._client is None:
            return self._remember_feedback_result(
                cache_key=cache_key,
                feedback_source=feedback_source,
                feedback_text=feedback,
                response_path=path,
                result=heuristic,
            )

        # Hybrid mode is intentionally conservative with LLM calls: accept
        # obvious deterministic incorporation and ask the LLM only for cases
        # where the heuristic did not find evidence.
        if self.mode == "hybrid" and heuristic.value is True:
            return self._remember_feedback_result(
                cache_key=cache_key,
                feedback_source=feedback_source,
                feedback_text=feedback,
                response_path=path,
                result=heuristic,
            )

        prompt = (
            "You are labeling whether a reasoning system incorporated system feedback after a failed try.\n"
            "Decide whether the response meaningfully addresses the specific feedback.\n"
            "Return JSON only with keys: incorporated_feedback (boolean), reason (string).\n\n"
            f"Feedback source:\n{feedback_source}\n\n"
            f"Feedback:\n{feedback}\n\n"
            f"Response:\n{path}\n"
        )
        parsed = self._ask_json(prompt)
        parsed_value = None
        if parsed is not None:
            if isinstance(parsed.get("incorporated_feedback"), bool):
                parsed_value = bool(parsed["incorporated_feedback"])
            elif isinstance(parsed.get("used_feedback"), bool):
                parsed_value = bool(parsed["used_feedback"])
        if parsed_value is not None:
            result = ContentJudgeResult(
                value=parsed_value,
                source="llm",
                reason=str(parsed.get("reason", "") or ""),
            )
            return self._remember_feedback_result(
                cache_key=cache_key,
                feedback_source=feedback_source,
                feedback_text=feedback,
                response_path=path,
                result=result,
            )

        return self._remember_feedback_result(
            cache_key=cache_key,
            feedback_source=feedback_source,
            feedback_text=feedback,
            response_path=path,
            result=heuristic,
        )

    def _heuristic_feedback_incorporation(
        self,
        feedback_text: str,
        response_path: str,
    ) -> ContentJudgeResult:
        feedback = feedback_text.lower()
        path = response_path.lower()

        if any(
            token in path
            for token in ("evaluator", "verifier", "feedback", "fail", "diagnosis", "system")
        ):
            return ContentJudgeResult(
                value=True,
                source="heuristic",
                reason="Repair path explicitly references system feedback.",
            )

        feedback_keywords = {
            token
            for token in re.findall(r"[a-z][a-z0-9_-]{3,}", feedback)
            if token not in _STOPWORDS
        }
        matched = sorted(token for token in feedback_keywords if token in path)
        if matched:
            return ContentJudgeResult(
                value=True,
                source="heuristic",
                reason=f"Repair path reuses feedback keywords: {', '.join(matched[:8])}.",
            )

        return ContentJudgeResult(
            value=False,
            source="heuristic",
            reason="No clear evidence that the repair path used system feedback.",
        )

    def _build_client(self):
        if OpenAI is None:
            return None

        openai_key = os.environ.get("OPENAI_API_KEY")
        openai_base_url = os.environ.get("OPENAI_BASE_URL")
        azure_key = os.environ.get("AZURE_OPENAI_API_KEY")
        azure_base = os.environ.get("AZURE_OPENAI_API_BASE")

        if openai_key:
            return OpenAI(api_key=openai_key, base_url=openai_base_url)

        if azure_key and AzureOpenAI is not None:
            return AzureOpenAI(
                api_key=azure_key,
                azure_endpoint=azure_base,
                api_version="2024-02-15-preview",
            )

        return None

    def _remember_feedback_result(
        self,
        *,
        cache_key: tuple[str, str],
        feedback_source: str,
        feedback_text: str,
        response_path: str,
        result: ContentJudgeResult,
    ) -> ContentJudgeResult:
        self._feedback_incorporation_cache[cache_key] = result
        if self._persistent_feedback_cache is not None and result.value is not None:
            self._persistent_feedback_cache.store(
                signature=self._feedback_cache_signature,
                feedback_source=feedback_source,
                feedback_text=feedback_text,
                response_path=response_path,
                label=CachedFeedbackLabel(
                    value=bool(result.value),
                    source=result.source,
                    reason=result.reason,
                ),
            )
        return result

    def _build_feedback_cache_signature(self) -> str:
        payload = {
            "mode": self.mode,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def _ask_json(self, prompt: str) -> Optional[dict]:
        if self._client is None:
            return None

        try:
            self.llm_request_count += 1
            response = self._client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                messages=[
                    {
                        "role": "system",
                        "content": "Return JSON only. Do not include markdown fences.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )
        except Exception:
            return None

        try:
            content = response.choices[0].message.content or ""
        except Exception:
            return None

        if not content:
            return None

        content = str(content).strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)

        match = _JSON_RE.search(content)
        candidate = match.group(0) if match else content
        try:
            return json.loads(candidate)
        except Exception:
            return None


def normalize_candidate_text(text: str) -> str:
    """Return a compact candidate answer string for offline analysis."""
    value = extract_final_answer(str(text or ""))
    return value.strip()
