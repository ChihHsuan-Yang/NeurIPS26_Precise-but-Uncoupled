"""Runtime Critique Intervention Policy helpers for PER.

The policy is disabled by default. When enabled, it can load offline-trained
credibility, repair/integration, and harm models and decide whether a reviewer
critique should be accepted, challenged, or blocked before the next PER actor
receives it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CritiqueInterventionDecision:
    enabled: bool
    mode: str = "disabled"
    action: str = "accept"
    would_action: str = ""
    credibility_prob: float = 0.0
    repair_prob: float = 0.0
    harm_prob: float = 0.0
    credibility_threshold: float = 1.0
    repair_threshold: float = 1.0
    harm_threshold: float = 1.0
    reason: str = ""

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "action": self.action,
            "would_action": self.would_action,
            "credibility_prob": self.credibility_prob,
            "repair_prob": self.repair_prob,
            "harm_prob": self.harm_prob,
            "credibility_threshold": self.credibility_threshold,
            "repair_threshold": self.repair_threshold,
            "harm_threshold": self.harm_threshold,
            "reason": self.reason,
        }


class RuntimeCritiqueInterventionPolicy:
    """Small runtime wrapper around offline-trained CIP models."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        mode: str = "disabled",
        credibility_model_path: str = "",
        repair_model_path: str = "",
        harm_model_path: str = "",
        credibility_threshold: float = 0.7,
        repair_threshold: float = 0.8,
        harm_threshold: float = 0.7,
        benchmark: str = "",
        model_name: str = "",
        challenge_route: str = "planner",
        harm_route: str = "evaluate",
        apply_to_routes: list[str] | None = None,
        log_only: bool = False,
        augment_scores: list[str] | str | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.mode = str(mode or "disabled").strip().lower()
        # For mode="augment": which predicted scores to expose to the solver as
        # metadata alongside the (always-passed) reviewer message. Subset of
        # {"correctness", "integration"}. Default both.
        if augment_scores is None:
            self.augment_scores = ["correctness", "integration"]
        elif isinstance(augment_scores, str):
            self.augment_scores = [s.strip().lower() for s in augment_scores.split(",") if s.strip()]
        else:
            self.augment_scores = [str(s).strip().lower() for s in augment_scores]
        self.credibility_model_path = str(credibility_model_path or "")
        self.repair_model_path = str(repair_model_path or "")
        self.harm_model_path = str(harm_model_path or "")
        self.credibility_threshold = float(credibility_threshold)
        self.repair_threshold = float(repair_threshold)
        self.harm_threshold = float(harm_threshold)
        self.benchmark = str(benchmark or "")
        self.model_name = str(model_name or "")
        self.challenge_route = _normalize_route(challenge_route, default="planner")
        self.harm_route = _normalize_route(harm_route, default="evaluate")
        self.apply_to_routes = {
            _normalize_route(route, default="")
            for route in (apply_to_routes or ["planner", "executor"])
        }
        self.apply_to_routes.discard("")
        self.log_only = bool(log_only)
        self._credibility_bundle: dict[str, Any] | None = None
        self._repair_bundle: dict[str, Any] | None = None
        self._harm_bundle: dict[str, Any] | None = None

        if not self.enabled:
            self.mode = "disabled"
        elif self.mode not in {"model", "two_score", "static", "augment"}:
            raise ValueError(
                "critique_intervention_policy.mode must be one of: model, two_score, static, augment"
            )
        elif self.mode == "augment":
            # Augment mode exposes predicted scores as metadata on the message;
            # it never changes routing. Load only the models needed for the
            # requested scores.
            if "correctness" in self.augment_scores:
                if not self.credibility_model_path:
                    raise ValueError(
                        "augment mode with 'correctness' requires credibility_model_path."
                    )
                self._credibility_bundle = _load_bundle(self.credibility_model_path)
            # "trajectory" is the canonical term; "integration" kept as a synonym.
            if ("trajectory" in self.augment_scores) or ("integration" in self.augment_scores):
                if not self.repair_model_path:
                    raise ValueError(
                        "augment mode with 'trajectory' requires repair_model_path."
                    )
                self._repair_bundle = _load_bundle(self.repair_model_path)
            if self.harm_model_path:
                self._harm_bundle = _load_bundle(self.harm_model_path)
        elif self.mode in {"model", "two_score"}:
            if not self.repair_model_path or not self.harm_model_path:
                raise ValueError(
                    "Model-mode critique_intervention_policy requires both "
                    "repair_model_path and harm_model_path."
                )
            if self.mode == "two_score" and not self.credibility_model_path:
                raise ValueError(
                    "Two-score critique_intervention_policy requires "
                    "credibility_model_path."
                )
            if self.credibility_model_path:
                self._credibility_bundle = _load_bundle(self.credibility_model_path)
            self._repair_bundle = _load_bundle(self.repair_model_path)
            self._harm_bundle = _load_bundle(self.harm_model_path)

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any] | None,
    ) -> "RuntimeCritiqueInterventionPolicy":
        cfg = dict(config or {})
        return cls(**cfg)

    def decide(
        self,
        *,
        route: str,
        review_text: str,
        candidate_text: str,
        proposed_revision: str = "",
        reviewer_role: str = "Reviewer",
        review_turn: int = 0,
        after_evaluator_fail: bool = False,
        stage: str = "",
    ) -> CritiqueInterventionDecision:
        if not self.enabled:
            return CritiqueInterventionDecision(enabled=False)
        normalized_route = _normalize_route(route, default="executor")
        if normalized_route not in self.apply_to_routes:
            return CritiqueInterventionDecision(
                enabled=True,
                mode=self.mode,
                action="accept",
                credibility_threshold=self.credibility_threshold,
                repair_threshold=self.repair_threshold,
                harm_threshold=self.harm_threshold,
                reason=f"route {normalized_route!r} is outside apply_to_routes",
            )

        if self.mode == "static":
            credibility_prob = float(self.credibility_threshold)
            repair_prob = float(self.repair_threshold)
            harm_prob = 0.0
        else:
            credibility_prob = self._predict_probability(
                self._credibility_bundle,
                route=normalized_route,
                review_text=review_text,
                candidate_text=candidate_text,
                proposed_revision=proposed_revision,
                reviewer_role=reviewer_role,
                review_turn=review_turn,
                after_evaluator_fail=after_evaluator_fail,
                stage=stage,
            )
            repair_prob = self._predict_probability(
                self._repair_bundle,
                route=normalized_route,
                review_text=review_text,
                candidate_text=candidate_text,
                proposed_revision=proposed_revision,
                reviewer_role=reviewer_role,
                review_turn=review_turn,
                after_evaluator_fail=after_evaluator_fail,
                stage=stage,
            )
            harm_prob = self._predict_probability(
                self._harm_bundle,
                route=normalized_route,
                review_text=review_text,
                candidate_text=candidate_text,
                proposed_revision=proposed_revision,
                reviewer_role=reviewer_role,
                review_turn=review_turn,
                after_evaluator_fail=after_evaluator_fail,
                stage=stage,
            )

        if self.mode == "augment":
            # Never change routing; the message is always passed. The action
            # "augment" signals the caller to prepend a score annotation.
            return CritiqueInterventionDecision(
                enabled=True,
                mode=self.mode,
                action="augment",
                would_action="augment",
                credibility_prob=credibility_prob,
                repair_prob=repair_prob,
                harm_prob=harm_prob,
                credibility_threshold=self.credibility_threshold,
                repair_threshold=self.repair_threshold,
                harm_threshold=self.harm_threshold,
                reason=f"augment: expose scores {self.augment_scores}",
            )

        if self.mode == "two_score":
            action, reason = _two_score_action(
                credibility_prob=credibility_prob,
                repair_prob=repair_prob,
                harm_prob=harm_prob,
                credibility_threshold=self.credibility_threshold,
                repair_threshold=self.repair_threshold,
                harm_threshold=self.harm_threshold,
            )
        else:
            if harm_prob >= self.harm_threshold:
                action = "ignore"
                reason = "harm probability meets threshold"
            elif repair_prob >= self.repair_threshold:
                action = "accept"
                reason = "repair probability meets threshold"
            else:
                action = "challenge"
                reason = "repair probability below threshold and harm below threshold"
        if self.log_only:
            would_action = action
            reason = f"log_only: would {action}; preserving PER route"
            action = "accept"
        else:
            would_action = action
        return CritiqueInterventionDecision(
            enabled=True,
            mode=self.mode,
            action=action,
            would_action=would_action,
            credibility_prob=credibility_prob,
            repair_prob=repair_prob,
            harm_prob=harm_prob,
            credibility_threshold=self.credibility_threshold,
            repair_threshold=self.repair_threshold,
            harm_threshold=self.harm_threshold,
            reason=reason,
        )

    def _predict_probability(
        self,
        bundle: dict[str, Any] | None,
        *,
        route: str,
        review_text: str,
        candidate_text: str,
        proposed_revision: str,
        reviewer_role: str,
        review_turn: int,
        after_evaluator_fail: bool,
        stage: str,
    ) -> float:
        if not bundle:
            return 0.0
        import pandas as pd

        from agentverse.metrics.critique_intervention.features import (
            build_policy_records,
        )
        from agentverse.metrics.critique_intervention.training import (
            _vectorizer_safe_text,
        )

        event_row = {
            "critique_event_id": f"runtime:{stage}:{review_turn}",
            "trace_path": "",
            "protocol": "per",
            "reviewer_role": reviewer_role,
            "action": "review_route",
            "reviewer_signal_type": _reviewer_signal_type(route, review_text),
            "after_evaluator_fail": after_evaluator_fail,
            "review_turn": review_turn,
            "candidate_before": candidate_text,
            "review_feedback": review_text,
            "proposed_revision": proposed_revision,
            "intervention_effect": "unknown",
            "gate_target_action": "unknown",
        }
        records = build_policy_records(
            [event_row],
            benchmark=self.benchmark,
            model=self.model_name,
        )
        if not records:
            return 0.0
        row = dict(records[0])
        feature_cols = (
            list(bundle.get("categorical_features") or [])
            + list(bundle.get("numeric_features") or [])
            + list(bundle.get("text_features") or [])
        )
        for col in bundle.get("text_features") or []:
            row[col] = _vectorizer_safe_text(row.get(col, ""))
        frame = pd.DataFrame.from_records([{col: row.get(col, "") for col in feature_cols}])
        pipeline = bundle["pipeline"]
        if hasattr(pipeline, "predict_proba"):
            probabilities = pipeline.predict_proba(frame)
            classes = list(getattr(pipeline, "classes_", []))
            if 1 in classes:
                return float(probabilities[0][classes.index(1)])
            if "1" in classes:
                return float(probabilities[0][classes.index("1")])
        prediction = pipeline.predict(frame)
        return 1.0 if str(prediction[0]) in {"1", "True", "true"} else 0.0


def _load_bundle(path: str) -> dict[str, Any]:
    import joblib

    model_path = Path(path).expanduser()
    if not model_path.exists():
        raise FileNotFoundError(f"CIP model path does not exist: {model_path}")
    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict) or "pipeline" not in bundle:
        raise ValueError(f"CIP model file is not a supported bundle: {model_path}")
    return bundle


def _two_score_action(
    *,
    credibility_prob: float,
    repair_prob: float,
    harm_prob: float,
    credibility_threshold: float,
    repair_threshold: float,
    harm_threshold: float,
) -> tuple[str, str]:
    high_credibility = credibility_prob >= credibility_threshold
    high_repair = repair_prob >= repair_threshold
    high_harm = harm_prob >= harm_threshold
    if high_harm and not high_credibility and not high_repair:
        return "ignore", "two_score: high harm without credibility or repair value"
    if high_credibility and high_repair:
        return "accept", "two_score: credibility and repair value meet thresholds"
    if high_credibility:
        return "challenge", "two_score: credible critique needs explicit response"
    if high_repair and not high_harm:
        return "challenge", "two_score: low-credibility critique may still be useful as probe"
    return "challenge", "two_score: uncertain critique routed for verification"


def _normalize_route(route: Any, *, default: str) -> str:
    value = str(route or "").strip().lower()
    if value == "evaluate":
        return "evaluate"
    if value in {"planner", "executor", "agree"}:
        return value
    return default


def _reviewer_signal_type(route: str, review_text: str) -> str:
    normalized_route = _normalize_route(route, default="executor")
    text = str(review_text or "").lower()
    if normalized_route in {"agree", "evaluate"}:
        return "approve"
    if "wrong" in text or "incorrect" in text or "fail" in text:
        return "reject"
    if normalized_route in {"planner", "executor"}:
        return "revise"
    return "unknown"
