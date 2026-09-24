from __future__ import annotations

import logging
import os
import platform
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .common import (
    exact_match,
    extract_boxed,
    extract_final_answer,
    normalize_omni_judge_report_text,
    parse_omni_judge_report,
)

logger = logging.getLogger(__name__)
_OMNI_JUDGE_CACHE: Dict[Tuple[str, int, str, str], "OmniJudgeEvaluator"] = {}
DEFAULT_OMNI_JUDGE_MODEL_ID = "KbsdJames/Omni-Judge"


def default_local_omni_judge_path() -> Path:
    return Path(__file__).resolve().parents[2] / ".local_models" / "Omni-Judge"


def resolve_omni_judge_model_path(model_path: str | None = None) -> str:
    requested = (model_path or "").strip()

    if requested:
        requested_path = Path(requested).expanduser()
        if requested_path.exists():
            return str(requested_path)
        if requested != DEFAULT_OMNI_JUDGE_MODEL_ID:
            return requested

    env_path = os.environ.get("MAS_AURORA_OMNI_JUDGE_PATH", "").strip()
    if env_path:
        expanded_env = Path(env_path).expanduser()
        if expanded_env.exists():
            return str(expanded_env)

    local_path = default_local_omni_judge_path()
    if local_path.exists():
        return str(local_path)

    return requested or DEFAULT_OMNI_JUDGE_MODEL_ID


class OmniJudgeEvaluator:
    """
    Thin wrapper around the official Omni-Judge usage pattern.
    This loads the open-source Omni-Judge checkpoint locally via transformers.
    The tokenizer is expected to expose `get_context(...)` through
    `trust_remote_code=True`, following the model card / repo examples.
    """

    def __init__(
        self,
        model_path: str,
        max_new_tokens: int = 300,
        device_preference: str = "auto",
        dtype_preference: str = "auto",
    ):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Omni-Judge evaluation requires `transformers` and `torch` in the runtime environment."
            ) from exc

        model_path = resolve_omni_judge_model_path(model_path)
        self._patch_torch_autocast_compat(torch)
        self.device = "cpu"
        self.dtype_name = "float32"
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=True,
        )

        self.model = self._load_model(
            AutoModelForCausalLM=AutoModelForCausalLM,
            model_path=model_path,
            torch=torch,
            device_preference=device_preference,
            dtype_preference=dtype_preference,
        )
        self.max_new_tokens = max_new_tokens

        terminators = []
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        if eos_id is not None:
            terminators.append(eos_id)
        eot_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if eot_id is not None and eot_id != self.tokenizer.unk_token_id:
            terminators.append(eot_id)
        self.terminators = terminators

    def _patch_torch_autocast_compat(self, torch: Any) -> None:
        """
        transformers 5.x may call torch.is_autocast_enabled(device_type),
        while torch 2.2 only supports torch.is_autocast_enabled() with no args.
        Patch that signature difference locally so Omni-Judge can still run in
        older envs without forcing an immediate torch upgrade.
        """
        try:
            torch.is_autocast_enabled("cpu")
            return
        except TypeError:
            original = torch.is_autocast_enabled

            def _compat_is_autocast_enabled(device_type: str | None = None) -> bool:
                return bool(original())

            torch.is_autocast_enabled = _compat_is_autocast_enabled
        except Exception:
            return

    def _dtype_for_name(self, torch: Any, dtype_name: str) -> Any:
        mapping = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        return mapping[dtype_name]

    def _resolve_dtype_name(self, device: str, dtype_preference: str) -> str:
        if dtype_preference != "auto":
            return dtype_preference
        if device == "cuda":
            return "auto"
        if device == "mps":
            return "float16"
        return "float32"

    def _candidate_devices(self, torch: Any, device_preference: str) -> List[str]:
        if device_preference == "auto":
            available: List[str] = []
            has_mps = bool(
                getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
            )
            if torch.cuda.is_available():
                available.append("cuda")
            if platform.system() == "Darwin":
                # Prefer CPU before MPS on macOS because Omni-Judge is a BF16 checkpoint
                # and Apple MPS has been noticeably less stable for this model family.
                available.append("cpu")
                if has_mps:
                    available.append("mps")
                return available
            if has_mps:
                available.append("mps")
            available.append("cpu")
            return available
        return [device_preference]

    def _load_model(
        self,
        AutoModelForCausalLM: Any,
        model_path: str,
        torch: Any,
        device_preference: str,
        dtype_preference: str,
    ) -> Any:
        failures: List[str] = []

        for device in self._candidate_devices(torch, device_preference):
            dtype_name = self._resolve_dtype_name(device, dtype_preference)
            if device == "mps" and dtype_name == "bfloat16":
                failures.append("mps/bfloat16 is unsupported on Apple MPS")
                continue

            model_kwargs: Dict[str, Any] = {
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
            }
            if device == "cuda":
                model_kwargs["device_map"] = "auto"
                model_kwargs["dtype"] = (
                    "auto" if dtype_name == "auto" else self._dtype_for_name(torch, dtype_name)
                )
            else:
                model_kwargs["device_map"] = None
                model_kwargs["dtype"] = self._dtype_for_name(
                    torch, "float32" if dtype_name == "auto" else dtype_name
                )
                if device == "mps":
                    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

            try:
                try:
                    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
                except TypeError as exc:
                    if "dtype" not in str(exc):
                        raise
                    legacy_kwargs = dict(model_kwargs)
                    legacy_kwargs["torch_dtype"] = legacy_kwargs.pop("dtype")
                    model = AutoModelForCausalLM.from_pretrained(model_path, **legacy_kwargs)
                if device != "cuda":
                    model.to(device)
                model.eval()
                self.device = device
                self.dtype_name = dtype_name
                logger.info(
                    "Loaded Omni-Judge on device=%s dtype=%s",
                    self.device,
                    self.dtype_name,
                )
                return model
            except Exception as exc:
                failures.append(f"{device}/{dtype_name}: {exc}")
                logger.warning(
                    "Failed to load Omni-Judge on device=%s dtype=%s; trying fallback.",
                    device,
                    dtype_name,
                )

        failure_text = "\n".join(failures)
        raise RuntimeError(
            "Unable to load Omni-Judge on any supported device.\n"
            f"Tried:\n{failure_text}"
        )

    def _prepare_prompt(self, record: Dict[str, Any]) -> str:
        if not hasattr(self.tokenizer, "get_context"):
            raise RuntimeError(
                "The Omni-Judge tokenizer does not expose `get_context`. "
                "Use the official Omni-Judge model/tokenizer checkpoint."
            )

        prompt = self.tokenizer.get_context(
            record["problem"],
            str(record["answer"]),
            str(record["model_generation"]),
        )
        if not prompt.endswith("\n"):
            prompt += "\n"
        return prompt

    def _generate_completion(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
        min_new_tokens: int = 0,
    ) -> str:
        model_inputs = self.tokenizer(prompt, return_tensors="pt")
        model_inputs = {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in model_inputs.items()
        }

        generated = self.model.generate(
            **model_inputs,
            do_sample=False,
            num_return_sequences=1,
            max_new_tokens=max_new_tokens or self.max_new_tokens,
            min_new_tokens=min_new_tokens,
        )[0].detach().cpu().tolist()

        prompt_ids = model_inputs["input_ids"][0].detach().cpu().tolist()
        pred = generated[len(prompt_ids):]
        for terminator in self.terminators:
            if terminator in pred:
                pred = pred[: pred.index(terminator)]

        return self.tokenizer.decode(
            pred,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()

    def _prepare_boolean_fallback_prompt(self, record: Dict[str, Any]) -> str:
        boxed = extract_boxed(str(record.get("model_generation", "")))
        student_final = boxed or str(record.get("model_generation", "")).strip()
        if hasattr(self.tokenizer, "apply_chat_template"):
            context = [
                {
                    "role": "system",
                    "content": "You are a careful mathematics judge.",
                },
                {
                    "role": "user",
                    "content": (
                        "Compare the student's final answer with the reference answer.\n"
                        "Return exactly one token: TRUE or FALSE.\n\n"
                        f"Question:\n{str(record.get('problem', '')).strip()}\n\n"
                        f"Reference Answer:\n{str(record.get('answer', '')).strip()}\n\n"
                        f"Student Final Answer:\n{student_final}\n"
                    ),
                },
            ]
            return self.tokenizer.apply_chat_template(
                context,
                tokenize=False,
                add_generation_prompt=True,
            )

        return (
            "You are a careful mathematics judge.\n"
            "Compare the student's final answer with the reference answer.\n"
            "Return exactly one token: TRUE or FALSE.\n\n"
            f"Question:\n{str(record.get('problem', '')).strip()}\n\n"
            f"Reference Answer:\n{str(record.get('answer', '')).strip()}\n\n"
            f"Student Final Answer:\n{student_final}\n"
        )

    def _apply_deterministic_fallback(
        self,
        parsed: Dict[str, Any],
        record: Dict[str, Any],
    ) -> Dict[str, Any]:
        reviewer_final = extract_boxed(str(record.get("model_generation", ""))) or extract_final_answer(
            str(record.get("model_generation", ""))
        )
        reference_final = extract_final_answer(str(record.get("answer", "")))
        matched = exact_match(str(record.get("model_generation", "")), str(record.get("answer", "")))

        if reviewer_final:
            parsed["Student Final Answer"] = reviewer_final
        elif reference_final:
            parsed["Student Final Answer"] = reference_final

        parsed["Equivalence Judgement"] = "TRUE" if matched else "FALSE"
        if not parsed.get("Justification"):
            parsed["Justification"] = (
                "Deterministic boxed-answer fallback used after incomplete Omni-Judge output."
            )
        return parsed

    def _generate_one(self, record: Dict[str, Any]) -> Dict[str, Any]:
        prompt = self._prepare_prompt(record)
        response = self._generate_completion(
            prompt,
            max_new_tokens=self.max_new_tokens,
            min_new_tokens=min(96, self.max_new_tokens),
        )

        raw_report = response
        if "## Student Final Answer" not in raw_report:
            raw_report = "## Student Final Answer\n" + raw_report
        report = normalize_omni_judge_report_text(raw_report)

        parsed: Dict[str, Any] = {}
        if hasattr(self.tokenizer, "parse_response"):
            try:
                official = self.tokenizer.parse_response(report)
                parsed = {
                    "Student Final Answer": str(official.get("answer") or "").strip(),
                    "Equivalence Judgement": str(official.get("judgement") or "").strip().upper(),
                    "Justification": str(official.get("justification") or "").strip(),
                }
            except Exception:
                parsed = {}

        if not parsed.get("Equivalence Judgement"):
            parsed = parse_omni_judge_report(report)

        if not parsed.get("Equivalence Judgement"):
            fallback_prompt = self._prepare_boolean_fallback_prompt(record)
            fallback_response = self._generate_completion(
                fallback_prompt,
                max_new_tokens=8,
                min_new_tokens=1,
            )
            fallback_match = re.search(r"TRUE|FALSE", fallback_response, flags=re.IGNORECASE)
            if fallback_match:
                parsed["Equivalence Judgement"] = fallback_match.group(0).upper()
                boxed = extract_boxed(str(record.get("model_generation", "")))
                if boxed:
                    parsed["Student Final Answer"] = boxed
                if not parsed.get("Justification"):
                    parsed["Justification"] = (
                        f"Fallback boolean judgement used: {fallback_response.strip()}"
                    )
        if not parsed.get("Equivalence Judgement"):
            parsed = self._apply_deterministic_fallback(parsed, record)
        judgement = parsed.get("Equivalence Judgement", "").strip().upper()
        student_final_answer = parsed.get("Student Final Answer", "").strip()

        return {
            **record,
            "omni_judge": report,
            "omni_judge_parsed": parsed,
            "correctness": judgement == "TRUE" if judgement else None,
            "judge_student_final_answer": student_final_answer,
            "judge_equivalence_judgement": judgement,
            "judge_device": self.device,
            "judge_dtype": self.dtype_name,
        }

    def evaluate(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self._generate_one(record) for record in records]


def get_cached_omni_judge_evaluator(
    model_path: str,
    max_new_tokens: int = 300,
    device_preference: str = "auto",
    dtype_preference: str = "auto",
) -> OmniJudgeEvaluator:
    model_path = resolve_omni_judge_model_path(model_path)
    key = (
        model_path,
        int(max_new_tokens),
        str(device_preference),
        str(dtype_preference),
    )
    evaluator = _OMNI_JUDGE_CACHE.get(key)
    if evaluator is None:
        evaluator = OmniJudgeEvaluator(
            model_path=model_path,
            max_new_tokens=max_new_tokens,
            device_preference=device_preference,
            dtype_preference=dtype_preference,
        )
        _OMNI_JUDGE_CACHE[key] = evaluator
    return evaluator
