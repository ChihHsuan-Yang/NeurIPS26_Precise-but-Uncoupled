# AgentVerse/agentverse_command/benchmark.py
import logging
import os
import json
import shutil
import re
import time
from argparse import ArgumentParser
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional

# from agentverse.agentverse import AgentVerse
from agentverse.evaluation.common import exact_match, extract_final_answer
from agentverse.evaluation.omni_judge import (
    DEFAULT_OMNI_JUDGE_MODEL_ID,
    get_cached_omni_judge_evaluator,
    resolve_omni_judge_model_path,
)
from agentverse.evaluation.omni_rule import get_cached_omni_rule_evaluator
from agentverse.tasksolving import TaskSolving
from agentverse.logging import get_logger
from dataloader import dataloader_registry
from agentverse_command.config_resolver import (
    ResolvedTaskConfig,
    resolve_benchmark_dataloader,
    resolve_dataset_path,
    resolve_optional_output_path,
    resolve_task_config_reference,
)

# Runtime metrics + trace-backed summaries
from agentverse.metrics import (
    MetricsTraceLogger,
    AgentMeta,
    MetricsCollector,
    RunAggregator,
)

parser = ArgumentParser()
EVAL_MODE_CHOICES = [
    "exact",
    "numeric-verifier",
    "omni-judge",
    "omni-verifier",
    "omni-rule",
    "none",
]

parser.add_argument(
    "--task",
    type=str,
    default="tasksolving/responsegen",
    help="Registered task id or a path to a folder/config.yaml/benchmark.yaml.",
)
parser.add_argument(
    "--tasks_dir",
    type=str,
    default=os.path.join(os.path.dirname(__file__), "..", "agentverse", "tasks"),
)
parser.add_argument(
    "--dataset_path",
    type=str,
    default=None,
    help="Dataset path. If omitted, benchmark.yaml may provide it.",
)
parser.add_argument("--output_path", type=str, default=None)
parser.add_argument("--has_tools", action="store_true")
parser.add_argument("--tool_tmp_path", type=str)
parser.add_argument("--overwrite", action="store_true")
parser.add_argument("--debug", action="store_true")
parser.add_argument(
    "--eval_mode",
    type=str,
    default=None,
    choices=EVAL_MODE_CHOICES,
    help=(
        "Deprecated alias for `--extra_post_eval`. "
        "If omitted, benchmark reuses the environment evaluator result when available."
    ),
)
parser.add_argument(
    "--extra_post_eval",
    type=str,
    default=None,
    choices=EVAL_MODE_CHOICES,
    help=(
        "Optional second evaluation layer run after the environment's own evaluator. "
        "If omitted, benchmark writes correctness/summary from the saved evaluator result "
        "instead of re-evaluating."
    ),
)
parser.add_argument(
    "--omni_judge_model_path",
    type=str,
    default=DEFAULT_OMNI_JUDGE_MODEL_ID,
    help=(
        "Local path or Hugging Face id for the official open-source Omni-Judge model. "
        "This evaluator runs locally through transformers, not through the ALCF endpoint."
    ),
)
parser.add_argument(
    "--omni_judge_max_new_tokens",
    type=int,
    default=300,
    help="Generation cap used for Omni-Judge evaluation.",
)
parser.add_argument(
    "--omni_judge_device",
    type=str,
    default="auto",
    choices=["auto", "cpu", "cuda", "mps"],
    help=(
        "Device preference for the local Omni-Judge model. "
        "`auto` tries CUDA first and prefers CPU before MPS on macOS for stability."
    ),
)
parser.add_argument(
    "--omni_judge_dtype",
    type=str,
    default="auto",
    choices=["auto", "float32", "float16", "bfloat16"],
    help=(
        "Dtype preference for the local Omni-Judge model. "
        "`auto` picks a safe default for the selected device."
    ),
)
parser.add_argument(
    "--per_verifier_mode",
    type=str,
    default=None,
    choices=["exact", "numeric-verifier", "omni-rule", "omni-judge", "omni-verifier"],
    help=(
        "Optional override for the verifier used inside the PER environment itself. "
        "If omitted, the task config default is used."
    ),
)
parser.add_argument(
    "--per_verifier_data_name",
    type=str,
    default=None,
    help=(
        "Optional PER verifier dataset family, for example `gsm8k` or `omni-math`. "
        "If omitted, the task config default is used."
    ),
)
parser.add_argument(
    "--per_numeric_tolerance",
    type=str,
    default=None,
    help=(
        "Optional numeric tolerance string forwarded to the PER numeric verifier. "
        "If omitted, the task config default is used."
    ),
)
parser.add_argument(
    "--per_feedback_mode",
    type=str,
    default=None,
    choices=["plain", "hint"],
    help=(
        "Optional override for how evaluator feedback is sent back to the Reviewer in PER mode. "
        "`plain` sends a generic pass/fail message. `hint` forwards sanitized mismatch guidance."
    ),
)

# DHD (Diverse Hypothesis Deliberation) options
parser.add_argument(
    "--dhd_k",
    type=int,
    default=None,
    help=(
        "DHD only: number of hypothesizers K to expose to the integrator "
        "(nested subset of the recruited roster). Overrides rule.n_hypothesizers."
    ),
)
parser.add_argument(
    "--dhd_exposure",
    type=str,
    default=None,
    choices=["none", "correctness", "trajectory", "both"],
    help="DHD only: live integrator score-exposure condition (default none/placebo).",
)
parser.add_argument(
    "--dhd_rounds",
    type=int,
    default=None,
    help="DHD only: revision rounds R (1 = DHD-v1; >1 = DHD-R, scaffolded).",
)
parser.add_argument(
    "--dhd_domain_hint",
    type=str,
    default=None,
    help="DHD only: light per-domain instruction appended to hypothesizer/integrator advice.",
)

# Trace options
parser.add_argument("--trace", action="store_true", help="Enable trace file output")
parser.add_argument(
    "--trace_dir",
    type=str,
    default=None,
    help="Directory to store trace files. Defaults to the resolved output directory.",
)
parser.add_argument(
    "--trace_tag",
    type=str,
    default=None,
    help="Tag included in trace filename. If omitted, benchmark.yaml may provide it.",
)
parser.add_argument(
    "--artifact_tag",
    type=str,
    default=None,
    help=(
        "Optional tag used for extra copies of result artifacts. "
        "If omitted and --trace is enabled, the trace tag is reused."
    ),
)
parser.add_argument(
    "--example_start",
    type=int,
    default=0,
    help=(
        "0-based index of the first example to process. Earlier examples "
        "in the dataset are skipped. Combine with --example_end to slice "
        "a chunk across parallel worker processes pointing at the same "
        "dataset file."
    ),
)
parser.add_argument(
    "--example_end",
    type=int,
    default=None,
    help=(
        "Exclusive 0-based index of the last example to process. "
        "Examples at this index and beyond are skipped. Default None "
        "processes through end of dataset."
    ),
)

args = parser.parse_args()

logger = get_logger()
logger.set_level(logging.DEBUG if args.debug else logging.INFO)


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


RUN_STAMP = _now_stamp()


def _sanitize_tag(raw: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("._-")


def _get_config_path(task: str, tasks_dir: str) -> Path:
    return Path(tasks_dir) / task / "config.yaml"


def _task_label() -> str:
    return str(getattr(args, "_resolved_task_label", args.task))


def _resolved_config_path() -> Path:
    path = getattr(args, "_resolved_config_path", None)
    if path is None:
        return _get_config_path(args.task, args.tasks_dir)
    return Path(path)


def _default_run_tag() -> str:
    return _artifact_tag() or _sanitize_tag(_task_label()) or "benchmark"


def _resolve_output_path() -> str:
    if args.output_path:
        return args.output_path
    if args.trace and args.trace_tag:
        trace_folder = _sanitize_tag(args.trace_tag) or "benchmark"
        return f"./results/{trace_folder}_{RUN_STAMP}"
    return f"./results/{_default_run_tag()}"


def _resolve_trace_dir(output_path: str) -> str:
    return args.trace_dir or output_path


def _task_run_output_path(output_path: str, example_idx: int) -> str:
    return os.path.join(output_path, "task_runs", f"example_{example_idx:04d}.txt")


def _build_per_env_overrides() -> Optional[Dict[str, Any]]:
    wants_per_override = any(
        [
            args.per_verifier_mode is not None,
            args.per_verifier_data_name is not None,
            args.per_numeric_tolerance is not None,
            args.per_feedback_mode is not None,
            args.omni_judge_model_path != DEFAULT_OMNI_JUDGE_MODEL_ID,
            args.omni_judge_max_new_tokens != 300,
            args.omni_judge_device != "auto",
            args.omni_judge_dtype != "auto",
        ]
    )
    if not wants_per_override:
        return None

    rule: Dict[str, Any] = {}
    if args.per_verifier_mode:
        rule["verifier_mode"] = args.per_verifier_mode
    if args.per_verifier_data_name:
        rule["verifier_data_name"] = args.per_verifier_data_name
    if args.per_numeric_tolerance is not None:
        rule["numeric_tolerance"] = args.per_numeric_tolerance
    if args.per_feedback_mode:
        rule["reviewer_feedback_mode"] = args.per_feedback_mode

    # These judge settings are shared by post-eval and PER-mode verifier paths.
    rule["omni_judge_model_path"] = resolve_omni_judge_model_path(args.omni_judge_model_path)
    rule["omni_judge_max_new_tokens"] = args.omni_judge_max_new_tokens
    rule["omni_judge_device"] = args.omni_judge_device
    rule["omni_judge_dtype"] = args.omni_judge_dtype

    return {"rule": rule} if rule else None


def _build_dhd_env_overrides() -> Optional[Dict[str, Any]]:
    """DHD rule overrides (K, exposure, rounds). Additive; independent of PER."""
    rule: Dict[str, Any] = {}
    if getattr(args, "dhd_k", None) is not None:
        rule["n_hypothesizers"] = int(args.dhd_k)
    if getattr(args, "dhd_exposure", None) is not None:
        rule["exposure_condition"] = str(args.dhd_exposure)
    if getattr(args, "dhd_rounds", None) is not None:
        rule["rounds"] = int(args.dhd_rounds)
    if getattr(args, "dhd_domain_hint", None) is not None:
        rule["domain_hint"] = str(args.dhd_domain_hint)
    return {"rule": rule} if rule else None


def _merge_env_overrides(*overrides: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    merged: Dict[str, Any] = {}
    for ov in overrides:
        if not ov:
            continue
        for key, value in ov.items():
            if key == "rule" and isinstance(value, dict):
                merged.setdefault("rule", {}).update(value)
            else:
                merged[key] = value
    return merged or None


def _load_agent_metas_from_yaml(config_path: Path):
    import yaml  # keep local to avoid adding dependency elsewhere

    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    agents = data.get("agents", [])
    metas: List[AgentMeta] = []
    name_to_A: Dict[str, str] = {}

    for i, a in enumerate(agents, start=1):
        A = f"A{i}"
        name = a.get("name", f"Agent{i}")
        llm = a.get("llm", {}) or {}
        metas.append(
            AgentMeta(
                agent_number=A,
                name=name,
                model=str(llm.get("model", "")),
                llm_type=str(llm.get("llm_type", "")),
                pre_prompt=str(a.get("role_description", "")).strip(),
            )
        )
        name_to_A[name] = A

    return metas, name_to_A


def _trace_write_example_header(trace: MetricsTraceLogger, idx: int, example: Dict[str, Any]) -> None:
    # append a readable per-example header into the same trace file
    trace.reset_turn_counter()
    sep = "\n" + ("=" * 80) + "\n"
    text = example.get("input", "")
    ans = example.get("answer", "")
    payload = (
        f"{sep}"
        f"EXAMPLE {idx + 1}\n"
        f"input: {text}\n"
        f"label: {ans}\n"
        f"{'-' * 80}\n"
    )
    trace._append(payload)  # using the logger's append for convenience


def _trace_log_event(
    trace: MetricsTraceLogger,
    item: Any,
    name_to_A: Dict[str, str],
) -> None:
    def is_broadcast_trace_activator(stage: str) -> bool:
        return stage.startswith(("poll_", "discussion_", "candidate_review_", "approval_")) or stage in {
            "final_proposal_selected",
        }

    def is_broadcast_main_event(stage: str) -> bool:
        return (
            stage.startswith("discussion_")
            or stage == "final_proposal_selected"
            or stage == "evaluation"
            or stage == "evaluation_hint"
        )

    def next_turn_label(stage: str) -> str | None:
        if not getattr(trace, "_broadcast_nested_mode", False):
            if not is_broadcast_trace_activator(stage):
                return None
            trace._broadcast_nested_mode = True

        if is_broadcast_main_event(stage):
            trace._broadcast_main_idx += 1
            trace._broadcast_sub_idx = 0
            return f"T{trace._broadcast_main_idx}"

        trace._broadcast_sub_idx += 1
        return f"T{trace._broadcast_main_idx}_{trace._broadcast_sub_idx}"

    def is_meaningful_trace_text(text: Any) -> bool:
        raw = str(text or "").strip()
        if not raw:
            return False
        normalized = re.sub(r"\s+", " ", raw).strip().lower()
        if normalized in {
            "",
            "[empty]",
            "[none]",
            "none",
            "n/a",
            "na",
            "no comment",
            "no comments",
            "no feedback",
            "no substantive feedback",
            "nothing to add",
            "skip",
            "pass",
            "[]",
            "{}",
            '""',
            "''",
        }:
            return False
        if "\\boxed{" in raw:
            return True
        return len(normalized) >= 4

    def log_one(
        sender: str,
        content: str,
        turn_label: int | str | None = None,
        force_numbered_system_trace: bool = False,
    ):
        sender = (sender or "").strip()
        content = (content or "").strip()
        if not content:
            return
        if sender.lower() == "system" and not force_numbered_system_trace:
            trace.log_system(content)
            return
        A = name_to_A.get(sender, sender)  # fall back to sender if unknown
        trace.log_turn(str(A), content, turn_idx=turn_label)

    if isinstance(item, dict):
        if item.get("trace_visible") is False:
            return

        entry_type = item.get("type")
        stage = str(item.get("stage", "") or "")

        # Surface confidence polls in the plain-text trace so they are easy to inspect.
        if entry_type == "summary" and stage.startswith("poll_"):
            sender = (
                item.get("sender")
                or item.get("role")
                or item.get("agent")
                or item.get("name")
                or item.get("module")
                or ""
            )
            discussion_turn = item.get("discussion_turn", "")
            score = item.get("score", "")
            reason = item.get("reason", "")
            intent = item.get("intent", "")
            candidate = item.get("candidate_answer", "") or "[None]"
            if (
                not is_meaningful_trace_text(reason)
                and not is_meaningful_trace_text(intent)
                and candidate == "[None]"
            ):
                return
            parse_status = "parsed_json" if item.get("parse_succeeded") else "fallback"
            content = (
                f"[confidence_poll round={item.get('round', 0)} turn={discussion_turn} "
                f"score={score} parse={parse_status}] reason={reason} intent={intent} candidate={candidate}"
            )
            turn_label = (
                None if str(sender).strip().lower() == "system" else next_turn_label(stage)
            )
            log_one(str(sender), content, turn_label=turn_label)
            return

        # Skip non-conversational events (e.g., meta/system summaries not shown in trace body)
        if entry_type is not None and entry_type != "message":
            return

        sender = (
            item.get("sender")
            or item.get("role")
            or item.get("agent")
            or item.get("name")
            or item.get("module")
            or ""
        )
        content = (
            item.get("content")
            or item.get("text")
            or item.get("message")
            or item.get("output")
            or ""
        )
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        force_numbered_system_trace = bool(item.get("force_numbered_system_trace"))
        turn_label = (
            next_turn_label(stage)
            if (force_numbered_system_trace or str(sender).strip().lower() != "system")
            else None
        )
        log_one(
            str(sender),
            str(content),
            turn_label=turn_label,
            force_numbered_system_trace=force_numbered_system_trace,
        )
        return

    trace.log_turn("LOG", str(item))


def _trace_from_logs(
    trace: MetricsTraceLogger,
    logs: Any,
    name_to_A: Dict[str, str],
) -> None:
    """
    Best-effort conversion of TaskSolving logs to trace turns.
    We try a few common shapes:
      - list[dict] with keys like sender/content
      - dict with a 'messages' list
      - string blob
    """

    if logs is None:
        return

    # dict wrapper
    if isinstance(logs, dict):
        if "messages" in logs and isinstance(logs["messages"], list):
            logs = logs["messages"]
        else:
            _trace_log_event(trace, logs, name_to_A)
            return

    # list of events
    if isinstance(logs, list):
        for item in logs:
            _trace_log_event(trace, item, name_to_A)
        return

    # string blob
    if isinstance(logs, str):
        lines = logs.splitlines()
        speaker_line = re.compile(r"^\s*(?P<speaker>[^:]{2,80})\s*:\s*(?P<msg>.+?)\s*$")
        any_parsed = False
        for line in lines:
            m = speaker_line.match(line)
            if m:
                any_parsed = True
                speaker = m.group("speaker").strip()
                msg = m.group("msg").strip()
                _trace_log_event(
                    trace,
                    {"type": "message", "sender": speaker, "content": msg},
                    name_to_A,
                )
        if not any_parsed:
            trace.log_turn("LOG", logs.strip())
        return

    # fallback
    trace.log_turn("LOG", str(logs))


def _build_live_trace_sink(
    trace: MetricsTraceLogger,
    name_to_A: Dict[str, str],
):
    def sink(event: Dict[str, Any]) -> None:
        _trace_log_event(trace, event, name_to_A)

    return sink


def get_dataloader(task, dataset_path):
    dataset_loader = getattr(args, "_resolved_dataset_loader", None)
    if dataset_loader or Path(str(task)).expanduser().exists():
        return resolve_benchmark_dataloader(
            task_label=_task_label(),
            dataset_path=dataset_path,
            dataset_loader=dataset_loader,
        )
    return dataloader_registry.build(task, path=dataset_path)


def _infer_rule_data_name(task: str, records: List[Dict[str, Any]]) -> str:
    task_lower = (task or "").lower()
    if "omni_math" in task_lower or "omni-math" in task_lower:
        return "omni-math"
    if "mgsm" in task_lower or "gsm8k" in task_lower:
        return "gsm8k"

    dataset_names = {
        str(record.get("dataset_name", "")).lower()
        for record in records
        if record.get("dataset_name")
    }
    if (
        "omni-math" in dataset_names
        or "omni-math-rule" in dataset_names
        or "omni-math-2-filtered" in dataset_names
    ):
        return "omni-math"
    if "mgsm" in dataset_names or "gsm8k" in dataset_names:
        return "gsm8k"

    return "omni-math"


def _result_paths(output_path: str) -> Dict[str, str]:
    paths = {
        "results": f"{output_path}/results.jsonl",
        "correctness_jsonl": f"{output_path}/correctness.jsonl",
        "correctness_txt": f"{output_path}/correctness.txt",
        "summary_json": f"{output_path}/accuracy_summary.json",
        "omni_judge_jsonl": f"{output_path}/omni_judge_results.jsonl",
        "omni_verifier_jsonl": f"{output_path}/omni_verifier_results.jsonl",
        "omni_rule_jsonl": f"{output_path}/omni_rule_results.jsonl",
        "config_yaml": f"{output_path}/config.yaml",
    }

    tag = _artifact_tag()
    if tag:
        for key, path in list(paths.items()):
            directory, filename = os.path.split(path)
            paths[f"tagged_{key}"] = os.path.join(directory, f"{tag}.{filename}")

    return paths


def _artifact_tag() -> Optional[str]:
    raw_tag = args.artifact_tag or (args.trace_tag if args.trace else None)
    if not raw_tag:
        return None

    sanitized = _sanitize_tag(raw_tag)
    return sanitized or None


def _write_jsonl_outputs(paths: Dict[str, str], key: str, rows: List[Dict[str, Any]]) -> None:
    _write_jsonl(paths[key], rows)

    tagged_key = f"tagged_{key}"
    if tagged_key in paths and paths[tagged_key] != paths[key]:
        _write_jsonl(paths[tagged_key], rows)


def _write_json_outputs(paths: Dict[str, str], key: str, payload: Dict[str, Any]) -> None:
    with open(paths[key], "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    tagged_key = f"tagged_{key}"
    if tagged_key in paths and paths[tagged_key] != paths[key]:
        with open(paths[tagged_key], "w") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)


def _write_text_outputs(paths: Dict[str, str], key: str, text: str) -> None:
    with open(paths[key], "w") as f:
        f.write(text)

    tagged_key = f"tagged_{key}"
    if tagged_key in paths and paths[tagged_key] != paths[key]:
        with open(paths[tagged_key], "w") as f:
            f.write(text)


def _copy_output_with_tag(paths: Dict[str, str], key: str) -> None:
    tagged_key = f"tagged_{key}"
    if tagged_key in paths and paths[tagged_key] != paths[key]:
        shutil.copyfile(paths[key], paths[tagged_key])


def _resolve_ground_truth(record: Dict[str, Any]) -> str:
    for key in ("answer", "answer_number", "label"):
        value = record.get(key)
        if value is None:
            continue
        normalized = str(value).strip()
        if normalized and normalized.lower() not in {"none", "null"}:
            return normalized
    return ""


def _resolve_reference_solution(record: Dict[str, Any]) -> str:
    for key in ("reference_solution", "equation_solution", "solution", "equation_answer"):
        value = record.get(key)
        if value is None:
            continue
        normalized = str(value).strip()
        if normalized and normalized.lower() not in {"none", "null"}:
            return normalized
    return ""


def _build_result_record(
    idx: int,
    example: Dict[str, Any],
    planner_output: str,
    model_generation: str,
    logs: Any,
) -> Dict[str, Any]:
    extra = {
        key: value
        for key, value in example.items()
        if key not in {"input", "answer", "tools"}
    }
    final_answer = extract_final_answer(model_generation)
    ground_truth = _resolve_ground_truth(example)
    if not ground_truth:
        raise ValueError(
            f"Missing ground truth for example {idx + 1}. "
            "Expected one of `answer`, `answer_number`, or `label`."
        )

    per_evaluation = _extract_per_evaluation(logs)

    return {
        "question_id": idx + 1,
        "input": example["input"],
        "problem": example["input"],
        "response": planner_output,
        "planner_output": planner_output,
        "model_generation": model_generation,
        "final_answer": final_answer,
        "label": ground_truth,
        "answer": ground_truth,
        "logs": logs,
        "per_evaluation": per_evaluation,
        "task_run_artifact": f"task_runs/example_{idx:04d}.txt",
        **extra,
    }


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return records

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _extract_per_evaluation(logs: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(logs, list):
        return None

    for item in reversed(logs):
        if not isinstance(item, dict):
            continue
        if item.get("type") != "meta":
            continue
        stage = str(item.get("stage", "") or "")
        if not stage.startswith("evaluation"):
            continue
        if item.get("correctness") not in {0, 1, True, False}:
            continue
        if not item.get("mode"):
            continue
        return {
            "mode": item.get("mode"),
            "correctness": item.get("correctness"),
            "final_answer": item.get("final_answer"),
            "judge_student_final_answer": item.get("judge_student_final_answer"),
            "judge_equivalence_judgement": item.get("judge_equivalence_judgement"),
            "judge_device": item.get("judge_device"),
            "judge_dtype": item.get("judge_dtype"),
            "rule_prediction": item.get("rule_prediction"),
            "reference_solution_used": item.get("reference_solution_used"),
        }
    return None


def _build_numeric_correctness(
    records: List[Dict[str, Any]],
    judge_name: str = "numeric-verifier",
) -> List[Dict[str, Any]]:
    scored: List[Dict[str, Any]] = []
    for idx, record in enumerate(records, start=1):
        generation = record.get("model_generation", record.get("response", ""))
        final_answer = extract_final_answer(generation)
        ground_truth = _resolve_ground_truth(record)
        correct = exact_match(generation, ground_truth)
        scored.append(
            {
                "question_id": idx,
                "problem": record.get("problem", record.get("input", "")),
                "final_answer": final_answer,
                "ground_truth": ground_truth,
                "correctness": 1 if correct else 0,
                "judge": judge_name,
                "difficulty": record.get("difficulty"),
                "difficulty_tier": record.get("difficulty_tier"),
                "source": record.get("source"),
                "domain": record.get("domain"),
            }
        )
    return scored


def _normalize_eval_mode_name(mode: str) -> str:
    return str(mode or "").strip().lower()


def _normalize_extra_post_eval(value: Optional[str]) -> Optional[str]:
    normalized = _normalize_eval_mode_name(value)
    if not normalized or normalized == "none":
        return None
    return normalized


def _requested_extra_post_eval() -> Optional[str]:
    requested = _normalize_extra_post_eval(args.extra_post_eval)
    legacy = _normalize_extra_post_eval(args.eval_mode)

    if requested and legacy and requested != legacy:
        raise ValueError(
            f"Conflicting evaluation flags: --extra_post_eval={requested!r} "
            f"and --eval_mode={legacy!r}."
        )

    if requested is not None:
        return requested

    if args.eval_mode is not None:
        logger.warn(
            "`--eval_mode` is deprecated. Use `--extra_post_eval` for a second "
            "benchmark-time evaluation layer, or omit both flags to reuse the "
            "saved environment evaluator result."
        )
    return legacy


def _infer_saved_per_mode(records: List[Dict[str, Any]]) -> Optional[str]:
    if not records:
        return None

    modes = set()
    for record in records:
        per_eval = record.get("per_evaluation")
        if not isinstance(per_eval, dict):
            return None
        mode = _normalize_eval_mode_name(per_eval.get("mode"))
        if not mode:
            return None
        modes.add(mode)

    if len(modes) != 1:
        return None
    return modes.pop()


def _can_reuse_per_eval(records: List[Dict[str, Any]], eval_mode: str) -> bool:
    expected = _normalize_eval_mode_name(eval_mode)
    if expected == "none":
        return False
    if not records:
        return False

    for record in records:
        per_eval = record.get("per_evaluation")
        if not isinstance(per_eval, dict):
            return False
        if _normalize_eval_mode_name(per_eval.get("mode")) != expected:
            return False
        correctness = per_eval.get("correctness")
        if correctness not in {0, 1, True, False}:
            return False
    return True


def _build_correctness_from_per_eval(
    records: List[Dict[str, Any]],
    eval_mode: str,
) -> List[Dict[str, Any]]:
    scored: List[Dict[str, Any]] = []
    raw_rows: List[Dict[str, Any]] = []
    mode = _normalize_eval_mode_name(eval_mode)

    for idx, record in enumerate(records, start=1):
        per_eval = record["per_evaluation"]
        correctness = int(bool(per_eval.get("correctness")))
        row = {
            "question_id": idx,
            "problem": record.get("problem", record.get("input", "")),
            "final_answer": per_eval.get("final_answer")
            or extract_final_answer(record.get("model_generation", record.get("response", ""))),
            "ground_truth": _resolve_ground_truth(record),
            "correctness": correctness,
            "judge": mode,
            "difficulty": record.get("difficulty"),
            "difficulty_tier": record.get("difficulty_tier"),
            "source": record.get("source"),
            "domain": record.get("domain"),
        }
        if mode in {"omni-judge", "omni-verifier"}:
            row["judge_student_final_answer"] = per_eval.get("judge_student_final_answer")
            row["judge_equivalence_judgement"] = per_eval.get("judge_equivalence_judgement")
            row["judge_device"] = per_eval.get("judge_device")
            row["judge_dtype"] = per_eval.get("judge_dtype")
            raw_rows.append(
                {
                    "question_id": idx,
                    "problem": record.get("problem", record.get("input", "")),
                    "answer": _resolve_ground_truth(record),
                    "model_generation": record.get("model_generation", record.get("response", "")),
                    "correctness": correctness,
                    "judge_student_final_answer": per_eval.get("judge_student_final_answer"),
                    "judge_equivalence_judgement": per_eval.get("judge_equivalence_judgement"),
                    "judge_device": per_eval.get("judge_device"),
                    "judge_dtype": per_eval.get("judge_dtype"),
                }
            )
        elif mode == "omni-rule":
            row["rule_data_name"] = _infer_rule_data_name(_task_label(), records)
            raw_rows.append(
                {
                    "question_id": idx,
                    "problem": record.get("problem", record.get("input", "")),
                    "answer": _resolve_ground_truth(record),
                    "model_generation": record.get("model_generation", record.get("response", "")),
                    "correctness": correctness,
                    "rule_prediction": per_eval.get("rule_prediction", ""),
                }
            )
        scored.append(row)

    paths = _result_paths(args.output_path)
    if mode == "omni-judge":
        _write_jsonl_outputs(paths, "omni_judge_jsonl", raw_rows)
    elif mode == "omni-verifier":
        _write_jsonl_outputs(paths, "omni_verifier_jsonl", raw_rows)
    elif mode == "omni-rule":
        _write_jsonl_outputs(paths, "omni_rule_jsonl", raw_rows)
    return scored


def _build_omni_judge_correctness(
    records: List[Dict[str, Any]],
    judge_name: str = "omni-judge",
) -> List[Dict[str, Any]]:
    if not args.omni_judge_model_path:
        raise ValueError(
            "--omni_judge_model_path is required when using the Omni-Judge evaluator."
        )

    judge = get_cached_omni_judge_evaluator(
        model_path=args.omni_judge_model_path,
        max_new_tokens=args.omni_judge_max_new_tokens,
        device_preference=args.omni_judge_device,
        dtype_preference=args.omni_judge_dtype,
    )

    judge_inputs = []
    for record in records:
        judge_inputs.append(
            {
                "question_id": record.get("question_id"),
                "problem": record.get("problem", record.get("input", "")),
                "answer": _resolve_ground_truth(record),
                "model_generation": record.get("model_generation", record.get("response", "")),
                "difficulty": record.get("difficulty"),
                "difficulty_tier": record.get("difficulty_tier"),
                "source": record.get("source"),
                "domain": record.get("domain"),
            }
        )

    judged = judge.evaluate(judge_inputs)
    paths = _result_paths(args.output_path)
    raw_key = "omni_judge_jsonl" if judge_name == "omni-judge" else "omni_verifier_jsonl"
    _write_jsonl_outputs(paths, raw_key, judged)

    scored: List[Dict[str, Any]] = []
    for idx, row in enumerate(judged, start=1):
        correctness = row.get("correctness")
        scored.append(
            {
                "question_id": idx,
                "problem": row["problem"],
                "final_answer": extract_final_answer(row["model_generation"]),
                "ground_truth": str(row["answer"]),
                "correctness": 1 if correctness else 0,
                "judge": judge_name,
                "difficulty": row.get("difficulty"),
                "difficulty_tier": row.get("difficulty_tier"),
                "source": row.get("source"),
                "domain": row.get("domain"),
                "omni_judge": row.get("omni_judge", ""),
                "judge_student_final_answer": row.get("judge_student_final_answer"),
                "judge_equivalence_judgement": row.get("judge_equivalence_judgement"),
                "judge_device": row.get("judge_device"),
                "judge_dtype": row.get("judge_dtype"),
            }
        )
    return scored


def _build_omni_rule_correctness(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    data_name = _infer_rule_data_name(_task_label(), records)
    evaluator = get_cached_omni_rule_evaluator(data_name=data_name)
    judged = evaluator.evaluate(
        [
            {
                "question_id": record.get("question_id"),
                "problem": record.get("problem", record.get("input", "")),
                "answer": _resolve_ground_truth(record),
                "model_generation": record.get("model_generation", record.get("response", "")),
                "difficulty": record.get("difficulty"),
                "difficulty_tier": record.get("difficulty_tier"),
                "source": record.get("source"),
                "domain": record.get("domain"),
            }
            for record in records
        ]
    )
    _write_jsonl_outputs(_result_paths(args.output_path), "omni_rule_jsonl", judged)

    scored: List[Dict[str, Any]] = []
    for idx, row in enumerate(judged, start=1):
        scored.append(
            {
                "question_id": idx,
                "problem": row["problem"],
                "final_answer": row.get("rule_prediction", ""),
                "ground_truth": row.get("rule_ground_truth", str(row["answer"])),
                "correctness": 1 if row.get("correctness") else 0,
                "judge": "omni-rule",
                "rule_data_name": data_name,
                "difficulty": row.get("difficulty"),
                "difficulty_tier": row.get("difficulty_tier"),
                "source": row.get("source"),
                "domain": row.get("domain"),
            }
        )
    return scored


def _write_accuracy_outputs(
    scored: List[Dict[str, Any]],
    *,
    judge_name: str,
    evaluation_source: str,
    extra_post_eval: Optional[str],
    per_evaluator_mode: Optional[str],
) -> None:
    paths = _result_paths(args.output_path)
    _write_jsonl_outputs(paths, "correctness_jsonl", scored)

    total = len(scored)
    correct = sum(int(row["correctness"]) for row in scored)
    accuracy = (correct / total) if total else 0.0

    by_tier: Dict[str, Dict[str, Any]] = {}
    for row in scored:
        tier = row.get("difficulty_tier")
        if tier is None:
            continue
        tier_key = str(tier)
        bucket = by_tier.setdefault(tier_key, {"correct": 0, "total": 0, "accuracy": 0.0})
        bucket["total"] += 1
        bucket["correct"] += int(row["correctness"])

    for bucket in by_tier.values():
        bucket["accuracy"] = (
            bucket["correct"] / bucket["total"] if bucket["total"] else 0.0
        )

    summary = {
        "judge": judge_name,
        "evaluation_source": evaluation_source,
        "extra_post_eval": extra_post_eval,
        "per_evaluator_mode": per_evaluator_mode,
        "eval_mode": extra_post_eval,  # backward-compatible alias
        "artifact_tag": _artifact_tag(),
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
        "accuracy_by_difficulty_tier": by_tier,
    }

    lines = []
    for row in scored:
        lines.append(f"#question_{int(row['question_id']):04d}: {int(row['correctness'])}")
    lines.append(f"overall_accuracy: {accuracy:.6f}")
    lines.append(f"correct: {correct}")
    lines.append(f"total: {total}")
    lines.append(f"judge: {judge_name}")
    lines.append(f"evaluation_source: {evaluation_source}")
    if per_evaluator_mode:
        lines.append(f"per_evaluator_mode: {per_evaluator_mode}")
    if extra_post_eval:
        lines.append(f"extra_post_eval: {extra_post_eval}")
    if summary["artifact_tag"]:
        lines.append(f"artifact_tag: {summary['artifact_tag']}")
    lines.append("")

    _write_text_outputs(paths, "correctness_txt", "\n".join(lines))
    _write_json_outputs(paths, "summary_json", summary)


def _run_post_eval() -> None:
    paths = _result_paths(args.output_path)
    records = _load_jsonl(paths["results"])
    requested_extra_post_eval = _requested_extra_post_eval()
    saved_per_mode = _infer_saved_per_mode(records)

    if requested_extra_post_eval is None:
        if saved_per_mode and _can_reuse_per_eval(records, saved_per_mode):
            scored = _build_correctness_from_per_eval(records, saved_per_mode)
            _write_accuracy_outputs(
                scored,
                judge_name=saved_per_mode,
                evaluation_source="per_evaluator",
                extra_post_eval=None,
                per_evaluator_mode=saved_per_mode,
            )
            logger.info(
                f"[EVAL] Reused saved environment evaluator result for {saved_per_mode}; "
                f"wrote correctness to {paths['correctness_jsonl']} and summary to "
                f"{paths['summary_json']}"
            )
            return

        scored = _build_numeric_correctness(records, judge_name="exact")
        _write_accuracy_outputs(
            scored,
            judge_name="exact",
            evaluation_source="fallback_exact",
            extra_post_eval=None,
            per_evaluator_mode=saved_per_mode,
        )
        logger.warn(
            "[EVAL] No saved environment evaluation was available, so benchmark "
            "fell back to exact post-eval."
        )
        return

    if requested_extra_post_eval == "exact":
        scored = _build_numeric_correctness(records, judge_name="exact")
    elif requested_extra_post_eval == "numeric-verifier":
        scored = _build_numeric_correctness(records, judge_name="numeric-verifier")
    elif requested_extra_post_eval == "omni-judge":
        scored = _build_omni_judge_correctness(records, judge_name="omni-judge")
    elif requested_extra_post_eval == "omni-verifier":
        scored = _build_omni_judge_correctness(records, judge_name="omni-verifier")
    elif requested_extra_post_eval == "omni-rule":
        scored = _build_omni_rule_correctness(records)
    else:
        raise ValueError(f"Unknown extra post-eval mode: {requested_extra_post_eval}")

    _write_accuracy_outputs(
        scored,
        judge_name=requested_extra_post_eval,
        evaluation_source="extra_post_eval",
        extra_post_eval=requested_extra_post_eval,
        per_evaluator_mode=saved_per_mode,
    )
    logger.info(
        f"[EVAL] Wrote correctness to {paths['correctness_jsonl']} and summary to "
        f"{paths['summary_json']} using extra post-eval={requested_extra_post_eval}"
    )


def cli_main():
    trace: Optional[MetricsTraceLogger] = None
    aggregator: Optional[RunAggregator] = None
    agentverse: Optional[TaskSolving] = None
    name_to_A: Dict[str, str] = {}
    paths: Dict[str, str]
    resolved_task: ResolvedTaskConfig = resolve_task_config_reference(args.task, args.tasks_dir)
    args._resolved_task_label = resolved_task.task_label
    args._resolved_config_path = str(resolved_task.config_path)
    args._resolved_dataset_loader = resolved_task.dataset_loader

    # Prepare output path
    args.dataset_path = str(resolve_dataset_path(args.dataset_path, resolved_task))
    args.trace_tag = args.trace_tag or resolved_task.default_trace_tag or _default_run_tag()
    if args.artifact_tag is None and resolved_task.default_artifact_tag:
        args.artifact_tag = resolved_task.default_artifact_tag
    if args.extra_post_eval is None and args.eval_mode is None and resolved_task.default_extra_post_eval:
        args.extra_post_eval = resolved_task.default_extra_post_eval
    if args.output_path is None:
        args.output_path = resolve_optional_output_path(args.output_path, resolved_task)

    dataloader = get_dataloader(args.task, args.dataset_path)
    args.omni_judge_model_path = resolve_omni_judge_model_path(args.omni_judge_model_path)
    env_overrides = _merge_env_overrides(
        _build_per_env_overrides(), _build_dhd_env_overrides()
    )
    args.output_path = _resolve_output_path()
    os.makedirs(args.output_path, exist_ok=True)
    paths = _result_paths(args.output_path)

    # Copy config for reproducibility
    shutil.copyfile(
        str(resolved_task.config_path),
        paths["config_yaml"],
    )
    _copy_output_with_tag(paths, "config_yaml")

    # Init trace + metrics (one file set per benchmark run)
    run_stamp = _now_stamp()
    if args.trace:
        args.trace_dir = _resolve_trace_dir(args.output_path)
        os.makedirs(args.trace_dir, exist_ok=True)
        config_path = resolved_task.config_path
        metas, name_to_A = _load_agent_metas_from_yaml(config_path)

        trace = MetricsTraceLogger(
            out_dir=args.trace_dir,
            tag=args.trace_tag,
            run_stamp=RUN_STAMP,
        )
        trace.write_header(
            metas,
            extra={
                "mode": "benchmark",
                "task": resolved_task.task_label,
                "tasks_dir": args.tasks_dir,
                "dataset_path": args.dataset_path,
                "output_path": args.output_path,
            },
        )
        aggregator = RunAggregator(
            run_id=run_stamp,
            task=resolved_task.task_label,
            model=metas[0].model if metas else "",
        )
        logger.info(f"[TRACE] Writing trace to: {trace.filepath}")
        logger.info(f"[METRICS] Writing metrics to: {trace.metrics_filepath}")

    # Resume behavior
    skip_cnt = 0
    if not args.overwrite and os.path.exists(paths["results"]):
        with open(paths["results"], "r") as f_in:
            for line in f_in:
                if line.strip():
                    skip_cnt += 1

    f_out = open(paths["results"], "w" if args.overwrite else "a")

    # Effective start = max(args.example_start, args.example_start + skip_cnt)
    # because skip_cnt rows in this worker's results.jsonl correspond to
    # global examples [example_start, example_start + skip_cnt).
    effective_start = args.example_start + skip_cnt
    for i, example in enumerate(dataloader):
        if i < effective_start:
            continue
        if args.example_end is not None and i >= args.example_end:
            break

        resolved_ground_truth = _resolve_ground_truth(example)
        logger.info(f"Input: {example['input']}\nAnswer: {resolved_ground_truth}")

        if trace is not None:
            _trace_write_example_header(trace, i, example)

        if args.has_tools:
            assert args.tool_tmp_path is not None
            with open(args.tool_tmp_path, "w") as f_tools:
                f_tools.write(json.dumps(example.get("tools", [])))

        if agentverse is None:
            agentverse = TaskSolving.from_config_path(
                str(resolved_task.config_path),
                task_name=resolved_task.task_label,
                env_overrides=env_overrides,
                result_output_path=_task_run_output_path(args.output_path, i),
            )
            if trace is not None:
                agentverse.environment.set_live_log_sink(
                    _build_live_trace_sink(trace, name_to_A)
                )

        agentverse.prepare_example(
            task_description=example["input"],
            ground_truth=resolved_ground_truth,
            reference_solution=_resolve_reference_solution(example),
            result_output_path=_task_run_output_path(args.output_path, i),
        )

        t0 = time.monotonic()
        plan, result, logs = agentverse.run()
        wall_time = time.monotonic() - t0

        record = _build_result_record(i, example, plan, result, logs)
        f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

        # Collect and write structured metrics
        if trace is not None:
            collector = MetricsCollector(i, example["input"], resolved_ground_truth)
            collector.extract_from_logs(logs)
            collector.snapshot_agent_tokens(agentverse.environment)
            example_metrics = collector.finalize(wall_time)
            trace.write_example_metrics(example_metrics)
            aggregator.add_example(example_metrics)
        f_out.flush()

    f_out.close()
    _run_post_eval()

    # Write aggregate run summary
    if trace is not None and aggregator is not None:
        summary = aggregator.summarize()
        trace.write_run_summary(summary)
        logger.info(
            f"[METRICS] Run summary: accuracy={summary.accuracy:.2%}, "
            f"avg_rounds={summary.avg_rounds:.1f}, "
            f"total_tokens={summary.total_tokens.total_tokens}"
        )


if __name__ == "__main__":
    cli_main()
