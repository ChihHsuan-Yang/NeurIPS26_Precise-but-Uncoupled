# AgentVerse/agentverse_command/main_tasksolving_cli.py
import os
import sys
import re
import logging
import atexit
from argparse import ArgumentParser
from pathlib import Path
from datetime import datetime

import yaml

from agentverse.tasksolving import TaskSolving
from agentverse.evaluation.omni_judge import (
    DEFAULT_OMNI_JUDGE_MODEL_ID,
    resolve_omni_judge_model_path,
)
from agentverse.logging import logger
from agentverse.metrics import MetricsTraceLogger, AgentMeta
from agentverse_command.config_resolver import resolve_task_config_reference


parser = ArgumentParser()

parser.add_argument(
    "--task",
    type=str,
    default="tasksolving/brainstorming",
    help="Registered task id or a path to a folder/config.yaml/benchmark.yaml.",
)
parser.add_argument("--debug", action="store_true")
parser.add_argument(
    "--tasks_dir",
    type=str,
    default=os.path.join(os.path.dirname(__file__), "..", "agentverse", "tasks"),
)

# Trace options (same flags as simulation CLI)
parser.add_argument("--trace", action="store_true", help="Enable clean trace file")
parser.add_argument("--trace_dir", type=str, default="traces", help="Directory to store trace files")
parser.add_argument("--trace_tag", type=str, default="tasksolving", help="Tag included in trace filename")
parser.add_argument(
    "--per_verifier_mode",
    type=str,
    default=None,
    choices=["exact", "numeric-verifier", "omni-rule", "omni-judge", "omni-verifier"],
    help="Optional override for the verifier used inside the PER environment.",
)
parser.add_argument(
    "--per_verifier_data_name",
    type=str,
    default=None,
    help="Optional PER verifier dataset family, for example `gsm8k` or `omni-math`.",
)
parser.add_argument(
    "--per_numeric_tolerance",
    type=str,
    default=None,
    help="Optional numeric tolerance string forwarded to the PER numeric verifier.",
)
parser.add_argument(
    "--omni_judge_model_path",
    type=str,
    default=DEFAULT_OMNI_JUDGE_MODEL_ID,
    help="Local path or Hugging Face id for the official open-source Omni-Judge model.",
)
parser.add_argument(
    "--omni_judge_max_new_tokens",
    type=int,
    default=300,
    help="Generation cap used for Omni-Judge evaluation inside PER mode.",
)
parser.add_argument(
    "--omni_judge_device",
    type=str,
    default="auto",
    choices=["auto", "cpu", "cuda", "mps"],
    help="Device preference for Omni-Judge. `auto` prefers CPU before MPS on macOS.",
)
parser.add_argument(
    "--omni_judge_dtype",
    type=str,
    default="auto",
    choices=["auto", "float32", "float16", "bfloat16"],
    help="Dtype preference for Omni-Judge. `auto` chooses a safe default per device.",
)

args = parser.parse_args()
logger.set_level(logging.DEBUG if args.debug else logging.INFO)


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _get_config_path(task: str, tasks_dir: str) -> Path:
    return Path(tasks_dir) / task / "config.yaml"


def _build_per_env_overrides():
    wants_per_override = any(
        [
            args.per_verifier_mode is not None,
            args.per_verifier_data_name is not None,
            args.per_numeric_tolerance is not None,
            args.omni_judge_model_path != DEFAULT_OMNI_JUDGE_MODEL_ID,
            args.omni_judge_max_new_tokens != 300,
            args.omni_judge_device != "auto",
            args.omni_judge_dtype != "auto",
        ]
    )
    if not wants_per_override:
        return None

    rule = {}
    if args.per_verifier_mode:
        rule["verifier_mode"] = args.per_verifier_mode
    if args.per_verifier_data_name:
        rule["verifier_data_name"] = args.per_verifier_data_name
    if args.per_numeric_tolerance is not None:
        rule["numeric_tolerance"] = args.per_numeric_tolerance

    rule["omni_judge_model_path"] = resolve_omni_judge_model_path(args.omni_judge_model_path)
    rule["omni_judge_max_new_tokens"] = args.omni_judge_max_new_tokens
    rule["omni_judge_device"] = args.omni_judge_device
    rule["omni_judge_dtype"] = args.omni_judge_dtype
    return {"rule": rule}


def _load_agent_metas_from_yaml(config_path: Path):
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    agents = data.get("agents", [])
    metas = []
    name_to_A = {}

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


class _TeeAndTrace:
    """
    Tee stdout/stderr to terminal, and also parse lines like:
      "AgentName: message"
    Then log them as turns into the trace file.

    NOTE:
    - This is "stdout parsing" (scraping prints/logs). It works if the runner prints
      messages in the "sender: content" style.
    - If tasksolving prints a different format, adjust SPEAKER_LINE accordingly.
    """

    SPEAKER_LINE = re.compile(r"^\s*(?P<speaker>[^:]{2,120})\s*:\s*(?P<msg>.+?)\s*$")

    def __init__(self, original_stream, trace: MetricsTraceLogger, name_to_A: dict):
        self._orig = original_stream
        self._trace = trace
        self._name_to_A = name_to_A
        self._buf = ""

    def write(self, s: str):
        self._orig.write(s)
        self._orig.flush()

        s = s.replace("\r\n", "\n").replace("\r", "\n")
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._handle_line(line)

    def flush(self):
        if self._buf.strip():
            self._handle_line(self._buf)
            self._buf = ""
        self._orig.flush()

    def _handle_line(self, line: str):
        m = self.SPEAKER_LINE.match(line)
        if not m:
            return
        speaker = m.group("speaker").strip()
        msg = m.group("msg").strip()

        if speaker in self._name_to_A:
            self._trace.log_turn(self._name_to_A[speaker], msg)


def cli_main():
    args.omni_judge_model_path = resolve_omni_judge_model_path(args.omni_judge_model_path)
    resolved_task = resolve_task_config_reference(args.task, args.tasks_dir)
    if args.trace:
        config_path = resolved_task.config_path
        metas, name_to_A = _load_agent_metas_from_yaml(config_path)

        trace = MetricsTraceLogger(
            out_dir=args.trace_dir,
            tag=args.trace_tag,
            run_stamp=_now_stamp(),
        )
        trace.write_header(
            metas,
            extra={"task": resolved_task.task_label, "tasks_dir": args.tasks_dir},
        )

        # wrap stdout/stderr so we capture printed lines
        sys.stdout = _TeeAndTrace(sys.stdout, trace, name_to_A)
        sys.stderr = _TeeAndTrace(sys.stderr, trace, name_to_A)

        # ensure partial buffered lines get parsed at exit
        atexit.register(lambda: getattr(sys.stdout, "flush", lambda: None)())
        atexit.register(lambda: getattr(sys.stderr, "flush", lambda: None)())

        print(f"[TRACE] Writing clean trace to: {trace.filepath}")

    agentversepipeline = TaskSolving.from_config_path(
        str(resolved_task.config_path),
        task_name=resolved_task.task_label,
        env_overrides=_build_per_env_overrides(),
    )
    agentversepipeline.run()


if __name__ == "__main__":
    cli_main()
