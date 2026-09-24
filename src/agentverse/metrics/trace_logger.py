"""Trace logger with metrics output.

Duplicates all plain-text trace functionality from agentverse/tracing/trace_logger.py,
then adds structured metrics output (.metrics.jsonl and .summary.json).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from agentverse.metrics.models import AgentMeta, ExampleMetrics, RunSummary


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _safe(s: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in s).strip("_")


class MetricsTraceLogger:
    """Writes .trace.txt (same format as tracing/) + .metrics.jsonl + .summary.json."""

    def __init__(
        self,
        out_dir: str | Path = "traces",
        tag: str = "benchmark",
        run_stamp: str | None = None,
    ):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.tag = tag
        self.run_stamp = run_stamp or _now_stamp()

        stem = f"{self.run_stamp}_{_safe(self.tag)}"
        self.filepath = self.out_dir / f"{stem}.trace.txt"
        self.metrics_filepath = self.out_dir / f"{stem}.metrics.jsonl"
        self.summary_filepath = self.out_dir / f"{stem}.summary.json"

        self._auto_idx = 0
        self._broadcast_nested_mode = False
        self._broadcast_main_idx = 0
        self._broadcast_sub_idx = 0

    def reset_turn_counter(self) -> None:
        """Restart plain-text turn numbering from T1 for a new example."""
        self._auto_idx = 0
        self._broadcast_nested_mode = False
        self._broadcast_main_idx = 0
        self._broadcast_sub_idx = 0

    # ------------------------------------------------------------------
    # Plain-text trace (identical to SimulationTraceLogger)
    # ------------------------------------------------------------------

    def write_header(
        self, agents: list[AgentMeta], extra: dict[str, Any] | None = None
    ) -> None:
        lines: list[str] = []
        lines.append(f"run_id: {self.run_stamp}")
        lines.append(f"tag: {self.tag}")
        lines.append(f"agents: {','.join(a.agent_number for a in agents)}")
        lines.append("")
        for a in agents:
            lines.append(
                f"{a.agent_number}: name={a.name} llm_type={a.llm_type} model={a.model}"
            )
            pre = (a.pre_prompt or "").strip().replace("\n", "\n    ")
            lines.append(f"    pre_prompt: {pre}")
            lines.append("")
        if extra:
            lines.append("extra:")
            extra_txt = json.dumps(extra, indent=2, ensure_ascii=False).replace(
                "\n", "\n    "
            )
            lines.append(f"    {extra_txt}")
            lines.append("")
        lines.append("---")
        self._append("\n".join(lines) + "\n")

    def log_turn(
        self, agent_number: str, text: str, turn_idx: int | str | None = None
    ) -> None:
        """Log a single turn to the plain-text trace."""
        if turn_idx is None:
            self._auto_idx += 1
            turn_label = f"T{self._auto_idx}"
        elif isinstance(turn_idx, str):
            turn_label = turn_idx if turn_idx.startswith("T") else f"T{turn_idx}"
        else:
            turn_label = f"T{turn_idx}"
        body = text.strip().replace("\n", " ")
        self._append(f"{turn_label}: {agent_number}: {body}\n")

    def log_system(self, text: str) -> None:
        """Log a visible system line without consuming a numbered turn label."""
        body = str(text or "").strip().replace("\n", " ")
        if not body:
            return
        self._append(f"System: {body}\n")

    def _append(self, s: str) -> None:
        with self.filepath.open("a", encoding="utf-8") as f:
            f.write(s)

    # ------------------------------------------------------------------
    # Structured metrics output
    # ------------------------------------------------------------------

    def write_example_metrics(self, metrics: ExampleMetrics) -> None:
        """Append one example's metrics as a JSON line to .metrics.jsonl."""
        with self.metrics_filepath.open("a", encoding="utf-8") as f:
            f.write(json.dumps(metrics.to_dict(), ensure_ascii=False) + "\n")

    def write_run_summary(self, summary: RunSummary) -> None:
        """Write the aggregate run summary to .summary.json."""
        with self.summary_filepath.open("w", encoding="utf-8") as f:
            json.dump(summary.to_dict(), f, indent=2, ensure_ascii=False)
            f.write("\n")
