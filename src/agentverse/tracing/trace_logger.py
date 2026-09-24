from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional
import json

def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def _safe(s: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in s).strip("_")

@dataclass
class AgentMeta:
    agent_number: str
    name: str
    model: str
    llm_type: str
    pre_prompt: str

class SimulationTraceLogger:
    def __init__(self, out_dir: str | Path = "traces", tag: str = "classroom", run_stamp: Optional[str] = None):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.tag = tag
        self.run_stamp = run_stamp or _now_stamp()
        self.filepath = self.out_dir / f"{self.run_stamp}_{_safe(self.tag)}.trace.txt"
        self._auto_idx = 0

    def reset_turn_counter(self) -> None:
        """Restart turn numbering from T1 for a new scenario/example."""
        self._auto_idx = 0

    def write_header(self, agents: List[AgentMeta], extra: Optional[Dict[str, Any]] = None) -> None:
        lines = []
        lines.append(f"run_id: {self.run_stamp}")
        lines.append(f"tag: {self.tag}")
        lines.append(f"agents: {','.join(a.agent_number for a in agents)}")
        lines.append("")
        for a in agents:
            lines.append(f"{a.agent_number}: name={a.name} llm_type={a.llm_type} model={a.model}")
            pre = (a.pre_prompt or "").strip().replace("\n", "\n    ")
            lines.append(f"    pre_prompt: {pre}")
            lines.append("")
        if extra:
            lines.append("extra:")
            extra_txt = json.dumps(extra, indent=2, ensure_ascii=False).replace("\n", "\n    ")
            lines.append(f"    {extra_txt}")
            lines.append("")
        lines.append("---")
        self._append("\n".join(lines) + "\n")

    def log_turn(self, agent_number: str, text: str, turn_idx: Optional[int] = None) -> None:
        """
        If turn_idx is provided, we log as T{turn_idx}.
        Otherwise we fall back to auto-increment (still works).
        """
        if turn_idx is None:
            self._auto_idx += 1
            turn_idx = self._auto_idx
        body = text.strip().replace("\n", " ")
        self._append(f"T{turn_idx}: {agent_number}: {body}\n")

    def _append(self, s: str) -> None:
        with self.filepath.open("a", encoding="utf-8") as f:
            f.write(s)
