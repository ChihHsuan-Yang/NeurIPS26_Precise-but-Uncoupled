"""Persistent cache for small-window feedback-incorporation labels."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CachedFeedbackLabel:
    """One persisted feedback-incorporation label."""

    value: bool
    source: str
    reason: str = ""


class PersistentFeedbackLabelCache:
    """Append-only JSONL cache for semantic feedback-incorporation labels."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        self._rows: dict[tuple[str, str, str, str], CachedFeedbackLabel] = {}
        self.hits = 0
        self.writes = 0
        self._load()

    def lookup(
        self,
        *,
        signature: str,
        feedback_source: str,
        feedback_text: str,
        response_path: str,
    ) -> CachedFeedbackLabel | None:
        key = self._key(signature, feedback_source, feedback_text, response_path)
        cached = self._rows.get(key)
        if cached is not None:
            self.hits += 1
        return cached

    def store(
        self,
        *,
        signature: str,
        feedback_source: str,
        feedback_text: str,
        response_path: str,
        label: CachedFeedbackLabel,
    ) -> None:
        key = self._key(signature, feedback_source, feedback_text, response_path)
        if key in self._rows:
            return

        self.path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "signature": signature,
            "feedback_source": str(feedback_source or "").strip(),
            "feedback_text": str(feedback_text or "").strip(),
            "response_path": str(response_path or "").strip(),
            "value": bool(label.value),
            "source": str(label.source or ""),
            "reason": str(label.reason or ""),
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._rows[key] = label
        self.writes += 1

    def _load(self) -> None:
        if not self.path.exists():
            return
        for raw_line in self.path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if not isinstance(row, dict):
                continue
            try:
                signature = str(row["signature"])
                feedback_source = str(row["feedback_source"])
                feedback_text = str(row["feedback_text"])
                response_path = str(row["response_path"])
                value = bool(row["value"])
                source = str(row.get("source", "") or "")
                reason = str(row.get("reason", "") or "")
            except Exception:
                continue
            self._rows[self._key(signature, feedback_source, feedback_text, response_path)] = (
                CachedFeedbackLabel(value=value, source=source, reason=reason)
            )

    def _key(
        self,
        signature: str,
        feedback_source: str,
        feedback_text: str,
        response_path: str,
    ) -> tuple[str, str, str, str]:
        return (
            str(signature or ""),
            str(feedback_source or "").strip(),
            str(feedback_text or "").strip(),
            str(response_path or "").strip(),
        )


def default_feedback_label_cache_path(base_dir: str | Path) -> Path:
    """Default shared cache path for feedback-incorporation labels."""

    return Path(base_dir).expanduser().resolve() / "feedback_label_cache.jsonl"
