"""Persistent candidate-label cache shared by post-trace analysis CLIs."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CachedCandidateLabel:
    """One persisted candidate-correctness label."""

    value: bool
    source: str
    reason: str = ""


class PersistentCandidateLabelCache:
    """Append-only JSONL cache for evaluator-style candidate labels.

    The cache is shared across the paper-facing post-trace CLIs so we do not
    replay the evaluator agent on the same `(problem, gold, candidate)` tuple
    multiple times across separate runs.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        self._rows: dict[tuple[str, str, str, str], CachedCandidateLabel] = {}
        self.hits = 0
        self.writes = 0
        # Guards both the in-memory dict and append-to-file so that thread-pool
        # callers (parallel post-hoc audits) can't double-write the same key
        # or interleave JSONL lines.
        self._lock = threading.Lock()
        self._load()

    def lookup(
        self,
        *,
        signature: str,
        problem: str,
        gold_answer: str,
        candidate_answer: str,
    ) -> CachedCandidateLabel | None:
        key = self._key(signature, problem, gold_answer, candidate_answer)
        with self._lock:
            cached = self._rows.get(key)
            if cached is not None:
                self.hits += 1
            return cached

    def store(
        self,
        *,
        signature: str,
        problem: str,
        gold_answer: str,
        candidate_answer: str,
        label: CachedCandidateLabel,
    ) -> None:
        key = self._key(signature, problem, gold_answer, candidate_answer)
        with self._lock:
            if key in self._rows:
                return

            self.path.parent.mkdir(parents=True, exist_ok=True)
            row = {
                "signature": signature,
                "problem": str(problem or "").strip(),
                "gold_answer": str(gold_answer or "").strip(),
                "candidate_answer": str(candidate_answer or "").strip(),
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
                problem = str(row["problem"])
                gold = str(row["gold_answer"])
                candidate = str(row["candidate_answer"])
                value = bool(row["value"])
                source = str(row.get("source", "") or "")
                reason = str(row.get("reason", "") or "")
            except Exception:
                continue
            self._rows[self._key(signature, problem, gold, candidate)] = CachedCandidateLabel(
                value=value,
                source=source,
                reason=reason,
            )

    def _key(
        self,
        signature: str,
        problem: str,
        gold_answer: str,
        candidate_answer: str,
    ) -> tuple[str, str, str, str]:
        return (
            str(signature or ""),
            str(problem or "").strip(),
            str(gold_answer or "").strip(),
            str(candidate_answer or "").strip(),
        )


def default_candidate_label_cache_path(base_dir: str | Path) -> Path:
    """Default shared cache path for post-trace evaluator-agent labels."""

    return Path(base_dir).expanduser().resolve() / "candidate_label_cache.jsonl"
