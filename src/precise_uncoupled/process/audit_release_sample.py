#!/usr/bin/env python3
"""Build a deterministic, stratified release-v2 provenance-audit worksheet.

The legacy canonical-review CSV is used only as a trajectory index so the
sample contains review opportunities. None of its before/after labels or
process metrics are reused.

PyArrow is imported lazily because the extractor and its unit tests are
dependency-free. See README.md for the full-Python invocation used locally.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from symmetric_transition_extractor import extract_symmetric_transitions


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
DEFAULT_RELEASE_ROOT = PROJECT / "analysis-4x2x5" / "data_cache"
DEFAULT_CANDIDATE_INDEX = (
    PROJECT / "analysis-4x2x5" / "outputs" / "canonical_review_events.csv"
)
DEFAULT_OUTPUT = HERE / "fixtures" / "release_v2_hand_audit_sample.csv"
DEFAULT_MANIFEST = (
    HERE / "fixtures" / "release_v2_hand_audit_sample_manifest.json"
)
DEFAULT_VERDICTS = HERE / "fixtures" / "release_v2_hand_audit_verdicts.csv"

AUDIT_FIELDS = [
    "dataset",
    "actor_family",
    "protocol",
    "trajectory_uid",
    "review_id",
    "sample_selection_reason",
    "status",
    "exclusion_reason",
    "review_action",
    "review_sequence_start",
    "review_sequence_end",
    "reviewer_senders_json",
    "review_stages_json",
    "reviewed_answer",
    "before_sequence_index",
    "before_sender",
    "before_stage",
    "answer_before",
    "before_content_excerpt",
    "review_feedback_excerpt",
    "reviewer_proposed_answers_json",
    "after_sequence_index",
    "after_sender",
    "after_stage",
    "answer_after",
    "after_content_excerpt",
    "boundary_sequence_index",
    "boundary_stage",
    "before_candidate_link_method",
    "after_candidate_link_method",
    "candidate_alignment",
    "actor_turn_observed",
    "actor_response_observed",
    "answer_pair_parsed",
    "strict_pre_gate_eligible",
    "intervening_system_candidate_updates",
    "system_selected_answers_json",
    "system_selected_answer_used_as_after",
    "manual_before_actor_provenance",
    "manual_review_provenance",
    "manual_after_actor_or_exclusion",
    "manual_system_selection_not_used_as_after",
    "manual_verdict",
    "manual_notes",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _candidate_trajectories(
    path: Path, per_cell: int
) -> dict[tuple[str, str, str], list[str]]:
    grouped: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    seen: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            protocol = row.get("protocol", "")
            if protocol not in {"per", "broadcast"}:
                continue
            key = (
                row.get("dataset", ""),
                row.get("actor_family", ""),
                protocol,
            )
            trajectory_uid = row.get("trajectory_uid", "")
            if (
                not trajectory_uid
                or trajectory_uid in seen[key]
                or len(grouped[key]) >= per_cell
            ):
                continue
            grouped[key].append(trajectory_uid)
            seen[key].add(trajectory_uid)
    return dict(grouped)


def _load_selected_messages(
    messages_root: Path, trajectory_uids: Sequence[str]
) -> dict[str, list[dict[str, Any]]]:
    try:
        import pyarrow.dataset as ds
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "PyArrow is required only for this release audit. Follow the "
            "PYTHONPATH invocation in process/README.md."
        ) from exc

    columns = [
        "event_uid",
        "message_uid",
        "trajectory_uid",
        "sequence_index",
        "event_type",
        "round_index",
        "stage",
        "sender",
        "role",
        "content",
    ]
    dataset = ds.dataset(messages_root, format="parquet")
    table = dataset.to_table(
        columns=columns,
        filter=ds.field("trajectory_uid").isin(list(trajectory_uids)),
    )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in table.to_pylist():
        grouped[str(row["trajectory_uid"])].append(row)
    return dict(grouped)


def _selection_key(record: Mapping[str, Any], protocol: str) -> tuple[Any, ...]:
    """Prefer rows that exercise the highest-risk provenance branch."""

    if protocol == "broadcast":
        status_order = {
            "actor_response_after_protocol_selection": 0,
            "actor_response_before_gate": 1,
            "no_actor_response_before_submission": 2,
            "no_actor_response_before_next_review": 3,
            "missing_actor_candidate_before": 4,
            "no_actor_response_observed": 5,
        }
        return (
            status_order.get(str(record.get("status")), 9),
            -int(record.get("intervening_system_candidate_updates") or 0),
            int(record.get("review_sequence_start") or 0),
        )

    return (
        0 if record.get("actor_response_observed") else 1,
        0 if record.get("answer_pair_parsed") else 1,
        int(record.get("review_sequence_start") or 0),
    )


def _selection_reason(record: Mapping[str, Any]) -> str:
    status = str(record.get("status") or "")
    if status == "actor_response_after_protocol_selection":
        return "purposive: actor response after system-selected reviewer proposal"
    if status == "actor_response_before_gate":
        return "purposive: genuine actor response before a recorded gate"
    if status == "no_actor_response_before_submission":
        return "purposive: consensus/submission with no next-actor response"
    return f"purposive: {status or 'fallback review opportunity'}"


def _choose_one_per_condition(
    candidates: Mapping[tuple[str, str, str], Sequence[str]],
    messages: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for (dataset, actor, protocol), trajectory_uids in sorted(candidates.items()):
        records: list[dict[str, Any]] = []
        for trajectory_uid in trajectory_uids:
            events = messages.get(trajectory_uid, ())
            if not events:
                continue
            for record in extract_symmetric_transitions(
                protocol, events, trajectory_uid=trajectory_uid
            ):
                enriched = dict(record)
                enriched.update(
                    {
                        "dataset": dataset,
                        "actor_family": actor,
                        "protocol": protocol,
                    }
                )
                records.append(enriched)
        if not records:
            continue
        chosen = min(records, key=lambda row: _selection_key(row, protocol))
        chosen["sample_selection_reason"] = _selection_reason(chosen)
        chosen.update(
            {
                "manual_before_actor_provenance": "PENDING",
                "manual_review_provenance": "PENDING",
                "manual_after_actor_or_exclusion": "PENDING",
                "manual_system_selection_not_used_as_after": "PENDING",
                "manual_verdict": "PENDING",
                "manual_notes": "",
            }
        )
        selected.append(chosen)
    return selected


def _write_audit(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError("No audit rows were selected")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _status_counts(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row.get("status") or "unknown")] += 1
    return dict(sorted(counts.items()))


def _validate_manual_verdicts(
    audit_rows: Sequence[Mapping[str, Any]], verdict_path: Path
) -> tuple[str, dict[str, int]]:
    if not verdict_path.exists():
        return "pending", {}
    with verdict_path.open(encoding="utf-8", newline="") as handle:
        verdicts = list(csv.DictReader(handle))
    audit_ids = {str(row["review_id"]) for row in audit_rows}
    verdict_ids = {str(row.get("review_id") or "") for row in verdicts}
    if audit_ids != verdict_ids:
        missing = sorted(audit_ids - verdict_ids)
        extra = sorted(verdict_ids - audit_ids)
        raise ValueError(
            "Manual verdict IDs do not match the generated audit worksheet: "
            f"missing={missing}, extra={extra}"
        )
    verdict_counts: dict[str, int] = defaultdict(int)
    for row in verdicts:
        verdict_counts[str(row.get("manual_verdict") or "missing")] += 1
    return "completed_separate_verdict_file", dict(sorted(verdict_counts.items()))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, default=DEFAULT_RELEASE_ROOT)
    parser.add_argument(
        "--candidate-index", type=Path, default=DEFAULT_CANDIDATE_INDEX
    )
    parser.add_argument("--candidate-traces-per-cell", type=int, default=24)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--verdicts", type=Path, default=DEFAULT_VERDICTS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    candidate_groups = _candidate_trajectories(
        args.candidate_index, args.candidate_traces_per_cell
    )
    all_trajectory_uids = sorted(
        {
            trajectory_uid
            for trajectory_uids in candidate_groups.values()
            for trajectory_uid in trajectory_uids
        }
    )
    messages = _load_selected_messages(
        args.release_root / "data" / "messages", all_trajectory_uids
    )
    rows = _choose_one_per_condition(candidate_groups, messages)
    _write_audit(rows, args.output_csv)

    release_manifest = args.release_root / "registry" / "release_manifest.json"
    embedded_manifest = json.loads(release_manifest.read_text(encoding="utf-8"))
    manual_status, manual_counts = _validate_manual_verdicts(
        rows, args.verdicts
    )
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Stratified purposive provenance audit; not an estimator and not "
            "reviewer-facing numerical evidence"
        ),
        "extractor_definition": (
            "actor candidate before review -> next actor candidate after review; "
            "reviewer/system/evaluator answers prohibited as answer_after"
        ),
        "source_hf_repository": embedded_manifest.get("source_hf_repository"),
        "embedded_source_revision": embedded_manifest.get("source_hf_revision"),
        "release_manifest_sha256": _sha256(release_manifest),
        "candidate_index_path": str(args.candidate_index),
        "candidate_index_sha256": _sha256(args.candidate_index),
        "candidate_index_use": (
            "trajectory discovery only; legacy before/after labels and metrics ignored"
        ),
        "candidate_traces_per_condition": args.candidate_traces_per_cell,
        "conditions_discovered": len(candidate_groups),
        "audit_rows": len(rows),
        "status_counts": _status_counts(rows),
        "sample_diagnostic_counts": {
            "actor_turn_observed": sum(
                bool(row.get("actor_turn_observed")) for row in rows
            ),
            "actor_response_observed": sum(
                bool(row.get("actor_response_observed")) for row in rows
            ),
            "strict_pre_gate_eligible": sum(
                bool(row.get("strict_pre_gate_eligible")) for row in rows
            ),
            "rows_with_system_candidate_update": sum(
                int(row.get("intervening_system_candidate_updates") or 0) > 0
                for row in rows
            ),
            "logical_system_candidate_updates": sum(
                int(row.get("intervening_system_candidate_updates") or 0)
                for row in rows
            ),
            "system_selected_answer_used_as_after": sum(
                bool(row.get("system_selected_answer_used_as_after"))
                for row in rows
            ),
        },
        "manual_audit_status": manual_status,
        "manual_verdict_counts": manual_counts,
        "manual_verdicts_path": str(args.verdicts),
        "manual_verdicts_sha256": (
            _sha256(args.verdicts) if args.verdicts.exists() else None
        ),
        "output_csv": str(args.output_csv),
        "output_csv_sha256": _sha256(args.output_csv),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"Wrote {args.output_csv} ({len(rows)} conditions); "
        f"status={manifest['status_counts']}"
    )
    print(f"Wrote {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
