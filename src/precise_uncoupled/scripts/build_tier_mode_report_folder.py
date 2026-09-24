#!/usr/bin/env python3
"""Build top-level tier/mode report tables from per-tier report folders."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


COMMON_FIELDS = [
    "tier",
    "mode",
    "examples_analyzed",
    "included_leaf_count",
    "skipped_empty_leaf_count",
]

EFFICIENCY_FIELDS = [
    "first_pass_rate",
    "final_pass_rate",
    "avg_correction_attempts_to_success",
    "avg_correction_loops",
    "avg_reflective_refinement_rounds",
    "avg_reasoning_turns",
    "avg_total_tokens",
    "avg_model_calls",
    "avg_wall_time_seconds",
    "total_correction_attempts_to_success",
    "total_evaluator_attempts",
    "total_model_calls",
    "total_wall_time_seconds",
]

EFFICIENCY_METRIC_MAP = {
    "first_pass_rate": "basic__Pass@1",
    "final_pass_rate": "basic__FinalPassRate",
    "avg_correction_attempts_to_success": "basic__AvgCorrectionAttemptsToSuccess",
    "avg_correction_loops": "basic__AvgCorrectionLoops",
    "avg_reflective_refinement_rounds": "basic__AvgReflectiveRefinementRounds",
    "avg_reasoning_turns": "basic__AvgReasoningTurns",
    "avg_total_tokens": "basic__AvgTotalTokens",
    "avg_model_calls": "basic__AvgModelCalls",
    "avg_wall_time_seconds": "basic__AvgWallTimeSeconds",
    "total_correction_attempts_to_success": "basic__TotalCorrectionAttemptsToSuccess",
    "total_evaluator_attempts": "basic__TotalEvaluatorAttempts",
    "total_model_calls": "basic__TotalModelCalls",
    "total_wall_time_seconds": "basic__TotalWallTimeSeconds",
}

BUNDLES = {
    "rq1_tier_mode_metrics.csv": ["rq1_outcome_cost.csv"],
    "rq2_tier_mode_metrics.csv": [
        "rq2_recovery.csv",
        "rq2_system_feedback_incorporation.csv",
    ],
    "rq3_protocol_tier_mode_metrics.csv": ["rq3_protocol_refinement.csv"],
    "rq3_collaboration_tier_mode_metrics.csv": [
        "rq3_collaboration_decomposition.csv",
        "rq3_review_conditioned_decomposition.csv",
        "rq3_review_feedback_incorporation.csv",
    ],
    "strict_coupling_tier_mode_metrics.csv": ["strict_coupling_rate_metrics.csv"],
}

PROTOCOL_SUMMARY_BUNDLES = {
    "strict_coupling_protocol_tier_mode_metrics.csv": "strict_coupling_rate_protocol_summary.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--mode-dirs", nargs="+", required=True)
    parser.add_argument("--tiers", nargs="+", required=True, help="Tier ids like 01 02 08")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--layout",
        choices=["tier_first", "mode_first"],
        default="tier_first",
        help="Result layout. 'tier_first' expects result_root/tierXX/mode_dir/report, "
        "'mode_first' expects result_root/mode_dir/tierXX/report.",
    )
    parser.add_argument(
        "--mode-label",
        action="append",
        default=[],
        help="Optional mapping like full_mode_dir=short_label. Can be repeated.",
    )
    return parser.parse_args()


def _resolve_mode_root(result_root: Path, tier_name: str, mode_dir: str, layout: str) -> Path:
    if layout == "tier_first":
        return result_root / tier_name / mode_dir
    if layout == "mode_first":
        return result_root / mode_dir / tier_name
    raise ValueError(f"Unsupported layout: {layout}")


def _read_scorecard(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        row = next(reader, None)
    return row or {}


def _read_metric_bundle(report_dir: Path, filenames: list[str]) -> dict[str, str]:
    bundle: dict[str, str] = {}
    for filename in filenames:
        path = report_dir / filename
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                section = row.get("section", "").strip()
                metric = row.get("metric", "").strip()
                if not section or not metric:
                    continue
                bundle[f"{section}__{metric}"] = row.get("value", "")
    return bundle


def _read_flat_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [{str(key): ("" if value is None else str(value)) for key, value in row.items()} for row in reader]


def _read_first_metric_denominator(report_dir: Path, filenames: list[str], keys: list[str]) -> str:
    metrics = _read_metric_bundle(report_dir, filenames)
    raw_rows = _read_metric_rows(report_dir, filenames)
    for key in keys:
        if key in metrics:
            for row in raw_rows:
                section = row.get("section", "").strip()
                metric = row.get("metric", "").strip()
                if f"{section}__{metric}" == key:
                    return (
                        row.get("denominator", "")
                        or row.get("denominator_sum", "")
                        or ""
                    )
    return ""


def _read_metric_rows(report_dir: Path, filenames: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for filename in filenames:
        path = report_dir / filename
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows.extend([{str(key): ("" if value is None else str(value)) for key, value in row.items()} for row in reader])
    return rows


def _chunk_dirs(mode_root: Path) -> list[Path]:
    if not mode_root.exists():
        return []
    return sorted(
        path
        for path in mode_root.iterdir()
        if path.is_dir() and path.name != "report" and path.name.startswith("chunk")
    )


def _leaf_counts(mode_root: Path) -> tuple[int, int]:
    included = 0
    skipped_empty = 0
    for chunk_dir in _chunk_dirs(mode_root):
        worker_summary = chunk_dir / "worker_summary.json"
        if worker_summary.exists():
            try:
                payload = json.loads(worker_summary.read_text(encoding="utf-8"))
            except Exception:
                payload = {}
            if payload.get("status") == "skipped_empty_shard":
                skipped_empty += 1
                continue
        included += 1
    return included, skipped_empty


def _parse_mode_labels(raw_mappings: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in raw_mappings:
        if "=" not in item:
            raise SystemExit(f"Invalid --mode-label mapping {item!r}; expected full_mode_dir=short_label")
        left, right = item.split("=", 1)
        mapping[left.strip()] = right.strip()
    return mapping


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _remove_if_exists(path: Path) -> bool:
    if path.exists():
        path.unlink()
        return True
    return False


def main() -> None:
    args = parse_args()
    mode_labels = _parse_mode_labels(args.mode_label)
    tiers = [f"tier{int(value):02d}" for value in args.tiers]

    bundle_rows: dict[str, list[dict[str, str]]] = {name: [] for name in BUNDLES}
    bundle_columns: dict[str, set[str]] = {name: set() for name in BUNDLES}
    protocol_rows: dict[str, list[dict[str, str]]] = {name: [] for name in PROTOCOL_SUMMARY_BUNDLES}
    protocol_columns: dict[str, set[str]] = {name: set() for name in PROTOCOL_SUMMARY_BUNDLES}
    efficiency_rows: list[dict[str, str]] = []

    for tier_name in tiers:
        for mode_dir in args.mode_dirs:
            mode_root = _resolve_mode_root(args.result_root, tier_name, mode_dir, args.layout)
            report_dir = mode_root / "report"
            if not report_dir.exists():
                continue

            scorecard_path = report_dir / "tier_mode_scorecard.tsv"
            scorecard = _read_scorecard(scorecard_path) if scorecard_path.exists() else {}
            mode_label = mode_labels.get(mode_dir, mode_dir)
            included_leaf_count, skipped_empty_leaf_count = _leaf_counts(mode_root)
            rq1_metrics = _read_metric_bundle(report_dir, ["rq1_outcome_cost.csv"])
            examples_analyzed = (
                scorecard.get("examples_analyzed", "")
                or _read_first_metric_denominator(
                    report_dir,
                    ["rq1_outcome_cost.csv"],
                    ["basic__FinalPassRate", "basic__Pass@1"],
                )
                or _read_first_metric_denominator(
                    report_dir,
                    ["rq2_recovery.csv"],
                    ["outer_loop__OuterPass@1"],
                )
            )
            common = {
                "tier": tier_name,
                "mode": mode_label,
                "examples_analyzed": examples_analyzed,
                "included_leaf_count": scorecard.get("included_leaf_count", "") or str(included_leaf_count),
                "skipped_empty_leaf_count": scorecard.get("skipped_empty_leaf_count", "") or str(skipped_empty_leaf_count),
            }

            efficiency_row = dict(common)
            for field in EFFICIENCY_FIELDS:
                efficiency_row[field] = rq1_metrics.get(EFFICIENCY_METRIC_MAP[field], "")
            efficiency_rows.append(efficiency_row)

            for output_name, source_files in BUNDLES.items():
                metrics = _read_metric_bundle(report_dir, source_files)
                row = dict(common)
                row.update(metrics)
                bundle_rows[output_name].append(row)
                bundle_columns[output_name].update(metrics.keys())

            for output_name, source_file in PROTOCOL_SUMMARY_BUNDLES.items():
                rows = _read_flat_csv_rows(report_dir / source_file)
                if not rows:
                    continue
                for extra in rows:
                    row = dict(common)
                    row.update(extra)
                    protocol_rows[output_name].append(row)
                    protocol_columns[output_name].update(extra.keys())

    written_outputs: list[Path] = []
    efficiency_path = args.output_dir / "efficiency_tier_mode_metrics.csv"
    if efficiency_rows:
        _write_csv(efficiency_path, efficiency_rows, COMMON_FIELDS + EFFICIENCY_FIELDS)
        written_outputs.append(efficiency_path)
    else:
        _remove_if_exists(efficiency_path)

    for output_name, rows in bundle_rows.items():
        path = args.output_dir / output_name
        dynamic_fields = sorted(bundle_columns[output_name])
        if not dynamic_fields:
            _remove_if_exists(path)
            continue
        _write_csv(path, rows, COMMON_FIELDS + dynamic_fields)
        written_outputs.append(path)

    for output_name, rows in protocol_rows.items():
        path = args.output_dir / output_name
        dynamic_fields = sorted(protocol_columns[output_name])
        if not dynamic_fields:
            _remove_if_exists(path)
            continue
        _write_csv(path, rows, COMMON_FIELDS + dynamic_fields)
        written_outputs.append(path)

    if not written_outputs:
        raise SystemExit(f"No tier-mode report inputs found under result_root={args.result_root}")

    for path in written_outputs:
        print(f"[tier-mode-report] wrote {path}")


if __name__ == "__main__":
    main()
