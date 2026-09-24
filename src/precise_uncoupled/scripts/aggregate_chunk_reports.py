#!/usr/bin/env python3
"""Aggregate per-chunk trace-analysis reports into a tier-level report folder."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


ANALYSIS_LAYOUT = {
    "rq1-outcome-cost": ["rq1_outcome_cost.csv"],
    "rq2-recovery": [
        "rq2_recovery.csv",
        "rq2_system_feedback_incorporation.csv",
    ],
    "rq3-protocol-refinement": ["rq3_protocol_refinement.csv"],
    "rq3-collaboration-decomposition": [
        "rq3_review_conditioned_decomposition.csv",
        "rq3_review_feedback_incorporation.csv",
    ],
}

REPORT_BUNDLES = [
    (
        "rq1_outcome_cost",
        lambda row: row["analysis"] == "rq1-outcome-cost",
        "Tier-level aggregate of chunk rq1 outcome/cost metrics.",
    ),
    (
        "rq2_recovery",
        lambda row: row["analysis"] == "rq2-recovery",
        "Tier-level aggregate of chunk rq2 recovery metrics, including feedback-incorporation sections.",
    ),
    (
        "rq3_protocol_refinement",
        lambda row: row["analysis"] == "rq3-protocol-refinement",
        "Tier-level aggregate of chunk rq3 protocol-refinement metrics.",
    ),
    (
        "rq3_collaboration_decomposition",
        lambda row: row["analysis"] == "rq3-collaboration-decomposition",
        "Tier-level aggregate of chunk rq3 collaboration-decomposition metrics, including reviewer-conditioned and reviewer-feedback sections.",
    ),
]


def _parse_float(value: str):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _is_integer_like(value: float) -> bool:
    return math.isfinite(value) and abs(value - round(value)) < 1e-9


def _format_number(value):
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if _is_integer_like(value):
            return str(int(round(value)))
        return f"{value:.10f}".rstrip("0").rstrip(".")
    return str(value)


def _resolve_mode_root(result_root: Path, tier_name: str, mode_dir: str, layout: str) -> Path:
    if layout == "tier_first":
        return result_root / tier_name / mode_dir
    if layout == "mode_first":
        return result_root / mode_dir / tier_name
    raise ValueError(f"Unsupported layout: {layout}")


def _collect_chunk_dirs(mode_root: Path):
    return sorted(
        path
        for path in mode_root.iterdir()
        if path.is_dir() and path.name != "report" and (path.name.startswith("chunck") or path.name.startswith("chunk"))
    )


def _collect_rows(result_root: Path, tier_name: str, mode_dir: str, layout: str):
    rows = []
    mode_root = _resolve_mode_root(result_root, tier_name, mode_dir, layout)
    chunk_dirs = _collect_chunk_dirs(mode_root)
    for chunk_dir in chunk_dirs:
        chunk_alias = chunk_dir.name
        for analysis_name, filenames in ANALYSIS_LAYOUT.items():
            analysis_dir = chunk_dir / analysis_name
            if not analysis_dir.is_dir():
                continue
            for filename in filenames:
                csv_path = analysis_dir / filename
                if not csv_path.exists():
                    continue
                with csv_path.open("r", encoding="utf-8", newline="") as handle:
                    reader = csv.DictReader(handle)
                    for row in reader:
                        rows.append(
                            {
                                "tier": tier_name,
                                "mode_dir": mode_dir,
                                "chunk_alias": chunk_alias,
                                "analysis": analysis_name,
                                "source_csv": filename,
                                "section": row.get("section", ""),
                                "metric": row.get("metric", ""),
                                "value": row.get("value", ""),
                                "numerator": row.get("numerator", ""),
                                "denominator": row.get("denominator", ""),
                                "source": row.get("source", ""),
                                "notes": row.get("notes", ""),
                            }
                        )
    return mode_root, chunk_dirs, rows


def _build_chunk_metric_index(rows):
    index = defaultdict(dict)
    for row in rows:
        key = (row["analysis"], row["source_csv"], row["section"], row["metric"])
        index[row["chunk_alias"]][key] = row
    return index


def _chunk_example_count(chunk_rows) -> float | None:
    candidates = [
        ("rq1-outcome-cost", "rq1_outcome_cost.csv", "basic", "FinalPassRate"),
        ("rq1-outcome-cost", "rq1_outcome_cost.csv", "basic", "Pass@1"),
        ("rq2-recovery", "rq2_recovery.csv", "outer_loop", "OuterPass@1"),
    ]
    for key in candidates:
        row = chunk_rows.get(key)
        if not row:
            continue
        denominator = _parse_float(row.get("denominator", ""))
        if denominator is not None:
            return denominator
    return None


def _parse_k_metric(metric: str, prefix: str) -> int | None:
    if not metric.startswith(prefix):
        return None
    try:
        return int(metric[len(prefix):])
    except ValueError:
        return None


def _filled_outer_pass_for_chunk(chunk_rows, k: int):
    prefix = "OuterPass@"
    pass_rows = {}
    for (analysis, source_csv, section, metric), row in chunk_rows.items():
        if analysis == "rq2-recovery" and source_csv == "rq2_recovery.csv" and section == "outer_loop":
            parsed_k = _parse_k_metric(metric, prefix)
            if parsed_k is not None:
                pass_rows[parsed_k] = row
    total_examples = _chunk_example_count(chunk_rows)
    if total_examples is None:
        return None, None
    if k in pass_rows:
        return _parse_float(pass_rows[k].get("numerator", "")), total_examples
    available = [value for value in pass_rows if value < k]
    if available:
        nearest = max(available)
        return _parse_float(pass_rows[nearest].get("numerator", "")), total_examples
    return 0.0, total_examples


def _filled_marginal_gain_for_chunk(chunk_rows, k: int):
    prefix = "MarginalRecoveryGain_"
    total_examples = _chunk_example_count(chunk_rows)
    if total_examples is None:
        return None, None
    for (analysis, source_csv, section, metric), row in chunk_rows.items():
        if analysis == "rq2-recovery" and source_csv == "rq2_recovery.csv" and section == "outer_loop":
            parsed_k = _parse_k_metric(metric, prefix)
            if parsed_k == k:
                return _parse_float(row.get("numerator", "")), total_examples
    return 0.0, total_examples


def _aggregate_metric_rows(rows, total_chunk_count: int):
    chunk_index = _build_chunk_metric_index(rows)
    grouped = defaultdict(list)
    for row in rows:
        key = (row["analysis"], row["source_csv"], row["section"], row["metric"])
        grouped[key].append(row)

    aggregate_rows = []
    for (analysis, source_csv, section, metric), metric_rows in sorted(grouped.items()):
        value_list = [_parse_float(row["value"]) for row in metric_rows]
        num_list = [_parse_float(row["numerator"]) for row in metric_rows]
        den_list = [_parse_float(row["denominator"]) for row in metric_rows]

        valid_values = [v for v in value_list if v is not None]
        valid_nums = [v for v in num_list if v is not None]
        valid_dens = [v for v in den_list if v is not None]

        aggregation = "unavailable"
        agg_value = None
        agg_num = None
        agg_den = None
        contributing_chunk_count = len(metric_rows)

        outer_k = None
        gain_k = None
        if analysis == "rq2-recovery" and source_csv == "rq2_recovery.csv" and section == "outer_loop":
            outer_k = _parse_k_metric(metric, "OuterPass@")
            gain_k = _parse_k_metric(metric, "MarginalRecoveryGain_")

        if outer_k is not None:
            nums = []
            dens = []
            for chunk_rows in chunk_index.values():
                num, den = _filled_outer_pass_for_chunk(chunk_rows, outer_k)
                if num is None or den is None:
                    continue
                nums.append(num)
                dens.append(den)
            if nums and dens:
                agg_num = sum(nums)
                agg_den = sum(dens)
                agg_value = (agg_num / agg_den) if agg_den else None
                aggregation = "sum_ratio_filled"
                contributing_chunk_count = len(nums)
        elif gain_k is not None:
            nums = []
            dens = []
            for chunk_rows in chunk_index.values():
                num, den = _filled_marginal_gain_for_chunk(chunk_rows, gain_k)
                if num is None or den is None:
                    continue
                nums.append(num)
                dens.append(den)
            if nums and dens:
                agg_num = sum(nums)
                agg_den = sum(dens)
                agg_value = (agg_num / agg_den) if agg_den else None
                aggregation = "sum_ratio_filled"
                contributing_chunk_count = len(nums)

        elif valid_nums and valid_dens and len(valid_nums) == len(metric_rows) and len(valid_dens) == len(metric_rows):
            agg_num = sum(valid_nums)
            agg_den = sum(valid_dens)
            agg_value = (agg_num / agg_den) if agg_den not in (None, 0) else None
            aggregation = "sum_ratio"
        elif valid_values and len(valid_values) == len(metric_rows):
            if all(_is_integer_like(v) for v in valid_values):
                agg_value = sum(valid_values)
                aggregation = "sum"
            else:
                agg_value = sum(valid_values) / len(valid_values)
                aggregation = "mean"

        aggregate_rows.append(
            {
                "analysis": analysis,
                "source_csv": source_csv,
                "section": section,
                "metric": metric,
                "aggregation": aggregation,
                "value": _format_number(agg_value),
                "numerator_sum": _format_number(agg_num),
                "denominator_sum": _format_number(agg_den),
                "chunk_count": str(contributing_chunk_count),
                "total_chunks": str(total_chunk_count),
                "source": metric_rows[0]["source"],
                "notes": metric_rows[0]["notes"],
            }
        )
    return _postprocess_aggregate_rows(aggregate_rows)


def _postprocess_aggregate_rows(aggregate_rows):
    keyed = {
        (row["analysis"], row["source_csv"], row["section"], row["metric"]): row
        for row in aggregate_rows
    }

    key = (
        "rq3-collaboration-decomposition",
        "rq3_review_conditioned_decomposition.csv",
        "reviewer_detection",
        "BalAcc_review",
    )
    bal_acc_row = keyed.get(key)
    if bal_acc_row and not bal_acc_row["value"]:
        tp = _parse_float(
            keyed[
                (
                    "rq3-collaboration-decomposition",
                    "rq3_review_conditioned_decomposition.csv",
                    "reviewer_detection",
                    "TP_review",
                )
            ]["value"]
        )
        fn = _parse_float(
            keyed[
                (
                    "rq3-collaboration-decomposition",
                    "rq3_review_conditioned_decomposition.csv",
                    "reviewer_detection",
                    "FN_review",
                )
            ]["value"]
        )
        fp = _parse_float(
            keyed[
                (
                    "rq3-collaboration-decomposition",
                    "rq3_review_conditioned_decomposition.csv",
                    "reviewer_detection",
                    "FP_review",
                )
            ]["value"]
        )
        tn = _parse_float(
            keyed[
                (
                    "rq3-collaboration-decomposition",
                    "rq3_review_conditioned_decomposition.csv",
                    "reviewer_detection",
                    "TN_review",
                )
            ]["value"]
        )
        if None not in (tp, fn, fp, tn):
            tpr = (tp / (tp + fn)) if (tp + fn) else None
            tnr = (tn / (tn + fp)) if (tn + fp) else None
            if tpr is not None and tnr is not None:
                bal = (tpr + tnr) / 2.0
                bal_acc_row["value"] = _format_number(bal)
                bal_acc_row["aggregation"] = "derived_from_counts"
                bal_acc_row["numerator_sum"] = ""
                bal_acc_row["denominator_sum"] = ""
    return aggregate_rows


def _write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_summary(path: Path, tier: str, mode_dir: str, chunk_count: int, aggregate_rows):
    lines = [
        "Tier Report Aggregate Summary",
        "",
        f"tier={tier}",
        f"mode_dir={mode_dir}",
        f"chunks_aggregated={chunk_count}",
        "",
        "Selected Aggregate Metrics",
        "",
    ]

    preferred_metrics = [
        "FinalPassRate",
        "Pass@1",
        "AvgCorrectionAttemptsToSuccess",
        "OuterPass@1",
        "MRRAttemptAccuracy",
        "HintAwareCorrectionScore",
        "ReflectiveRepairRate",
        "ReflectivePreservationRate",
        "ReviewerFeedbackIncorporationRate",
        "UsefulReviseFeedbackIncorporationRate",
    ]
    preferred = {row["metric"]: row for row in aggregate_rows}
    for metric in preferred_metrics:
        row = preferred.get(metric)
        if not row:
            continue
        lines.append(
            f"{row['analysis']} / {row['metric']}: {row['value']} "
            f"(aggregation={row['aggregation']}, chunks={row['chunk_count']})"
        )

    lines.extend(
        [
            "",
            "Artifacts",
            f"- report_metrics_by_chunk.csv",
            f"- report_metrics_aggregated.csv",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_bundle_txt(path: Path, title: str, description: str, rows) -> None:
    lines = [
        title,
        "",
        description,
        "",
    ]
    last_source = None
    last_section = None
    for row in rows:
        source_csv = row["source_csv"]
        section = row["section"]
        if source_csv != last_source:
            lines.extend(["", f"[{source_csv}]"])
            last_source = source_csv
            last_section = None
        if section != last_section:
            lines.extend(["", f"{section}"])
            last_section = section
        metric = row["metric"]
        value = row["value"] or "n/a"
        agg = row["aggregation"]
        chunk_count = row["chunk_count"]
        num = row["numerator_sum"]
        den = row["denominator_sum"]
        detail = f"{metric}: {value} # aggregation={agg}; chunks={chunk_count}"
        total_chunks = row["total_chunks"]
        if total_chunks and total_chunks != chunk_count:
            detail = f"{metric}: {value} # aggregation={agg}; chunks={chunk_count}/{total_chunks}"
        if num or den:
            detail += f"; numerator_sum={num or 'n/a'}; denominator_sum={den or 'n/a'}"
        if row["notes"]:
            detail += f"; notes={row['notes']}"
        lines.append(detail)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_bundle_reports(report_dir: Path, aggregate_rows) -> None:
    fieldnames = [
        "analysis",
        "source_csv",
        "section",
        "metric",
        "aggregation",
        "value",
        "numerator_sum",
        "denominator_sum",
        "chunk_count",
        "total_chunks",
        "source",
        "notes",
    ]
    for stem, predicate, description in REPORT_BUNDLES:
        rows = [row for row in aggregate_rows if predicate(row)]
        if not rows:
            continue
        csv_path = report_dir / f"{stem}.csv"
        txt_path = report_dir / f"{stem}.txt"
        _write_csv(csv_path, rows, fieldnames)
        _write_bundle_txt(txt_path, stem, description, rows)


def build_report(result_root: Path, tier_name: str, mode_dir: str, layout: str) -> None:
    mode_root = _resolve_mode_root(result_root, tier_name, mode_dir, layout)
    report_dir = mode_root / "report"
    mode_root, chunk_dirs, rows = _collect_rows(result_root, tier_name, mode_dir, layout)
    if not rows:
        raise SystemExit(f"No analysis CSVs found under {mode_root}")

    by_chunk_path = report_dir / "report_metrics_by_chunk.csv"
    agg_path = report_dir / "report_metrics_aggregated.csv"
    summary_path = report_dir / "report_summary.txt"

    _write_csv(
        by_chunk_path,
        rows,
        [
            "tier",
            "mode_dir",
            "chunk_alias",
            "analysis",
            "source_csv",
            "section",
            "metric",
            "value",
            "numerator",
            "denominator",
            "source",
            "notes",
        ],
    )
    aggregate_rows = _aggregate_metric_rows(rows, len(chunk_dirs))
    _write_csv(
        agg_path,
        aggregate_rows,
        [
            "analysis",
            "source_csv",
            "section",
            "metric",
            "aggregation",
            "value",
            "numerator_sum",
            "denominator_sum",
            "chunk_count",
            "total_chunks",
            "source",
            "notes",
        ],
    )
    _write_summary(summary_path, tier_name, mode_dir, len(chunk_dirs), aggregate_rows)
    _write_bundle_reports(report_dir, aggregate_rows)
    print(f"[aggregate-report] wrote {report_dir}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate chunk-level reports into a tier-level report folder.")
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--tiers", nargs="+", required=True, help="Tier ids like 01 02 08")
    parser.add_argument("--mode-dir", required=True, help="broadcast / per / single_agent / baseline_llm")
    parser.add_argument(
        "--layout",
        choices=["tier_first", "mode_first"],
        default="tier_first",
        help="Result layout. 'tier_first' expects result_root/tierXX/mode_dir/chunk..., 'mode_first' expects result_root/mode_dir/tierXX/chunk....",
    )
    args = parser.parse_args()

    for tier in args.tiers:
        build_report(args.result_root, f"tier{tier}", args.mode_dir, args.layout)


if __name__ == "__main__":
    main()
