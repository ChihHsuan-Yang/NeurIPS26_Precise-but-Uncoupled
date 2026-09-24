#!/usr/bin/env python3
"""Bootstrap paper metrics from saved `paper_experiment` results.

This script provides a more honest uncertainty story for episode-derived
review metrics by resampling whole problems instead of individual review
episodes. It is intended for saved-result analysis only; it does not rerun
benchmark inference.
"""

from __future__ import annotations

import argparse
import os
import csv
import json
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Callable

# RELEASE PATCH: importable from a fresh clone with no install.
# Only src/ is added. src/precise_uncoupled must NEVER go on sys.path -- it holds
# subpackages named `io` and `scripts` that would shadow the standard library.
SRC_ROOT = Path(__file__).resolve().parents[2]      # src
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from agentverse.metrics.collaboration_decomposition import (
    load_collaboration_decomposition_dataset,
    summarize_reviewer_conditioned_decomposition,
    summarize_reviewer_feedback_incorporation,
)
from agentverse.metrics.collaboration_decomposition.strict_coupling import (
    build_strict_coupling_episode_records,
    summarize_strict_coupling_rate,
)
from agentverse.metrics.main_paper.inputs import _discover_run_config_path
from agentverse.metrics.main_paper.records import MetricRecord


RQ1_METRIC_NAMES = (
    "FinalPassRate",
    "PassAt1Rate",
    "AvgTotalTokens",
    "AvgModelCalls",
)

RQ3_METRIC_NAMES = (
    "reviewer_detection__Prec_review",
    "reviewer_detection__Rec_review",
    "reviewer_detection__FAR_review",
    "reviewer_detection__BalAcc_review",
    "reviewer_conditioned_response__ReviewerGuidedRepairRate",
    "reviewer_conditioned_response__ReviewerDetectedButNotFixedRate",
    "reviewer_conditioned_response__MisleadingReviewHarmRate",
    "reviewer_conditioned_response__MisleadingReviewResistanceRate",
    "reviewer_feedback_incorporation__ReviewerFeedbackIncorporationRate",
    "reviewer_feedback_incorporation__UsefulReviseFeedbackIncorporationRate",
    "reviewer_feedback_incorporation__MisleadingReviewSusceptibilityRate",
)

STRICT_METRIC_NAMES = (
    "strict_coupling__StrictUsefulCouplingRate",
    "strict_coupling__EquivalenceAwareUsefulCouplingRate",
    "strict_coupling__ChangedAndRepairedRateAmongUsefulRevise",
    "strict_coupling__ChangedButStillWrongRateAmongUsefulRevise",
    "strict_coupling__UnchangedRateAmongUsefulRevise",
    "strict_coupling__RepairGivenChangeRate",
    "strict_coupling__StrictLegacyDisagreementRate",
)


@dataclass(frozen=True)
class ProblemRow:
    problem_id: str
    tier: str
    protocol: str
    global_example_idx: int
    final_passed: bool
    first_passed: bool
    total_tokens: float
    model_calls: float


# RELEASE PATCH (NeurIPS26_Precise-but-Uncoupled).
# Upstream `--result_root` defaulted to an author-private absolute path pointing
# at `results/paper_experiment`, a directory that was later renamed and no longer
# exists. The released default is repository-relative and env-overridable:
#   PU_TRACE_ROOT -> root of the downloaded per-tier trace tree
#                    (layout: tierNN/{baseline_llm,single_agent,per,broadcast}/)
# Pass --result_root to override on the command line. See docs/DATA.md.
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULT_ROOT = os.environ.get(
    "PU_TRACE_ROOT", str(REPO_ROOT / "data" / "traces" / "gpt_oss_120b")
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate problem-level and problem-clustered bootstrap intervals "
            "for saved paper_experiment results."
        )
    )
    parser.add_argument(
        "--result_root",
        type=str,
        default=DEFAULT_RESULT_ROOT,
        help="Root directory containing tierXX/mode result folders.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["per", "broadcast"],
        help="Protocols to analyze for clustered review metrics.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory. Defaults to <result_root>/report/bootstrap_problem_clustered.",
    )
    parser.add_argument(
        "--n_bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap replicates.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Bootstrap RNG seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_root = Path(args.result_root).expanduser().resolve()
    if not result_root.exists():
        raise FileNotFoundError(f"Missing result root: {result_root}")

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else result_root / "report" / "bootstrap_problem_clustered"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(int(args.seed))
    modes = list(dict.fromkeys(args.modes))
    tiers = discover_tiers(result_root)
    if not tiers:
        raise FileNotFoundError(f"No tierXX folders found under: {result_root}")

    problem_rows_by_mode: dict[str, list[ProblemRow]] = {}
    trace_key_to_problem_id_by_mode: dict[str, dict[tuple[str, int], str]] = {}
    trace_files_by_mode: dict[str, list[Path]] = {}

    for mode in modes:
        rows, mapping = load_problem_rows_and_mapping(result_root, tiers, mode)
        if not rows:
            raise FileNotFoundError(f"No tier_mode_problem_table.csv rows found for mode={mode}")
        problem_rows_by_mode[mode] = rows
        trace_key_to_problem_id_by_mode[mode] = mapping
        trace_files = sorted((result_root).glob(f"tier*/{mode}/**/*.trace.txt"))
        if not trace_files:
            raise FileNotFoundError(f"No trace files found for mode={mode} under {result_root}")
        trace_files_by_mode[mode] = trace_files

    coverage_rows: list[dict[str, object]] = []
    rq1_summary_rows: list[dict[str, object]] = []
    rq3_summary_rows: list[dict[str, object]] = []
    strict_summary_rows: list[dict[str, object]] = []
    paired_rq1_rows: list[dict[str, object]] = []
    paired_rq3_rows: list[dict[str, object]] = []
    paired_strict_rows: list[dict[str, object]] = []
    review_bundles_by_mode: dict[str, dict[str, object]] = {}

    for mode in modes:
        rows = problem_rows_by_mode[mode]
        rq1_point = compute_rq1_metrics(rows)
        rq1_boot = bootstrap_metric_dict(
            problem_ids=[row.problem_id for row in rows],
            metric_fn=lambda sampled_ids, *, mode=mode: compute_rq1_metrics(
                resample_problem_rows(problem_rows_by_mode[mode], sampled_ids)
            ),
            metric_names=RQ1_METRIC_NAMES,
            n_bootstrap=int(args.n_bootstrap),
            rng=rng,
        )
        rq1_summary_rows.extend(
            summarize_bootstrap_dict(
                family="rq1_problem_level",
                protocol=mode,
                point_metrics=rq1_point,
                bootstrap_metrics=rq1_boot,
            )
        )

        merged_cache_path = build_merged_candidate_cache(
            result_root=result_root,
            mode=mode,
            output_dir=output_dir / "_merged_caches",
        )
        review_bundle = load_review_bundle(
            trace_files=trace_files_by_mode[mode],
            trace_key_to_problem_id=trace_key_to_problem_id_by_mode[mode],
            candidate_label_cache_path=merged_cache_path,
        )
        review_bundles_by_mode[mode] = review_bundle
        coverage_rows.append(
            {
                "protocol": mode,
                "problems_total": len(rows),
                "trace_files": len(trace_files_by_mode[mode]),
                "review_episodes": len(review_bundle["review_episodes"]),
                "feedback_episodes": len(review_bundle["reviewer_feedback_episodes"]),
                "strict_records": len(review_bundle["strict_records"]),
                "mapped_review_episodes": review_bundle["mapped_review_episodes"],
                "unmapped_review_episodes": review_bundle["unmapped_review_episodes"],
                "mapped_feedback_episodes": review_bundle["mapped_feedback_episodes"],
                "unmapped_feedback_episodes": review_bundle["unmapped_feedback_episodes"],
                "mapped_strict_records": review_bundle["mapped_strict_records"],
                "unmapped_strict_records": review_bundle["unmapped_strict_records"],
                "candidate_cache_entries": review_bundle["candidate_cache_entries"],
                "analysis_runtime_label_hits": review_bundle["analysis_runtime_label_hits"],
                "analysis_candidate_cache_hits": review_bundle["analysis_candidate_cache_hits"],
                "analysis_candidate_cache_writes": review_bundle["analysis_candidate_cache_writes"],
                "analysis_feedback_labeler_mode": review_bundle["analysis_feedback_labeler_mode"],
            }
        )

        rq3_point = compute_rq3_metrics(
            review_bundle["review_episodes"],
            review_bundle["reviewer_feedback_episodes"],
        )
        rq3_boot = bootstrap_metric_dict(
            problem_ids=[row.problem_id for row in rows],
            metric_fn=lambda sampled_ids, *, mode=mode: compute_rq3_metrics(
                flatten_grouped(review_bundle["review_episodes_by_problem"], sampled_ids),
                flatten_grouped(review_bundle["reviewer_feedback_episodes_by_problem"], sampled_ids),
            ),
            metric_names=RQ3_METRIC_NAMES,
            n_bootstrap=int(args.n_bootstrap),
            rng=rng,
        )
        rq3_summary_rows.extend(
            summarize_bootstrap_dict(
                family="rq3_problem_clustered",
                protocol=mode,
                point_metrics=rq3_point,
                bootstrap_metrics=rq3_boot,
            )
        )

        strict_point = compute_strict_metrics(review_bundle["strict_records"])
        strict_boot = bootstrap_metric_dict(
            problem_ids=[row.problem_id for row in rows],
            metric_fn=lambda sampled_ids, *, mode=mode: compute_strict_metrics(
                flatten_grouped(review_bundle["strict_records_by_problem"], sampled_ids)
            ),
            metric_names=STRICT_METRIC_NAMES,
            n_bootstrap=int(args.n_bootstrap),
            rng=rng,
        )
        strict_summary_rows.extend(
            summarize_bootstrap_dict(
                family="strict_problem_clustered",
                protocol=mode,
                point_metrics=strict_point,
                bootstrap_metrics=strict_boot,
            )
        )

    if len(modes) >= 2:
        left_mode = modes[0]
        right_mode = modes[1]
        delta_name = f"{right_mode}_minus_{left_mode}"
        common_problem_ids = sorted(
            set(row.problem_id for row in problem_rows_by_mode[left_mode])
            & set(row.problem_id for row in problem_rows_by_mode[right_mode])
        )

        paired_rq1_point = paired_delta_metrics(
            common_problem_ids,
            left_fn=lambda ids: compute_rq1_metrics(
                resample_problem_rows(problem_rows_by_mode[left_mode], ids)
            ),
            right_fn=lambda ids: compute_rq1_metrics(
                resample_problem_rows(problem_rows_by_mode[right_mode], ids)
            ),
            metric_names=RQ1_METRIC_NAMES,
        )
        paired_rq1_boot = bootstrap_metric_dict(
            problem_ids=common_problem_ids,
            metric_fn=lambda sampled_ids: paired_delta_metrics(
                sampled_ids,
                left_fn=lambda ids: compute_rq1_metrics(
                    resample_problem_rows(problem_rows_by_mode[left_mode], ids)
                ),
                right_fn=lambda ids: compute_rq1_metrics(
                    resample_problem_rows(problem_rows_by_mode[right_mode], ids)
                ),
                metric_names=RQ1_METRIC_NAMES,
            ),
            metric_names=RQ1_METRIC_NAMES,
            n_bootstrap=int(args.n_bootstrap),
            rng=rng,
        )
        paired_rq1_rows.extend(
            summarize_bootstrap_dict(
                family="rq1_problem_level_delta",
                protocol=delta_name,
                point_metrics=paired_rq1_point,
                bootstrap_metrics=paired_rq1_boot,
            )
        )

        left_bundle = review_bundles_by_mode[left_mode]
        right_bundle = review_bundles_by_mode[right_mode]

        paired_rq3_point = paired_delta_metrics(
            common_problem_ids,
            left_fn=lambda ids: compute_rq3_metrics(
                flatten_grouped(left_bundle["review_episodes_by_problem"], ids),
                flatten_grouped(left_bundle["reviewer_feedback_episodes_by_problem"], ids),
            ),
            right_fn=lambda ids: compute_rq3_metrics(
                flatten_grouped(right_bundle["review_episodes_by_problem"], ids),
                flatten_grouped(right_bundle["reviewer_feedback_episodes_by_problem"], ids),
            ),
            metric_names=RQ3_METRIC_NAMES,
        )
        paired_rq3_boot = bootstrap_metric_dict(
            problem_ids=common_problem_ids,
            metric_fn=lambda sampled_ids: paired_delta_metrics(
                sampled_ids,
                left_fn=lambda ids: compute_rq3_metrics(
                    flatten_grouped(left_bundle["review_episodes_by_problem"], ids),
                    flatten_grouped(left_bundle["reviewer_feedback_episodes_by_problem"], ids),
                ),
                right_fn=lambda ids: compute_rq3_metrics(
                    flatten_grouped(right_bundle["review_episodes_by_problem"], ids),
                    flatten_grouped(right_bundle["reviewer_feedback_episodes_by_problem"], ids),
                ),
                metric_names=RQ3_METRIC_NAMES,
            ),
            metric_names=RQ3_METRIC_NAMES,
            n_bootstrap=int(args.n_bootstrap),
            rng=rng,
        )
        paired_rq3_rows.extend(
            summarize_bootstrap_dict(
                family="rq3_problem_clustered_delta",
                protocol=delta_name,
                point_metrics=paired_rq3_point,
                bootstrap_metrics=paired_rq3_boot,
            )
        )

        paired_strict_point = paired_delta_metrics(
            common_problem_ids,
            left_fn=lambda ids: compute_strict_metrics(
                flatten_grouped(left_bundle["strict_records_by_problem"], ids)
            ),
            right_fn=lambda ids: compute_strict_metrics(
                flatten_grouped(right_bundle["strict_records_by_problem"], ids)
            ),
            metric_names=STRICT_METRIC_NAMES,
        )
        paired_strict_boot = bootstrap_metric_dict(
            problem_ids=common_problem_ids,
            metric_fn=lambda sampled_ids: paired_delta_metrics(
                sampled_ids,
                left_fn=lambda ids: compute_strict_metrics(
                    flatten_grouped(left_bundle["strict_records_by_problem"], ids)
                ),
                right_fn=lambda ids: compute_strict_metrics(
                    flatten_grouped(right_bundle["strict_records_by_problem"], ids)
                ),
                metric_names=STRICT_METRIC_NAMES,
            ),
            metric_names=STRICT_METRIC_NAMES,
            n_bootstrap=int(args.n_bootstrap),
            rng=rng,
        )
        paired_strict_rows.extend(
            summarize_bootstrap_dict(
                family="strict_problem_clustered_delta",
                protocol=delta_name,
                point_metrics=paired_strict_point,
                bootstrap_metrics=paired_strict_boot,
            )
        )

    write_csv(output_dir / "coverage_summary.csv", coverage_rows)
    write_csv(output_dir / "rq1_problem_level_bootstrap_summary.csv", rq1_summary_rows)
    write_csv(output_dir / "rq3_problem_clustered_bootstrap_summary.csv", rq3_summary_rows)
    write_csv(output_dir / "strict_problem_clustered_bootstrap_summary.csv", strict_summary_rows)
    if paired_rq1_rows:
        write_csv(output_dir / "rq1_problem_level_delta_bootstrap_summary.csv", paired_rq1_rows)
    if paired_rq3_rows:
        write_csv(output_dir / "rq3_problem_clustered_delta_bootstrap_summary.csv", paired_rq3_rows)
    if paired_strict_rows:
        write_csv(output_dir / "strict_problem_clustered_delta_bootstrap_summary.csv", paired_strict_rows)

    metadata = {
        "result_root": str(result_root),
        "modes": modes,
        "tiers": tiers,
        "n_bootstrap": int(args.n_bootstrap),
        "seed": int(args.seed),
        "output_dir": str(output_dir),
    }
    (output_dir / "bootstrap_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote bootstrap summaries to {output_dir}")


def discover_tiers(result_root: Path) -> list[str]:
    tiers = [path.name for path in result_root.glob("tier*") if path.is_dir()]
    return sorted(tiers)


def load_problem_rows_and_mapping(
    result_root: Path,
    tiers: list[str],
    mode: str,
) -> tuple[list[ProblemRow], dict[tuple[str, int], str]]:
    rows: list[ProblemRow] = []
    mapping: dict[tuple[str, int], str] = {}
    for tier in tiers:
        table_path = result_root / tier / mode / "report" / "tier_mode_problem_table.csv"
        if not table_path.exists():
            continue
        with table_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for raw in reader:
                global_example_idx = int(raw["global_example_idx"])
                source_local_example_idx = int(raw["source_local_example_idx"])
                problem_id = f"{tier}:{global_example_idx}"
                rows.append(
                    ProblemRow(
                        problem_id=problem_id,
                        tier=tier,
                        protocol=str(raw["protocol"]).strip(),
                        global_example_idx=global_example_idx,
                        final_passed=parse_bool(raw["final_passed"]),
                        first_passed=parse_bool(raw["first_passed"]),
                        total_tokens=float(raw["total_tokens"]),
                        model_calls=float(raw["model_calls"]),
                    )
                )
                trace_key = (
                    normalize_trace_suffix(str(raw["source_trace_path"])),
                    source_local_example_idx,
                )
                mapping[trace_key] = problem_id
    return rows, mapping


def build_merged_candidate_cache(
    *,
    result_root: Path,
    mode: str,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_path = output_dir / f"{mode}_candidate_label_cache.jsonl"
    merged_rows: dict[tuple[str, str, str, str], dict[str, object]] = {}
    for cache_path in sorted(result_root.glob(f"tier*/{mode}/**/candidate_label_cache.jsonl")):
        with cache_path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
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
                    key = (
                        str(row["signature"]),
                        str(row["problem"]).strip(),
                        str(row["gold_answer"]).strip(),
                        str(row["candidate_answer"]).strip(),
                    )
                except Exception:
                    continue
                merged_rows.setdefault(key, row)

    with merged_path.open("w", encoding="utf-8") as handle:
        for key in sorted(merged_rows):
            handle.write(json.dumps(merged_rows[key], ensure_ascii=False) + "\n")
    return merged_path


def load_review_bundle(
    *,
    trace_files: list[Path],
    trace_key_to_problem_id: dict[tuple[str, int], str],
    candidate_label_cache_path: Path,
) -> dict[str, object]:
    run_config_path = _discover_run_config_path(trace_files[0].parent) if trace_files else None
    dataset = load_collaboration_decomposition_dataset(
        trace_files,
        run_config_path=str(run_config_path) if run_config_path is not None else None,
        reuse_runtime_final_labels=True,
        cache_candidate_labels=True,
        candidate_label_cache_path=candidate_label_cache_path,
        reviewer_feedback_labeler_mode="heuristic",
        cache_feedback_labels=False,
    )
    strict_records = build_strict_coupling_episode_records(
        dataset.review_episodes,
        dataset.reviewer_feedback_episodes,
    )

    review_grouped, review_mapped, review_unmapped = group_by_problem_id(
        dataset.review_episodes,
        trace_key_to_problem_id,
    )
    feedback_grouped, feedback_mapped, feedback_unmapped = group_by_problem_id(
        dataset.reviewer_feedback_episodes,
        trace_key_to_problem_id,
    )
    strict_grouped, strict_mapped, strict_unmapped = group_by_problem_id(
        strict_records,
        trace_key_to_problem_id,
    )

    accounting = dataset.analysis_accounting or {}
    candidate_accounting = accounting.get("candidate_correctness_evaluator", {}) or {}
    return {
        "review_episodes": dataset.review_episodes,
        "reviewer_feedback_episodes": dataset.reviewer_feedback_episodes,
        "strict_records": strict_records,
        "review_episodes_by_problem": review_grouped,
        "reviewer_feedback_episodes_by_problem": feedback_grouped,
        "strict_records_by_problem": strict_grouped,
        "mapped_review_episodes": review_mapped,
        "unmapped_review_episodes": review_unmapped,
        "mapped_feedback_episodes": feedback_mapped,
        "unmapped_feedback_episodes": feedback_unmapped,
        "mapped_strict_records": strict_mapped,
        "unmapped_strict_records": strict_unmapped,
        "candidate_cache_entries": sum(1 for _ in candidate_label_cache_path.open("r", encoding="utf-8")),
        "analysis_runtime_label_hits": candidate_accounting.get("runtime_label_hits", 0),
        "analysis_candidate_cache_hits": candidate_accounting.get("cache_hits", 0),
        "analysis_candidate_cache_writes": candidate_accounting.get("cache_writes", 0),
        "analysis_feedback_labeler_mode": dataset.reviewer_feedback_labeler_mode,
    }


def group_by_problem_id(
    records: list[object],
    trace_key_to_problem_id: dict[tuple[str, int], str],
) -> tuple[dict[str, list[object]], int, int]:
    grouped: dict[str, list[object]] = defaultdict(list)
    mapped = 0
    unmapped = 0
    for record in records:
        trace_path = str(getattr(record, "trace_path"))
        example_idx = int(getattr(record, "example_idx"))
        key = (normalize_trace_suffix(trace_path), example_idx)
        problem_id = trace_key_to_problem_id.get(key)
        if problem_id is None:
            unmapped += 1
            continue
        grouped[problem_id].append(record)
        mapped += 1
    return dict(grouped), mapped, unmapped


def flatten_grouped(grouped: dict[str, list[object]], sampled_problem_ids: list[str]) -> list[object]:
    flat: list[object] = []
    for problem_id in sampled_problem_ids:
        flat.extend(grouped.get(problem_id, ()))
    return flat


def resample_problem_rows(rows: list[ProblemRow], sampled_problem_ids: list[str]) -> list[ProblemRow]:
    row_by_problem_id = {row.problem_id: row for row in rows}
    return [row_by_problem_id[problem_id] for problem_id in sampled_problem_ids if problem_id in row_by_problem_id]


def compute_rq1_metrics(rows: list[ProblemRow]) -> dict[str, float | None]:
    if not rows:
        return {name: None for name in RQ1_METRIC_NAMES}
    return {
        "FinalPassRate": fmean(1.0 if row.final_passed else 0.0 for row in rows),
        "PassAt1Rate": fmean(1.0 if row.first_passed else 0.0 for row in rows),
        "AvgTotalTokens": fmean(row.total_tokens for row in rows),
        "AvgModelCalls": fmean(row.model_calls for row in rows),
    }


def compute_rq3_metrics(
    review_episodes: list[object],
    reviewer_feedback_episodes: list[object],
) -> dict[str, float | None]:
    reviewer_records = summarize_reviewer_conditioned_decomposition(review_episodes)
    feedback_records = summarize_reviewer_feedback_incorporation(reviewer_feedback_episodes)
    metric_map = metrics_to_dict(reviewer_records + feedback_records)
    return {name: metric_map.get(name) for name in RQ3_METRIC_NAMES}


def compute_strict_metrics(strict_records: list[object]) -> dict[str, float | None]:
    metric_map = metrics_to_dict(summarize_strict_coupling_rate(strict_records))
    return {name: metric_map.get(name) for name in STRICT_METRIC_NAMES}


def paired_delta_metrics(
    sampled_problem_ids: list[str],
    *,
    left_fn: Callable[[list[str]], dict[str, float | None]],
    right_fn: Callable[[list[str]], dict[str, float | None]],
    metric_names: tuple[str, ...],
) -> dict[str, float | None]:
    left = left_fn(sampled_problem_ids)
    right = right_fn(sampled_problem_ids)
    deltas: dict[str, float | None] = {}
    for metric_name in metric_names:
        left_value = left.get(metric_name)
        right_value = right.get(metric_name)
        deltas[metric_name] = (
            None if left_value is None or right_value is None else right_value - left_value
        )
    return deltas


def bootstrap_metric_dict(
    *,
    problem_ids: list[str],
    metric_fn: Callable[[list[str]], dict[str, float | None]],
    metric_names: tuple[str, ...],
    n_bootstrap: int,
    rng: random.Random,
) -> dict[str, list[float | None]]:
    replicates: dict[str, list[float | None]] = {name: [] for name in metric_names}
    if not problem_ids:
        return replicates
    for _ in range(n_bootstrap):
        sampled = [rng.choice(problem_ids) for _ in range(len(problem_ids))]
        metric_values = metric_fn(sampled)
        for metric_name in metric_names:
            replicates[metric_name].append(metric_values.get(metric_name))
    return replicates


def summarize_bootstrap_dict(
    *,
    family: str,
    protocol: str,
    point_metrics: dict[str, float | None],
    bootstrap_metrics: dict[str, list[float | None]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for metric_name, point_value in point_metrics.items():
        values = [value for value in bootstrap_metrics.get(metric_name, []) if value is not None]
        lo = percentile(values, 2.5) if values else None
        hi = percentile(values, 97.5) if values else None
        rows.append(
            {
                "family": family,
                "protocol": protocol,
                "metric": metric_name,
                "point_estimate": point_value,
                "ci95_lower": lo,
                "ci95_upper": hi,
                "bootstrap_replicates": len(values),
            }
        )
    return rows


def metrics_to_dict(records: list[MetricRecord]) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {}
    for record in records:
        key = f"{record.section}__{record.metric}"
        value = record.value
        if isinstance(value, (int, float)) or value is None:
            metrics[key] = value
    return metrics


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    rank = (len(ordered) - 1) * (q / 100.0)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() == "true"


def normalize_trace_suffix(path_str: str) -> str:
    parts = Path(path_str).parts
    if "paper_experiment" in parts:
        idx = parts.index("paper_experiment")
        return "/".join(parts[idx + 1 :])
    return str(Path(path_str).name)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
