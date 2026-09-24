#!/usr/bin/env python3
"""Export compact per-problem tables from saved paper_experiment results.

This script produces reviewer-friendly per-problem CSVs that complement the
paper-facing aggregate tables without requiring raw trace inspection. It reads
the saved `tier_mode_problem_table.csv` outputs for each protocol and, for
review-based modes, reconstructs compact per-problem review summaries from the
same trace-backed analysis pipeline used by the paper.
"""

from __future__ import annotations

import argparse
import os
import csv
import json
import sys
from pathlib import Path

# RELEASE PATCH: importable from a fresh clone with no install.
# Only src/ is added. src/precise_uncoupled must NEVER go on sys.path -- it holds
# subpackages named `io` and `scripts` that would shadow the standard library.
SRC_ROOT = Path(__file__).resolve().parents[2]      # src
REPO_ROOT = Path(__file__).resolve().parents[3]     # repository root
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from precise_uncoupled.scripts.bootstrap_problem_clustered_metrics import (  # noqa: E402
    build_merged_candidate_cache,
    discover_tiers,
    load_problem_rows_and_mapping,
    load_review_bundle,
)


# RELEASE PATCH (NeurIPS26_Precise-but-Uncoupled).
# Upstream this defaulted to an author-private absolute path pointing at
# `results/paper_experiment`, a directory that was later renamed and no longer
# exists. The released default is repository-relative and env-overridable:
#   PU_TRACE_ROOT  -> root of the downloaded per-tier trace tree
#                     (layout: tierNN/{baseline_llm,single_agent,per,broadcast}/)
# Pass --result_root to override on the command line. See docs/DATA.md.
DEFAULT_RESULT_ROOT = Path(
    os.environ.get(
        "PU_TRACE_ROOT",
        str(REPO_ROOT / "data" / "traces" / "gpt_oss_120b"),
    )
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export compact per-problem outcomes and review summaries from "
            "saved paper_experiment results."
        )
    )
    parser.add_argument(
        "--result_root",
        type=str,
        default=str(DEFAULT_RESULT_ROOT),
        help="Root directory containing tierXX/mode result folders.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["baseline_llm", "single_agent", "per", "broadcast"],
        help="Modes to export in the outcome table.",
    )
    parser.add_argument(
        "--review_modes",
        nargs="+",
        default=["per", "broadcast"],
        help="Modes to export in the per-problem review summary table.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=(
            "Output directory. Defaults to "
            "<result_root>/report/per_problem."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_root = Path(args.result_root).expanduser().resolve()
    if not result_root.exists():
        raise FileNotFoundError(f"Missing result root: {result_root}")

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else result_root / "report" / "per_problem"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    tiers = discover_tiers(result_root)
    if not tiers:
        raise FileNotFoundError(f"No tierXX folders found under: {result_root}")

    outcome_rows: list[dict[str, object]] = []
    metadata: dict[str, object] = {
        "result_root": str(result_root),
        "tiers": tiers,
        "modes": list(args.modes),
        "review_modes": list(args.review_modes),
    }

    for mode in args.modes:
        outcome_rows.extend(load_outcome_rows(result_root, tiers, mode))

    outcome_rows.sort(
        key=lambda row: (
            str(row["protocol"]),
            str(row["tier"]),
            int(row["global_example_idx"]),
        )
    )
    write_csv(output_dir / "per_problem_outcomes.csv", outcome_rows)

    review_rows: list[dict[str, object]] = []
    for mode in args.review_modes:
        review_rows.extend(load_review_rows(result_root, tiers, mode, output_dir))

    review_rows.sort(
        key=lambda row: (
            str(row["protocol"]),
            str(row["tier"]),
            int(row["global_example_idx"]),
        )
    )
    write_csv(output_dir / "per_problem_review_summary.csv", review_rows)

    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote per-problem tables to {output_dir}")


def load_outcome_rows(
    result_root: Path,
    tiers: list[str],
    mode: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for tier in tiers:
        table_path = result_root / tier / mode / "report" / "tier_mode_problem_table.csv"
        if not table_path.exists():
            continue
        with table_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for raw in reader:
                global_example_idx = int(raw["global_example_idx"])
                rows.append(
                    {
                        "problem_id": make_problem_id(tier, global_example_idx),
                        "tier": tier,
                        "global_example_idx": global_example_idx,
                        "protocol": str(raw["protocol"]).strip(),
                        "source_dataset": str(raw.get("source_dataset", "")).strip(),
                        "domain": str(raw.get("domain", "")).strip(),
                        "final_passed": parse_bool(raw["final_passed"]),
                        "first_passed": parse_bool(raw["first_passed"]),
                        "first_success_attempt": parse_optional_int(
                            raw.get("first_success_attempt", "")
                        ),
                        "correction_loops": int(raw.get("correction_loops", "0") or 0),
                        "hints_used": int(raw.get("hints_used", "0") or 0),
                        "reflective_rounds": int(raw.get("reflective_rounds", "0") or 0),
                        "reasoning_turns": int(raw.get("reasoning_turns", "0") or 0),
                        "evaluator_submission_count": int(
                            raw.get("evaluator_submission_count", "0") or 0
                        ),
                        "prompt_tokens": int(float(raw.get("prompt_tokens", "0") or 0)),
                        "completion_tokens": int(
                            float(raw.get("completion_tokens", "0") or 0)
                        ),
                        "total_tokens": int(float(raw.get("total_tokens", "0") or 0)),
                        "model_calls": int(float(raw.get("model_calls", "0") or 0)),
                        "wall_time_seconds": float(raw.get("wall_time_seconds", "0") or 0),
                    }
                )
    return rows


def load_review_rows(
    result_root: Path,
    tiers: list[str],
    mode: str,
    output_dir: Path,
) -> list[dict[str, object]]:
    problem_rows, trace_key_to_problem_id = load_problem_rows_and_mapping(
        result_root,
        tiers,
        mode,
    )
    row_by_problem_id = {row.problem_id: row for row in problem_rows}
    if not row_by_problem_id:
        return []

    trace_files = sorted(result_root.glob(f"tier*/{mode}/**/*.trace.txt"))
    if not trace_files:
        return []

    merged_cache_path = build_merged_candidate_cache(
        result_root=result_root,
        mode=mode,
        output_dir=output_dir / "_merged_caches",
    )
    review_bundle = load_review_bundle(
        trace_files=trace_files,
        trace_key_to_problem_id=trace_key_to_problem_id,
        candidate_label_cache_path=merged_cache_path,
    )

    feedback_by_problem = review_bundle["reviewer_feedback_episodes_by_problem"]
    review_by_problem = review_bundle["review_episodes_by_problem"]
    strict_by_problem = review_bundle["strict_records_by_problem"]

    rows: list[dict[str, object]] = []
    for problem_id in sorted(row_by_problem_id):
        base = row_by_problem_id[problem_id]
        feedback_episodes = feedback_by_problem.get(problem_id, [])
        review_episodes = review_by_problem.get(problem_id, [])
        strict_records = strict_by_problem.get(problem_id, [])

        useful_revise = [
            episode
            for episode in feedback_episodes
            if episode.action == "revise" and episode.pre_correct is False
        ]
        misleading_revise = [
            episode
            for episode in feedback_episodes
            if episode.action == "revise" and episode.pre_correct is True
        ]
        useful_incorporated = sum(
            1 for episode in useful_revise if episode.incorporated_feedback is True
        )
        misleading_incorporated = sum(
            1 for episode in misleading_revise if episode.incorporated_feedback is True
        )
        useful_repaired = sum(
            1 for episode in useful_revise if episode.post_correct is True
        )
        misleading_harmed = sum(
            1 for episode in misleading_revise if episode.post_correct is False
        )

        strict_useful = [
            record
            for record in strict_records
            if record.action == "revise" and record.pre_correct is False
        ]
        strict_changed = sum(
            1 for record in strict_useful if record.strict_answer_changed is True
        )
        strict_changed_and_repaired = sum(
            1
            for record in strict_useful
            if record.strict_answer_changed is True and record.post_correct is True
        )
        strict_unchanged = sum(
            1 for record in strict_useful if record.outcome_bucket == "unchanged"
        )

        rows.append(
            {
                "problem_id": problem_id,
                "tier": base.tier,
                "global_example_idx": base.global_example_idx,
                "protocol": base.protocol,
                "review_episodes": len(review_episodes),
                "review_revise_episodes": sum(
                    1 for episode in review_episodes if episode.action == "revise"
                ),
                "review_agree_episodes": sum(
                    1 for episode in review_episodes if episode.action == "agree"
                ),
                "useful_revise_episodes": len(useful_revise),
                "misleading_revise_episodes": len(misleading_revise),
                "useful_revise_incorporated": useful_incorporated,
                "misleading_revise_incorporated": misleading_incorporated,
                "useful_revise_repaired": useful_repaired,
                "misleading_revise_harmed": misleading_harmed,
                "strict_useful_revise_episodes": len(strict_useful),
                "strict_useful_revise_changed": strict_changed,
                "strict_useful_revise_changed_and_repaired": strict_changed_and_repaired,
                "strict_useful_revise_unchanged": strict_unchanged,
            }
        )
    return rows


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


def make_problem_id(tier: str, global_example_idx: int) -> str:
    return f"{tier}:{global_example_idx}"


def parse_bool(raw: str) -> bool:
    value = str(raw).strip().lower()
    if value == "true":
        return True
    if value == "false":
        return False
    raise ValueError(f"Expected boolean string, got: {raw!r}")


def parse_optional_int(raw: str) -> int | None:
    value = str(raw).strip()
    if not value:
        return None
    return int(value)


if __name__ == "__main__":
    main()
