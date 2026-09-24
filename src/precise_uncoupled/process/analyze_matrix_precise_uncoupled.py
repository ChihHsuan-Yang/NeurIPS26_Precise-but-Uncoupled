#!/usr/bin/env python3
"""Audit "precise but uncoupled" across the release-v2 2x5 matrix.

This analysis deliberately separates three quantities:

1. explicit-decision precision: P(the candidate was wrong | reviewer said revise),
2. strict uptake: P(the next actor-authored answer changed | revise on a wrong candidate),
3. verified repair: P(the next labeled actor answer was correct | revise on a wrong candidate).

Uptake and repair use only strict pre-gate actor transitions. Broadcast responses
that occur after a system-selected candidate update are excluded. This prevents
the approval gate from being counted as though it were an immediate actor response.

The script is an audit of saved release-v2 traces; it makes no model calls.
"""

from __future__ import annotations

import argparse
import os
import hashlib
import math
from pathlib import Path

import pandas as pd


DATASET_ORDER = ["omnimath2", "jeebench", "scibench", "labbench", "mascqa"]
DATASET_DISPLAY = {
    "omnimath2": "Omni-MATH",
    "jeebench": "JEEBench",
    "scibench": "SciBench",
    "labbench": "LAB-Bench",
    "mascqa": "MaScQA",
}
ACTOR_ORDER = ["gpt_oss_120b", "gemma_4_31b"]
ACTOR_DISPLAY = {
    "gpt_oss_120b": "GPT-OSS-120B",
    "gemma_4_31b": "Gemma-4-31B",
}
PROTOCOLS = ["per", "broadcast"]


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().eq("true")


def _wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return math.nan, math.nan
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return center - half, center + half


def _rate_record(prefix: str, successes: int, total: int) -> dict[str, float | int]:
    lo, hi = _wilson(successes, total)
    return {
        f"{prefix}_successes": successes,
        f"{prefix}_n": total,
        f"{prefix}_rate": successes / total if total else math.nan,
        f"{prefix}_ci_lo": lo,
        f"{prefix}_ci_hi": hi,
    }


def _fmt(rate: float, successes: int, total: int) -> str:
    if total == 0 or pd.isna(rate):
        return "--"
    return f"{100 * rate:.1f}% ({successes}/{total})"


def analyze(input_csv: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    df = pd.read_csv(input_csv, low_memory=False)
    for column in [
        "matrix_4x2x5",
        "strict_pre_gate_eligible",
        "answer_pair_parsed",
        "answer_changed",
        "before_correct",
        "after_correct",
    ]:
        if column in df:
            df[column] = _as_bool(df[column]) if not df[column].isna().any() else df[column]

    matrix = df[
        _as_bool(df["matrix_4x2x5"])
        & df["protocol"].isin(PROTOCOLS)
        & df["dataset"].isin(DATASET_ORDER)
        & df["actor_family"].isin(ACTOR_ORDER)
    ].copy()

    rows: list[dict[str, object]] = []
    for dataset in DATASET_ORDER:
        for actor in ACTOR_ORDER:
            for protocol in PROTOCOLS:
                cell = matrix[
                    (matrix["dataset"] == dataset)
                    & (matrix["actor_family"] == actor)
                    & (matrix["protocol"] == protocol)
                ]

                # Precision is a reviewer-decision statistic and does not require
                # a subsequent actor response. "Explicit" matters because some
                # structured PER review actions cannot be semantically recovered.
                precision_base = cell[
                    cell["review_action"].eq("revise") & cell["before_correct"].notna()
                ]
                precision_success = int(precision_base["before_correct"].eq(False).sum())

                # Uptake and repair share the strict, pre-gate, actor-authored
                # transition population. Repair has a smaller labeled denominator.
                strict = cell[
                    _as_bool(cell["strict_pre_gate_eligible"])
                    & _as_bool(cell["answer_pair_parsed"])
                    & cell["review_action"].eq("revise")
                    & cell["before_correct"].eq(False)
                ]
                uptake_success = int(strict["answer_changed"].eq(True).sum())
                repair_base = strict[strict["after_correct"].notna()]
                repair_success = int(repair_base["after_correct"].eq(True).sum())

                row: dict[str, object] = {
                    "dataset": dataset,
                    "dataset_display": DATASET_DISPLAY[dataset],
                    "actor_family": actor,
                    "actor_display": ACTOR_DISPLAY[actor],
                    "protocol": protocol,
                }
                row.update(_rate_record("precision", precision_success, len(precision_base)))
                row.update(_rate_record("uptake", uptake_success, len(strict)))
                row.update(_rate_record("repair", repair_success, len(repair_base)))
                row["post_label_coverage"] = len(repair_base) / len(strict) if len(strict) else math.nan
                row["sparse_strict_followup"] = len(strict) < 20
                rows.append(row)

    long = pd.DataFrame(rows)
    expected = len(DATASET_ORDER) * len(ACTOR_ORDER) * len(PROTOCOLS)
    if len(long) != expected:
        raise RuntimeError(f"Expected {expected} protocol rows, found {len(long)}")

    wide = long.pivot(index=["dataset", "dataset_display", "actor_family", "actor_display"],
                      columns="protocol")
    summary = {
        "cells": len(DATASET_ORDER) * len(ACTOR_ORDER),
        "per_precision_above_broadcast": 0,
        "per_uptake_below_broadcast": 0,
        "per_repair_below_broadcast": 0,
        "per_precision_above_own_repair": 0,
        "broadcast_precision_above_own_repair": 0,
        "cells_with_sparse_per_strict_followup": 0,
        "cells_with_sparse_broadcast_strict_followup": 0,
    }
    for _, row in wide.iterrows():
        summary["per_precision_above_broadcast"] += int(
            row[("precision_rate", "per")] > row[("precision_rate", "broadcast")]
        )
        summary["per_uptake_below_broadcast"] += int(
            row[("uptake_rate", "per")] < row[("uptake_rate", "broadcast")]
        )
        summary["per_repair_below_broadcast"] += int(
            row[("repair_rate", "per")] < row[("repair_rate", "broadcast")]
        )
        summary["per_precision_above_own_repair"] += int(
            row[("precision_rate", "per")] > row[("repair_rate", "per")]
        )
        summary["broadcast_precision_above_own_repair"] += int(
            row[("precision_rate", "broadcast")] > row[("repair_rate", "broadcast")]
        )
        summary["cells_with_sparse_per_strict_followup"] += int(
            row[("sparse_strict_followup", "per")]
        )
        summary["cells_with_sparse_broadcast_strict_followup"] += int(
            row[("sparse_strict_followup", "broadcast")]
        )
    return long, summary


_REPO_ROOT = Path(__file__).resolve().parents[3]   # <repo>/src/precise_uncoupled/process -> <repo>


def _display_source(source: Path) -> str:
    """Repo-relative path if the input is inside the repository, else basename."""
    resolved = Path(source).resolve()
    try:
        return str(resolved.relative_to(_REPO_ROOT))
    except ValueError:
        return resolved.name


def write_markdown(long: pd.DataFrame, summary: dict[str, int], source: Path, output: Path) -> None:
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    keyed = long.set_index(["dataset", "actor_family", "protocol"])
    table_rows = []
    for dataset in DATASET_ORDER:
        for actor in ACTOR_ORDER:
            record: list[str] = [DATASET_DISPLAY[dataset], ACTOR_DISPLAY[actor]]
            sparse = []
            for protocol in PROTOCOLS:
                row = keyed.loc[(dataset, actor, protocol)]
                record.extend([
                    _fmt(row.precision_rate, int(row.precision_successes), int(row.precision_n)),
                    _fmt(row.uptake_rate, int(row.uptake_successes), int(row.uptake_n)),
                    _fmt(row.repair_rate, int(row.repair_successes), int(row.repair_n)),
                ])
                if bool(row.sparse_strict_followup):
                    sparse.append(protocol.upper())
            record.append(", ".join(sparse) if sparse else "--")
            table_rows.append(record)

    headers = [
        "Dataset", "Actor", "PER precision", "PER strict uptake", "PER repair",
        "Broadcast precision", "Broadcast strict uptake", "Broadcast repair",
        "Strict follow-up n<20",
    ]
    lines = [
        "# Symmetric 2×5 audit: is the matrix precise but uncoupled?",
        "",
        "## Takeaway",
        "",
        f"- PER reviewer decision precision is higher than Broadcast in "
        f"**{summary['per_precision_above_broadcast']}/{summary['cells']}** dataset×actor cells.",
        f"- Within PER, reviewer precision is higher than verified next-answer repair in "
        f"**{summary['per_precision_above_own_repair']}/{summary['cells']}** cells. "
        "The exception is JEEBench/Gemma-4, where only two strict useful-review "
        "follow-ups are evaluable, so it is not a stable counterexample.",
        f"- Within Broadcast, reviewer precision is higher than verified next-answer repair in "
        f"**{summary['broadcast_precision_above_own_repair']}/{summary['cells']}** cells.",
        "- The old protocol-ordering claim does **not** survive the symmetric transition audit: "
        f"PER has lower strict uptake in {summary['per_uptake_below_broadcast']}/{summary['cells']} cells "
        f"and lower verified repair in {summary['per_repair_below_broadcast']}/{summary['cells']} cells. "
        "The earlier 10/10 ordering counted Broadcast responses after system-selected candidate "
        "updates together with immediate actor responses.",
        "",
        "The defensible cross-setting conclusion is therefore a **detection-to-repair separation**, "
        "not a universal PER-versus-Broadcast uptake ordering. Accurate reviewer decisions often "
        "fail to become verified repairs across both actor families and all five datasets; the size "
        "and protocol ordering of the intermediate uptake rate depend on the transition definition.",
        "",
        "## Per-cell results",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---", "---"] + ["---:"] * 6 + ["---"]) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in table_rows)
    lines.extend([
        "",
        "Rates are shown as percentage (successes/eligible episodes). `Precision` is the fraction "
        "of explicit reviewer `revise` decisions whose pre-review candidate is frozen-labeled wrong. "
        "`Strict uptake` is the fraction of those useful-review episodes for which the immediate "
        "next actor-authored answer changes before any system candidate update. `Repair` is the "
        "fraction of the post-labeled strict episodes whose immediate next actor answer is correct.",
        "",
        "## Interpretation guardrails",
        "",
        "- This is a saved-trace analysis, not a new model run.",
        "- `review_action=revise` is a reviewer-issued decision. It does not independently verify "
        "that every sentence of the critique is semantically correct.",
        "- Some PER structured review actions are not recoverable as explicit `agree` or `revise`; "
        "the precision values therefore use explicit decisions only.",
        "- Uptake and repair use strict pre-gate actor transitions. Post-system-update Broadcast "
        "transitions are intentionally excluded.",
        "- Small denominators, especially JEEBench/Gemma-4 and MaScQA/Gemma-4, should be described "
        "as exploratory rather than definitive.",
        "- These process rates do not isolate the causal effect of the approval gate, prompt, routing, "
        "round count, or compute.",
        "",
        "## Provenance",
        "",
        # RELEASE PATCH (NeurIPS26_Precise-but-Uncoupled): upstream echoed the
        # caller's path verbatim, so a generated report carried whatever absolute
        # path the operator happened to run from. Emit a repository-relative path
        # when the input lies inside the repo, so the report is portable and
        # cannot leak a private directory. The SHA-256 below is the real identity
        # of the input and is unchanged.
        f"- Input: `{_display_source(source)}`",
        f"- Input SHA-256: `{digest}`",
        "- Matrix: release-v2, PER and Broadcast, 2 actor families × 5 datasets",
        "- Evaluator family in the released matrix: `gpt-oss-120b`",
    ])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")


# RELEASE PATCH (NeurIPS26_Precise-but-Uncoupled): upstream these three defaults
# were paths relative to the authors' working directory ("rebuttal-code/outputs/...")
# which do not exist in a fresh clone. They are now repository-relative and
# env-overridable. The analysis logic below is UNCHANGED.
_DEFAULT_INPUT = Path(
    os.environ.get(
        "PU_TRANSITIONS",
        str(_REPO_ROOT / "data" / "derived" / "full_release_symmetric_transitions.csv"),
    )
)
_DEFAULT_OUT = Path(
    os.environ.get("PU_OUTPUT_DIR", str(_REPO_ROOT / "results" / "derived_tables"))
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=_DEFAULT_INPUT)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=_DEFAULT_OUT / "matrix_2x5_precise_uncoupled_strict.csv",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=_DEFAULT_OUT / "matrix_2x5_precise_uncoupled_strict.md",
    )
    args = parser.parse_args()
    long, summary = analyze(args.input)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    long.to_csv(args.output_csv, index=False)
    write_markdown(long, summary, args.input, args.output_md)
    print(summary)
    print(args.output_csv)
    print(args.output_md)


if __name__ == "__main__":
    main()
