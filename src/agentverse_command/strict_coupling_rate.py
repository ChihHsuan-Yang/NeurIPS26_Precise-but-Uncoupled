"""Strict useful-coupling CLI for paper robustness checks."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Any

from agentverse.logging import get_logger
from agentverse.metrics.api_accounting import write_api_accounting_sidecar
from agentverse.metrics.collaboration_decomposition import (
    load_collaboration_decomposition_dataset,
)
from agentverse.metrics.collaboration_decomposition.strict_coupling import (
    build_audit_sample,
    build_strict_coupling_episode_records,
    render_strict_coupling_text,
    summarize_strict_coupling_by_protocol,
    summarize_strict_coupling_rate,
    write_metric_csv,
    write_protocol_summary_csv,
    write_strict_coupling_episode_csv,
)
from agentverse.metrics.main_paper.inputs import _discover_run_config_path
from agentverse_command.post_eval_common import (
    assert_eligible_protocols,
    load_post_eval_section,
    protocol_set_from_examples,
    resolve_output_dir,
    resolve_primary_input,
    resolve_trace_files,
)


logger = get_logger()


def _build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description=(
            "Analyze saved `.trace.txt` files for a strict useful-review "
            "coupling-rate definition, equivalence-aware sensitivity check, "
            "and audit-ready exports."
        )
    )
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--run_config_path", type=str, default=None)
    parser.add_argument("--evaluator_type", type=str, default=None)
    parser.add_argument("--candidate_label_cache_path", type=str, default=None)
    parser.add_argument(
        "--reviewer_feedback_labeler",
        type=str,
        default=None,
        choices=["heuristic", "hybrid", "llm"],
    )
    parser.add_argument("--feedback_llm_model", type=str, default=None)
    parser.add_argument("--feedback_label_cache_path", type=str, default=None)
    parser.add_argument("--audit_sample_size", type=int, default=120)
    parser.add_argument("--audit_seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true", default=None)
    return parser


def cli_main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    input_path = resolve_primary_input(args.input_path)
    trace_files = resolve_trace_files(input_path)
    if not trace_files:
        raise FileNotFoundError(f"No `.trace.txt` files found under: {input_path}")

    resolved = _resolve_args(args, base_dir=trace_files[0].parent)
    out_dir = resolve_output_dir(
        input_path=input_path,
        output_dir=args.output_dir,
        default_dirname="strict-coupling-rate",
        overwrite=bool(args.overwrite),
    )

    dataset = load_collaboration_decomposition_dataset(
        trace_files,
        correctness_evaluator_type=str(resolved["evaluator_type"]),
        numeric_tolerance=str(resolved["numeric_tolerance"]),
        data_name=str(resolved["data_name"]),
        omni_judge_model_path=resolved.get("omni_judge_model_path"),
        omni_judge_max_new_tokens=int(resolved["omni_judge_max_new_tokens"]),
        omni_judge_device=str(resolved["omni_judge_device"]),
        omni_judge_dtype=str(resolved["omni_judge_dtype"]),
        run_config_path=resolved.get("run_config_path"),
        reuse_runtime_final_labels=bool(resolved["reuse_runtime_final_labels"]),
        cache_candidate_labels=bool(resolved["cache_candidate_labels"]),
        candidate_label_cache_path=resolved.get("candidate_label_cache_path"),
        reviewer_feedback_labeler_mode=str(resolved["reviewer_feedback_labeler"]),
        reviewer_feedback_llm_model=str(resolved["reviewer_feedback_llm_model"]),
        reviewer_feedback_temperature=float(resolved["reviewer_feedback_temperature"]),
        reviewer_feedback_max_tokens=int(resolved["reviewer_feedback_max_tokens"]),
        cache_feedback_labels=bool(resolved["cache_feedback_labels"]),
        feedback_label_cache_path=resolved.get("feedback_label_cache_path"),
    )

    assert_eligible_protocols(
        cli_name="agentverse-strict-coupling-rate",
        protocols=protocol_set_from_examples(dataset.examples),
        allowed={"per", "broadcast"},
    )

    strict_records = build_strict_coupling_episode_records(
        dataset.review_episodes,
        dataset.reviewer_feedback_episodes,
        numeric_tolerance=str(resolved["numeric_tolerance"]),
    )
    metrics = summarize_strict_coupling_rate(strict_records)
    protocol_rows = summarize_strict_coupling_by_protocol(strict_records)
    audit_sample = build_audit_sample(
        strict_records,
        sample_size=max(int(args.audit_sample_size or 0), 0),
        seed=int(args.audit_seed or 0),
    )

    metrics_csv = out_dir / "strict_coupling_rate_metrics.csv"
    metrics_txt = out_dir / "strict_coupling_rate_metrics.txt"
    protocol_csv = out_dir / "strict_coupling_rate_protocol_summary.csv"
    episodes_csv = out_dir / "strict_coupling_rate_episode_details.csv"
    audit_csv = out_dir / "strict_coupling_rate_audit_sample.csv"

    write_metric_csv(metrics_csv, metrics)
    write_protocol_summary_csv(protocol_csv, protocol_rows)
    metrics_txt.write_text(
        render_strict_coupling_text(
            metrics,
            protocol_rows,
            strict_records,
            audit_sample_size=len(audit_sample),
        ),
        encoding="utf-8",
    )
    write_strict_coupling_episode_csv(episodes_csv, strict_records)
    write_strict_coupling_episode_csv(audit_csv, audit_sample)

    accounting_path = write_api_accounting_sidecar(
        out_dir,
        {
            "kind": "analysis_api_accounting",
            "cli_name": "agentverse-strict-coupling-rate",
            "trace_files": dataset.trace_files,
            "example_count": len(dataset.examples),
            **dataset.analysis_accounting,
        },
    )

    logger.info(
        "[STRICT COUPLING RATE] "
        f"examples={len(dataset.examples)} "
        f"correctness_evaluator={dataset.correctness_evaluator_type} "
        f"reviewer_feedback_labeler={dataset.reviewer_feedback_labeler_mode} "
        f"output_dir={out_dir}"
    )
    logger.info(
        "[STRICT COUPLING RATE] Wrote metrics CSV/TXT, protocol summary CSV, "
        "per-episode CSV, audit sample CSV, and "
        f"{accounting_path.name}."
    )


def _resolve_args(args, *, base_dir: Path) -> dict[str, Any]:
    section, _config_path = load_post_eval_section(
        config_path=args.config_path,
        section_names=("trace_collaboration_decomposition", "collaboration_decomposition"),
    )
    resolved: dict[str, Any] = {
        "evaluator_type": "llm",
        "numeric_tolerance": "0",
        "data_name": "omni-math",
        "reuse_runtime_final_labels": True,
        "cache_candidate_labels": True,
        "candidate_label_cache_path": None,
        "omni_judge_model_path": None,
        "omni_judge_max_new_tokens": 300,
        "omni_judge_device": "auto",
        "omni_judge_dtype": "auto",
        "run_config_path": None,
        "reviewer_feedback_labeler": "heuristic",
        "reviewer_feedback_llm_model": "openai/gpt-oss-120b",
        "reviewer_feedback_temperature": 0.0,
        "reviewer_feedback_max_tokens": 400,
        "cache_feedback_labels": True,
        "feedback_label_cache_path": None,
    }

    for key in ("numeric_tolerance", "data_name"):
        if key in section:
            resolved[key] = section[key]

    evaluator = section.get("evaluator") or section.get("correctness") or {}
    if isinstance(evaluator, dict):
        evaluator_type = evaluator.get("type", evaluator.get("correctness_source"))
        if evaluator_type is not None:
            resolved["evaluator_type"] = evaluator_type
        for key in (
            "reuse_runtime_final_labels",
            "cache_candidate_labels",
            "candidate_label_cache_path",
            "omni_judge_model_path",
            "omni_judge_max_new_tokens",
            "omni_judge_device",
            "omni_judge_dtype",
            "run_config_path",
        ):
            if key in evaluator:
                resolved[key] = evaluator[key]

    reviewer_feedback = section.get("reviewer_feedback") or {}
    if isinstance(reviewer_feedback, dict):
        if "labeler" in reviewer_feedback:
            resolved["reviewer_feedback_labeler"] = reviewer_feedback["labeler"]
        if "llm_model" in reviewer_feedback:
            resolved["reviewer_feedback_llm_model"] = reviewer_feedback["llm_model"]
        if "model" in reviewer_feedback:
            resolved["reviewer_feedback_llm_model"] = reviewer_feedback["model"]
        if "temperature" in reviewer_feedback:
            resolved["reviewer_feedback_temperature"] = reviewer_feedback["temperature"]
        if "max_tokens" in reviewer_feedback:
            resolved["reviewer_feedback_max_tokens"] = reviewer_feedback["max_tokens"]
        if "cache_labels" in reviewer_feedback:
            resolved["cache_feedback_labels"] = reviewer_feedback["cache_labels"]
        if "cache_path" in reviewer_feedback:
            resolved["feedback_label_cache_path"] = reviewer_feedback["cache_path"]

    if args.evaluator_type is not None:
        resolved["evaluator_type"] = args.evaluator_type
    if args.candidate_label_cache_path is not None:
        resolved["candidate_label_cache_path"] = args.candidate_label_cache_path
    if args.run_config_path is not None:
        resolved["run_config_path"] = args.run_config_path
    if args.reviewer_feedback_labeler is not None:
        resolved["reviewer_feedback_labeler"] = args.reviewer_feedback_labeler
    if args.feedback_llm_model is not None:
        resolved["reviewer_feedback_llm_model"] = args.feedback_llm_model
    if args.feedback_label_cache_path is not None:
        resolved["feedback_label_cache_path"] = args.feedback_label_cache_path

    if not resolved.get("run_config_path"):
        discovered = _discover_run_config_path(base_dir)
        if discovered is not None:
            resolved["run_config_path"] = str(discovered)
    return resolved


if __name__ == "__main__":
    cli_main()
