"""RQ2 outer-loop recovery CLI."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Any

from agentverse.logging import get_logger
from agentverse.metrics.api_accounting import (
    combine_api_accounting_summaries,
    write_api_accounting_sidecar,
)
from agentverse.metrics.main_paper import (
    extract_system_feedback_episode_bundle,
    load_main_paper_dataset,
    summarize_system_feedback_incorporation,
    summarize_hint_comparison,
    summarize_outer_loop_metrics,
)
from agentverse.metrics.main_paper.reports import (
    render_outer_text,
    render_system_feedback_text,
    write_metric_csv,
)
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
        description="Analyze saved `.trace.txt` files for RQ2 recovery metrics."
    )
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--results_path", type=str, default=None)
    parser.add_argument("--metrics_path", type=str, default=None)
    parser.add_argument("--plain_input_path", type=str, default=None)
    parser.add_argument("--hint_input_path", type=str, default=None)
    parser.add_argument("--plain_results_path", type=str, default=None)
    parser.add_argument("--hint_results_path", type=str, default=None)
    parser.add_argument("--plain_metrics_path", type=str, default=None)
    parser.add_argument("--hint_metrics_path", type=str, default=None)
    parser.add_argument("--hint_lambda", type=float, default=None)
    parser.add_argument("--max_k", type=int, default=None)
    parser.add_argument(
        "--system_feedback_labeler",
        "--evaluator_feedback_labeler",
        type=str,
        default=None,
        dest="system_feedback_labeler",
        choices=["heuristic", "hybrid", "llm"],
    )
    parser.add_argument("--feedback_llm_model", type=str, default=None)
    parser.add_argument("--feedback_label_cache_path", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true", default=None)
    return parser


def cli_main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    input_path = resolve_primary_input(args.input_path)
    trace_files = resolve_trace_files(input_path)
    if not trace_files:
        raise FileNotFoundError(f"No `.trace.txt` files found under: {input_path}")

    resolved = _resolve_args(args)
    out_dir = resolve_output_dir(
        input_path=input_path,
        output_dir=args.output_dir,
        default_dirname="rq2-recovery",
        overwrite=bool(args.overwrite),
    )

    dataset = load_main_paper_dataset(
        trace_files,
        results_path=args.results_path,
        metrics_path=args.metrics_path,
        include_inner_transitions=False,
    )
    assert_eligible_protocols(
        cli_name="agentverse-rq2-recovery",
        protocols=protocol_set_from_examples(dataset.examples),
        allowed={"single_agent", "per", "broadcast"},
    )

    recovery_records, outer_scores = summarize_outer_loop_metrics(
        dataset.problems,
        hint_lambda=float(resolved["hint_lambda"]),
        max_k=(
            int(resolved["max_k"])
            if resolved.get("max_k") is not None
            else None
        ),
    )
    comparison_records = _load_hint_comparison_records(args, resolved)
    if comparison_records:
        recovery_records.extend(comparison_records)

    feedback_episodes, feedback_accounting = extract_system_feedback_episode_bundle(
        dataset.examples,
        labeler_mode=str(resolved["system_feedback_labeler"]),
        llm_model=str(resolved["system_feedback_llm_model"]),
        temperature=float(resolved["system_feedback_temperature"]),
        max_tokens=int(resolved["system_feedback_max_tokens"]),
        cache_feedback_labels=bool(resolved["cache_feedback_labels"]),
        feedback_label_cache_path=resolved.get("feedback_label_cache_path"),
        base_dir=trace_files[0].parent,
    )
    feedback_records = summarize_system_feedback_incorporation(feedback_episodes)

    recovery_csv = out_dir / "rq2_recovery.csv"
    recovery_txt = out_dir / "rq2_recovery.txt"
    feedback_csv = out_dir / "rq2_system_feedback_incorporation.csv"
    feedback_txt = out_dir / "rq2_system_feedback_incorporation.txt"
    legacy_feedback_csv = out_dir / "rq2_feedback_incorporation.csv"
    legacy_feedback_txt = out_dir / "rq2_feedback_incorporation.txt"

    write_metric_csv(recovery_csv, recovery_records)
    recovery_txt.write_text(
        render_outer_text(recovery_records, dataset.problems, outer_scores),
        encoding="utf-8",
    )
    write_metric_csv(feedback_csv, feedback_records)
    feedback_rendered = render_system_feedback_text(feedback_records, feedback_episodes)
    feedback_txt.write_text(feedback_rendered, encoding="utf-8")
    write_metric_csv(legacy_feedback_csv, feedback_records)
    legacy_feedback_txt.write_text(feedback_rendered, encoding="utf-8")
    accounting_path = write_api_accounting_sidecar(
        out_dir,
        {
            "kind": "analysis_api_accounting",
            "cli_name": "agentverse-rq2-recovery",
            "trace_files": dataset.trace_files,
            "example_count": len(dataset.examples),
            **combine_api_accounting_summaries(dataset.analysis_accounting, feedback_accounting),
        },
    )

    logger.info(
        "[RQ2 RECOVERY] "
        f"examples={len(dataset.examples)} "
        f"system_feedback_labeler={resolved['system_feedback_labeler']} "
        f"output_dir={out_dir}"
    )
    logger.info(
        f"[RQ2 RECOVERY] Wrote recovery/system-feedback CSV/TXT files and {accounting_path.name}."
    )


def _resolve_args(args) -> dict[str, Any]:
    section, _config_path = load_post_eval_section(
        config_path=args.config_path,
        section_names=("trace_rq2_recovery", "rq2_recovery", "trace_analysis"),
    )
    resolved: dict[str, Any] = {
        "hint_lambda": 1.0,
        "max_k": None,
        "system_feedback_labeler": "hybrid",
        "system_feedback_llm_model": "openai/gpt-oss-120b",
        "system_feedback_temperature": 0.0,
        "system_feedback_max_tokens": 400,
        "cache_feedback_labels": True,
        "feedback_label_cache_path": None,
    }

    for key in ("hint_lambda", "max_k"):
        if key in section:
            resolved[key] = section[key]

    for feedback_key in ("evaluator_feedback", "system_feedback"):
        feedback_section = section.get(feedback_key) or {}
        if not isinstance(feedback_section, dict):
            continue
        if "labeler" in feedback_section:
            resolved["system_feedback_labeler"] = feedback_section["labeler"]
        if "llm_model" in feedback_section:
            resolved["system_feedback_llm_model"] = feedback_section["llm_model"]
        if "model" in feedback_section:
            resolved["system_feedback_llm_model"] = feedback_section["model"]
        if "temperature" in feedback_section:
            resolved["system_feedback_temperature"] = feedback_section["temperature"]
        if "max_tokens" in feedback_section:
            resolved["system_feedback_max_tokens"] = feedback_section["max_tokens"]
        if "cache_labels" in feedback_section:
            resolved["cache_feedback_labels"] = feedback_section["cache_labels"]
        if "cache_path" in feedback_section:
            resolved["feedback_label_cache_path"] = feedback_section["cache_path"]

    if args.hint_lambda is not None:
        resolved["hint_lambda"] = args.hint_lambda
    if args.max_k is not None:
        resolved["max_k"] = args.max_k
    if args.system_feedback_labeler is not None:
        resolved["system_feedback_labeler"] = args.system_feedback_labeler
    if args.feedback_llm_model is not None:
        resolved["system_feedback_llm_model"] = args.feedback_llm_model
    if args.feedback_label_cache_path is not None:
        resolved["feedback_label_cache_path"] = args.feedback_label_cache_path
    return resolved


def _load_hint_comparison_records(args, resolved: dict[str, Any]):
    if not args.plain_input_path and not args.hint_input_path:
        return []
    if not args.plain_input_path or not args.hint_input_path:
        raise ValueError(
            "`--plain_input_path` and `--hint_input_path` must be provided together."
        )

    plain_files = resolve_trace_files(Path(args.plain_input_path).expanduser().resolve())
    hint_files = resolve_trace_files(Path(args.hint_input_path).expanduser().resolve())
    if not plain_files:
        raise FileNotFoundError(f"No plain `.trace.txt` files found under: {args.plain_input_path}")
    if not hint_files:
        raise FileNotFoundError(f"No hint `.trace.txt` files found under: {args.hint_input_path}")

    plain = load_main_paper_dataset(
        plain_files,
        results_path=args.plain_results_path,
        metrics_path=args.plain_metrics_path,
        include_inner_transitions=False,
    )
    hint = load_main_paper_dataset(
        hint_files,
        results_path=args.hint_results_path,
        metrics_path=args.hint_metrics_path,
        include_inner_transitions=False,
    )
    return summarize_hint_comparison(
        plain.problems,
        hint.problems,
        hint_lambda=float(resolved["hint_lambda"]),
    )


if __name__ == "__main__":
    cli_main()
