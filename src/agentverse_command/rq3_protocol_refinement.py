"""RQ3 protocol-level refinement CLI."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Any

from agentverse.logging import get_logger
from agentverse.metrics.api_accounting import write_api_accounting_sidecar
from agentverse.metrics.main_paper import (
    load_main_paper_dataset,
    summarize_inner_loop_metrics,
)
from agentverse.metrics.main_paper.inputs import _discover_run_config_path
from agentverse.metrics.main_paper.reports import (
    render_inner_text,
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
        description=(
            "Analyze saved `.trace.txt` files for RQ3 protocol-level "
            "reflective refinement metrics."
        )
    )
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--results_path", type=str, default=None)
    parser.add_argument("--metrics_path", type=str, default=None)
    parser.add_argument("--run_config_path", type=str, default=None)
    parser.add_argument("--evaluator_type", type=str, default=None)
    parser.add_argument("--candidate_label_cache_path", type=str, default=None)
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
        default_dirname="rq3-protocol-refinement",
        overwrite=bool(args.overwrite),
    )

    dataset = load_main_paper_dataset(
        trace_files,
        results_path=args.results_path,
        metrics_path=args.metrics_path,
        evaluator_type=str(resolved["evaluator_type"]),
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
        include_inner_transitions=True,
    )
    assert_eligible_protocols(
        cli_name="agentverse-rq3-protocol-refinement",
        protocols=protocol_set_from_examples(dataset.examples),
        allowed={"per", "broadcast"},
    )

    records = summarize_inner_loop_metrics(dataset.inner_transitions)
    csv_path = out_dir / "rq3_protocol_refinement.csv"
    txt_path = out_dir / "rq3_protocol_refinement.txt"
    write_metric_csv(csv_path, records)
    txt_path.write_text(
        render_inner_text(records, dataset.inner_transitions),
        encoding="utf-8",
    )
    accounting_path = write_api_accounting_sidecar(
        out_dir,
        {
            "kind": "analysis_api_accounting",
            "cli_name": "agentverse-rq3-protocol-refinement",
            "trace_files": dataset.trace_files,
            "example_count": len(dataset.examples),
            **dataset.analysis_accounting,
        },
    )

    logger.info(
        "[RQ3 PROTOCOL REFINEMENT] "
        f"examples={len(dataset.examples)} evaluator_type={dataset.evaluator_type} "
        f"output_dir={out_dir}"
    )
    logger.info(
        f"[RQ3 PROTOCOL REFINEMENT] Wrote CSV/TXT files and {accounting_path.name}."
    )


def _resolve_args(args, *, base_dir: Path) -> dict[str, Any]:
    section, _config_path = load_post_eval_section(
        config_path=args.config_path,
        section_names=("trace_rq3_protocol_refinement", "rq3_protocol_refinement", "trace_analysis"),
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
    }

    for key in ("numeric_tolerance", "data_name"):
        if key in section:
            resolved[key] = section[key]

    evaluator = section.get("evaluator") or {}
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

    if args.evaluator_type is not None:
        resolved["evaluator_type"] = args.evaluator_type
    if args.candidate_label_cache_path is not None:
        resolved["candidate_label_cache_path"] = args.candidate_label_cache_path
    if args.run_config_path is not None:
        resolved["run_config_path"] = args.run_config_path

    if not resolved.get("run_config_path"):
        discovered = _discover_run_config_path(base_dir)
        if discovered is not None:
            resolved["run_config_path"] = str(discovered)
    return resolved


if __name__ == "__main__":
    cli_main()
