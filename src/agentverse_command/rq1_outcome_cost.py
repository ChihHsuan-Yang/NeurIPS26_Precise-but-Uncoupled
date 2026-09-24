"""RQ1 outcome and cost CLI."""

from __future__ import annotations

from argparse import ArgumentParser

from agentverse.logging import get_logger
from agentverse.metrics.api_accounting import write_api_accounting_sidecar
from agentverse.metrics.main_paper import load_main_paper_dataset, summarize_basic_metrics
from agentverse.metrics.main_paper.reports import (
    render_basic_text,
    write_metric_csv,
)
from agentverse_command.post_eval_common import (
    load_post_eval_section,
    resolve_output_dir,
    resolve_primary_input,
    resolve_trace_files,
)


logger = get_logger()


def _build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Analyze saved `.trace.txt` files for RQ1 outcome and cost metrics."
    )
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--results_path", type=str, default=None)
    parser.add_argument("--metrics_path", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true", default=None)
    return parser


def cli_main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    input_path = resolve_primary_input(args.input_path)
    trace_files = resolve_trace_files(input_path)
    if not trace_files:
        raise FileNotFoundError(f"No `.trace.txt` files found under: {input_path}")

    _section, _config_path = load_post_eval_section(
        config_path=args.config_path,
        section_names=("trace_rq1_outcome_cost", "rq1_outcome_cost"),
    )
    out_dir = resolve_output_dir(
        input_path=input_path,
        output_dir=args.output_dir,
        default_dirname="rq1-outcome-cost",
        overwrite=bool(args.overwrite),
    )

    dataset = load_main_paper_dataset(
        trace_files,
        results_path=args.results_path,
        metrics_path=args.metrics_path,
        include_inner_transitions=False,
    )
    records = summarize_basic_metrics(dataset.problems)

    csv_path = out_dir / "rq1_outcome_cost.csv"
    txt_path = out_dir / "rq1_outcome_cost.txt"
    write_metric_csv(csv_path, records)
    txt_path.write_text(render_basic_text(records, dataset.problems), encoding="utf-8")
    accounting_path = write_api_accounting_sidecar(
        out_dir,
        {
            "kind": "analysis_api_accounting",
            "cli_name": "agentverse-rq1-outcome-cost",
            "trace_files": dataset.trace_files,
            "example_count": len(dataset.examples),
            **dataset.analysis_accounting,
        },
    )

    logger.info(
        "[RQ1 OUTCOME/COST] "
        f"examples={len(dataset.examples)} output_dir={out_dir}"
    )
    logger.info(f"[RQ1 OUTCOME/COST] Wrote CSV/TXT files and {accounting_path.name}.")


if __name__ == "__main__":
    cli_main()
