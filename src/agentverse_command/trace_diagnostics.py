"""Supplemental trace diagnostics CLI.

This command intentionally excludes the main-paper must-have metrics computed
by `agentverse-trace-analysis` and the decomposition metrics computed by
`agentverse-collaboration-decomposition`. It reports lightweight process
diagnostics only.
"""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Any

from agentverse.logging import get_logger
from agentverse.metrics.diagnostics import (
    load_diagnostics_dataset,
    summarize_process_diagnostics,
)
from agentverse.metrics.diagnostics.reports import (
    render_process_diagnostics_text,
    write_diagnostics_csv,
)
from agentverse_command.trace_analysis import (
    _default_post_eval_config_path,
    _parse_config_mapping,
)


logger = get_logger()


def _build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description=(
            "Analyze saved `.trace.txt` files for supplemental process "
            "diagnostics only."
        )
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Optional YAML/JSON config with a top-level `trace_diagnostics:` block.",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to one `.trace.txt` file or a directory containing trace files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for diagnostics CSV/TXT files. Defaults to `trace-diagnose` next to input.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=None,
        help="Allow writing into an existing non-empty output directory.",
    )
    return parser


def cli_main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    input_path = Path(args.input_path).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    trace_files = _resolve_trace_files(input_path)
    if not trace_files:
        raise FileNotFoundError(f"No `.trace.txt` files found under: {input_path}")

    resolved = _resolve_args(args)
    out_dir = _resolve_output_dir(input_path, args.output_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not bool(args.overwrite):
        raise FileExistsError(
            f"Output directory already exists and is not empty: {out_dir}. "
            "Re-run with --overwrite to reuse it."
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_diagnostics_dataset(
        trace_files,
        repeated_candidate_threshold=int(resolved["repeated_candidate_threshold"]),
        repeated_route_threshold=int(resolved["repeated_route_threshold"]),
    )

    process_records = summarize_process_diagnostics(dataset.process_records)

    process_csv = out_dir / "process_diagnostics.csv"
    process_txt = out_dir / "process_diagnostics.txt"

    write_diagnostics_csv(process_csv, process_records)
    process_txt.write_text(
        render_process_diagnostics_text(process_records, dataset.process_records),
        encoding="utf-8",
    )

    logger.info(
        "[TRACE DIAGNOSTICS] "
        f"examples={len(dataset.examples)} "
        f"output_dir={out_dir}"
    )
    logger.info("[TRACE DIAGNOSTICS] Wrote process diagnostics CSV/TXT files.")


def _resolve_trace_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(input_path.rglob("*.trace.txt"))
    return []


def _resolve_output_dir(input_path: Path, output_dir: str | None) -> Path:
    if output_dir:
        return Path(output_dir).expanduser().resolve()
    if input_path.is_file():
        return input_path.parent / "trace-diagnose"
    return input_path / "trace-diagnose"


def _resolve_args(args) -> dict[str, Any]:
    config_path = (
        Path(args.config_path).expanduser().resolve()
        if args.config_path
        else _default_post_eval_config_path()
    )
    if config_path is not None and not config_path.exists():
        raise FileNotFoundError(f"Config path does not exist: {config_path}")

    resolved = {
        "repeated_candidate_threshold": 3,
        "repeated_route_threshold": 3,
    }
    resolved.update(_load_diagnostics_config(config_path))
    return resolved


def _load_diagnostics_config(config_path: Path | None) -> dict[str, Any]:
    if config_path is None:
        return {}

    payload = _parse_config_mapping(config_path.read_text(encoding="utf-8"), config_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Trace-diagnostics config must be a mapping: {config_path}")

    diagnostics = payload.get("trace_diagnostics") or {}
    if not isinstance(diagnostics, dict):
        diagnostics = {}

    config: dict[str, Any] = {}

    process = diagnostics.get("process") or {}
    if isinstance(process, dict):
        for key in ("repeated_candidate_threshold", "repeated_route_threshold"):
            if key in process:
                config[key] = process[key]

    return config


if __name__ == "__main__":
    cli_main()
