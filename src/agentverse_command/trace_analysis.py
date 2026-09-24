"""Main-paper trace analysis CLI.

This command intentionally computes only the must-have paper metrics:
basic outcome/cost metrics, outer-loop verifier-guided correction metrics, and
inner-loop reflective-refinement transition metrics.
"""

from __future__ import annotations

import json
import re
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

from agentverse.logging import get_logger
from agentverse.metrics.main_paper import (
    load_main_paper_dataset,
    summarize_basic_metrics,
    summarize_hint_comparison,
    summarize_inner_loop_metrics,
    summarize_outer_loop_metrics,
)
from agentverse.metrics.main_paper.reports import (
    render_basic_text,
    render_inner_text,
    render_outer_text,
    write_metric_csv,
)


logger = get_logger()


def _build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description=(
            "Analyze saved `.trace.txt` files for main-paper basic, outer-loop, "
            "and inner-loop metrics only."
        )
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Optional YAML/JSON config with a top-level `trace_analysis:` block.",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default=None,
        help="Path to one `.trace.txt` file or a directory containing trace files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for the six main-paper metric files. Defaults next to input.",
    )
    parser.add_argument(
        "--results_path",
        type=str,
        default=None,
        help="Optional results.jsonl sidecar for difficulty/domain metadata.",
    )
    parser.add_argument(
        "--metrics_path",
        type=str,
        default=None,
        help="Optional *.metrics.jsonl sidecar for token/call counts.",
    )
    parser.add_argument(
        "--run_config_path",
        type=str,
        default=None,
        help=(
            "Optional original run config. Needed to replay benchmark LLM "
            "evaluator-agent labels for intermediate candidates."
        ),
    )
    parser.add_argument(
        "--plain_input_path",
        type=str,
        default=None,
        help="Optional plain-feedback trace for matched plain-vs-hint comparison.",
    )
    parser.add_argument(
        "--hint_input_path",
        type=str,
        default=None,
        help="Optional hinted-feedback trace for matched plain-vs-hint comparison.",
    )
    parser.add_argument(
        "--plain_results_path",
        type=str,
        default=None,
        help="Optional results sidecar for `--plain_input_path`.",
    )
    parser.add_argument(
        "--hint_results_path",
        type=str,
        default=None,
        help="Optional results sidecar for `--hint_input_path`.",
    )
    parser.add_argument(
        "--plain_metrics_path",
        type=str,
        default=None,
        help="Optional metrics sidecar for `--plain_input_path`.",
    )
    parser.add_argument(
        "--hint_metrics_path",
        type=str,
        default=None,
        help="Optional metrics sidecar for `--hint_input_path`.",
    )
    parser.add_argument(
        "--plain_run_config_path",
        type=str,
        default=None,
        help="Optional original run config for `--plain_input_path`.",
    )
    parser.add_argument(
        "--hint_run_config_path",
        type=str,
        default=None,
        help="Optional original run config for `--hint_input_path`.",
    )
    parser.add_argument(
        "--evaluator_type",
        type=str,
        default=None,
        help=(
            "Candidate correctness evaluator for inner transitions. Use "
            "`llm`, `same_as_benchmark`, `exact`, `numeric-verifier`, "
            "`omni-rule`, `omni-judge`, or `omni-verifier`."
        ),
    )
    parser.add_argument(
        "--candidate_label_cache_path",
        type=str,
        default=None,
        help=(
            "Optional shared JSONL cache for intermediate candidate "
            "correctness labels. Defaults next to the trace."
        ),
    )
    parser.add_argument(
        "--hint_lambda",
        type=float,
        default=None,
        help="Hint penalty coefficient for HintAwareCorrectionScore.",
    )
    parser.add_argument(
        "--max_k",
        type=int,
        default=None,
        help="Maximum system-try k for OuterPass@k and MarginalRecoveryGain_k.",
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
    resolved = _resolve_args(args)

    primary_input = _resolve_primary_input(args)
    primary_trace_files = _resolve_trace_files(primary_input)
    if not primary_trace_files:
        raise FileNotFoundError(f"No `.trace.txt` files found under: {primary_input}")

    out_dir = _resolve_output_dir(primary_input, args.output_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not bool(args.overwrite):
        raise FileExistsError(
            f"Output directory already exists and is not empty: {out_dir}. "
            "Re-run with --overwrite to reuse it."
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_main_paper_dataset(
        primary_trace_files,
        results_path=args.results_path,
        metrics_path=args.metrics_path,
        evaluator_type=str(resolved["evaluator_type"]),
        numeric_tolerance=str(resolved["numeric_tolerance"]),
        data_name=str(resolved["data_name"]),
        omni_judge_model_path=resolved.get("omni_judge_model_path"),
        omni_judge_max_new_tokens=int(resolved["omni_judge_max_new_tokens"]),
        omni_judge_device=str(resolved["omni_judge_device"]),
        omni_judge_dtype=str(resolved["omni_judge_dtype"]),
        run_config_path=args.run_config_path or resolved.get("run_config_path"),
        reuse_runtime_final_labels=bool(resolved["reuse_runtime_final_labels"]),
        cache_candidate_labels=bool(resolved["cache_candidate_labels"]),
        candidate_label_cache_path=resolved.get("candidate_label_cache_path"),
    )

    basic_records = summarize_basic_metrics(dataset.problems)
    outer_records, outer_scores = summarize_outer_loop_metrics(
        dataset.problems,
        hint_lambda=float(resolved["hint_lambda"]),
        max_k=(
            int(resolved["max_k"])
            if resolved.get("max_k") is not None
            else None
        ),
    )
    inner_records = summarize_inner_loop_metrics(dataset.inner_transitions)

    comparison_records = _load_hint_comparison_records(args, resolved)
    if comparison_records:
        outer_records.extend(comparison_records)

    basic_csv = out_dir / "basic_metrics.csv"
    basic_txt = out_dir / "basic_metrics.txt"
    outer_csv = out_dir / "outer_loop_metrics.csv"
    outer_txt = out_dir / "outer_loop_metrics.txt"
    inner_csv = out_dir / "inner_loop_metrics.csv"
    inner_txt = out_dir / "inner_loop_metrics.txt"

    write_metric_csv(basic_csv, basic_records)
    basic_txt.write_text(
        render_basic_text(basic_records, dataset.problems),
        encoding="utf-8",
    )
    write_metric_csv(outer_csv, outer_records)
    outer_txt.write_text(
        render_outer_text(outer_records, dataset.problems, outer_scores),
        encoding="utf-8",
    )
    write_metric_csv(inner_csv, inner_records)
    inner_txt.write_text(
        render_inner_text(inner_records, dataset.inner_transitions),
        encoding="utf-8",
    )

    logger.info(
        "[TRACE ANALYSIS] "
        f"examples={len(dataset.examples)} "
        f"evaluator_type={dataset.evaluator_type} "
        f"output_dir={out_dir}"
    )
    logger.info("[TRACE ANALYSIS] Wrote basic, outer-loop, and inner-loop CSV/TXT files.")


def _load_hint_comparison_records(args, resolved: dict[str, Any]):
    if not args.plain_input_path and not args.hint_input_path:
        return []
    if not args.plain_input_path or not args.hint_input_path:
        raise ValueError(
            "`--plain_input_path` and `--hint_input_path` must be provided together."
        )

    plain_files = _resolve_trace_files(Path(args.plain_input_path).expanduser().resolve())
    hint_files = _resolve_trace_files(Path(args.hint_input_path).expanduser().resolve())
    if not plain_files:
        raise FileNotFoundError(f"No plain `.trace.txt` files found under: {args.plain_input_path}")
    if not hint_files:
        raise FileNotFoundError(f"No hint `.trace.txt` files found under: {args.hint_input_path}")

    plain = load_main_paper_dataset(
        plain_files,
        results_path=args.plain_results_path,
        metrics_path=args.plain_metrics_path,
        evaluator_type=str(resolved["evaluator_type"]),
        numeric_tolerance=str(resolved["numeric_tolerance"]),
        data_name=str(resolved["data_name"]),
        omni_judge_model_path=resolved.get("omni_judge_model_path"),
        omni_judge_max_new_tokens=int(resolved["omni_judge_max_new_tokens"]),
        omni_judge_device=str(resolved["omni_judge_device"]),
        omni_judge_dtype=str(resolved["omni_judge_dtype"]),
        run_config_path=args.plain_run_config_path or resolved.get("plain_run_config_path"),
        reuse_runtime_final_labels=bool(resolved["reuse_runtime_final_labels"]),
        cache_candidate_labels=bool(resolved["cache_candidate_labels"]),
        candidate_label_cache_path=resolved.get("candidate_label_cache_path"),
    )
    hint = load_main_paper_dataset(
        hint_files,
        results_path=args.hint_results_path,
        metrics_path=args.hint_metrics_path,
        evaluator_type=str(resolved["evaluator_type"]),
        numeric_tolerance=str(resolved["numeric_tolerance"]),
        data_name=str(resolved["data_name"]),
        omni_judge_model_path=resolved.get("omni_judge_model_path"),
        omni_judge_max_new_tokens=int(resolved["omni_judge_max_new_tokens"]),
        omni_judge_device=str(resolved["omni_judge_device"]),
        omni_judge_dtype=str(resolved["omni_judge_dtype"]),
        run_config_path=args.hint_run_config_path or resolved.get("hint_run_config_path"),
        reuse_runtime_final_labels=bool(resolved["reuse_runtime_final_labels"]),
        cache_candidate_labels=bool(resolved["cache_candidate_labels"]),
        candidate_label_cache_path=resolved.get("candidate_label_cache_path"),
    )
    return summarize_hint_comparison(
        plain.problems,
        hint.problems,
        hint_lambda=float(resolved["hint_lambda"]),
    )


def _resolve_primary_input(args) -> Path:
    raw = args.input_path or args.hint_input_path or args.plain_input_path
    if not raw:
        raise ValueError("Provide `--input_path`, or paired `--plain_input_path` and `--hint_input_path`.")
    input_path = Path(raw).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    return input_path


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
        return input_path.parent / "trace_analysis"
    return input_path / "trace_analysis"


def _resolve_args(args) -> dict[str, Any]:
    config_path = (
        Path(args.config_path).expanduser().resolve()
        if args.config_path
        else _default_post_eval_config_path()
    )
    if config_path is not None and not config_path.exists():
        raise FileNotFoundError(f"Config path does not exist: {config_path}")

    defaults: dict[str, Any] = {
        "hint_lambda": 1.0,
        "max_k": None,
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
        "plain_run_config_path": None,
        "hint_run_config_path": None,
    }
    resolved = dict(defaults)
    resolved.update(_load_trace_analysis_config(config_path))

    if args.evaluator_type is not None:
        resolved["evaluator_type"] = args.evaluator_type
    if args.candidate_label_cache_path is not None:
        resolved["candidate_label_cache_path"] = args.candidate_label_cache_path
    if args.hint_lambda is not None:
        resolved["hint_lambda"] = args.hint_lambda
    if args.max_k is not None:
        resolved["max_k"] = args.max_k

    resolved["config_path"] = str(config_path) if config_path is not None else None
    return resolved


def _default_post_eval_config_path() -> Path | None:
    candidate = Path(__file__).resolve().parents[1] / "configs" / "post_eval" / "default.yaml"
    return candidate if candidate.exists() else None


def _load_trace_analysis_config(config_path: Path | None) -> dict[str, Any]:
    if config_path is None:
        return {}

    raw_text = config_path.read_text(encoding="utf-8")
    payload = _parse_config_mapping(raw_text, config_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Trace-analysis config must be a mapping: {config_path}")

    for section_name in ("trace_analysis", "main_paper_metrics"):
        section = payload.get(section_name)
        if isinstance(section, dict):
            payload = dict(section)
            break

    config: dict[str, Any] = {}
    for key in ("hint_lambda", "max_k", "numeric_tolerance", "data_name"):
        if key in payload:
            config[key] = payload[key]

    evaluator = payload.get("evaluator") or {}
    if isinstance(evaluator, dict):
        evaluator_type = evaluator.get("type", evaluator.get("correctness_source"))
        if evaluator_type is not None:
            config["evaluator_type"] = evaluator_type
        for key in (
            "reuse_runtime_final_labels",
            "cache_candidate_labels",
            "candidate_label_cache_path",
            "omni_judge_model_path",
            "omni_judge_max_new_tokens",
            "omni_judge_device",
            "omni_judge_dtype",
            "run_config_path",
            "plain_run_config_path",
            "hint_run_config_path",
        ):
            if key in evaluator:
                config[key] = evaluator[key]

    return config


def _parse_config_mapping(raw_text: str, config_path: Path) -> dict[str, Any]:
    try:
        import yaml
    except Exception:
        yaml = None

    if yaml is not None:
        payload = yaml.safe_load(raw_text) or {}
        if isinstance(payload, dict):
            return payload
        raise ValueError(f"Trace-analysis config must be a mapping: {config_path}")

    try:
        payload = json.loads(raw_text)
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass

    return _parse_simple_yaml_mapping(raw_text, config_path)


def _parse_simple_yaml_mapping(raw_text: str, config_path: Path) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]

    for line_no, raw_line in enumerate(raw_text.splitlines(), start=1):
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        stripped = raw_line.split("#", 1)[0].strip()
        if not stripped:
            continue
        if ":" not in stripped:
            raise ValueError(
                f"Unsupported config line {line_no} in {config_path}: {raw_line!r}"
            )

        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"Missing key on line {line_no} in {config_path}")

        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()
        current = stack[-1][1]

        if not value:
            child: dict[str, Any] = {}
            current[key] = child
            stack.append((indent, child))
            continue

        current[key] = _parse_scalar(value)

    return root


def _parse_scalar(value: str) -> Any:
    normalized = value.strip()
    if normalized.startswith(("'", '"')) and normalized.endswith(("'", '"')):
        return normalized[1:-1]

    lowered = normalized.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None

    if re.fullmatch(r"-?\d+", normalized):
        try:
            return int(normalized)
        except Exception:
            return normalized
    if re.fullmatch(r"-?\d+\.\d+", normalized):
        try:
            return float(normalized)
        except Exception:
            return normalized

    return normalized


if __name__ == "__main__":
    cli_main()
