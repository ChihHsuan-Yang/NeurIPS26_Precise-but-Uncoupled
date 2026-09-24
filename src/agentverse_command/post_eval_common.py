"""Shared helpers for post-trace paper CLIs."""

from __future__ import annotations

import json
import re
from argparse import Namespace
from pathlib import Path
from typing import Any


def default_post_eval_config_path() -> Path | None:
    candidate = Path(__file__).resolve().parents[1] / "configs" / "post_eval" / "default.yaml"
    return candidate if candidate.exists() else None


def load_post_eval_section(
    *,
    config_path: str | None,
    section_names: tuple[str, ...],
) -> tuple[dict[str, Any], str | None]:
    resolved_path = (
        Path(config_path).expanduser().resolve()
        if config_path
        else default_post_eval_config_path()
    )
    if resolved_path is not None and not resolved_path.exists():
        raise FileNotFoundError(f"Config path does not exist: {resolved_path}")
    if resolved_path is None:
        return {}, None

    raw_text = resolved_path.read_text(encoding="utf-8")
    payload = parse_config_mapping(raw_text, resolved_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Post-eval config must be a mapping: {resolved_path}")

    section_payload: dict[str, Any] = {}
    for section_name in section_names:
        section = payload.get(section_name)
        if isinstance(section, dict):
            section_payload = dict(section)
            break
    return section_payload, str(resolved_path)


def resolve_primary_input(raw_input: str | None) -> Path:
    if not raw_input:
        raise ValueError("Provide `--input_path`.")
    input_path = Path(raw_input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    return input_path


def resolve_trace_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(input_path.rglob("*.trace.txt"))
    return []


def resolve_output_dir(
    *,
    input_path: Path,
    output_dir: str | None,
    default_dirname: str,
    overwrite: bool,
) -> Path:
    if output_dir:
        out_dir = Path(output_dir).expanduser().resolve()
    elif input_path.is_file():
        out_dir = input_path.parent / default_dirname
    else:
        out_dir = input_path / default_dirname

    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory already exists and is not empty: {out_dir}. "
            "Re-run with --overwrite to reuse it."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def parse_config_mapping(raw_text: str, config_path: Path) -> dict[str, Any]:
    try:
        import yaml
    except Exception:
        yaml = None

    if yaml is not None:
        payload = yaml.safe_load(raw_text) or {}
        if isinstance(payload, dict):
            return payload
        raise ValueError(f"Config must be a mapping: {config_path}")

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


def protocol_set_from_examples(examples: list[Any]) -> set[str]:
    return {str(getattr(example, "protocol", "") or "").strip() for example in examples}


def assert_eligible_protocols(
    *,
    cli_name: str,
    protocols: set[str],
    allowed: set[str],
) -> None:
    normalized = {protocol for protocol in protocols if protocol}
    if normalized and normalized.issubset(allowed):
        return
    raise ValueError(
        f"{cli_name} is only applicable to protocols {sorted(allowed)}; "
        f"detected {sorted(normalized) or ['unknown']}."
    )


def optional_bool_flag(value: Any, fallback: bool) -> bool:
    return fallback if value is None else bool(value)


def namespace_with_defaults(**kwargs: Any) -> Namespace:
    return Namespace(**kwargs)
