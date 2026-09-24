"""Helpers for path-based task configs and benchmark presets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from dataloader.gsm8k import GSM8KLoader
from dataloader.mgsm import MGSMLoader
from dataloader.omni_math import OmniMathLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_CONFIG_NAME = "config.yaml"
BENCHMARK_CONFIG_NAME = "benchmark.yaml"


@dataclass
class ResolvedTaskConfig:
    task_label: str
    config_path: Path
    source_dir: Path
    benchmark_config_path: Path | None = None
    dataset_loader: str | None = None
    default_dataset_path: str | None = None
    default_output_path: str | None = None
    default_trace_tag: str | None = None
    default_artifact_tag: str | None = None
    default_extra_post_eval: str | None = None


def resolve_task_config_reference(task: str, tasks_dir: str) -> ResolvedTaskConfig:
    """Resolve a registered task id or a path-based experiment folder."""

    tasks_root = Path(tasks_dir).expanduser().resolve()
    maybe_path = _resolve_existing_input_path(task)
    if maybe_path is not None:
        return _resolve_path_based_task(maybe_path)

    config_path = (tasks_root / task / RUNTIME_CONFIG_NAME).resolve()
    if not config_path.exists():
        raise FileNotFoundError(
            f"Could not resolve task {task!r} as a registered task or path-based config."
        )
    return ResolvedTaskConfig(
        task_label=str(task),
        config_path=config_path,
        source_dir=config_path.parent,
    )


def resolve_dataset_path(
    cli_value: str | None,
    resolved_task: ResolvedTaskConfig,
) -> Path:
    raw = cli_value if cli_value else resolved_task.default_dataset_path
    if not raw:
        raise ValueError(
            "Dataset path was not provided. Pass `--dataset_path`, or set "
            "`dataset_path` in benchmark.yaml."
        )
    path = resolve_repo_or_local_path(raw, base_dir=resolved_task.source_dir)
    if not path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {path}")
    return path


def resolve_optional_output_path(
    cli_value: str | None,
    resolved_task: ResolvedTaskConfig,
) -> str | None:
    raw = cli_value if cli_value else resolved_task.default_output_path
    if not raw:
        return None
    raw_path = Path(raw).expanduser()
    if not raw_path.is_absolute() and raw_path.parts and raw_path.parts[0] == "results":
        return str((REPO_ROOT / raw_path).resolve())
    return str(resolve_repo_or_local_path(raw, base_dir=resolved_task.source_dir))


def resolve_benchmark_dataloader(
    *,
    task_label: str,
    dataset_path: str,
    dataset_loader: str | None,
):
    """Build a dataset loader without requiring task registration."""

    loader_name = normalize_dataset_loader(dataset_loader) or infer_dataset_loader(
        task_label=task_label,
        dataset_path=dataset_path,
    )
    if loader_name == "omni-math":
        return OmniMathLoader(path=dataset_path)
    if loader_name == "mgsm":
        return MGSMLoader(path=dataset_path)
    if loader_name == "gsm8k":
        return GSM8KLoader(path=dataset_path)
    if loader_name == "jsonl":
        return GenericJsonlLoader(path=dataset_path)
    raise ValueError(
        f"Unsupported dataset_loader={dataset_loader!r}. "
        "Use one of: omni-math, mgsm, gsm8k, jsonl."
    )


def normalize_dataset_loader(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower().replace("_", "-")
    aliases = {
        "omni-math-2-filtered": "omni-math",
        "omni-math-rule": "omni-math",
        "omnimath": "omni-math",
        "gsm": "gsm8k",
        "generic": "jsonl",
        "generic-jsonl": "jsonl",
    }
    return aliases.get(normalized, normalized)


def infer_dataset_loader(*, task_label: str, dataset_path: str) -> str:
    lowered_task = str(task_label or "").lower()
    lowered_path = str(dataset_path or "").lower()
    combined = f"{lowered_task} {lowered_path}"
    if "omni_math" in combined or "omni-math" in combined:
        return "omni-math"
    if "mgsm" in combined:
        return "mgsm"
    if "gsm8k" in combined:
        return "gsm8k"
    return "jsonl"


def resolve_repo_or_local_path(value: str | Path, *, base_dir: Path) -> Path:
    raw = Path(value).expanduser()
    if raw.is_absolute():
        return raw.resolve()

    local = (base_dir / raw).resolve()
    if local.exists():
        return local

    repo = (REPO_ROOT / raw).resolve()
    if repo.exists():
        return repo

    return local


def _resolve_existing_input_path(raw: str) -> Path | None:
    direct = Path(raw).expanduser()
    if direct.exists():
        return direct.resolve()
    repo_relative = (REPO_ROOT / raw).resolve()
    if repo_relative.exists():
        return repo_relative
    return None


def _resolve_path_based_task(path: Path) -> ResolvedTaskConfig:
    config_path: Path | None = None
    source_dir = path if path.is_dir() else path.parent
    benchmark_path: Path | None = None
    payload: dict[str, Any] = {}

    if path.is_file():
        if path.name == BENCHMARK_CONFIG_NAME:
            benchmark_path = path
            payload = _load_yaml(benchmark_path)
        elif path.name == RUNTIME_CONFIG_NAME:
            config_path = path
        else:
            raise ValueError(
                f"Path-based task input must be a folder, {RUNTIME_CONFIG_NAME}, "
                f"or {BENCHMARK_CONFIG_NAME}: {path}"
            )
    else:
        candidate_benchmark = path / BENCHMARK_CONFIG_NAME
        candidate_config = path / RUNTIME_CONFIG_NAME
        if candidate_benchmark.exists():
            benchmark_path = candidate_benchmark
            payload = _load_yaml(benchmark_path)
        if candidate_config.exists():
            config_path = candidate_config

    explicit_config = payload.get("task_config_path") or payload.get("config_path")
    if explicit_config:
        config_path = resolve_repo_or_local_path(str(explicit_config), base_dir=source_dir)

    if config_path is None or not config_path.exists():
        raise FileNotFoundError(
            f"Could not find {RUNTIME_CONFIG_NAME} for path-based task input: {path}"
        )

    task_label = (
        str(payload.get("task_name") or "").strip()
        or _relative_task_label(config_path)
        or source_dir.name
    )
    return ResolvedTaskConfig(
        task_label=task_label,
        config_path=config_path,
        source_dir=source_dir,
        benchmark_config_path=benchmark_path,
        dataset_loader=normalize_dataset_loader(payload.get("dataset_loader")),
        default_dataset_path=_coerce_optional_str(payload.get("dataset_path")),
        default_output_path=_coerce_optional_str(payload.get("output_path")),
        default_trace_tag=_coerce_optional_str(payload.get("trace_tag")),
        default_artifact_tag=_coerce_optional_str(payload.get("artifact_tag")),
        default_extra_post_eval=_coerce_optional_str(
            payload.get("extra_post_eval", payload.get("eval_mode"))
        ),
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def _relative_task_label(config_path: Path) -> str:
    task_root = REPO_ROOT / "agentverse" / "tasks"
    try:
        return str(config_path.parent.relative_to(task_root))
    except Exception:
        return config_path.parent.name


def _coerce_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


class GenericJsonlLoader:
    """Small fallback loader for simple JSONL task-solving datasets."""

    def __init__(self, path: str):
        self.path = path
        self.examples: list[dict[str, Any]] = []
        self.load()

    def load(self) -> None:
        with open(self.path, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                row = json.loads(line)
                question = row.get("input", row.get("question", row.get("problem", "")))
                answer = row.get("answer", row.get("answer_number", row.get("label", "")))
                if not question:
                    raise ValueError(
                        f"Generic JSONL loader could not find input/question/problem in {self.path}"
                    )
                self.examples.append(
                    {
                        "input": question,
                        "answer": str(answer).strip(),
                        **row,
                    }
                )

    def __iter__(self):
        return iter(self.examples)
