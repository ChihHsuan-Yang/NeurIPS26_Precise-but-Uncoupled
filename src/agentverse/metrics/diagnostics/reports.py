"""CSV and concise text reports for supplemental diagnostics."""

from __future__ import annotations

from typing import Any

from agentverse.metrics.diagnostics.records import ProcessExampleRecord
from agentverse.metrics.main_paper.records import MetricRecord
from agentverse.metrics.main_paper.reports import metric_explanation, write_metric_csv


def render_process_diagnostics_text(
    records: list[MetricRecord],
    process_records: list[ProcessExampleRecord],
) -> str:
    lines = ["Process Diagnostics", ""]
    lines.extend(_metric_lines(records))
    lines.extend(["", "Per Problem", ""])
    for record in sorted(
        process_records,
        key=lambda item: (item.trace_path, item.example_idx),
    ):
        loop = "yes" if record.looping_signal else "no"
        lines.append(
            (
                f"Example {record.example_idx}: agreements={record.agreement_events}/"
                f"{record.review_events}; wrong_convergence="
                f"{record.agreement_followed_by_fail}; repeated_adjacent="
                f"{record.repeated_adjacent_answers}/{record.adjacent_submission_pairs}; "
                f"no_change_after_fail={record.no_change_after_fail}/"
                f"{record.fail_with_next_count}; loop_signal={loop}; "
                f"distinct_candidates={record.distinct_candidate_count}; "
                f"turn_imbalance={_fmt(record.turn_imbalance)}"
            )
        )
    return "\n".join(lines).rstrip() + "\n"


def write_diagnostics_csv(path, records: list[MetricRecord]) -> None:
    write_metric_csv(path, records)


def _metric_lines(records: list[MetricRecord]) -> list[str]:
    lines: list[str] = []
    for record in records:
        detail = f"{record.metric}: {_fmt(record.value)}"
        if record.numerator is not None or record.denominator is not None:
            detail += f" ({_fmt(record.numerator)}/{_fmt(record.denominator)})"
        if record.notes:
            detail += f"; {record.notes}"
        explanation = metric_explanation(record.metric, notes=record.notes)
        if explanation:
            detail += f" # {explanation}"
        lines.append(detail)
    return lines


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)
