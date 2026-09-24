"""Strict coupling-rate summaries and audit exports for paper robustness checks."""

from __future__ import annotations

import csv
import random
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from agentverse.evaluation.common import exact_match
from agentverse.metrics.collaboration_decomposition.records import (
    ReviewerFeedbackIncorporationEpisode,
)
from agentverse.metrics.collaboration_decomposition.reports import write_metric_csv
from agentverse.metrics.main_paper.records import MetricRecord
from agentverse.metrics.repair import ReviewEpisode


@dataclass
class StrictCouplingEpisodeRecord:
    trace_path: str
    protocol: str
    example_idx: int
    episode_id: str
    review_turn: int
    reviewer: str
    action: str
    pre_correct: bool | None
    post_correct: bool | None
    after_evaluator_fail: bool
    evidence_turns: str
    review_feedback: str
    answer_before: str
    answer_after: str
    answer_before_key: str
    answer_after_key: str
    strict_answer_eligible: bool
    strict_answer_changed: bool | None
    equivalence_eligible: bool
    equivalence_changed: bool | None
    normalized_change_but_equivalent: bool | None
    legacy_incorporated_feedback: bool | None
    legacy_incorporated_feedback_source: str
    strict_vs_legacy_disagreement: bool | None
    outcome_bucket: str
    audit_stratum: str
    human_substantive_uptake: str = ""
    human_review_aligned_change: str = ""
    human_notes: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class StrictCouplingProtocolSummaryRow:
    protocol: str
    useful_revise_episodes: int
    strict_eligible_useful_revise_episodes: int
    strict_changed_useful_revise_episodes: int
    strict_useful_coupling_rate: float | None
    equivalence_eligible_useful_revise_episodes: int
    equivalence_changed_useful_revise_episodes: int
    equivalence_aware_useful_coupling_rate: float | None
    normalized_change_but_equivalent_episodes: int
    changed_and_repaired_episodes: int
    changed_and_repaired_rate_among_useful_revise: float | None
    changed_but_still_wrong_episodes: int
    changed_but_still_wrong_rate_among_useful_revise: float | None
    unchanged_episodes: int
    unchanged_rate_among_useful_revise: float | None
    repair_given_change_rate: float | None
    legacy_eligible_useful_revise_episodes: int
    legacy_incorporated_useful_revise_episodes: int
    legacy_useful_coupling_rate: float | None
    strict_vs_legacy_disagreement_episodes: int
    strict_vs_legacy_disagreement_rate: float | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_strict_coupling_episode_records(
    review_episodes: list[ReviewEpisode],
    reviewer_feedback_episodes: list[ReviewerFeedbackIncorporationEpisode],
    *,
    numeric_tolerance: str = "0",
) -> list[StrictCouplingEpisodeRecord]:
    legacy_by_episode = {
        (episode.trace_path, episode.example_idx, episode.episode_id): episode
        for episode in reviewer_feedback_episodes
    }

    records: list[StrictCouplingEpisodeRecord] = []
    for episode in review_episodes:
        answer_before = str(episode.candidate_before or "").strip()
        answer_after = str(episode.candidate_after or "").strip()
        answer_before_key = _answer_key(answer_before)
        answer_after_key = _answer_key(answer_after)

        strict_answer_eligible = bool(answer_before_key and answer_after_key)
        strict_answer_changed = None
        if strict_answer_eligible:
            strict_answer_changed = answer_before_key != answer_after_key

        equivalence_eligible = bool(answer_before and answer_after)
        equivalence_changed = None
        if equivalence_eligible:
            equivalence_changed = not exact_match(
                answer_before,
                answer_after,
                tol=str(numeric_tolerance or "0"),
            )

        normalized_change_but_equivalent = None
        if strict_answer_changed is not None and equivalence_changed is not None:
            normalized_change_but_equivalent = strict_answer_changed and not equivalence_changed

        legacy = legacy_by_episode.get((episode.trace_path, episode.example_idx, episode.episode_id))
        legacy_flag = None if legacy is None else legacy.incorporated_feedback
        strict_vs_legacy_disagreement = None
        if strict_answer_changed is not None and legacy_flag is not None:
            strict_vs_legacy_disagreement = strict_answer_changed != legacy_flag

        outcome_bucket = _outcome_bucket(
            strict_answer_changed=strict_answer_changed,
            normalized_change_but_equivalent=normalized_change_but_equivalent,
            post_correct=episode.post_correct,
        )
        audit_stratum = _audit_stratum(
            strict_vs_legacy_disagreement=strict_vs_legacy_disagreement,
            outcome_bucket=outcome_bucket,
        )

        records.append(
            StrictCouplingEpisodeRecord(
                trace_path=episode.trace_path,
                protocol=episode.protocol,
                example_idx=episode.example_idx,
                episode_id=episode.episode_id,
                review_turn=episode.review_turn,
                reviewer=episode.reviewer,
                action=episode.action,
                pre_correct=episode.pre_correct,
                post_correct=episode.post_correct,
                after_evaluator_fail=episode.after_evaluator_fail,
                evidence_turns=",".join(str(turn) for turn in episode.evidence_turns),
                review_feedback=episode.review_feedback,
                answer_before=answer_before,
                answer_after=answer_after,
                answer_before_key=answer_before_key,
                answer_after_key=answer_after_key,
                strict_answer_eligible=strict_answer_eligible,
                strict_answer_changed=strict_answer_changed,
                equivalence_eligible=equivalence_eligible,
                equivalence_changed=equivalence_changed,
                normalized_change_but_equivalent=normalized_change_but_equivalent,
                legacy_incorporated_feedback=legacy_flag,
                legacy_incorporated_feedback_source=(
                    "" if legacy is None else legacy.incorporated_feedback_source
                ),
                strict_vs_legacy_disagreement=strict_vs_legacy_disagreement,
                outcome_bucket=outcome_bucket,
                audit_stratum=audit_stratum,
            )
        )
    return records


def summarize_strict_coupling_rate(
    records: list[StrictCouplingEpisodeRecord],
) -> list[MetricRecord]:
    useful_revise = _useful_revise(records)
    strict_eligible = [record for record in useful_revise if record.strict_answer_eligible]
    equivalence_eligible = [record for record in useful_revise if record.equivalence_eligible]
    legacy_eligible = [
        record for record in useful_revise if record.legacy_incorporated_feedback is not None
    ]

    strict_changed = sum(
        1 for record in strict_eligible if record.strict_answer_changed is True
    )
    equivalence_changed = sum(
        1 for record in equivalence_eligible if record.equivalence_changed is True
    )
    normalized_change_but_equivalent = sum(
        1 for record in strict_eligible if record.normalized_change_but_equivalent is True
    )
    legacy_incorporated = sum(
        1 for record in legacy_eligible if record.legacy_incorporated_feedback is True
    )
    disagreements = sum(
        1 for record in legacy_eligible if record.strict_vs_legacy_disagreement is True
    )

    changed_and_repaired = sum(
        1 for record in strict_eligible if record.outcome_bucket == "changed_and_repaired"
    )
    changed_but_still_wrong = sum(
        1 for record in strict_eligible if record.outcome_bucket == "changed_but_still_wrong"
    )
    unchanged = sum(1 for record in strict_eligible if record.outcome_bucket == "unchanged")

    return [
        _count(
            "strict_coupling",
            "UsefulReviseEpisodes",
            len(useful_revise),
            source="trace+benchmark_evaluator",
            notes="Wrong-before-review episodes where the reviewer says revise.",
        ),
        _count(
            "strict_coupling",
            "StrictEligibleUsefulReviseEpisodes",
            len(strict_eligible),
            source="trace+benchmark_evaluator",
            notes="Useful revise episodes with both before/after extracted answers present.",
        ),
        _count(
            "strict_coupling",
            "StrictChangedUsefulReviseEpisodes",
            strict_changed,
            source="trace+benchmark_evaluator",
        ),
        _rate(
            "strict_coupling",
            "StrictUsefulCouplingRate",
            strict_changed,
            len(strict_eligible),
            source="trace+benchmark_evaluator",
            notes=(
                "Primary strict definition: among evaluator-verified useful revise episodes, "
                "fraction whose immediately following extracted answer changes."
            ),
        ),
        _count(
            "strict_coupling",
            "EquivalenceAwareEligibleUsefulReviseEpisodes",
            len(equivalence_eligible),
            source="trace+benchmark_evaluator",
            notes="Useful revise episodes eligible for deterministic answer equivalence checks.",
        ),
        _count(
            "strict_coupling",
            "EquivalenceAwareChangedUsefulReviseEpisodes",
            equivalence_changed,
            source="trace+benchmark_evaluator",
        ),
        _rate(
            "strict_coupling",
            "EquivalenceAwareUsefulCouplingRate",
            equivalence_changed,
            len(equivalence_eligible),
            source="trace+benchmark_evaluator",
            notes=(
                "Sensitivity check that treats deterministically equivalent before/after "
                "answers as unchanged."
            ),
        ),
        _count(
            "strict_coupling",
            "NormalizedChangeButEquivalentUsefulReviseEpisodes",
            normalized_change_but_equivalent,
            source="trace+benchmark_evaluator",
            notes="Text-normalized answer changes that collapse under deterministic equivalence.",
        ),
        _count(
            "strict_coupling",
            "ChangedAndRepairedUsefulReviseEpisodes",
            changed_and_repaired,
            source="trace+benchmark_evaluator",
            notes="Strict answer change followed by a correct post-review candidate.",
        ),
        _rate(
            "strict_coupling",
            "ChangedAndRepairedRateAmongUsefulRevise",
            changed_and_repaired,
            len(strict_eligible),
            source="trace+benchmark_evaluator",
            notes=(
                "Among evaluator-verified useful revise episodes with both extracted answers present, "
                "fraction where the answer changes and the post-review candidate is correct."
            ),
        ),
        _count(
            "strict_coupling",
            "ChangedButStillWrongUsefulReviseEpisodes",
            changed_but_still_wrong,
            source="trace+benchmark_evaluator",
            notes="Strict answer change followed by a still-wrong post-review candidate.",
        ),
        _rate(
            "strict_coupling",
            "ChangedButStillWrongRateAmongUsefulRevise",
            changed_but_still_wrong,
            len(strict_eligible),
            source="trace+benchmark_evaluator",
            notes=(
                "Among evaluator-verified useful revise episodes with both extracted answers present, "
                "fraction where the answer changes but the post-review candidate remains wrong."
            ),
        ),
        _count(
            "strict_coupling",
            "UnchangedUsefulReviseEpisodes",
            unchanged,
            source="trace+benchmark_evaluator",
            notes="Useful revise episodes where the immediately following extracted answer does not change.",
        ),
        _rate(
            "strict_coupling",
            "UnchangedRateAmongUsefulRevise",
            unchanged,
            len(strict_eligible),
            source="trace+benchmark_evaluator",
            notes=(
                "Among evaluator-verified useful revise episodes with both extracted answers present, "
                "fraction where the immediately following extracted answer does not change."
            ),
        ),
        _rate(
            "strict_coupling",
            "RepairGivenChangeRate",
            changed_and_repaired,
            strict_changed,
            source="trace+benchmark_evaluator",
            notes=(
                "Among useful revise episodes whose immediately following extracted answer changes, "
                "fraction where the post-review candidate is correct."
            ),
        ),
        _count(
            "strict_coupling",
            "LegacyEligibleUsefulReviseEpisodes",
            len(legacy_eligible),
            source="trace+feedback_labeler+benchmark_evaluator",
            notes="Useful revise episodes eligible under the current reviewer-feedback pipeline.",
        ),
        _rate(
            "strict_coupling",
            "LegacyUsefulReviseFeedbackIncorporationRate",
            legacy_incorporated,
            len(legacy_eligible),
            source="trace+feedback_labeler+benchmark_evaluator",
            notes="Current decomposition metric for comparison against the stricter answer-change definition.",
        ),
        _count(
            "strict_coupling",
            "StrictLegacyDisagreementEpisodes",
            disagreements,
            source="trace+feedback_labeler+benchmark_evaluator",
            notes="Useful revise episodes where the strict answer-change label disagrees with the legacy incorporation label.",
        ),
        _rate(
            "strict_coupling",
            "StrictLegacyDisagreementRate",
            disagreements,
            len(legacy_eligible),
            source="trace+feedback_labeler+benchmark_evaluator",
            notes=(
                "Among useful revise episodes eligible under the legacy pipeline, "
                "fraction where the strict answer-change label disagrees with the legacy incorporation label."
            ),
        ),
    ]


def summarize_strict_coupling_by_protocol(
    records: list[StrictCouplingEpisodeRecord],
) -> list[StrictCouplingProtocolSummaryRow]:
    by_protocol: dict[str, list[StrictCouplingEpisodeRecord]] = defaultdict(list)
    for record in _useful_revise(records):
        by_protocol[str(record.protocol)].append(record)

    rows: list[StrictCouplingProtocolSummaryRow] = []
    for protocol in sorted(by_protocol):
        group = by_protocol[protocol]
        strict_eligible = [record for record in group if record.strict_answer_eligible]
        equivalence_eligible = [record for record in group if record.equivalence_eligible]
        legacy_eligible = [record for record in group if record.legacy_incorporated_feedback is not None]
        strict_changed = sum(1 for record in strict_eligible if record.strict_answer_changed is True)
        equivalence_changed = sum(
            1 for record in equivalence_eligible if record.equivalence_changed is True
        )
        rows.append(
            StrictCouplingProtocolSummaryRow(
                protocol=protocol,
                useful_revise_episodes=len(group),
                strict_eligible_useful_revise_episodes=len(strict_eligible),
                strict_changed_useful_revise_episodes=strict_changed,
                strict_useful_coupling_rate=_safe_rate(strict_changed, len(strict_eligible)),
                equivalence_eligible_useful_revise_episodes=len(equivalence_eligible),
                equivalence_changed_useful_revise_episodes=equivalence_changed,
                equivalence_aware_useful_coupling_rate=_safe_rate(
                    equivalence_changed, len(equivalence_eligible)
                ),
                normalized_change_but_equivalent_episodes=sum(
                    1
                    for record in strict_eligible
                    if record.normalized_change_but_equivalent is True
                ),
                changed_and_repaired_episodes=sum(
                    1
                    for record in strict_eligible
                    if record.outcome_bucket == "changed_and_repaired"
                ),
                changed_and_repaired_rate_among_useful_revise=_safe_rate(
                    sum(
                        1
                        for record in strict_eligible
                        if record.outcome_bucket == "changed_and_repaired"
                    ),
                    len(strict_eligible),
                ),
                changed_but_still_wrong_episodes=sum(
                    1
                    for record in strict_eligible
                    if record.outcome_bucket == "changed_but_still_wrong"
                ),
                changed_but_still_wrong_rate_among_useful_revise=_safe_rate(
                    sum(
                        1
                        for record in strict_eligible
                        if record.outcome_bucket == "changed_but_still_wrong"
                    ),
                    len(strict_eligible),
                ),
                unchanged_episodes=sum(
                    1 for record in strict_eligible if record.outcome_bucket == "unchanged"
                ),
                unchanged_rate_among_useful_revise=_safe_rate(
                    sum(
                        1 for record in strict_eligible if record.outcome_bucket == "unchanged"
                    ),
                    len(strict_eligible),
                ),
                repair_given_change_rate=_safe_rate(
                    sum(
                        1
                        for record in strict_eligible
                        if record.outcome_bucket == "changed_and_repaired"
                    ),
                    strict_changed,
                ),
                legacy_eligible_useful_revise_episodes=len(legacy_eligible),
                legacy_incorporated_useful_revise_episodes=sum(
                    1
                    for record in legacy_eligible
                    if record.legacy_incorporated_feedback is True
                ),
                legacy_useful_coupling_rate=_safe_rate(
                    sum(
                        1
                        for record in legacy_eligible
                        if record.legacy_incorporated_feedback is True
                    ),
                    len(legacy_eligible),
                ),
                strict_vs_legacy_disagreement_episodes=sum(
                    1
                    for record in legacy_eligible
                    if record.strict_vs_legacy_disagreement is True
                ),
                strict_vs_legacy_disagreement_rate=_safe_rate(
                    sum(
                        1
                        for record in legacy_eligible
                        if record.strict_vs_legacy_disagreement is True
                    ),
                    len(legacy_eligible),
                ),
            )
        )
    return rows


def build_audit_sample(
    records: list[StrictCouplingEpisodeRecord],
    *,
    sample_size: int,
    seed: int,
) -> list[StrictCouplingEpisodeRecord]:
    if sample_size <= 0:
        return []

    eligible = [
        record
        for record in _useful_revise(records)
        if record.strict_answer_eligible or record.legacy_incorporated_feedback is not None
    ]
    grouped: dict[tuple[str, str], list[StrictCouplingEpisodeRecord]] = defaultdict(list)
    for record in eligible:
        grouped[(record.protocol, record.audit_stratum)].append(record)

    rng = random.Random(seed)
    pools: list[tuple[tuple[str, str], list[StrictCouplingEpisodeRecord]]] = []
    for key in sorted(grouped, key=lambda item: (_protocol_rank(item[0]), _stratum_rank(item[1]))):
        group = list(grouped[key])
        rng.shuffle(group)
        pools.append((key, group))

    sample: list[StrictCouplingEpisodeRecord] = []
    while len(sample) < sample_size:
        added = False
        for _, group in pools:
            if not group:
                continue
            sample.append(group.pop())
            added = True
            if len(sample) >= sample_size:
                break
        if not added:
            break
    return sample


def render_strict_coupling_text(
    metrics: list[MetricRecord],
    protocol_rows: list[StrictCouplingProtocolSummaryRow],
    records: list[StrictCouplingEpisodeRecord],
    *,
    audit_sample_size: int = 0,
) -> str:
    lines = ["Strict Coupling Rate", ""]
    for metric in metrics:
        line = f"{metric.metric}: {_fmt(metric.value)}"
        if metric.numerator is not None or metric.denominator is not None:
            line += f" ({_fmt(metric.numerator)}/{_fmt(metric.denominator)})"
        if metric.notes:
            line += f" # {metric.notes}"
        lines.append(line)

    lines.extend(["", "By Protocol", ""])
    for row in protocol_rows:
        lines.append(
            (
                f"{row.protocol}: strict={_fmt(row.strict_useful_coupling_rate)} "
                f"({row.strict_changed_useful_revise_episodes}/"
                f"{row.strict_eligible_useful_revise_episodes}); "
                f"equiv-aware={_fmt(row.equivalence_aware_useful_coupling_rate)} "
                f"({row.equivalence_changed_useful_revise_episodes}/"
                f"{row.equivalence_eligible_useful_revise_episodes}); "
                f"unchanged-rate={_fmt(row.unchanged_rate_among_useful_revise)}; "
                f"changed+repaired-rate={_fmt(row.changed_and_repaired_rate_among_useful_revise)}; "
                f"repair-given-change={_fmt(row.repair_given_change_rate)}; "
                f"legacy={_fmt(row.legacy_useful_coupling_rate)} "
                f"({row.legacy_incorporated_useful_revise_episodes}/"
                f"{row.legacy_eligible_useful_revise_episodes}); "
                f"changed+repaired={row.changed_and_repaired_episodes}; "
                f"changed+wrong={row.changed_but_still_wrong_episodes}; "
                f"unchanged={row.unchanged_episodes}; "
                f"strict_vs_legacy_disagreement={row.strict_vs_legacy_disagreement_episodes} "
                f"({_fmt(row.strict_vs_legacy_disagreement_rate)})"
            )
        )

    stratum_counts = defaultdict(int)
    for record in _useful_revise(records):
        stratum_counts[(record.protocol, record.audit_stratum)] += 1

    lines.extend(["", "Audit Strata", ""])
    for protocol, stratum in sorted(
        stratum_counts,
        key=lambda item: (_protocol_rank(item[0]), _stratum_rank(item[1])),
    ):
        lines.append(f"{protocol} / {stratum}: {stratum_counts[(protocol, stratum)]}")

    if audit_sample_size > 0:
        lines.extend(["", f"Audit sample size requested: {audit_sample_size}"])
    return "\n".join(lines).rstrip() + "\n"


def write_strict_coupling_episode_csv(
    path: Path,
    records: list[StrictCouplingEpisodeRecord],
) -> None:
    _write_dataclass_csv(path, records, StrictCouplingEpisodeRecord)


def write_protocol_summary_csv(
    path: Path,
    rows: list[StrictCouplingProtocolSummaryRow],
) -> None:
    _write_dataclass_csv(path, rows, StrictCouplingProtocolSummaryRow)


def _write_dataclass_csv(path: Path, rows: list[object], schema) -> None:
    fieldnames = list(schema.__dataclass_fields__.keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _useful_revise(
    records: Iterable[StrictCouplingEpisodeRecord],
) -> list[StrictCouplingEpisodeRecord]:
    return [
        record
        for record in records
        if record.action == "revise" and record.pre_correct is False
    ]


def _count(
    section: str,
    metric: str,
    value: int,
    *,
    source: str,
    notes: str = "",
) -> MetricRecord:
    return MetricRecord(section=section, metric=metric, value=value, source=source, notes=notes)


def _rate(
    section: str,
    metric: str,
    numerator: int,
    denominator: int,
    *,
    source: str,
    notes: str = "",
) -> MetricRecord:
    return MetricRecord(
        section=section,
        metric=metric,
        value=_safe_rate(numerator, denominator),
        numerator=numerator,
        denominator=denominator,
        source=source,
        notes=notes,
    )


def _safe_rate(numerator: int, denominator: int) -> float | None:
    return (numerator / denominator) if denominator else None


def _answer_key(answer: str) -> str:
    return re.sub(r"\s+", "", str(answer or "").strip()).lower()


def _outcome_bucket(
    *,
    strict_answer_changed: bool | None,
    normalized_change_but_equivalent: bool | None,
    post_correct: bool | None,
) -> str:
    if strict_answer_changed is None:
        return "ineligible"
    if strict_answer_changed is False:
        return "unchanged"
    if normalized_change_but_equivalent:
        return "changed_but_equivalent"
    if post_correct is True:
        return "changed_and_repaired"
    if post_correct is False:
        return "changed_but_still_wrong"
    return "changed_unknown"


def _audit_stratum(
    *,
    strict_vs_legacy_disagreement: bool | None,
    outcome_bucket: str,
) -> str:
    if strict_vs_legacy_disagreement:
        return "legacy_disagreement"
    return outcome_bucket


def _protocol_rank(protocol: str) -> int:
    order = {"per": 0, "broadcast": 1}
    return order.get(str(protocol), 99)


def _stratum_rank(stratum: str) -> int:
    order = {
        "legacy_disagreement": 0,
        "changed_but_equivalent": 1,
        "changed_but_still_wrong": 2,
        "changed_and_repaired": 3,
        "unchanged": 4,
        "changed_unknown": 5,
        "ineligible": 6,
    }
    return order.get(str(stratum), 99)


def _fmt(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


__all__ = [
    "StrictCouplingEpisodeRecord",
    "StrictCouplingProtocolSummaryRow",
    "build_audit_sample",
    "build_strict_coupling_episode_records",
    "render_strict_coupling_text",
    "summarize_strict_coupling_by_protocol",
    "summarize_strict_coupling_rate",
    "write_metric_csv",
    "write_protocol_summary_csv",
    "write_strict_coupling_episode_csv",
]
