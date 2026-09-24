#!/usr/bin/env python3
"""Inventory and stage a unified paper result layout.

This script is intentionally non-destructive by default. It scans two result
roots:

- results/paper_experiment
- results/paper_experiment_chunk2plus

and produces machine-readable inventories for a canonical layout:

    tierXX/<mode>/chunkNNN_partMMM

Optionally, it can create a staging mirror using symlinks for the "winning"
source directory for each canonical destination.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

KNOWN_MODES = {
    "baseline_llm",
    "single_agent_hint_llm",
    "per_hint_llm",
    "broadcast_hint_llm",
}

SUBCHUNK_RE = re.compile(r"(tier(?P<tier>\d{2})_chunk(?P<chunk>\d{3})_part(?P<part>\d{3}))")
LEAF_RE = re.compile(r"chunk(?P<chunk>\d{3})_part(?P<part>\d{3})")
TIER_DIR_RE = re.compile(r"tier(?P<tier>\d{2})$")


@dataclass
class Record:
    source_root: str
    source_dir: Path
    worker_summary: Path
    mode: str
    tier: str
    chunk: str
    part: str
    canonical_rel: Path
    mtime: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-root", type=Path, required=True)
    parser.add_argument("--chunk2plus-root", type=Path, required=True)
    parser.add_argument("--inventory-tsv", type=Path, required=True)
    parser.add_argument("--chosen-tsv", type=Path, required=True)
    parser.add_argument("--stage-root", type=Path, default=None)
    parser.add_argument(
        "--link-mode",
        choices=["none", "symlink"],
        default="none",
        help="Whether to create a staging mirror for chosen winners.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing links under --stage-root.",
    )
    return parser.parse_args()


def find_mode(parts: Iterable[str]) -> str | None:
    for part in parts:
        if part in KNOWN_MODES:
            return part
    return None


def find_subchunk(parts: Iterable[str]) -> tuple[str, str, str] | None:
    for part in parts:
        match = SUBCHUNK_RE.fullmatch(part)
        if match:
            return match.group("tier"), match.group("chunk"), match.group("part")
    return None


def find_tier(parts: Iterable[str]) -> str | None:
    for part in parts:
        match = TIER_DIR_RE.fullmatch(part)
        if match:
            return match.group("tier")
    return None


def find_leaf(parts: Iterable[str]) -> tuple[str, str] | None:
    for part in parts:
        match = LEAF_RE.fullmatch(part)
        if match:
            return match.group("chunk"), match.group("part")
    return None


def discover_records(root_name: str, root: Path) -> list[Record]:
    records: list[Record] = []
    if not root.exists():
        return records

    for worker_summary in root.rglob("worker_summary.json"):
        source_dir = worker_summary.parent
        parts = source_dir.parts
        mode = find_mode(parts)
        if not mode:
            continue

        subchunk = find_subchunk(parts)
        if subchunk:
            tier, chunk, part = subchunk
        else:
            tier = find_tier(parts)
            leaf = find_leaf(parts)
            if not tier or not leaf:
                continue
            chunk, part = leaf

        canonical_rel = Path(f"tier{tier}") / mode / f"chunk{chunk}_part{part}"
        records.append(
            Record(
                source_root=root_name,
                source_dir=source_dir,
                worker_summary=worker_summary,
                mode=mode,
                tier=tier,
                chunk=chunk,
                part=part,
                canonical_rel=canonical_rel,
                mtime=worker_summary.stat().st_mtime,
            )
        )
    return records


def write_inventory(records: list[Record], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_root",
                "source_dir",
                "worker_summary",
                "mode",
                "tier",
                "chunk",
                "part",
                "canonical_rel",
                "mtime",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for record in sorted(records, key=lambda r: (str(r.canonical_rel), r.mtime)):
            writer.writerow(
                {
                    "source_root": record.source_root,
                    "source_dir": str(record.source_dir),
                    "worker_summary": str(record.worker_summary),
                    "mode": record.mode,
                    "tier": record.tier,
                    "chunk": record.chunk,
                    "part": record.part,
                    "canonical_rel": str(record.canonical_rel),
                    "mtime": f"{record.mtime:.6f}",
                }
            )


def choose_winners(records: list[Record]) -> list[Record]:
    winners: dict[str, Record] = {}
    for record in records:
        key = str(record.canonical_rel)
        current = winners.get(key)
        if current is None or record.mtime > current.mtime:
            winners[key] = record
    return [winners[key] for key in sorted(winners)]


def write_chosen(records: list[Record], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "canonical_rel",
                "source_root",
                "source_dir",
                "worker_summary",
                "mode",
                "tier",
                "chunk",
                "part",
                "mtime",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "canonical_rel": str(record.canonical_rel),
                    "source_root": record.source_root,
                    "source_dir": str(record.source_dir),
                    "worker_summary": str(record.worker_summary),
                    "mode": record.mode,
                    "tier": record.tier,
                    "chunk": record.chunk,
                    "part": record.part,
                    "mtime": f"{record.mtime:.6f}",
                }
            )


def stage_symlinks(records: list[Record], stage_root: Path, overwrite: bool) -> None:
    for record in records:
        destination = stage_root / record.canonical_rel
        destination.parent.mkdir(parents=True, exist_ok=True)

        if destination.exists() or destination.is_symlink():
            if not overwrite:
                continue
            if destination.is_symlink() or destination.is_file():
                destination.unlink()
            else:
                raise RuntimeError(f"Refusing to replace non-link directory: {destination}")

        destination.symlink_to(record.source_dir)


def main() -> None:
    args = parse_args()

    all_records: list[Record] = []
    all_records.extend(discover_records("paper_experiment", args.paper_root))
    all_records.extend(discover_records("paper_experiment_chunk2plus", args.chunk2plus_root))

    write_inventory(all_records, args.inventory_tsv)
    chosen = choose_winners(all_records)
    write_chosen(chosen, args.chosen_tsv)

    if args.link_mode == "symlink":
        if not args.stage_root:
            raise SystemExit("--stage-root is required when --link-mode=symlink")
        args.stage_root.mkdir(parents=True, exist_ok=True)
        stage_symlinks(chosen, args.stage_root, overwrite=args.overwrite)

    print(f"inventory_tsv={args.inventory_tsv}")
    print(f"chosen_tsv={args.chosen_tsv}")
    print(f"records={len(all_records)}")
    print(f"chosen={len(chosen)}")
    if args.link_mode == "symlink":
        print(f"stage_root={args.stage_root}")


if __name__ == "__main__":
    main()
