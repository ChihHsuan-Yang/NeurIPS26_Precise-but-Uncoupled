#!/usr/bin/env python3
"""Pre-publication self-check of this repository.

Stricter than the test suite: it runs against a CLEAN tree and fails on build
artifacts that a test run would itself create. Intended to be run immediately
before publishing, and by anyone who wants to confirm what they downloaded.

    python3 scripts/validate_release.py          # human-readable
    python3 scripts/validate_release.py --json   # machine-readable

Exit code 0 = PASS, 1 = FAIL.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()
INTEGRITY_TEST = REPO_ROOT / "tests" / "test_release_integrity.py"

# Artifacts with a digest the release commits to.
PINNED = {
    "results/derived_tables/matrix_2x5_precise_uncoupled_strict.csv":
        "6c77c01e6a02889cbf69b77d3f33d0f8e5784a150d3d1b29d9311b06435daef2",
    "data/derived/full_release_symmetric_transitions.csv":
        "7441a770e590da04cb0accbb381d9a012e905e93b1d7120f2c04b1b53e21023c",
    "results/derived_tables/rq1_tier_mode_metrics.csv":
        "80c4eb0b5cccbc12f7e0c0dfc355f9bf2ec5b02251da6d4eca81100d54957d79",
    "results/derived_tables/rq2_tier_mode_metrics.csv":
        "cbec609fb51a441f455fb550eb39918961c3b2a2f4797660dbdfddd7b198b5da",
    "results/derived_tables/rq3_collaboration_tier_mode_metrics.csv":
        "a16157c7f2d1de6e18fe637252e17421e3ab8b9c61e58267ad8025c1063e77ba",
    "results/derived_tables/rq3_protocol_tier_mode_metrics.csv":
        "82ded7fdcbc96beffe7cca554500034d52617341581cdbffd7b2e5958ee86e87",
    "results/derived_tables/efficiency_tier_mode_metrics.csv":
        "0b3e4fa4d1e382f830c36ea1007d3fb6576aeab85f8a7e7c5ad15a19d2ad5fbf",
}

REQUIRED_PATHS = [
    "README.md", "LICENSE", "NOTICE", "CITATION.cff", "Makefile",
    "requirements.txt", "pyproject.toml",
    "configs/paper/per_hint_llm/config.yaml",
    "configs/paper/broadcast_hint_llm/config.yaml",
    "configs/paper/baseline_llm/config.yaml",
    "configs/paper/single_agent_hint_llm/config.yaml",
    "src/precise_uncoupled/__init__.py",
    "src/precise_uncoupled/process/analyze_matrix_precise_uncoupled.py",
    "src/agentverse/metrics/main_paper/__init__.py",
    "scripts/reproduce_main_results.sh", "scripts/smoke_live.sh",
    "scripts/fetch_data.sh", "scripts/verify_checksums.sh",
    "docs/REPRODUCIBILITY.md", "docs/PROVENANCE.md",
    "docs/PAPER_TO_ARTIFACT_MAP.md", "docs/LIMITATIONS.md",
    "data/manifests/release_artifacts.sha256",
]

SCAN_SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", "data", "figures", "logs"}
SCAN_SKIP_SUFFIX = {".csv", ".jsonl", ".parquet", ".pdf", ".png", ".svg", ".pyc"}

PATTERNS = {
    "private_absolute_path": r"/Users/|/home/|/lus/|/grand/",
    "credential":            (r"sk-[A-Za-z0-9]{16,}|ghp_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}"
                              r"|AKIA[0-9A-Z]{16}|xox[baprs]-[A-Za-z0-9-]{10,}"),
    "private_endpoint":      r"inference-api\.[A-Za-z0-9.-]+|/resource_server/",
    "retired_claim":         r"lower strict uptake in all settings",
    "withdrawn_cost_value":  r"\b(18,?385|48,?123|400,?351|402,?\d{3}|422,?\d{3}|557,?\d{3}|616,?499)\b",
}

# Strings each pattern MUST match. If a positive control fails, the scanner is
# broken and a clean result would be meaningless.
POSITIVE_CONTROLS = {
    "private_absolute_path": "/home/someone/work",
    "credential":            "sk-abcdefghijklmnopqrstuvw",
    "private_endpoint":      "https://inference-api.example.gov/resource_server/x/v1",
    "retired_claim":         "it has lower strict uptake in all settings",
    "withdrawn_cost_value":  "Avg. Tokens 400,351",
}


def iter_text_files():
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file():
            continue
        rp = path.resolve()
        # These two files legitimately contain the forbidden patterns as their
        # own positive controls; measuring them would be self-referential.
        if rp == SELF or rp == INTEGRITY_TEST.resolve():
            continue
        rel = path.relative_to(REPO_ROOT)
        if any(part in SCAN_SKIP_DIRS for part in rel.parts[:-1]):
            continue
        if path.suffix in SCAN_SKIP_SUFFIX:
            continue
        try:
            yield rel, path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="emit JSON")
    args = ap.parse_args()

    report = {"checks": [], "pass": True}

    def record(name, ok, detail=""):
        report["checks"].append({"check": name, "ok": bool(ok), "detail": detail})
        if not ok:
            report["pass"] = False

    # 0. the scanner must work before its results mean anything
    broken = [n for n, s in POSITIVE_CONTROLS.items() if not re.search(PATTERNS[n], s)]
    record("scanner_positive_controls", not broken,
           "patterns that failed their own control: %s" % broken if broken
           else "all %d patterns match their positive control" % len(PATTERNS))
    scanned = list(iter_text_files())
    record("scanner_reaches_files", len(scanned) > 50, "%d text files scanned" % len(scanned))

    # 1. required paths
    missing = [p for p in REQUIRED_PATHS if not (REPO_ROOT / p).exists()]
    record("required_paths_present", not missing, "missing: %s" % missing if missing else
           "all %d required paths present" % len(REQUIRED_PATHS))

    # 2. pinned digests
    bad = []
    for rel, expected in PINNED.items():
        p = REPO_ROOT / rel
        if not p.exists():
            bad.append("%s MISSING" % rel)
            continue
        got = hashlib.sha256(p.read_bytes()).hexdigest()
        if got != expected:
            bad.append("%s expected %s got %s" % (rel, expected[:16], got[:16]))
    record("pinned_digests", not bad, "; ".join(bad) if bad else
           "%d pinned artifacts match" % len(PINNED))

    # 3. content scan
    for name, pat in PATTERNS.items():
        rx = re.compile(pat)
        hits = []
        for rel, text in scanned:
            for i, line in enumerate(text.splitlines(), 1):
                if rx.search(line):
                    hits.append("%s:%d" % (rel, i))
        record("no_" + name, not hits, "%d hit(s): %s" % (len(hits), hits[:8]) if hits else "0 hits")

    # 4. build artifacts -- strict, no allowances
    arts = []
    for p in REPO_ROOT.rglob("*"):
        rel = p.relative_to(REPO_ROOT)
        if ".git" in rel.parts:
            continue
        n = p.name
        if n in {".DS_Store", "Thumbs.db"} or n.endswith(".pyc"):
            arts.append(str(rel))
        elif p.is_dir() and n in {"__pycache__", ".pytest_cache", ".ipynb_checkpoints",
                                  ".mypy_cache", ".venv", "venv", "logs"}:
            arts.append(str(rel))
        elif p.is_file() and n.endswith(".log"):
            arts.append(str(rel))
    record("no_build_artifacts", not arts, "%d found: %s" % (len(arts), arts[:8]) if arts else "clean")

    # 5. no nested .git copied in from a source tree
    nested = [str(p.relative_to(REPO_ROOT)) for p in REPO_ROOT.rglob(".git")
              if p.relative_to(REPO_ROOT) != Path(".git")]
    record("no_nested_git", not nested, "found: %s" % nested if nested else "none")

    # 6. no reader-run output masquerading as reference data
    live = (REPO_ROOT / "results" / "live").exists()
    regen = [str(p.relative_to(REPO_ROOT)) for p in (REPO_ROOT / "results").rglob("_regenerated")]
    record("no_run_output_shipped", not live and not regen,
           "results/live=%s regenerated=%s" % (live, regen))

    # 7. single definition of the analysis modules
    dup = (REPO_ROOT / "src" / "precise_uncoupled" / "analysis" / "main_paper").exists()
    record("single_analysis_definition", not dup,
           "a duplicate analysis tree exists" if dup else
           "analysis is a re-export facade over agentverse.metrics")

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        width = max(len(c["check"]) for c in report["checks"])
        print("=" * 70)
        print("validate_release.py --", REPO_ROOT.name)
        print("=" * 70)
        for c in report["checks"]:
            print("  %-4s %-*s  %s" % ("PASS" if c["ok"] else "FAIL", width, c["check"], c["detail"]))
        print("=" * 70)
        print("RESULT:", "PASS" if report["pass"] else "FAIL")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
