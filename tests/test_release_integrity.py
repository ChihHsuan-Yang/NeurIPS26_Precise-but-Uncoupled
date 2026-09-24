"""Guards on the RELEASE ITSELF, not on the science.

Each test here corresponds to a defect that was actually present in the source
material and fixed during packaging. They exist so the defect cannot silently
return. See docs/PROVENANCE.md.
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"

SKIP_DIR_NAMES = {".git", "__pycache__", ".pytest_cache", "data", "figures", "logs"}

#: These two files necessarily CONTAIN the forbidden patterns -- they are the
#: positive controls of this scanner and of scripts/validate_release.py (see the
#: *_not_vacuous tests). A scanner that measured the files defining its own
#: patterns would be wrong on arrival, so both are excluded. Everything else,
#: including every other file under tests/ and scripts/, is still scanned.
#: The exclusion is by RESOLVED PATH, not by directory, so it cannot be widened
#: accidentally: dropping a new file into scripts/ does not exempt it.
SELF = Path(__file__).resolve()
VALIDATOR = (REPO_ROOT / "scripts" / "validate_release.py").resolve()
SCANNER_SELVES = {SELF, VALIDATOR}


def iter_text_files():
    """Every text file in the release, excluding data blobs, caches and self."""
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if path.resolve() in SCANNER_SELVES:
            continue
        if any(part in SKIP_DIR_NAMES for part in path.relative_to(REPO_ROOT).parts[:-1]):
            continue
        if path.suffix in {".csv", ".jsonl", ".parquet", ".pdf", ".png", ".svg", ".pyc"}:
            continue
        try:
            yield path, path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue


class TestNoPrivatePaths(unittest.TestCase):
    """AGENT_CONSTRAINTS section 6: no private absolute paths or credentials."""

    def test_no_private_absolute_paths(self):
        pattern = re.compile(r"/Users/|/home/|/lus/|/grand/")
        hits = []
        for path, text in iter_text_files():
            for i, line in enumerate(text.splitlines(), 1):
                if pattern.search(line):
                    hits.append("%s:%d: %s" % (path.relative_to(REPO_ROOT), i, line.strip()[:120]))
        self.assertEqual(hits, [], "private absolute paths found:\n" + "\n".join(hits))

    def test_no_credentials(self):
        pattern = re.compile(
            r"sk-[A-Za-z0-9]{16,}|ghp_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}"
            r"|AKIA[0-9A-Z]{16}|xox[baprs]-[A-Za-z0-9-]{10,}"
        )
        hits = []
        for path, text in iter_text_files():
            for i, line in enumerate(text.splitlines(), 1):
                if pattern.search(line):
                    hits.append("%s:%d" % (path.relative_to(REPO_ROOT), i))
        self.assertEqual(hits, [], "credential-shaped strings found: %s" % hits)

    def test_no_resolvable_private_endpoint(self):
        """Facility NAMES are permitted (orchestrator ruling R1); resolvable
        private endpoint URLs and gateway URL paths are not."""
        pattern = re.compile(r"inference-api\.[A-Za-z0-9.-]+|/resource_server/")
        hits = []
        for path, text in iter_text_files():
            for i, line in enumerate(text.splitlines(), 1):
                if pattern.search(line):
                    hits.append("%s:%d: %s" % (path.relative_to(REPO_ROOT), i, line.strip()[:120]))
        self.assertEqual(hits, [], "private endpoint references found:\n" + "\n".join(hits))

    def test_exemptions_are_exactly_two_files(self):
        """Guard the guard: only the two scanner files are exempt.

        A plant in any OTHER file under scripts/ or tests/ must still be caught.
        """
        self.assertEqual(len(SCANNER_SELVES), 2)
        # iter_text_files yields ABSOLUTE paths; compare as paths, not strings.
        scanned = {p.resolve() for p, _ in iter_text_files()}
        # sibling files in the same two directories ARE still scanned
        self.assertIn((REPO_ROOT / "scripts" / "reproduce_main_results.sh").resolve(), scanned)
        self.assertIn((REPO_ROOT / "tests" / "conftest.py").resolve(), scanned)
        # only the two scanner files are exempt
        self.assertNotIn(VALIDATOR, scanned)
        self.assertNotIn(SELF, scanned)

    def test_scanner_is_not_vacuous(self):
        """Positive control: the patterns above must match a known-bad string.

        Without this, a broken regex would make the three tests above pass by
        matching nothing at all.
        """
        self.assertTrue(re.search(r"/Users/|/home/|/lus/|/grand/", "/home/someone/x"))
        self.assertTrue(re.search(r"sk-[A-Za-z0-9]{16,}", "key: sk-abcdefghijklmnopqrstuvwxyz"))
        self.assertTrue(re.search(r"inference-api\.[A-Za-z0-9.-]+", "https://inference-api.example.gov/v1"))
        self.assertTrue(re.search(r"/resource_server/", "https://h/resource_server/x/v1"))
        # and the walker must actually be reaching files
        self.assertGreater(sum(1 for _ in iter_text_files()), 50)


class TestNoBuildArtifacts(unittest.TestCase):
    """Artifacts that must never be in the tree.

    NOTE: __pycache__, .pytest_cache and logs/ are deliberately NOT checked here.
    Running this very suite creates them, so asserting their absence from inside
    the suite tests the test runner, not the release. They are covered by
    scripts/validate_release.py, which runs against a clean tree before publish,
    and by .gitignore. This test covers what a test run does NOT produce.
    """

    def test_no_stray_artifacts(self):
        bad = []
        generated_by_running_tests = {"__pycache__", ".pytest_cache", "logs"}
        for path in REPO_ROOT.rglob("*"):
            rel = path.relative_to(REPO_ROOT)
            if ".git" in rel.parts or generated_by_running_tests & set(rel.parts):
                continue
            name = path.name
            if name in {".DS_Store", "Thumbs.db"}:
                bad.append(str(rel))
            elif path.is_dir() and name in {".ipynb_checkpoints", ".venv", "venv", ".mypy_cache"}:
                bad.append(str(rel))
            elif path.is_file() and name.endswith(".log"):
                bad.append(str(rel))
        self.assertEqual(bad, [], "stray artifacts present: %s" % bad)

    def test_no_nested_git_repo(self):
        nested = [str(p.relative_to(REPO_ROOT)) for p in REPO_ROOT.rglob(".git")
                  if p.relative_to(REPO_ROOT) != Path(".git")]
        self.assertEqual(nested, [], "a .git directory was copied from a source tree: %s" % nested)

    def test_no_reader_run_output_shipped(self):
        """A live run must never leave outputs that could be read as reference data."""
        self.assertFalse((REPO_ROOT / "results" / "live").exists(),
                         "results/live/ contains one run's output; it must not ship")
        stray = list((REPO_ROOT / "results").rglob("_regenerated"))
        self.assertEqual(stray, [], "regenerated outputs present: %s" % stray)


class TestSingleDefinition(unittest.TestCase):
    """The duplicate-module hazard: code/coupling/ duplicated agentverse/metrics/.

    The release must carry exactly ONE definition of each analysis function.
    """

    def test_no_duplicate_analysis_tree(self):
        self.assertFalse(
            (SRC / "precise_uncoupled" / "analysis" / "main_paper").exists(),
            "a second copy of the analysis modules is present; the release must "
            "re-export from agentverse.metrics, not duplicate it",
        )

    def test_analysis_facade_reexports_only(self):
        """precise_uncoupled.analysis must not define metric functions itself."""
        text = (SRC / "precise_uncoupled" / "analysis" / "__init__.py").read_text()
        self.assertNotIn("\ndef summarize", text)
        self.assertIn("from agentverse.metrics", text)


class TestImportability(unittest.TestCase):
    """The tree must run standalone. As extracted it could not: code/coupling/
    imported agentverse.metrics.* which was not shipped alongside it."""

    ENTRY_POINTS = [
        "precise_uncoupled",
        "precise_uncoupled.analysis",
        "precise_uncoupled.io",
        "precise_uncoupled.evaluation",
        "agentverse_command.rq1_outcome_cost",
        "agentverse_command.rq2_recovery",
        "agentverse_command.rq3_protocol_refinement",
        "agentverse_command.collaboration_decomposition",
        "agentverse_command.strict_coupling_rate",
        "agentverse_command.trace_diagnostics",
    ]

    def test_entry_points_import(self):
        import importlib
        if str(SRC) not in sys.path:
            sys.path.insert(0, str(SRC))
        for name in self.ENTRY_POINTS:
            with self.subTest(module=name):
                importlib.import_module(name)


class TestNoImportSideEffects(unittest.TestCase):
    """Importing the analysis code must not write into the source tree.

    Upstream, agentverse/logging.py created <package>/../logs/ and
    agentverse/llms/openai.py opened api_rate_monitor.log in the CWD, both at
    import time -- so merely importing dirtied a clean checkout.
    """

    def test_import_creates_no_files_in_tree(self):
        import tempfile
        before = {p for p in REPO_ROOT.rglob("*") if p.is_file()}
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [sys.executable, "-c", "import agentverse_command.rq1_outcome_cost"],
                cwd=tmp,
                env={**os.environ, "PYTHONPATH": str(SRC), "PYTHONDONTWRITEBYTECODE": "1"},
                capture_output=True,
                check=True,
            )
        after = {p for p in REPO_ROOT.rglob("*") if p.is_file()}
        new = sorted(str(p.relative_to(REPO_ROOT)) for p in after - before)
        self.assertEqual(new, [], "import wrote files into the release tree: %s" % new)


class TestPinnedArtifacts(unittest.TestCase):
    """The two load-bearing digests, checked directly rather than via a manifest."""

    def test_2x5_table_digest(self):
        p = REPO_ROOT / "results" / "derived_tables" / "matrix_2x5_precise_uncoupled_strict.csv"
        self.assertTrue(p.exists(), "released 2x5 table is missing")
        self.assertEqual(
            hashlib.sha256(p.read_bytes()).hexdigest(),
            "6c77c01e6a02889cbf69b77d3f33d0f8e5784a150d3d1b29d9311b06435daef2",
        )

    def test_transitions_input_digest(self):
        p = REPO_ROOT / "data" / "derived" / "full_release_symmetric_transitions.csv"
        self.assertTrue(p.exists(), "pinned transitions CSV is missing")
        self.assertEqual(
            hashlib.sha256(p.read_bytes()).hexdigest(),
            "7441a770e590da04cb0accbb381d9a012e905e93b1d7120f2c04b1b53e21023c",
        )

    def test_omni_math_corpus_size(self):
        """N=4,181 over ten tiers is the paper's scope; a changed corpus would
        silently change every table."""
        import json
        from collections import Counter
        p = REPO_ROOT / "data" / "omni-math-2-filtered" / "all.jsonl"
        rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
        self.assertEqual(len(rows), 4181)
        counts = Counter(r["difficulty_tier"] for r in rows)
        self.assertEqual(
            {k: counts[k] for k in sorted(counts)},
            {1: 116, 2: 517, 3: 20, 4: 1106, 5: 1078, 6: 497, 7: 358, 8: 328, 9: 146, 10: 15},
        )


class TestForbiddenClaims(unittest.TestCase):
    """Claims the release must never make (AGENT_CONSTRAINTS section 5)."""

    RETIRED = re.compile(r"lower strict uptake in all settings", re.I)
    # legacy absolute token/call values withdrawn by P0E
    LEGACY_COST = re.compile(r"\b(18,?385|48,?123|400,?351|402,?\d{3}|422,?\d{3}|557,?\d{3}|616,?499)\b")

    def test_retired_claim_absent(self):
        hits = [str(p.relative_to(REPO_ROOT)) for p, t in iter_text_files() if self.RETIRED.search(t)]
        self.assertEqual(hits, [], "the retired uptake claim reappeared in: %s" % hits)

    def test_no_legacy_cost_values(self):
        hits = []
        for p, t in iter_text_files():
            for i, line in enumerate(t.splitlines(), 1):
                if self.LEGACY_COST.search(line):
                    hits.append("%s:%d: %s" % (p.relative_to(REPO_ROOT), i, line.strip()[:110]))
        self.assertEqual(hits, [], "withdrawn absolute token/call values present:\n" + "\n".join(hits))

    def test_claim_patterns_are_not_vacuous(self):
        self.assertTrue(self.RETIRED.search("PER has lower strict uptake in all settings"))
        self.assertTrue(self.LEGACY_COST.search("Avg. Tokens 400,351"))
        self.assertTrue(self.LEGACY_COST.search("18385 tokens"))


if __name__ == "__main__":
    unittest.main()
