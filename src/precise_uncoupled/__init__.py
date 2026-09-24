"""Precise but Uncoupled -- paper-facing analysis, protocol and evaluation code.

This package is the reader-facing surface of the release:

  precise_uncoupled.process   the 2x5 symmetric transition pipeline (Track A)
  precise_uncoupled.analysis  the paper's trace-metric modules
  precise_uncoupled.protocols the four protocol runtimes (Track B)
  precise_uncoupled.evaluation the evaluator/verifier contract
  precise_uncoupled.io        trace parsing and dataset loading
  precise_uncoupled.scripts   report-building and bootstrap scripts
  precise_uncoupled.jobs      shell drivers for batch re-analysis

`analysis`, `protocols`, `evaluation` and `io` are thin re-export facades over the
vendored `agentverse` runtime in `src/agentverse/`, so that a reader can import
the paper's functions by paper-facing names while the underlying modules remain
BYTE-IDENTICAL to the code that produced the submitted results. Nothing here
redefines a metric; see docs/PROVENANCE.md for the per-file blob verification.
"""

__version__ = "1.0.0"
