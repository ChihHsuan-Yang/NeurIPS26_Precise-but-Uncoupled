"""Track A: the 2x5 symmetric transition pipeline.

Three stages, all offline, no model calls:

  1. aggregate_release_transitions.py   HF parquet traces -> full_release_symmetric_transitions.csv
  2. symmetric_transition_extractor.py  the transition labeller applied by stage 1
  3. analyze_matrix_precise_uncoupled.py  transitions CSV -> the 2x5 precision/uptake/repair table

These four files are verbatim recoveries from the post-rebuttal analysis branch,
with ONE documented edit: `aggregate_release_transitions.py`'s default release
root and output dir are now repository-relative and env-overridable (upstream they
pointed at an author-private cache). See docs/PROVENANCE.md for the sha256 of each
file as recovered.

The modules are run as standalone scripts (they import each other by bare module
name, as they did upstream), so run them with
`PYTHONPATH=src/precise_uncoupled/process` or via scripts/reproduce_main_results.sh.
"""
