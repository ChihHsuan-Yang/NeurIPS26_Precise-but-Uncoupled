#!/usr/bin/env bash
# TRACK A -- reproduce the paper's headline process table. NO MODEL CALLS.
#
# Chain:
#   data/derived/full_release_symmetric_transitions.csv   (pinned, sha 7441a770...)
#     -> src/precise_uncoupled/process/analyze_matrix_precise_uncoupled.py
#     -> the 2x5 precision / strict-uptake / verified-repair table
#
# The output must be BYTE-IDENTICAL to the released reference. This script ends
# by comparing sha256 and printing PASS or FAIL. Exit code follows the verdict.
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
cd "$REPO_ROOT"

EXPECTED_SHA="6c77c01e6a02889cbf69b77d3f33d0f8e5784a150d3d1b29d9311b06435daef2"
INPUT="${PU_TRANSITIONS:-$REPO_ROOT/data/derived/full_release_symmetric_transitions.csv}"
OUTDIR="$REPO_ROOT/results/derived_tables/_regenerated"
REFERENCE="$REPO_ROOT/results/derived_tables/matrix_2x5_precise_uncoupled_strict.csv"

rule
say "reproduce-analysis  (TRACK A -- offline, deterministic, no model calls)"
say "  input     : $INPUT"
say "  output    : $OUTDIR/matrix_2x5_precise_uncoupled_strict.csv"
say "  reference : $REFERENCE"
rule

[ -f "$INPUT" ] || die "transitions CSV not found: $INPUT
       It ships with the repository. If you removed it, restore it or set
       PU_TRANSITIONS. See docs/DATA.md."

say "verifying the INPUT before using it..."
INPUT_SHA="$(sha256_of "$INPUT")"
INPUT_EXPECTED="7441a770e590da04cb0accbb381d9a012e905e93b1d7120f2c04b1b53e21023c"
say "  input sha256 : $INPUT_SHA"
if [ "$INPUT_SHA" != "$INPUT_EXPECTED" ]; then
  say "  WARNING: input does not match the pinned digest $INPUT_EXPECTED"
  say "           A regenerated (rather than pinned) input is expected to differ"
  say "           slightly in bytes; see docs/REPRODUCIBILITY.md. Continuing."
else
  say "  -> matches the pinned digest"
fi
rule

mkdir -p "$OUTDIR"
"$PYTHON" "$REPO_ROOT/src/precise_uncoupled/process/analyze_matrix_precise_uncoupled.py" \
  --input "$INPUT" \
  --output-csv "$OUTDIR/matrix_2x5_precise_uncoupled_strict.csv" \
  --output-md  "$OUTDIR/matrix_2x5_precise_uncoupled_strict.md"
rc=$?
[ "$rc" -eq 0 ] || die "the analysis script exited $rc"

rule
GOT="$(sha256_of "$OUTDIR/matrix_2x5_precise_uncoupled_strict.csv")"
say "2x5 table sha256"
say "  expected : $EXPECTED_SHA"
say "  actual   : $GOT"
rule

if [ "$GOT" = "$EXPECTED_SHA" ]; then
  say "reproduce-analysis: PASS  (byte-identical to the released reference)"
  say ""
  say "What this table licenses you to say:"
  say "  * PER reviewer precision > Broadcast precision in 10/10 dataset x actor cells"
  say "  * within PER, precision > verified repair in 9/10 cells"
  say "    (exception JEEBench/Gemma-4, only n=2 follow-ups evaluable)"
  say "  * within Broadcast, precision > verified repair in 10/10 cells"
  say "  i.e. reviewer DETECTION quality and successful critique UPTAKE/REPAIR are"
  say "  empirically separable."
  say ""
  say "What it does NOT license:"
  say "  * any universal PER-vs-Broadcast ranking (Gemma-3 actors REVERSE it)"
  say "  * any causal claim about the approval gate"
  say "  * any claim that one protocol takes up critique less often in EVERY"
  say "    setting -- the symmetric audit measures 0/10 such cells"
  exit 0
fi

say "reproduce-analysis: FAIL  (output differs from the released reference)"
say ""
say "Diff of the rendered table rows:"
if [ -f "$REFERENCE" ]; then
  diff "$REFERENCE" "$OUTDIR/matrix_2x5_precise_uncoupled_strict.csv" | head -40
else
  say "  (reference file missing: $REFERENCE)"
fi
say ""
say "See docs/TROUBLESHOOTING.md."
exit 1
