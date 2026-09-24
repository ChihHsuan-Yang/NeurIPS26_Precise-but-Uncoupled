#!/usr/bin/env bash
# TRACK A -- regenerate the POST-REBUTTAL robustness tables. No model calls.
#
# SCOPE WARNING: everything this script produces is post-rebuttal ROBUSTNESS
# evidence over a 5-dataset x 2-actor matrix. The PAPER is Omni-MATH only
# (N=4,181, ten tiers, gpt-oss-120b actor and evaluator). Do not present these
# tables as the paper's main result.
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
cd "$REPO_ROOT"

INPUT="${PU_TRANSITIONS:-$REPO_ROOT/data/derived/full_release_symmetric_transitions.csv}"
OUTDIR="$REPO_ROOT/results/derived_tables/_regenerated"

rule
say "reproduce-rebuttal  (TRACK A -- post-rebuttal robustness, no model calls)"
say "  input  : $INPUT"
say "  output : $OUTDIR"
rule
say "SCOPE: 5 datasets x 2 actor families. This is ROBUSTNESS evidence."
say "       The paper itself is Omni-MATH only (N=4,181)."
rule

[ -f "$INPUT" ] || die "transitions CSV not found: $INPUT  (see docs/DATA.md)"
mkdir -p "$OUTDIR"

say "[1/2] 2x5 symmetric audit (precision / strict uptake / verified repair)"
"$PYTHON" "$REPO_ROOT/src/precise_uncoupled/process/analyze_matrix_precise_uncoupled.py" \
  --input "$INPUT" \
  --output-csv "$OUTDIR/matrix_2x5_precise_uncoupled_strict.csv" \
  --output-md  "$OUTDIR/matrix_2x5_precise_uncoupled_strict.md" || die "audit failed"

say ""
say "[2/2] rendered table"
if [ -f "$OUTDIR/matrix_2x5_precise_uncoupled_strict.md" ]; then
  cat "$OUTDIR/matrix_2x5_precise_uncoupled_strict.md"
fi

rule
say "Interpretation guardrails, from the script's own docstring:"
say "  * uptake and repair use only STRICT PRE-GATE actor transitions;"
say "  * Broadcast responses after a system-selected candidate update are EXCLUDED,"
say "    so the approval gate is never counted as an immediate actor response;"
say "  * cells where either protocol has zero strict observations are flagged sparse"
say "    and must not be read as a comparison."
say ""
say "The JEEBench/Gemma-4 cell has only n=2 evaluable follow-ups. It is the one"
say "cell where within-PER precision does not exceed repair. Report it as such."
say ""
say "Do NOT pool Gemma 3 with Gemma 4: different model generations. The paper's"
say "second family is Gemma 3; this matrix's is Gemma 4."
rule
say "reproduce-rebuttal: done"
