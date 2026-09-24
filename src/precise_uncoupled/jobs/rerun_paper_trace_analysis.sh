#!/usr/bin/env bash
# Re-run the four RQ analyses over saved per-chunk trace files.
#
# This is Track A: it reads SAVED traces and makes no model calls.
#
# RELEASE PATCH (NeurIPS26_Precise-but-Uncoupled).
# Upstream (as jobs/rerun_paper_broadcast_trace_analysis.sh) this script began with
#     source "$HOME/miniforge3/etc/profile.d/conda.sh"; conda activate agentverse_env
#     export REPO_ROOT="${REPO_ROOT:-<an absolute path on the authors' cluster>}"
#     RESULT_ROOT="${RESULT_ROOT:-$REPO_ROOT/results/paper_experiment}"
#     source "$REPO_ROOT/use_alcf.sh"
# Four site-specific dependencies, all removed here:
#   * the conda bootstrap assumed one machine's miniforge layout;
#   * REPO_ROOT was an absolute private path;
#   * `results/paper_experiment` was later renamed and no longer exists;
#   * use_alcf.sh set a private ALCF endpoint + a Globus token, and is NOT part
#     of this release. These analyses read saved traces, so no endpoint is needed.
# The script now derives the repository root from its own location and takes the
# data root from PU_TRACE_ROOT. See docs/REPRODUCIBILITY.md.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$HERE/../../.." && pwd)}"
cd "$REPO_ROOT"

RESULT_ROOT="${RESULT_ROOT:-${PU_TRACE_ROOT:-$REPO_ROOT/data/traces/gpt_oss_120b}}"
TIERS="${TIERS:-08 07}"
MODE_DIR="${MODE_DIR:-broadcast}"
PYTHON="${PYTHON:-python}"
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT/src/precise_uncoupled${PYTHONPATH:+:$PYTHONPATH}"

echo "Re-running trace analysis (Track A -- saved traces, no model calls)"
echo "  REPO_ROOT=$REPO_ROOT"
echo "  RESULT_ROOT=$RESULT_ROOT"
echo "  TIERS=$TIERS"
echo "  MODE_DIR=$MODE_DIR"

if [ ! -d "$RESULT_ROOT" ]; then
  echo "ERROR: trace root not found: $RESULT_ROOT" >&2
  echo "       Download the released traces first (see docs/DATA.md), or set" >&2
  echo "       PU_TRACE_ROOT / RESULT_ROOT to where they live." >&2
  exit 2
fi

for TIER_PAD in $TIERS; do
  BASE_DIR="$RESULT_ROOT/tier${TIER_PAD}/${MODE_DIR}"
  echo
  echo "=== tier${TIER_PAD} / ${MODE_DIR} ==="

  for IDX in 1 2 3 4 5 6 7 8 9 10; do
    CHUNK_DIR="$BASE_DIR/chunck1_${IDX}"
    if [ ! -d "$CHUNK_DIR" ]; then
      echo "skip missing: $CHUNK_DIR"
      continue
    fi

    TRACE=$(ls -t "$CHUNK_DIR"/*.trace.txt 2>/dev/null | head -n 1 || true)
    if [ -z "$TRACE" ]; then
      echo "skip no trace: $CHUNK_DIR"
      continue
    fi

    echo "[ANALYZE] tier${TIER_PAD} chunck1_${IDX}"
    echo "          trace=$TRACE"

    "$PYTHON" -m agentverse_command.rq1_outcome_cost \
      --input_path "$TRACE" --output_dir "$CHUNK_DIR/rq1-outcome-cost" --overwrite

    "$PYTHON" -m agentverse_command.rq2_recovery \
      --input_path "$TRACE" --output_dir "$CHUNK_DIR/rq2-recovery" --overwrite

    "$PYTHON" -m agentverse_command.rq3_protocol_refinement \
      --input_path "$TRACE" --output_dir "$CHUNK_DIR/rq3-protocol-refinement" --overwrite

    "$PYTHON" -m agentverse_command.collaboration_decomposition \
      --input_path "$TRACE" --output_dir "$CHUNK_DIR/rq3-collaboration-decomposition" --overwrite
  done
done

echo
echo "Trace analysis rerun complete."
