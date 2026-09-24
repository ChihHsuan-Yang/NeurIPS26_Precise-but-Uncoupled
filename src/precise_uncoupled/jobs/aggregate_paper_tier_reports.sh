#!/usr/bin/env bash
# Aggregate per-chunk report folders into tier-level reports.
#
# RELEASE PATCH (NeurIPS26_Precise-but-Uncoupled).
# Upstream this script began with
#     source "$HOME/miniforge3/etc/profile.d/conda.sh"; conda activate agentverse_env
#     export REPO_ROOT="${REPO_ROOT:-<an absolute path on the authors' cluster>}"
#     RESULT_ROOT="${RESULT_ROOT:-$REPO_ROOT/results/paper_experiment}"
# All three were site-specific: the conda bootstrap assumed one machine's
# miniforge layout, REPO_ROOT was an absolute private path, and
# `results/paper_experiment` was later renamed and no longer exists.
# This version derives the repository root from the script's own location and
# takes the data root from PU_TRACE_ROOT. Activate whatever environment you
# like before running it (see docs/REPRODUCIBILITY.md).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$HERE/../../.." && pwd)}"
cd "$REPO_ROOT"

RESULT_ROOT="${RESULT_ROOT:-${PU_TRACE_ROOT:-$REPO_ROOT/data/traces/gpt_oss_120b}}"
TIERS="${TIERS:-01}"
MODE_DIR="${MODE_DIR:-broadcast}"
PYTHON="${PYTHON:-python}"

echo "[aggregate-report] REPO_ROOT=$REPO_ROOT"
echo "[aggregate-report] RESULT_ROOT=$RESULT_ROOT"
echo "[aggregate-report] TIERS=$TIERS"
echo "[aggregate-report] MODE_DIR=$MODE_DIR"

if [ ! -d "$RESULT_ROOT" ]; then
  echo "ERROR: trace root not found: $RESULT_ROOT" >&2
  echo "       Download the released traces first (see docs/DATA.md), or set" >&2
  echo "       PU_TRACE_ROOT / RESULT_ROOT to where they live." >&2
  exit 2
fi

PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT/src/precise_uncoupled${PYTHONPATH:+:$PYTHONPATH}" \
"$PYTHON" "$REPO_ROOT/src/precise_uncoupled/scripts/aggregate_chunk_reports.py" \
  --result-root "$RESULT_ROOT" \
  --tiers $TIERS \
  --mode-dir "$MODE_DIR"
