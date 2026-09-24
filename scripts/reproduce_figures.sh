#!/usr/bin/env bash
# TRACK A -- assemble the per-tier data behind the paper's figures. No model calls.
#
# READ THIS FIRST -- what this script does and does not do.
#
# It exports the tier x protocol tables that the paper's figures are drawn FROM.
# It does NOT re-render the published PDFs. Two honest reasons:
#
#  1. The figure-plotting script that emits all 20 figures is a 1,539-line variant
#     that lives outside the analysis repository; the copy bundled with the paper
#     source emits only 14 of the 20. Shipping the short one would silently fail
#     to produce 6 figures, including main-paper Figure 2.
#  2. Even the full variant is a NEAR match, not a byte-exact regenerator: four of
#     its own PDFs differ in byte size from the published ones.
#
# So the release publishes the figure INPUT DATA, which is exactly reproducible
# and checksummed, rather than a rendering pipeline that would quietly disagree
# with the paper. See docs/PAPER_TO_ARTIFACT_MAP.md for the per-figure status.
#
# KNOWN GAP, stated rather than hidden: the coupling panel of main-paper Figure 4
# shows CouplingRate 0.175 (ACK) / 0.227 (EMB). Those two values are NOT
# reconstructable from any released aggregate -- they do not appear in the saved
# CSVs at all. Appendix Table 14's 0.200 / 0.247 ARE reproducible and are the
# values this release quotes. Do not average the two, and do not silently pick one.
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
cd "$REPO_ROOT"

SRC_TABLES="$REPO_ROOT/results/derived_tables"
OUT="$REPO_ROOT/results/figure_data"

rule
say "reproduce-figures  (TRACK A -- figure INPUT data, no model calls)"
rule

mkdir -p "$OUT"
missing=0
for f in rq1_tier_mode_metrics.csv rq2_tier_mode_metrics.csv \
         rq3_protocol_tier_mode_metrics.csv rq3_collaboration_tier_mode_metrics.csv \
         efficiency_tier_mode_metrics.csv; do
  if [ -f "$SRC_TABLES/$f" ]; then
    cp "$SRC_TABLES/$f" "$OUT/$f"
    say "  ok      $f  ($(sha256_of "$OUT/$f" | cut -c1-16)...)"
  else
    say "  MISSING $f"
    missing=1
  fi
done

rule
say "Figure -> data mapping (see docs/PAPER_TO_ARTIFACT_MAP.md for the full table):"
say "  Fig 2  collaboration gain by tier   <- rq1_tier_mode_metrics.csv"
say "  Fig 3  precision / coupling / repair <- rq3_collaboration_tier_mode_metrics.csv"
say "  Fig 4  within-PER interventions      <- rq3_protocol_tier_mode_metrics.csv"
say "         (coupling panel NOT reproducible -- see the header of this script)"
say "  Fig 7  outer-loop recovery           <- rq2_tier_mode_metrics.csv"
say "  Fig 17 evaluator call counts         <- efficiency_tier_mode_metrics.csv"
say ""
say "NOT reproducible and NOT to be republished: Figures 14, 15, 16 and one panel"
say "of Figure 19 are built on the withdrawn token-cost fields. See docs/LIMITATIONS.md."
rule
if [ "$missing" -eq 0 ]; then
  say "reproduce-figures: done -- data in results/figure_data/"
  exit 0
fi
say "reproduce-figures: FAILED -- some source tables are missing"
exit 1
