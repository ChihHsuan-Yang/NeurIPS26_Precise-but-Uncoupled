#!/usr/bin/env bash
# Validate every released artifact against data/manifests/*.sha256.
#
# This is a REAL check: it recomputes each digest and compares. A missing file is
# a FAILURE, not a skip -- an absent file must never read as a pass.
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
cd "$REPO_ROOT"

rule
say "verify-checksums"
rule

total=0; ok=0; bad=0; missing=0

for manifest in data/manifests/*.sha256; do
  [ -f "$manifest" ] || continue
  say "manifest: $manifest"
  while read -r expected path; do
    [ -z "${expected:-}" ] && continue
    case "$expected" in \#*) continue;; esac
    total=$((total+1))
    if [ ! -f "$path" ]; then
      printf '  MISSING  %s\n' "$path"
      missing=$((missing+1))
      continue
    fi
    actual="$(sha256_of "$path")"
    if [ "$actual" = "$expected" ]; then
      ok=$((ok+1))
    else
      printf '  MISMATCH %s\n' "$path"
      printf '           expected %s\n' "$expected"
      printf '           actual   %s\n' "$actual"
      bad=$((bad+1))
    fi
  done < "$manifest"
done

# The single most load-bearing artifact gets named explicitly.
HEADLINE="results/derived_tables/matrix_2x5_precise_uncoupled_strict.csv"
HEADLINE_SHA="6c77c01e6a02889cbf69b77d3f33d0f8e5784a150d3d1b29d9311b06435daef2"
rule
if [ -f "$HEADLINE" ]; then
  got="$(sha256_of "$HEADLINE")"
  say "headline 2x5 table"
  say "  expected $HEADLINE_SHA"
  say "  actual   $got"
  if [ "$got" != "$HEADLINE_SHA" ]; then
    say "  -> MISMATCH"
    bad=$((bad+1))
  else
    say "  -> match"
  fi
else
  say "headline 2x5 table: MISSING ($HEADLINE)"
  missing=$((missing+1))
fi

rule
printf 'checked %d   ok %d   mismatch %d   missing %d\n' "$total" "$ok" "$bad" "$missing"
if [ "$bad" -eq 0 ] && [ "$missing" -eq 0 ] && [ "$total" -gt 0 ]; then
  say "verify-checksums: PASS"
  exit 0
fi
if [ "$total" -eq 0 ]; then
  say "verify-checksums: FAIL (no manifest entries were read -- the check was vacuous)"
  exit 1
fi
say "verify-checksums: FAIL"
exit 1
