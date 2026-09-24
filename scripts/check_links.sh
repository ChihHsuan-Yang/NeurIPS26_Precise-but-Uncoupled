#!/usr/bin/env bash
# Re-verify every URL in docs/RELEASE_ARTIFACTS.md.
# Anonymous requests only: this checks what a stranger sees, not what you see
# while logged in. Exit 0 if all reachable, 1 otherwise.
# Portable to bash 3.2 (macOS default) — no mapfile, no associative arrays.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DOC="$ROOT/docs/RELEASE_ARTIFACTS.md"
[ -f "$DOC" ] || { echo "missing $DOC" >&2; exit 1; }

URLFILE="$(mktemp)"; trap 'rm -f "$URLFILE"' EXIT
grep -oE 'https://[^ )|`<>]+' "$DOC" | sed 's/[.,]$//' | sort -u > "$URLFILE"
total=$(wc -l < "$URLFILE" | tr -d ' ')
echo "checking $total unique URLs from docs/RELEASE_ARTIFACTS.md"
echo

# Positive control: a URL that must resolve. If this fails, the network or the
# checker is broken and every other result is meaningless rather than clean.
ctl=$(curl -s -o /dev/null -w '%{http_code}' -L --max-time 25 "https://arxiv.org/abs/2607.15388")
if [ "$ctl" != "200" ]; then
  echo "POSITIVE CONTROL FAILED (arXiv returned $ctl) — results are not trustworthy." >&2
  exit 1
fi
echo "positive control: arXiv 200 — checker is live"
echo

fail=0
while IFS= read -r u; do
  [ -n "$u" ] || continue
  code=$(curl -s -o /dev/null -w '%{http_code}' -L --max-time 25 "$u")
  case "$code" in
    200|301|302) printf '  %-4s %s\n' "$code" "$u" ;;
    *)           printf '  %-4s %s   <-- UNREACHABLE\n' "$code" "$u"; fail=1 ;;
  esac
done < "$URLFILE"

echo
if [ "$fail" -eq 0 ]; then echo "all links reachable"; else echo "one or more links unreachable"; fi
exit "$fail"
