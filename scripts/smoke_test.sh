#!/usr/bin/env bash
# TRACK A smoke test -- fast, offline, no model calls.
# Proves the environment, imports and the analysis chain work before you commit
# to anything larger. Runs in a few seconds.
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
cd "$REPO_ROOT"

fail=0
step() { printf '\n[%s] %s\n' "$1" "$2"; }

rule; say "smoke-test  (TRACK A -- offline, no model calls)"; rule

step 1/5 "python version"
"$PYTHON" -c 'import sys; print("  ", sys.version.split()[0])' || fail=1

step 2/5 "required third-party packages"
"$PYTHON" - <<'PYEOF' || fail=1
import importlib, sys
need = ["pandas", "yaml"]
missing = []
for m in need:
    try:
        mod = importlib.import_module(m)
        print("   ok  %-10s %s" % (m, getattr(mod, "__version__", "")))
    except ImportError:
        missing.append(m); print("   MISSING", m)
if missing:
    print("   -> pip install -r requirements.txt", file=sys.stderr); sys.exit(1)
PYEOF

step 3/5 "paper analysis modules import"
"$PYTHON" - <<'PYEOF' || fail=1
mods = [
    "precise_uncoupled",
    "precise_uncoupled.analysis",
    "precise_uncoupled.io",
    "precise_uncoupled.evaluation",
    "agentverse_command.rq1_outcome_cost",
    "agentverse_command.rq2_recovery",
    "agentverse_command.rq3_protocol_refinement",
    "agentverse_command.collaboration_decomposition",
    "agentverse_command.strict_coupling_rate",
]
import importlib
for m in mods:
    importlib.import_module(m)
    print("   ok ", m)
PYEOF

step 4/5 "released data is present and intact"
bash "$REPO_ROOT/scripts/verify_checksums.sh" >/dev/null 2>&1 \
  && say "   ok  all manifest entries verified" \
  || { say "   FAIL  run scripts/verify_checksums.sh for detail"; fail=1; }

step 5/5 "unit tests"
"$PYTHON" -m pytest "$REPO_ROOT/tests" -q 2>&1 | tail -3 || fail=1

rule
if [ "$fail" -eq 0 ]; then
  say "smoke-test: PASS"
  say "Next:  make reproduce-analysis   (rebuilds the 2x5 table and checks its sha256)"
  exit 0
fi
say "smoke-test: FAIL -- see docs/TROUBLESHOOTING.md"
exit 1
