#!/usr/bin/env bash
# TRACK B -- a tiny LIVE run against YOUR OpenAI-compatible endpoint.
#
# This CALLS A MODEL and COSTS MONEY/COMPUTE. It does not reproduce the paper's
# numbers and is not supposed to: see the variability warning printed below.
#
# Required environment:
#   OPENAI_BASE_URL   your OpenAI-compatible endpoint, e.g. https://host/v1
#   OPENAI_API_KEY    your key for that endpoint
# Optional:
#   PU_SMOKE_TASK     config dir (default configs/paper/per_hint_llm)
#   PU_SMOKE_DATA     jsonl (default data/example/omni_math_example_3.jsonl)
#
# No PBS, no Slurm, no ALCF, no facility login is required or used.
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
cd "$REPO_ROOT"

TASK="${PU_SMOKE_TASK:-configs/paper/per_hint_llm}"
DATA="${PU_SMOKE_DATA:-data/example/omni_math_example_3.jsonl}"
OUT="${PU_SMOKE_OUT:-results/live/smoke}"

rule
say "smoke-live  (TRACK B -- calls YOUR model endpoint)"
rule
say ""
say "  *** OUTPUTS WILL VARY BETWEEN RUNS. THIS IS EXPECTED. ***"
say ""
say "  The paper ran at temperature 0, but a hosted OpenAI-compatible endpoint"
say "  exposes NO SEED and its runtime is not frozen, so two executions of the"
say "  same config are independent. Measured envelope (P1A, frozen N=195 subset,"
say "  two independent executions on the same backend):"
say ""
say "      PER       FinalPass  range 0.51 pp   (SD 0.36 pp)"
say "      Broadcast FinalPass  range 2.05 pp   (SD 1.45 pp)"
say ""
say "  Run-to-run variation is SMALLER than the Broadcast-PER effect (+4.10 to"
say "  +6.67 pp), which is why the effect survives; but a single 3-problem smoke"
say "  run tells you NOTHING about accuracy. Its purpose is to prove your"
say "  endpoint, config and parsing work end to end."
say ""
say "  These protocol runtimes are vendored at AgentVerse HEAD b4a2db6, which"
say "  POST-DATES the submitted run's base be9c47c9. A live re-run reproduces the"
say "  experiment's DESIGN, not the submitted run's outputs. See docs/PROVENANCE.md."
say ""
rule

missing=0
if [ -z "${OPENAI_BASE_URL:-}" ]; then
  say "MISSING: OPENAI_BASE_URL is not set."
  missing=1
fi
if [ -z "${OPENAI_API_KEY:-}" ]; then
  say "MISSING: OPENAI_API_KEY is not set."
  missing=1
fi
if [ "$missing" -ne 0 ]; then
  say ""
  say "Point the code at your own endpoint, for example:"
  say '    export OPENAI_BASE_URL="https://your-endpoint.example/v1"'
  say '    export OPENAI_API_KEY="sk-..."'
  say '    make smoke-live'
  say ""
  say "Any OpenAI-compatible server works (vLLM, SGLang, llama.cpp server, a"
  say "commercial API). The paper served the PUBLIC open-weight checkpoint"
  say "openai/gpt-oss-120b (Apache-2.0, https://huggingface.co/openai/gpt-oss-120b)."
  say "No endpoint of the authors' is contacted, and none is embedded anywhere in"
  say "this repository."
  exit 2
fi

# Show where we are pointing WITHOUT echoing the key.
# Read the model id OUT OF THE CONFIG rather than asserting one, so the line
# printed below is the id that will actually be used.
MODEL="$(grep -m1 -E '^[[:space:]]*model:' "$TASK/config.yaml" 2>/dev/null | sed -E 's/^[[:space:]]*model:[[:space:]]*//' || true)"
[ -n "$MODEL" ] || MODEL="(not found in $TASK/config.yaml)"

say "endpoint : $OPENAI_BASE_URL"
say "api key  : set (${#OPENAI_API_KEY} chars, not shown)"
say "model    : $MODEL   (read from $TASK/config.yaml)"
say "task     : $TASK"
say "data     : $DATA"
say "output   : $OUT"
rule

[ -d "$TASK" ] || die "config dir not found: $TASK"
[ -f "$DATA" ] || die "dataset not found: $DATA"

# benchmark.py resolves a RELATIVE --dataset_path against the CONFIG directory
# (see config_resolver.resolve_repo_or_local_path), not the working directory, so
# a bare "data/example/..." would be looked for inside configs/paper/<task>/.
# Pass absolute paths and the ambiguity disappears.
DATA_ABS="$(cd "$(dirname "$DATA")" && pwd)/$(basename "$DATA")"
mkdir -p "$OUT"
OUT_ABS="$(cd "$OUT" && pwd)"
# NOTE: benchmark.py has no --model flag. The actor/evaluator model id lives in
# the config's `model:` fields. To use a different model id, copy the config dir
# and edit `model:` there -- see docs/REPRODUCIBILITY.md "Using a different model".
"$PYTHON" -m agentverse_command.benchmark \
  --task "$TASK" \
  --dataset_path "$DATA_ABS" \
  --output_path "$OUT_ABS" \
  --overwrite
rc=$?

rule
if [ "$rc" -eq 0 ]; then
  say "smoke-live: completed (exit 0). Artifacts under $OUT"
  say ""
  say "REMINDER: do not compare these three problems' accuracy to any paper"
  say "number. For a meaningful live comparison use the frozen N=195 subset"
  say "described in docs/REPRODUCIBILITY.md, and expect the envelope above."
else
  say "smoke-live: FAILED (exit $rc). See docs/TROUBLESHOOTING.md."
fi
exit "$rc"
