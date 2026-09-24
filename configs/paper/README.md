# Paper configs — the exact protocol definitions used for the main results

Every config here runs on **Omni-MATH 2 (filtered), N=4,181 problems, ten tiers**,
with actor AND evaluator both `openai/gpt-oss-120b`, **temperature 0**,
`max_tokens 4096`, and `evaluator: {type: llm}` (the in-loop LLM evaluator).

## The four-rung protocol ladder (main results, Table 1)

| Directory | `env_type` | What it is |
|---|---|---|
| `baseline_llm/` | `task-single-llm-benchmark` | one-shot single model |
| `single_agent_hint_llm/` | `task-single-agent-reflect` | single agent, iterative self-revision |
| `per_hint_llm/` | `task-per` | Planner -> Executor -> Reviewer pipeline |
| `broadcast_hint_llm/` | `task-broadcast-deliberation` | three peers, confidence-polled speaker, then an approval phase |

Under the matched configuration, Broadcast submission requires **unanimous approval**
(`approval_rounds: 2` in `broadcast_hint_llm/config.yaml`).

## Within-PER probes (Appendix Table 14, Figure 4)

Each toggles exactly one key relative to `per_hint_llm/`:

| Directory | Key | Effect |
|---|---|---|
| `per_ack_required_hint_llm/` | `require_explicit_critique_uptake: true` | mandatory acknowledgment preface; routing unchanged |
| `per_embedded_advice_hint_llm/` | `embed_action_feedback_in_history: true` | reviewer feedback inserted into the executor's primary working context |
| `per_6_inner_rounds_hint_llm/` | `inner_rounds: 6` (vs 2) | deeper local reflective budget |

Read these results carefully: ACK-required **lowers** final accuracy (85.2% -> 82.5%)
and raises NeglectRate (0.488 -> 0.792); EMB partially recovers (-> 86.3%). The
release states these as **directional within-PER probes, not causal comparisons**,
and `per_6_inner_rounds_hint_llm` is an accuracy stress test only — it is
**never** a token- or compute-matched comparison. See `docs/LIMITATIONS.md`.

## Second actor family

`different_model_family/` — Gemma-3-27b-it and Llama-3.1-70B-Instruct actors with the
evaluator still fixed to gpt-oss-120b. See that directory's README; the Gemma-3 arm
is the paper's ranking-reversal evidence.

## Prompt surface

`prompt_presets/omni_math/` holds the Omni-MATH prompt and evaluator wording exported
from the four primary configs, so you can read the exact prompt surface without
parsing the full YAML.

## Pointing a config at your own endpoint

No config contains an endpoint, key, or hostname — the client reads
`OPENAI_BASE_URL` and `OPENAI_API_KEY` from the environment. See
`docs/REPRODUCIBILITY.md` (Track B).

## Not included, deliberately

`*_pilot/` and `*_intervention/` subset configs (100- and 500-problem shards) and
the LAB-Bench / ScienceQA / DHD config bundles that exist in the authors' working
repository are **not** part of this release. The paper's main results are
full-benchmark N=4,181; see `docs/PROVENANCE.md` for the complete exclusion list
and the reason for each.
