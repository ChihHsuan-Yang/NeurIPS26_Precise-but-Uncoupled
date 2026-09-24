# Methods — what was actually run

---

## Benchmark

**Omni-MATH 2 (filtered)** — 4,181 competition-level mathematics problems in ten
benchmark-provided difficulty tiers, tier 10 hardest. Each problem carries an
evaluator-usable final answer, which is what makes verifier-grounded recovery
measurable rather than judge-preference measurable.

Shipped at `data/omni-math-2-filtered/all.jsonl`
(sha256 `7f4691ec…`), with per-tier counts:

| Tier | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| N | 116 | 517 | 20 | 1,106 | 1,078 | 497 | 358 | 328 | 146 | 15 |

Tiers 3 (n=20) and 10 (n=15) are small-N and are banded in the paper's tier figures.
Tier-group sizes used in the paper: 1–4 = 1,759, 5–6 = 1,575, 7–10 = 847.

Licensing: the public cards for Omni-MATH and Omni-MATH-2 both list MIT. See
[NOTICE](../NOTICE).

---

## Models

| Role | Identifier | Where |
|---|---|---|
| Actor (planner, executor, reviewer, peers) | `openai/gpt-oss-120b` | all main results |
| Evaluator / verifier | `openai/gpt-oss-120b` — the same checkpoint, held fixed | all cells |
| Second actor family | `google/gemma-3-27b-it` | Appendix Table 6, N=835, evaluator still gpt-oss-120b |
| Replay evaluators | `Meta-Llama-3.1-70B-Instruct`, `gemma-3-27b-it` | robustness replay only |
| Second family, post-rebuttal matrix | `gemma-4-31b` | 4×2×5 matrix only — **not** in the paper |

`openai/gpt-oss-120b` is public and open-weight (Apache-2.0), served through an
OpenAI-compatible server. **No model was trained or fine-tuned for this work.**

**Decoding:** temperature **0** throughout, `max_tokens` **4096**. Also held fixed
across protocols: the evaluator contract, the outer-loop budget, and the memory reset
policy.

**There is no seed.** A hosted endpoint exposes none and its runtime is not frozen.
Temperature 0 is not determinism. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) §2.3.

---

## The four-rung protocol ladder

### 1. Baseline LLM — `task-single-llm-benchmark`
One-shot. `configs/paper/baseline_llm/`.

### 2. Single-Agent Iterative — `task-single-agent-reflect`
One agent, up to 3 outer rounds, revising its own answer.
`configs/paper/single_agent_hint_llm/`.

### 3. PER — `task-per`
Planner → Executor → Reviewer. `configs/paper/per_hint_llm/`, 3 outer rounds,
2 inner rounds.

The planner is held to **hard role separation**: it may not perform arithmetic or
symbolic manipulation, only plan; the executor carries out the computation. Reviewer
feedback reaches the next actor through a separable `advice` field. **The solver can
acknowledge a critique and still keep its candidate** — this is the structural fact the
paper's coupling measurement is about.

### 4. Broadcast deliberation — `task-broadcast-deliberation`
`configs/paper/broadcast_hint_llm/`. Three peers share one candidate. A confidence
poll selects one speaker at a time, so critique enters shared deliberation state that
every peer sees. An approval phase follows: under the matched configuration
(`approval_rounds: 2`) submission requires **unanimous approval**.

PER and Broadcast differ **jointly** in routing, shared state, feedback delivery and
approval. The comparison is a *configuration-level difference*; it does not isolate
any one factor. See [LIMITATIONS.md](LIMITATIONS.md) §2.

---

## Within-PER probes

Each toggles exactly one key against `per_hint_llm`:

| Variant | Key | Result |
|---|---|---|
| **ACK-required** | `require_explicit_critique_uptake: true` | FinalPass 85.2 → **82.5%**, Neglect 0.488 → **0.792**. A failed compliance control |
| **EMB** | `embed_action_feedback_in_history: true` | 82.5 → **86.3%** vs the matched ACK control. Partial recovery, **directional only** |
| **PER-inner6** | `inner_rounds: 6` (vs 2) | 85.2 → 86.3%; reflective rounds 1.17 → 2.06. **Accuracy stress test only — never token-matched** |

On ACK, the only supported statement is *the tested ACK instruction did not help*. The
experiment cannot separate compliance from distraction, format effects, or reasoning
perturbation.

---

## Evaluation

The primary path uses `evaluator: {type: llm}` — an evaluator agent running the same
public checkpoint, with a fixed contract, parsed by the `mgsm-evaluator` output parser.

Optional alternatives exist in `src/agentverse/evaluation/` (`omni-rule`, a
sympy-based rule evaluator, and `omni-judge`, which downloads the public
`KbsdJames/Omni-Judge` model). **Neither is on the paper's default path**; both are
lazily imported so the base install stays light.

**Cross-evaluator robustness.** 16,724 saved submissions (10,590 unique candidates)
were replayed under three evaluators: instance-level disagreement 3.38–5.76%,
Cohen's κ 0.850–0.915. On the hard slice (PER+Broadcast, tiers 7–10, N=1,694):
disagreement 9.68 / 9.39 / 5.25%, κ 0.755 / 0.761 / 0.875. Direct re-tabulation
preserves the Broadcast−PER ordering under all three: +2.88 [1.82, 3.91],
+3.21 [2.13, 4.26], +3.33 [2.22, 4.45] pp.

---

## Uncertainty

* Wilson intervals on every rate; Newcombe intervals on differences.
* Paired **problem-clustered bootstrap**, 500 replicates, for the headline contrasts
  (`src/precise_uncoupled/scripts/bootstrap_problem_clustered_metrics.py`).
* Run-to-run variability was measured separately on a frozen 195-problem subset
  (paired bootstrap, 10,000 iterations, seed 20260724) — see
  [REPRODUCIBILITY.md](REPRODUCIBILITY.md) §2.3.

The published intervals are **within-run** and do not substitute for multi-seed
variance; that is what the frozen-subset study addresses.

---

## Compute

Runs were executed on GPU-backed HPC infrastructure serving the public checkpoint
through an OpenAI-compatible server. **None of that is required to use this release:**
Track A needs no model at all, and Track B needs only an OpenAI-compatible endpoint of
your choosing.
