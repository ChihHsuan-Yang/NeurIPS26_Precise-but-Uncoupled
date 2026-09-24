# Metrics — exact definitions

Every metric the paper reports, defined operationally, with the module that
computes it. The recurring caution: these are **transition statistics over saved
traces**, not judgements about what a model understood.

---

## Outcome metrics

| Metric | Definition | Module |
|---|---|---|
| **FinalPassRate** | fraction of problems eventually solved, as judged by the in-loop evaluator at the end of the protocol | `metrics/main_paper/basic.py` |
| **Pass@1** | fraction solved on the first system-level attempt | same |
| **OuterPass@k**, **Δ@k**, **CRR@k** | recovery across outer-loop attempts | `metrics/main_paper/outer_loop.py` |

For the primary results the evaluator is `openai/gpt-oss-120b` with
`evaluator: {type: llm}` — the same checkpoint as the actors, held fixed across
protocols so that protocol comparison is not confounded by verifier choice.

---

## The three metrics that carry the paper's claim

These are deliberately separated. The paper's point is that they come apart.

### 1. Reviewer precision — *detection*

> Given that the reviewer issued an explicit "this is wrong" decision, how often was
> the candidate actually wrong?

Measured: **PER 0.861**, **Broadcast 0.644**.

Denominator: reviewer turns carrying an explicit revise/route decision. Correctness
is evaluator-verified, not reviewer-asserted.

### 2. CouplingRate — *transmission*

> Given a **correct** warning on a **wrong** candidate, how often did the next answer
> the protocol carries forward actually **change**?

Measured: **PER 0.336**, **Broadcast 0.935**.

**Read this carefully.** CouplingRate asks only whether the answer *changed*. It does
**not** ask whether the change was good, and it says nothing about whether the solver
"understood", "ignored" or "deprioritized" the critique. Those would be semantic
claims; this is an answer-transition count. A blinded neglect taxonomy would require
per-problem intervention traces that were not retained.

### 3. ReviewerGuidedRepairRate — *outcome*

> Of those same episodes, how often was the next answer **correct**?

Measured: **PER 0.051**, **Broadcast 0.286**.

Computed as
`C_wrong_revise_correct / (C_wrong_revise_correct + C_wrong_revise_wrong)`.

### Why the separation is the finding

PER's reviewer is **better at detection** (0.861 vs 0.644) and **much worse at
transmission** (0.336 vs 0.935), giving lower repair (0.051 vs 0.286) and lower final
accuracy (85.2% vs 89.2%). Detection quality did not carry through to outcomes.

---

## Inner-loop decomposition

Each episode where a reviewer flagged a wrong candidate lands in exactly one bucket:

| Bucket | Meaning | PER | Broadcast |
|---|---|---|---|
| **Repair** | the answer changed and became correct | 11.0% | 25.7% |
| **Neglect** | the answer did not change | 48.8% | 26.2% |
| **TryButFail** | the answer changed but stayed wrong | 40.2% | 48.1% |

Module: `metrics/collaboration_decomposition/summaries.py`.

**MisleadingResistance** (PER 0.921, Broadcast 0.708) is the complement: when a
reviewer wrongly flags a *correct* candidate, how often the protocol keeps it. PER's
weak coupling is protective here — the same property that costs it repair.

---

## The strict-coupling audit

An independent check that CouplingRate is not an artifact of a lenient string match.
Three definitions were computed side by side:

| Definition | PER | Broadcast |
|---|---|---|
| legacy | 0.336 (1,766/5,253) | 0.935 (4,876/5,214) |
| strict deterministic | 0.336 | 0.935 |
| equivalence-aware | 0.340 | 0.935 |

Legacy-vs-strict disagreement: **0 of 5,253** and **0 of 5,214**. The metric is
definition-stable. Module: `metrics/collaboration_decomposition/strict_coupling.py`.

---

## The 2×5 symmetric audit

The post-rebuttal generalisation, computed by
`src/precise_uncoupled/process/analyze_matrix_precise_uncoupled.py`. Three
quantities per dataset × actor cell:

1. **explicit-decision precision** — P(candidate was wrong | reviewer said revise)
2. **strict uptake** — P(next actor-authored answer changed | revise on a wrong candidate)
3. **verified repair** — P(next labelled actor answer correct | revise on a wrong candidate)

Two methodological guardrails are **implemented in the code**, not merely asserted:

* uptake and repair use **only strict pre-gate actor transitions**. Broadcast
  responses occurring after a system-selected candidate update are **excluded**, so
  the approval gate is never counted as if it were an immediate actor response.
* a cell is **refused** (flagged sparse) when either protocol has zero strict
  observations, rather than being reported as a comparison.

Never used as `answer_after`: a reviewer-proposed correction, a system candidate-memory
record, a protocol-selected candidate update, a submission, or an evaluator message.
Only an actor's own next answer counts.

Wilson intervals accompany every rate.

---

## Resource metrics — read [LIMITATIONS.md](LIMITATIONS.md) first

| Metric | Status |
|---|---|
| **Evaluator/verifier calls per problem** (PER 2.19, Broadcast 1.35) | **Valid** — separate counter. But see the `evaluator_calls` field caveat in LIMITATIONS §1 |
| **Wall-clock time** | **Valid** for all matched trajectories. Not a compute-matched measure |
| **Tokens, model calls, cost per solve, λ\*** | **WITHDRAWN.** The counters accumulated across problems within a worker. Exact recovery is impossible |

---

## A note on what "the gate" does to these numbers

Broadcast changes its next actor candidate in **322/1,479 (21.77%)** of episodes
*before* a system candidate update, but **685/1,603 (42.73%)** *after* one. This is
descriptive: it shows the approval gate changes the measured transition regime, which
is why the 2×5 audit excludes post-gate transitions. It is **not** evidence that the
gate causes the accuracy gap — the available toggle changes 6 of 7 required invariants
at once and cannot isolate it.
