# Limitations, withdrawn results, and claims this release does not make

This document exists because several things the paper printed did not survive
post-submission scrutiny. They are listed here in full rather than quietly dropped.

---

## 1. Withdrawn: every absolute token and model-call value

### What went wrong

The harness created **one agent object per worker process** and reused it across
problems. `BaseAgent.reset()` cleared memory but not the token counters, and those
counters only ever incremented — nothing set them back to zero. The metrics
collector then snapshotted them per problem, while its own docstring assumed fresh
agents per example.

So a per-problem "token count" was actually a **running total for that worker**.

Three independent confirmations:

1. **Code trace** — the counters are class-level, incremented in six places, reset
   nowhere.
2. **Sawtooth in the exported data** — one protocol's first eleven rows show
   `model_calls` 8, 12, 16, 19, 25, 30, 34, 38, 42, 46, then **reset to 8** as a new
   worker starts. Tokens show the same pattern.
3. **Exact reproduction of the published means** — averaging the cumulative
   snapshots reproduces the submitted table values exactly.

### What is withdrawn

* all **average-token** values in Table 1, Table 5 and Table 14;
* all **tokens-per-extra-solve** figures;
* all **cost-per-solve**, cost-frontier and verifier-cost-threshold (λ\*) values
  (Tables 19, 20, 21);
* **Figures 14, 15, 16** and the token panel of Figure 19;
* the tier-group deployment guidance derived from them;
* a token-caliper analysis that matched on the affected field (withdrawn entirely:
  its matching variable is the defective one).

Also: **PER-inner6 must never be described as token- or compute-matched.** No
aggregate generation-token ceiling exists in this harness — the only `token_budget`
is prompt-context truncation — so a compute-matched PER/Broadcast comparison was
not merely unperformed, it is infeasible in this design. PER-inner6 is an accuracy
stress test.

### Why no corrected numbers are supplied

Exact recovery is **impossible**. No worker-boundary metadata survived in the
export and no lower-level request logs were retained. Heuristic boundary inference
makes the corruption worse for one protocol: 1,242 counter resets across 4,181
rows, with inferred block sizes 1–11 — sizes above 10 are impossible for 10-example
workers, because dynamic role-agent replacement restarts a counter mid-worker. So
no corrected value is emitted rather than a plausible-looking wrong one.

### What survives

* **Every accuracy result** — FinalPass, Pass@1, per-tier accuracy. Unaffected.
* **Every process metric** — precision, coupling, repair, the transition
  decomposition, the three-evaluator replay. Unaffected.
* **Evaluator/verifier call counts** (PER 2.19 vs Broadcast 1.35 per problem). A
  separate counter. *Caveat, stated openly:* a data-side audit found the
  `evaluator_calls` **field** in the released per-row data showing the same
  cumulative signature for PER rows, while a paper-side audit listed evaluator call
  counts among surviving claims. These conflict. The release fails safe: that
  **field** is marked invalid for PER pending explicit re-derivation. The
  per-problem figures quoted above come from the paper's own tables.
* **Wall-clock time** — valid for all matched trajectories. Caveat: wall time is
  not a compute-matched measure.
* A clean per-process resource table measured with one problem per fresh process,
  which shows a persistent realized-resource imbalance (Broadcast > PER). Bounded
  evidence; explicitly not a compute control.

### The fix

The harness now resets each agent's counters in `prepare_example()`, iterating
whatever agents currently exist so dynamic role replacement is handled, with three
regression tests. A live check under the patched harness gives per-problem calls
8/8/8 instead of the pre-patch 8/12/16. **This fix post-dates the runs whose
numbers were withdrawn** — it makes future runs correct; it does not retroactively
repair the published values.

---

## 2. Claims this release does not make

| Not claimed | Why |
|---|---|
| A universal PER-vs-Broadcast ranking | With Gemma-3-27b-it actors the ranking **reverses**: PER 65.6% vs Broadcast 58.7%. The precision–uptake separation persists, the ordering does not. The ranking is setting-dependent. |
| Any causal claim about the approval gate | The only available toggle changes 6 of 7 required invariants at once, so it cannot isolate the gate. The isolated gate effect is an admitted limitation, not a result. |
| That the comparison "isolates" routing, shared state, feedback delivery, or rounds | PER and Broadcast are complete protocol designs that differ jointly. The right phrase is *configuration-level difference*. |
| Any semantic claim about critique text | CouplingRate is an operational answer-transition statistic. It does not show that a solver "understood", "ignored" or "deprioritized" a critique. A blinded neglect taxonomy would need per-problem intervention traces that were not retained. |
| "Superficial compliance" as the ACK mechanism | The tested ACK instruction does not distinguish compliance from distraction, format effects or reasoning perturbation. Only this is claimed: *the tested ACK instruction did not help.* |
| ACK vs EMB as a causal comparison | Downgraded to a **directional within-PER probe**. (The phrase "cleanest within-family causal comparison" appears in the arXiv v1 appendix; treat it as a known overstatement.) |
| The EMB upstream-cohort mechanism | On a frozen 195-problem replication EMB is **−1.03 pp** vs base PER and −2.82 pp within the retained cohort. Cohort membership churns (Jaccard 0.623), and the gained and lost cohorts are both 100% FinalPass — consistent with cohort entry tracking problem difficulty rather than a responsiveness gain. |
| Generalization beyond the tested model families | Cross-cell correlations rest on ten contexts. Descriptive only. |
| Any released project-trained model | None was trained. See [PROVENANCE.md](PROVENANCE.md) §5.3. |
| That Gemma 3 and Gemma 4 results can be pooled | Different model generations. The paper's second family is Gemma 3; the post-rebuttal matrix's is Gemma 4. |
| That execution backends are model families | They are provenance strata, not experimental conditions. |

### A claim that must never appear

An earlier *planned* revision claim — that one protocol has lower strict uptake in
**all** settings, sometimes cited as a 10/10 precision–repair inversion — is
**retired**. It came from a mock analysis that pooled immediate actor responses
with responses observed only after a system-selected candidate update. The
symmetric audit measures **0/10** and **3/10** for those two orderings.

It was **never in the submitted paper or arXiv v1** (grep: zero hits), so no
correction of record is owed. It simply must not appear. A test in
`tests/test_release_integrity.py` fails the build if that sentence reappears
anywhere in the repository.

---

## 3. Known reproducibility gaps, stated rather than hidden

### 3.1 Figure 4's coupling panel is not reproducible

Main-paper Figure 4 shows CouplingRate **0.175** (ACK) and **0.227** (EMB).
Appendix Table 14 shows **0.200** and **0.247** for the same quantities.

The saved aggregate CSV matches the **Table 14** values exactly. The Figure 4 values
**do not appear in any released aggregate**. The correctness-based tensors are
internally consistent and reproduce the repair column exactly, but CouplingRate is a
change/incorporation metric and cannot be rebuilt from a correctness tensor — so the
0.025 gap is consistent with different denominator logic in the two generators, and
the exact Figure 4 logic has not been recovered.

**This release quotes 0.200 / 0.247** and marks the Figure 4 panel as *not
reproducible from released aggregates*. The two definitions are not averaged and
neither is silently preferred in a regenerated figure.

### 3.2 Figure regeneration is not byte-exact

The bundled plotting script emits only 14 of the 20 figures; the complete variant
lives outside the analysis repository, and even it is a near, not exact, regenerator
— four of its own outputs differ in byte size from the published PDFs. This release
therefore publishes the **figure input data**, which is exactly reproducible and
checksummed, rather than a pipeline that would quietly disagree with the paper.

### 3.3 Layer 1 of the Track A chain was verified from one revision

The end-to-end regeneration was run against the current public data revision, where
it reproduced all ten table rows cell-for-cell. The earlier revision's link rests on
the byte-identical archived intermediate rather than on a fresh aggregate run from
that revision.

### 3.4 Two marginal cells

In the 2x5 audit, two repair comparisons differ only in the fourth decimal place
(0.035912 vs 0.035955; 0.076923 vs 0.075377). They should not be presented as clean
separations. They do not affect the precision counts that carry the paper's claim.

### 3.5 Sparse cells

The audit flags cells with too few strict observations, and one — JEEBench/Gemma-4,
with **n=2** evaluable follow-ups — is the single exception to the within-PER 9/10
count. Quote the denominator whenever quoting that cell.

---

## 4. Scope

The paper is **Omni-MATH only**: N=4,181, ten tiers, actor and evaluator both
`openai/gpt-oss-120b`, temperature 0, no seed. The 5-dataset × 2-actor matrix is
**post-rebuttal robustness evidence**. Keeping that distinction is a condition of
using this release honestly.
