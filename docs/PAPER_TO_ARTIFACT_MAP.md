# Paper → artifact map

For each table and figure: the claim, the data behind it, the command that
rebuilds it, whether it needs model calls, and whether it is a primary result or
robustness evidence.

**Track A** = offline, no model calls, deterministic, checksum-verifiable.
**Track B** = requires your own endpoint; outputs vary between runs.

All paths are relative to the repository root. Run commands from there.

**Setup assumed by every Track A command below:**
```bash
pip install -r requirements.txt
```

---

## Reference digests

| Artifact | sha256 |
|---|---|
| `results/derived_tables/matrix_2x5_precise_uncoupled_strict.csv` | `6c77c01e6a02889cbf69b77d3f33d0f8e5784a150d3d1b29d9311b06435daef2` |
| `data/derived/full_release_symmetric_transitions.csv` | `7441a770e590da04cb0accbb381d9a012e905e93b1d7120f2c04b1b53e21023c` |
| `results/derived_tables/rq1_tier_mode_metrics.csv` | `80c4eb0b5cccbc12f7e0c0dfc355f9bf2ec5b02251da6d4eca81100d54957d79` |
| `results/derived_tables/rq2_tier_mode_metrics.csv` | `cbec609fb51a441f455fb550eb39918961c3b2a2f4797660dbdfddd7b198b5da` |
| `results/derived_tables/rq3_collaboration_tier_mode_metrics.csv` | `a16157c7f2d1de6e18fe637252e17421e3ab8b9c61e58267ad8025c1063e77ba` |
| `results/derived_tables/rq3_protocol_tier_mode_metrics.csv` | `82ded7fdcbc96beffe7cca554500034d52617341581cdbffd7b2e5958ee86e87` |
| `results/derived_tables/efficiency_tier_mode_metrics.csv` | `0b3e4fa4d1e382f830c36ea1007d3fb6576aeab85f8a7e7c5ad15a19d2ad5fbf` |
| `data/omni-math-2-filtered/all.jsonl` | `7f4691ecfba92b97e1071bbb3270cc22716e02634f9df939d255602fff37a87e` |

Verify all of them at once: `make verify-checksums` (recomputes each digest; a
missing file is a failure, not a skip).

---

## Main paper

### Figure 1 — conceptual schematic
**Claim:** why reviewer quality and realized repair can diverge (PER's routed,
non-binding critique vs Broadcast's shared, collectively re-approved candidate).
**Evidence class:** primary, conceptual. **Track:** n/a — no data.
**Note:** in the paper this figure is a LaTeX table, not an image file. There is no
PDF asset to reproduce.

### Table 1 — overall outcome and cost, N=4,181
| | |
|---|---|
| **Claim (valid part)** | FinalPass 56.8 / 78.8 / 85.2 / 89.2 % and Pass@1 56.8 / 57.4 / 72.8 / 78.6 % for Baseline / Single-Agent / PER / Broadcast; evaluator calls 1.00 / 1.69 / 2.19 / 1.35 per problem |
| **Evidence class** | **PRIMARY** |
| **Track** | **A** |
| **Source data** | `results/derived_tables/rq1_tier_mode_metrics.csv` (`80c4eb0b…`), `efficiency_tier_mode_metrics.csv` (`0b3e4fa4…`) |
| **Producing code** | `src/agentverse/metrics/main_paper/basic.py` via `agentverse_command.rq1_outcome_cost` |
| **Command** | `make verify-checksums` to validate; `make reproduce-figures` to export |
| **Validation** | sha256 equality against the digests above |
| **Model calls** | **No** |
| **⚠ WITHDRAWN COLUMNS** | `Avg. Tokens` and `Tokens per Extra Solve` are **invalid** and must not be quoted or recomputed. See [LIMITATIONS.md](LIMITATIONS.md) §1. |

### Figure 2 — collaboration gain by tier
**Claim:** gains ≈0 at tiers 1–2, opening sharply from tier 4, reaching ~10–20 pp at
tiers 6–9. **Evidence class:** PRIMARY. **Track:** A.
**Data:** `results/figure_data/rq1_tier_mode_metrics.csv`.
**Command:** `make reproduce-figures`. **Model calls:** No.
**Caveat:** tier 3 (n=20) and tier 10 (n=15) are small-N and are banded in the paper.
**Gap:** the bundled plot script does not emit this figure (see §Figures below).

### Figure 3 — the core figure: precision vs coupling vs repair
**Claim:** PER precision 0.861 vs Broadcast 0.644; CouplingRate 0.336 vs 0.935;
ReviewerGuidedRepairRate 0.051 vs 0.286. NeglectRate dominates PER.
**Evidence class:** **PRIMARY — this is the paper's central result.** **Track:** A.
**Data:** `results/derived_tables/rq3_collaboration_tier_mode_metrics.csv` (`a16157c7…`).
**Producing code:** `src/agentverse/metrics/collaboration_decomposition/` via
`agentverse_command.collaboration_decomposition`.
**Command:** `make verify-checksums && make reproduce-figures`. **Model calls:** No.

### Figure 4 — within-PER interventions (ACK, EMB)
**Claim:** FinalPass 85.2 → 82.5 → 86.3; NeglectRate 0.488 → 0.792 → 0.698.
**Evidence class:** PRIMARY but **directional, not causal**. **Track:** A.
**Data:** `results/derived_tables/rq3_protocol_tier_mode_metrics.csv` (`82ded7fd…`).
**Model calls:** No.
**⚠ KNOWN GAP:** the **coupling panel** of this figure is **NOT reproducible from
released aggregates**. It shows 0.175 / 0.227; those values appear in no saved
aggregate. Use Appendix **Table 14's 0.200 / 0.247**, which the saved CSV matches
exactly. Do not average the two definitions. See [LIMITATIONS.md](LIMITATIONS.md) §3.1.

---

## Appendix tables

| № | What it shows | Class | Track | Source / command | Model calls |
|---|---|---|---|---|---|
| 3 | Strict-coupling audit: legacy = strict for both protocols (PER 0.336 = 1,766/5,253; Broadcast 0.935 = 4,876/5,214; disagreement 0/5,253 and 0/5,214) | primary robustness — a strong anchor | **A** | `agentverse_command.strict_coupling_rate`; `rq3_collaboration_tier_mode_metrics.csv` | No |
| 4 | Three-evaluator replay: 16,724 instances, disagreement 3.38–5.76%, κ 0.850–0.915; hard slice N=1,694 | robustness | **A** | released replay labels | No |
| 5 | PER / PER-inner6 / Broadcast accuracy (85.2 / 86.3 / 89.2) | robustness | **A** | `rq1_tier_mode_metrics.csv` | No |
| | ⚠ its `AvgTokens` column is **withdrawn**; and inner6 is **never** token-matched | | | | |
| 6 | **Gemma-3 replication, N=835: ranking REVERSES** (PER 65.6 vs Broadcast 58.7) while precision 0.881 vs 0.722 and repair 0.009 vs 0.174 | **robustness — the ranking-reversal evidence** | **A** (analysis) / **B** (to regenerate traces) | `configs/paper/different_model_family/` | No / Yes |
| 9 | Inner-loop decomposition: PER 11.0 / 48.8 / 40.2; Broadcast 25.7 / 26.2 / 48.1 | PRIMARY | **A** | `rq3_collaboration_tier_mode_metrics.csv` | No |
| 10 | Review instances 10,804 / 11,587; precision .861/.644; recall .754/.872 | PRIMARY | **A** | same | No |
| 11 | **Reviewer-conditioned decomposition — the most quotable table**: precision .861/.644, recall .754/.872, Coupling .336/.935, Repair .051/.286, MisleadingResistance .921/.708 | **PRIMARY** | **A** | same | No |
| 12 | Matched protocol knobs | primary, descriptive | **A** | `configs/paper/*/config.yaml` | No |
| 14 | PER/ACK/EMB/Broadcast: FinalPass 85.2/82.5/86.3/89.2, Neglect .488/.792/.698/.262, Repair .051/.032/.044/.286, **UsefulCoupling .200/.247** | primary, directional | **A** | `rq3_protocol_tier_mode_metrics.csv` | No |
| | ⚠ `AvgTokens` column **withdrawn**. Its UsefulCoupling values are the **verified** ones — prefer them over Figure 4's | | | | |
| 15, 18 | FinalPass by tier | PRIMARY | **A** | `rq1_tier_mode_metrics.csv` | No |
| 16 | Critique routing / approval structure | descriptive | **A** | `configs/paper/` | No |
| | ⚠ read as a topology/confound inventory, **not** "mechanistic support" | | | | |
| 17 | Outer-loop recovery: Δ@k, CRR@k, OuterPass@3 | PRIMARY | **A** | `rq2_tier_mode_metrics.csv` (`cbec609f…`) | No |
| 22, 23 | Wilson / Newcombe intervals | PRIMARY | **A** | `rq1_tier_mode_metrics.csv` | No |
| 24 | Paired problem-clustered bootstrap, 500 replicates | PRIMARY | **A** | `src/precise_uncoupled/scripts/bootstrap_problem_clustered_metrics.py` (needs the per-tier trace tree; set `PU_TRACE_ROOT`) | No |
| 25 | Published Omni-MATH rates as unmatched context | context | n/a | literature | No |
| **19, 20, 21** | reader guidance, λ\* thresholds, tier-group thresholds | **WITHDRAWN — do not republish** | — | derived from the invalid cost fields | — |

## Appendix figures

Figures 5, 7–13, 17, 18, 20 are **Track A**, reproducible from the checksummed
per-tier tables (`make reproduce-figures`), no model calls.

| Figure | Status |
|---|---|
| 6 (`per_inner6`) | panel (b) accuracy is fine; **panel (a) is a token-cost frontier — withdrawn** |
| 14, 15, 16 | **WITHDRAWN — do not republish.** Built entirely on the invalid cost fields |
| 19 | one of its six panels is average generated tokens — **that panel is withdrawn** |
| `fig_crr_by_tier` | orphan: emitted by the plot script but never included in the paper. Not a paper figure |

**Figure rendering gap (applies to all of the above):** this release ships figure
**input data**, not a re-rendering pipeline. The plotting script bundled with the
paper source emits only **14 of 20** figures — including, notably, not main-paper
Figure 2 — and the complete 1,539-line variant is a *near* regenerator, with four of
its own outputs differing in byte size from the published PDFs. Publishing exactly
reproducible input data was preferred to a pipeline that would quietly disagree with
the paper. See [LIMITATIONS.md](LIMITATIONS.md) §3.2.

---

## Post-rebuttal robustness: the 2×5 symmetric audit

**Not a table in the paper.** This is the post-rebuttal evidence that the
precision/uptake separation is broad.

| | |
|---|---|
| **Claim** | PER precision > Broadcast precision in **10/10** dataset×actor cells; within PER, precision > verified repair in **9/10** (exception JEEBench/Gemma-4, n=2); within Broadcast, **10/10** |
| **Evidence class** | **POST-REBUTTAL ROBUSTNESS** — never the paper's main claim |
| **Track** | **A** |
| **Source data** | `data/derived/full_release_symmetric_transitions.csv` (`7441a770…`), 104,729 lines × 59 columns |
| **Producing code** | `src/precise_uncoupled/process/analyze_matrix_precise_uncoupled.py` |
| **Command** | `make reproduce-analysis` |
| **Expected output** | `results/derived_tables/_regenerated/matrix_2x5_precise_uncoupled_strict.csv`, sha256 **`6c77c01e…daef2`** |
| **Validation** | the script prints PASS/FAIL on sha256 equality; also verified byte-identical across py3.9/pandas2 and py3.13/pandas3 |
| **Model calls** | **No** |
| **Also measures** | PER strict uptake lower than Broadcast in **0/10** cells, repair lower in **3/10** — which is why no universal ranking is claimed |

To rebuild the 65 MB intermediate itself from public traces (layer 1→2), see
[REPRODUCIBILITY.md](REPRODUCIBILITY.md) §1.3. Expect byte-level differences in the
intermediate and cell-for-cell identity in the table.

---

## Track B — anything requiring model calls

| Goal | Command | Reproduces exactly? |
|---|---|---|
| Verify your endpoint works | `make smoke-live` | No — 3 problems, sanity only |
| Regenerate one protocol's traces | `python3 -m agentverse_command.benchmark --task configs/paper/per_hint_llm --dataset_path "$PWD/data/omni-math-2-filtered/all.jsonl" --output_path "$PWD/results/live/per" --overwrite` | **No** |
| Second actor family | same with `configs/paper/different_model_family/per_hint_gemma3_27b_eval_oss120b` | **No** |

No Track B output reproduces the submitted run. Two independent reasons, both
documented: there is **no seed** available from a hosted endpoint (measured
envelope: PER FinalPass range 0.51 pp, Broadcast 2.05 pp), and the protocol
runtime here **post-dates** the submitted run's base commit. Compare directions and
separations, not absolute percentages.
