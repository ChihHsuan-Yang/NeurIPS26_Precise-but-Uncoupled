# Symmetric 2×5 audit: is the matrix precise but uncoupled?

## Takeaway

- PER reviewer decision precision is higher than Broadcast in **10/10** dataset×actor cells.
- Within PER, reviewer precision is higher than verified next-answer repair in **9/10** cells. The exception is JEEBench/Gemma-4, where only two strict useful-review follow-ups are evaluable, so it is not a stable counterexample.
- Within Broadcast, reviewer precision is higher than verified next-answer repair in **10/10** cells.
- The old protocol-ordering claim does **not** survive the symmetric transition audit: PER has lower strict uptake in 0/10 cells and lower verified repair in 3/10 cells. The earlier 10/10 ordering counted Broadcast responses after system-selected candidate updates together with immediate actor responses.

The defensible cross-setting conclusion is therefore a **detection-to-repair separation**, not a universal PER-versus-Broadcast uptake ordering. Accurate reviewer decisions often fail to become verified repairs across both actor families and all five datasets; the size and protocol ordering of the intermediate uptake rate depend on the transition definition.

## Per-cell results

| Dataset | Actor | PER precision | PER strict uptake | PER repair | Broadcast precision | Broadcast strict uptake | Broadcast repair | Strict follow-up n<20 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Omni-MATH | GPT-OSS-120B | 74.8% (1992/2663) | 38.3% (145/379) | 3.6% (13/362) | 53.7% (1742/3244) | 29.5% (157/533) | 3.6% (16/445) | -- |
| Omni-MATH | Gemma-4-31B | 68.4% (1390/2033) | 25.4% (81/319) | 5.7% (18/316) | 34.6% (1550/4485) | 9.6% (33/345) | 1.2% (4/326) | -- |
| JEEBench | GPT-OSS-120B | 64.7% (22/34) | 33.3% (2/6) | 16.7% (1/6) | 23.7% (130/548) | 14.3% (7/49) | 8.3% (4/48) | PER |
| JEEBench | Gemma-4-31B | 100.0% (11/11) | 100.0% (2/2) | 100.0% (2/2) | 30.2% (80/265) | 27.8% (5/18) | 17.6% (3/17) | PER, BROADCAST |
| SciBench | GPT-OSS-120B | 91.1% (41/45) | 63.6% (7/11) | 0.0% (0/11) | 58.3% (214/367) | 16.9% (11/65) | 3.2% (2/62) | PER |
| SciBench | Gemma-4-31B | 97.6% (82/84) | 65.0% (13/20) | 10.0% (2/20) | 61.1% (299/489) | 16.2% (6/37) | 3.1% (1/32) | -- |
| LAB-Bench | GPT-OSS-120B | 70.7% (94/133) | 28.6% (8/28) | 7.7% (2/26) | 41.3% (612/1482) | 25.7% (58/226) | 7.5% (15/199) | -- |
| LAB-Bench | Gemma-4-31B | 79.4% (27/34) | 40.0% (2/5) | 40.0% (2/5) | 57.6% (724/1256) | 22.8% (37/162) | 4.9% (7/144) | PER |
| MaScQA | GPT-OSS-120B | 75.0% (30/40) | 22.2% (2/9) | 11.1% (1/9) | 34.0% (85/250) | 18.2% (6/33) | 3.4% (1/29) | PER |
| MaScQA | Gemma-4-31B | 100.0% (12/12) | 50.0% (1/2) | 0.0% (0/2) | 43.8% (63/144) | 18.2% (2/11) | 9.1% (1/11) | PER, BROADCAST |

Rates are shown as percentage (successes/eligible episodes). `Precision` is the fraction of explicit reviewer `revise` decisions whose pre-review candidate is frozen-labeled wrong. `Strict uptake` is the fraction of those useful-review episodes for which the immediate next actor-authored answer changes before any system candidate update. `Repair` is the fraction of the post-labeled strict episodes whose immediate next actor answer is correct.

## Interpretation guardrails

- This is a saved-trace analysis, not a new model run.
- `review_action=revise` is a reviewer-issued decision. It does not independently verify that every sentence of the critique is semantically correct.
- Some PER structured review actions are not recoverable as explicit `agree` or `revise`; the precision values therefore use explicit decisions only.
- Uptake and repair use strict pre-gate actor transitions. Post-system-update Broadcast transitions are intentionally excluded.
- Small denominators, especially JEEBench/Gemma-4 and MaScQA/Gemma-4, should be described as exploratory rather than definitive.
- These process rates do not isolate the causal effect of the approval gate, prompt, routing, round count, or compute.

## Provenance

- Input: `rebuttal-code/outputs/process/full_release_symmetric_transitions.csv`
- Input SHA-256: `7441a770e590da04cb0accbb381d9a012e905e93b1d7120f2c04b1b53e21023c`
- Matrix: release-v2, PER and Broadcast, 2 actor families × 5 datasets
- Evaluator family in the released matrix: `gpt-oss-120b`
