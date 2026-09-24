# Rebuttal / robustness configs — NOT the paper's main scope

Everything in this directory belongs to **post-rebuttal robustness evidence**, not
to the paper's headline claims.

The paper is **Omni-MATH-only**: N=4,181 problems, ten difficulty tiers, with the
actor AND the evaluator both `openai/gpt-oss-120b` at temperature 0. That scope is
defined by `configs/paper/`.

## `different_model_family/` (Gemma-4 arms)

Second-actor-family bundles for `google/gemma-4-31B-it` and `google/gemma-4-E4B-it`,
evaluator fixed to `openai/gpt-oss-120b`. These are the actor configs behind the
Gemma-4 column of the post-rebuttal **4x2x5 matrix** (4 protocols x 2 actor
families x 5 datasets).

**Do not pool Gemma 3 with Gemma 4.** They are different model generations. The
submitted paper's second family is Gemma **3** (`configs/paper/different_model_family/`);
the 4x2x5 matrix's second family is Gemma **4**. Reporting them together would
misstate both.

## The 2x5 audit these support

The released 2x5 symmetric audit (`results/derived_tables/matrix_2x5_precise_uncoupled_strict.csv`)
covers 5 datasets x 2 actor families and reports, per cell, reviewer explicit-decision
precision, strict uptake, and verified repair. Its headline counts are:

* PER precision > Broadcast precision in **10/10** cells
* within PER, precision > verified repair in **9/10** cells
  (the exception is JEEBench/Gemma-4, where only n=2 follow-ups are evaluable)
* within Broadcast, precision > verified repair in **10/10** cells

Reproducing that table needs **no model calls and no config in this directory** —
it runs offline from released data. See `docs/REPRODUCIBILITY.md` (Track A).
The configs here are only needed if you want to regenerate the underlying traces
yourself (Track B), which will not reproduce the released traces byte-for-byte.
