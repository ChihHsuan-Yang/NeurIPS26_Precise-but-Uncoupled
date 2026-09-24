# Second actor family (appendix robustness arm)

Actor-family replication bundles for the four main protocols. The **evaluator is
held fixed at `openai/gpt-oss-120b`** in every bundle, matching the main paper's
evaluator contract; only the acting model changes.

Actor families in this directory:

| Actor model | Used by |
|---|---|
| `google/gemma-3-27b-it` | **Appendix Table 6** of the paper (N=835 tier-sampled) |
| `meta-llama/Meta-Llama-3.1-70B-Instruct` | additional family sweep, not a paper table |

Bundles: `{baseline,single_agent_hint,per_hint,broadcast_hint}_{gemma3_27b,llama31_70b}_eval_oss120b`.

## What the Gemma-3 arm shows, precisely

Appendix Table 6 (N=835, evaluator fixed to gpt-oss-120b) **REVERSES the outcome
ranking** seen with gpt-oss-120b actors: PER 65.6% vs Broadcast 58.7% FinalPass.
The precision/uptake separation nonetheless persists (precision 0.881 vs 0.722;
uptake 0.092 vs 0.742; repair 0.009 vs 0.174).

This is why the release **never states a universal PER-vs-Broadcast ranking**. The
ranking is setting-dependent. See `docs/LIMITATIONS.md`.

## Scope notes

* Gemma-**4** bundles live in `configs/rebuttal/different_model_family/`, not here.
  Gemma 3 and Gemma 4 are different model generations and **must not be pooled**:
  the paper uses Gemma 3; the post-rebuttal 4x2x5 matrix uses Gemma 4.
* These are Track B configs: running them calls a model. See `docs/REPRODUCIBILITY.md`.

## Running one

    export OPENAI_BASE_URL="https://your-endpoint.example/v1"
    export OPENAI_API_KEY="..."
    PYTHONPATH=src python -m agentverse_command.benchmark \
      --task configs/paper/different_model_family/per_hint_gemma3_27b_eval_oss120b \
      --dataset_path data/example/omni_math_example.jsonl \
      --output_path results/live/per_gemma3_smoke

The upstream copy of this README additionally contained a machine-specific
`cd <absolute path>` and a `REPO_ROOT=<absolute path>` export for the authors'
cluster checkouts. Both were removed for the release; nothing here depends on a
particular directory.
