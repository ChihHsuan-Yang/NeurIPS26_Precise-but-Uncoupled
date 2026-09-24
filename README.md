# Precise but Uncoupled

### Reviewer Precision Does Not Guarantee Critique Uptake in Multi-Agent Math Reasoning

Chih-Hsuan Yang¹\*, Jingyan Jiang¹, Vikram Vasudevan², Cheng-Hau Yang¹, Huihuo Zheng¹,
Le Chen¹, Eliu A. Huerta¹ ³, Venkatram Vishwanath¹, Ian T. Foster¹ ³, Rajeev Thakur¹

¹ Argonne National Laboratory  ² Oregon State University  ³ University of Chicago
\* corresponding author, `bellayang@anl.gov`

**Preprint.** [arXiv:2607.15388](https://arxiv.org/abs/2607.15388) [cs.AI], v1, 16 July 2026.
DOI [10.48550/arXiv.2607.15388](https://doi.org/10.48550/arXiv.2607.15388).
*No acceptance decision is claimed.*

| | |
|---|---|
| Paper | https://arxiv.org/abs/2607.15388 |
| Website | https://chihhsuan-yang.github.io/NeurIPS26_Precise-but-Uncoupled/ |
| Code (this repo) | https://github.com/ChihHsuan-Yang/NeurIPS26_Precise-but-Uncoupled |
| Data | https://huggingface.co/datasets/AgentsSci/NeurIPS26_Precise-but-Uncoupled |

---

## The question

If you put a **reviewer** in a multi-agent system, and that reviewer is good at spotting
wrong answers, does the system solve more problems?

## The finding

**No — not on its own.** Being right about the error and *acting* on it are separable.

On 4,181 Omni-MATH problems with `gpt-oss-120b` as both actor and evaluator, the
planner–executor–reviewer pipeline (**PER**) has the *more precise* reviewer
(**0.861** vs **0.644**), yet the critique is far less likely to change the next
answer the protocol carries forward (**CouplingRate 0.336** vs **0.935**) and yields
less repair (**ReviewerGuidedRepairRate 0.051** vs **0.286**). Broadcast reaches
higher final accuracy (**89.2%** vs **85.2%**).

> A system can look strong at *spotting* errors and still fail to *solve* more problems,
> because the protocol never acts on what the reviewer found.

This separation is the paper's claim, and it holds broadly: across a 5-dataset × 2-actor
matrix, reviewer precision exceeds Broadcast's in **10/10** cells, and exceeds verified
repair within PER in **9/10** cells and within Broadcast in **10/10**.

**What we do *not* claim:** that one protocol beats the other in general. With
**Gemma-3-27b-it** actors the accuracy ranking *reverses* (PER 65.6% vs Broadcast 58.7%)
while the precision–uptake separation persists. See [docs/LIMITATIONS.md](docs/LIMITATIONS.md).

## The vocabulary, in one table

| Term | Meaning |
|---|---|
| **PER** | Planner → Executor → Reviewer. Critique travels to the solver in a separable `advice` field; the solver may acknowledge it and still keep its answer. |
| **Broadcast** | Three peers discuss a shared candidate, one speaks at a time by confidence poll, then an approval phase. Under the matched config, submission needs unanimous approval. |
| **Reviewer precision** | When the reviewer says "this is wrong", how often is it actually wrong. A property of *detection*. |
| **CouplingRate (uptake)** | Given a correct warning on a wrong candidate, how often the *next* answer the protocol carries forward actually changes. A property of *transmission*. This is an operational answer-transition statistic — **not** a judgement about whether the solver "understood" the critique. |
| **ReviewerGuidedRepairRate** | Of those, how often the next answer is *correct*. A property of *outcome*. |

## Two ways to use this repository

| | Track A | Track B |
|---|---|---|
| What | Reproduce the paper's tables from released data | Re-run the protocols live |
| Model calls | **None** | **Yes — against *your* endpoint** |
| Private infrastructure | **None** | None (any OpenAI-compatible server) |
| Determinism | **Byte-identical, verified by sha256** | **Outputs vary** (see below) |
| Time | ~1 minute | hours to days at full scale |
| Entry point | `make reproduce-analysis` | `make smoke-live` |

## Fastest thing to run

```bash
pip install -r requirements.txt
make smoke-test          # offline, seconds: imports + checksums + unit tests
make reproduce-analysis  # offline, ~1 minute: rebuilds the headline table, checks its sha256
```

`make reproduce-analysis` prints `PASS` only if the regenerated table is **byte-identical**
to the released one (sha256 `6c77c01e…daef2`). It has been verified byte-identical across
Python 3.9.6/pandas 2.3.3, Python 3.13.0/pandas 3.0.3, and the original analysis
environment — and it reproduces cell-for-cell when regenerated end-to-end from the public
trace data (34,446 trajectories → 95,799 transitions → the same ten rows).

Make is a convenience, never a requirement — every target is a script:

| Make target | Equivalent script |
|---|---|
| `make fetch-data` | `bash scripts/fetch_data.sh` |
| `make verify-checksums` | `bash scripts/verify_checksums.sh` |
| `make reproduce-analysis` | `bash scripts/reproduce_main_results.sh` |
| `make reproduce-rebuttal` | `bash scripts/reproduce_rebuttal_results.sh` |
| `make reproduce-figures` | `bash scripts/reproduce_figures.sh` |
| `make smoke-test` | `bash scripts/smoke_test.sh` |
| `make smoke-live` | `bash scripts/smoke_live.sh` |
| `make validate` | `python3 scripts/validate_release.py` |
| `make test` | `python3 -m pytest tests/ -q` |

(GNU Make 3.81, the version macOS ships, is supported.)

## Running Track B against your own endpoint

```bash
export OPENAI_BASE_URL="https://your-endpoint.example/v1"
export OPENAI_API_KEY="..."
make smoke-live          # 3 problems through the PER protocol
```

Any OpenAI-compatible server works (vLLM, SGLang, llama.cpp, a commercial API). The paper
served the **public, open-weight** checkpoint
[`openai/gpt-oss-120b`](https://huggingface.co/openai/gpt-oss-120b) (Apache-2.0) at
temperature 0. **No endpoint of ours is contacted and none is embedded in this
repository.** No PBS, Slurm, or facility account is needed.

### Live outputs will not match the paper exactly. This is expected.

* **There is no seed.** The paper ran at temperature 0, but a hosted endpoint exposes no
  seed and its runtime is not frozen. Independent executions differ.
  Measured envelope on a frozen 195-problem subset: **PER FinalPass range 0.51 pp**,
  **Broadcast FinalPass range 2.05 pp**. That is smaller than the Broadcast−PER effect
  (+4.10 to +6.67 pp), which is why the effect survives — but a small live run tells you
  nothing about accuracy.
* **The protocol code post-dates the submitted run.** These runtimes are vendored at
  AgentVerse `b4a2db6`, which post-dates the submitted run's base `be9c47c9`. Track B
  reproduces the experiment's *design*, not the submitted run's outputs. (The *analysis*
  code is byte-identical to `be9c47c9` — see [docs/PROVENANCE.md](docs/PROVENANCE.md).)

## Is there a project model to download?

**No.** No model was trained, fine-tuned, or distilled for this paper, and no weights are
released. There is **no model card**, because there is no model. Every model used is a
public third-party checkpoint you obtain and serve yourself. This was verified by an
exhaustive filesystem walk plus an all-refs git history search: zero checkpoints, zero
training code on the paper path.

## Which measurements are INVALID — read before quoting any cost number

A post-submission audit found that the token and model-call counters **accumulated across
problems within a worker process** instead of resetting per problem. Averaging those
cumulative snapshots is what produced the published per-problem figures, so:

**Withdrawn — do not cite, do not recompute from released aggregates:**

* all absolute **average-token** values in the paper (Table 1, Table 5, Table 14) and every
  "tokens per extra solve" figure;
* all **cost-per-solve**, cost-frontier and verifier-cost-threshold (λ\*) values
  (Tables 19–21, Figures 14–16);
* any deployment guidance derived from them.

Exact recovery is **impossible** — no worker-boundary metadata and no lower-level request
logs were retained — so no corrected value is offered in their place.

**Still valid:**

* every accuracy result (FinalPass, Pass@1) and every process metric
  (precision, coupling, repair, the transition decomposition);
* **evaluator/verifier call counts** (PER 2.19 vs Broadcast 1.35 per problem) — a separate
  counter, unaffected. *Caveat:* a data-side check found a related `evaluator_calls` field
  in the released per-row data showing the same cumulative signature for PER rows; that
  **field** is marked invalid for PER as a fail-safe. The paper's per-problem call counts
  above come from the paper's own tables and stand.
* **wall-clock time**, which is valid for all matched trajectories (though wall time is not
  a compute-matched measure).

The harness has since been fixed (counter reset per example, with regression tests). See
[docs/LIMITATIONS.md](docs/LIMITATIONS.md) for the full account.

## What is in here

```
configs/paper/         the exact protocol configs behind the main results
configs/rebuttal/      Gemma-4 arms -- post-rebuttal robustness only
src/precise_uncoupled/ paper-facing API: analysis, protocols, evaluation, io, process
src/agentverse/        the vendored runtime (analysis modules byte-identical to be9c47c9)
data/                  the 4,181-problem corpus, examples, the pinned Track A input, manifests
results/derived_tables/ the released reference tables (checksummed)
scripts/               one script per make target
docs/                  reproducibility, data, methods, metrics, provenance, limitations
tests/                 25 recovered pipeline tests + release integrity guards
```

Start with [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md); to find the artifact behind
a specific number in the paper, use
[docs/PAPER_TO_ARTIFACT_MAP.md](docs/PAPER_TO_ARTIFACT_MAP.md).

## Scope discipline

The **paper** is Omni-MATH only: N=4,181, ten tiers, actor and evaluator both
`openai/gpt-oss-120b`, temperature 0. The 5-dataset × 2-actor matrix is **post-rebuttal
robustness evidence** and is labelled as such everywhere it appears. Please preserve that
distinction if you build on this work.

## Licence and citation

Apache-2.0 (see [LICENSE](LICENSE)). This is a derivative work of
[AgentVerse](https://github.com/OpenBMB/AgentVerse); attribution and the full list of
modifications are in [NOTICE](NOTICE), as Apache-2.0 §4(b) requires. The benchmark derives
from Omni-MATH / Omni-MATH-2 (MIT). No model weights are redistributed.

**Note on the released dataset:** the Hugging Face dataset carries per-source licence terms
including a NonCommercial source (MaScQA, CC-BY-NC-SA-4.0) and a ShareAlike, do-not-train
source (LAB-Bench, CC-BY-SA-4.0). Those terms are inherited — read the dataset card before
redistribution or commercial use. This code repository is Apache-2.0.

```bibtex
@misc{yang2026preciseuncoupled,
  title         = {Precise but Uncoupled: Reviewer Precision Does Not Guarantee
                   Critique Uptake in Multi-Agent Math Reasoning},
  author        = {Yang, Chih-Hsuan and Jiang, Jingyan and Vasudevan, Vikram and
                   Yang, Cheng-Hau and Zheng, Huihuo and Chen, Le and
                   Huerta, Eliu A. and Vishwanath, Venkatram and
                   Foster, Ian T. and Thakur, Rajeev},
  year          = {2026},
  eprint        = {2607.15388},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  doi           = {10.48550/arXiv.2607.15388},
  url           = {https://arxiv.org/abs/2607.15388}
}
```

## Acknowledgments

This research used resources of the Argonne Leadership Computing Facility, a U.S.
Department of Energy (DOE) Office of Science user facility at Argonne National Laboratory
(ANL) operated under Contract No. DE-AC02-06CH11357. The work was also supported under the
same contract by the DOE Office of Science's Advanced Scientific Computing Research Program
and by Laboratory Directed Research and Development (LDRD) funding from ANL, provided by
the Director, DOE Office of Science.
